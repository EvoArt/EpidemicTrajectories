# Fitting the badger bovine-TB model with PracticalBayes.
#
# The parameters split three ways:
#
#   * NUTS for the continuous epidemiological parameters — transmission, the
#     density exponent, progression, the Siler survival curve, and the six tests'
#     sensitivity/specificity;
#   * conjugate Gibbs for the capture probabilities and the initial-state mixing,
#     which have closed forms given X and are cheaper and better-mixing that way
#     than through NUTS;
#   * iFFBS for the hidden trajectory X itself, once per sweep — which is the
#     whole reason this package exists (see CLAUDE.md).
#
# Run:  julia --project=examples examples/badger_fit.jl
# Env:  BADGER_SWEEPS / BADGER_BURN / BADGER_ADAPTS / BADGER_SEED / BADGER_OUT

using EpidemicTrajectories
using PracticalBayes
using Distributions
using Random
using AdvancedHMC: NUTS
import AbstractMCMC
using StableRNGs: StableRNG
using Statistics: mean, std
using LinearAlgebra: I
using Dates
using JLD2: @save

include(joinpath(@__DIR__, "badger_model.jl"))

const DATA_DIR = joinpath(@__DIR__, "..", "badger_ref", "RData2")

b = badger_data(DATA_DIR)
data, raw = b.data, b.raw
const G = raw.n_groups
const NT = raw.n_tests
const NS = raw.n_seasons
const NNU = raw.n_nu_times

println("Badger base model: ", data.n_individuals, " badgers x ", data.n_timepoints,
        " timepoints x ", G, " groups, ", NT, " tests")
println("Brock changepoint fixed at t=", BROCK_CHANGEPOINT, " (not inferred)")

loglik = epidemic_loglik(data)
latent! = epidemic_latent_sampler(data)

## ---------------------------------------------------------------------------
## The latent trajectory
## ---------------------------------------------------------------------------

# One whole-matrix discrete latent: PracticalBayes routes it to the value store,
# so it is AD-constant during the NUTS gradients and updated once per sweep here.
struct TrajectoryLatent <: Distributions.DiscreteMatrixDistribution
    n_time::Int
    n_ind::Int
end
Base.size(d::TrajectoryLatent) = (d.n_time, d.n_ind)
Distributions.logpdf(::TrajectoryLatent, X::AbstractMatrix) = 0.0
Distributions.rand(rng::AbstractRNG, d::TrajectoryLatent) = fill(1, d.n_time, d.n_ind)

# Gather the parameters a rate function needs out of the Gibbs state.
_pars(c) = (; tau=c.values.tau, alpha=c.values.alpha, lambda=c.values.lambda,
            beta=c.values.beta, q=c.values.q,
            a1=c.values.a1, b1=c.values.b1, a2=c.values.a2, b2=c.values.b2,
            c1=c.values.c1,
            thetas=c.values.thetas, rhos=c.values.rhos, phis=c.values.phis,
            etas=c.values.etas, nu=c.values.nu)

struct iFFBSKernel <: PracticalBayes.AbstractLatentKernel
    latent!::Function
end
PracticalBayes.latent_step(rng, k::iFFBSKernel, block_names, c::PracticalBayes.ModelConditional) = begin
    X = copy(c.values.X)
    k.latent!(rng, _pars(c), X)
    (; X=X)
end

## ---------------------------------------------------------------------------
## Conjugate updates
## ---------------------------------------------------------------------------

# Capture probability per season: Beta, from how often a badger that was alive AND
# in a trapped group was actually caught.
struct EtaKernel <: PracticalBayes.AbstractLatentKernel
    a::Float64
    b::Float64
end
function PracticalBayes.latent_step(rng, k::EtaKernel, block_names, c::PracticalBayes.ModelConditional)
    X = c.values.X
    etas = zeros(Float64, NS)
    for s in 1:NS
        caught = 0
        available = 0
        for t in 1:data.n_timepoints
            data.season[t] == s || continue
            for i in 1:data.n_individuals
                g = data.social_group[i, t]
                (g > 0 && data.capt_effort[g, t] == 1) || continue
                X[t, i] == 4 && continue                  # dead: not available
                available += 1
                caught += data.capture[t, i] == 1
            end
        end
        etas[s] = rand(rng, Beta(k.a + caught, k.b + max(available - caught, 0)))
    end
    (; etas=etas)
end

# Initial-state mixing at each nu-time: Dirichlet over the (S, E, I) counts of the
# badgers that were already alive when the study began.
#
# nuE and nuI are two components of ONE simplex — the starting state is
# (1 - nuE - nuI, nuE, nuI), so they must sum to at most 1. Drawing them from
# independent Betas would let the susceptible probability go negative, and the
# likelihood takes its log. The reference draws a Dirichlet triple for exactly
# this reason; so does this kernel, and `NuSimplex` below keeps the prior honest.
struct NuKernel <: PracticalBayes.AbstractLatentKernel
    hp::Vector{Float64}
end
function PracticalBayes.latent_step(rng, k::NuKernel, block_names, c::PracticalBayes.ModelConditional)
    X = c.values.X
    nu = Matrix{Float64}(undef, NNU, 2)     # columns: nuE, nuI
    for (idx, nt) in enumerate(data.nu_times)
        counts = zeros(Int, 3)
        for i in 1:data.n_individuals
            start_time = data.sampling_period[i][1]
            (start_time == nt && data.birth_time[i] < start_time) || continue
            s = X[nt, i]
            s <= 3 && (counts[s] += 1)
        end
        p = rand(rng, Dirichlet(counts .+ k.hp))
        nu[idx, 1], nu[idx, 2] = p[2], p[3]
    end
    (; nu=nu)
end

# The prior over the nu simplex, as one matrix-valued latent (n_nu x 2, holding
# nuE and nuI). It is only ever updated by NuKernel's closed form, so this supplies
# a valid draw and a constant density — the constraint nuE + nuI <= 1 lives here,
# where independent Betas would have broken it.
#
# Declared as a *Discrete* matrix distribution even though its values are
# continuous. That is not a mistake: in PracticalBayes, discrete-valued sites are
# what get routed to the value store, where a latent kernel owns them and NUTS
# never touches them. A continuous declaration would send `nu` to NUTS instead —
# which cannot work, since its conjugate update is the whole point, and NUTS would
# also need an unconstrained transform for a simplex it doesn't know about. The
# same mechanism carries the trajectory `X`.
struct NuSimplex <: Distributions.DiscreteMatrixDistribution
    n_nu::Int
    hp::Vector{Float64}
end
Base.size(d::NuSimplex) = (d.n_nu, 2)
Distributions.logpdf(::NuSimplex, nu::AbstractMatrix) = 0.0
function Distributions.rand(rng::AbstractRNG, d::NuSimplex)
    nu = Matrix{Float64}(undef, d.n_nu, 2)
    for i in 1:d.n_nu
        p = rand(rng, Dirichlet(d.hp))
        nu[i, 1], nu[i, 2] = p[2], p[3]
    end
    nu
end

## ---------------------------------------------------------------------------
## The model — priors exactly as the reference's base model
## ---------------------------------------------------------------------------

@model function badger_base(data, n_time, n_ind, n_groups, n_tests, n_seasons, n_nu, loglik_fn)
    tau ~ Exponential(10.0)                          # progression scale
    alpha ~ PracticalBayes.filldist(Exponential(1.0), n_groups)   # per-group external FOI
    lambda ~ Exponential(1.0)
    beta ~ Exponential(1.0)
    q ~ Beta(1, 1)                                   # density-dependence exponent

    c1 ~ Exponential(1.0)                            # Siler
    a1 ~ Exponential(1.0)
    b1 ~ Exponential(1.0)
    a2 ~ Exponential(1.0)
    b2 ~ Exponential(1.0)

    thetas ~ PracticalBayes.filldist(Beta(1, 1), n_tests)   # sensitivity
    rhos ~ PracticalBayes.filldist(Beta(1, 1), n_tests)     # exposed-state discount
    phis ~ PracticalBayes.filldist(Beta(1, 1), n_tests)     # specificity

    # updated by conjugate kernels, but still declared so they are part of the state
    etas ~ PracticalBayes.filldist(Beta(1, 1), n_seasons)
    nu ~ NuSimplex(n_nu, [8.0, 1.0, 1.0])       # (nuE, nuI) per nu-time, on the simplex

    X ~ TrajectoryLatent(n_time, n_ind)

    pars = (; tau=tau, alpha=alpha, lambda=lambda, beta=beta, q=q,
            a1=a1, b1=b1, a2=a2, b2=b2, c1=c1,
            thetas=thetas, rhos=rhos, phis=phis,
            etas=etas, nu=nu)
    @addlogprob! loglik_fn(pars, data, X)
end

## ---------------------------------------------------------------------------
## Run
## ---------------------------------------------------------------------------

function run_badger_fit(; n_sweeps=parse(Int, get(ENV, "BADGER_SWEEPS", "10000")),
                          n_burn=parse(Int, get(ENV, "BADGER_BURN", "2000")),
                          n_adapts=parse(Int, get(ENV, "BADGER_ADAPTS", "1000")),
                          seed=parse(Int, get(ENV, "BADGER_SEED", "13")))
    m = badger_base(data, data.n_timepoints, data.n_individuals, G, NT, NS, NNU, loglik)

    spl = Gibbs(
        (:tau, :alpha, :lambda, :beta, :q, :c1, :a1, :b1, :a2, :b2,
         :thetas, :rhos, :phis) => NUTS(0.8),
        :etas => EtaKernel(1.0, 1.0),
        :nu => NuKernel([1.0, 1.0, 1.0]),
        :X => iFFBSKernel(latent!),
    )

    init0 = badger_initial_params(raw; rng=StableRNG(seed))
    X0 = copy(raw.X_init)
    init = (; X=X0, tau=init0.tau, alpha=init0.alpha, lambda=init0.lambda,
            beta=init0.beta, q=init0.q, c1=init0.c1, a1=init0.a1, b1=init0.b1,
            a2=init0.a2, b2=init0.b2, thetas=init0.thetas, rhos=init0.rhos,
            phis=init0.phis, etas=init0.etas,
            nu=hcat(init0.nuE, init0.nuI))

    # The aggregates must agree with the starting X before the first likelihood
    # call; the sampler keeps them consistent from then on.
    reset_aggregates!(data)
    apply_derived_summaries!(init0, data, X0)

    println("\nGibbs: NUTS(13 continuous) + conjugate(etas, nu) + iFFBS(X)")
    println("sweeps=$n_sweeps burn=$n_burn adapts=$n_adapts seed=$seed")
    t0 = time()
    chn = AbstractMCMC.sample(StableRNG(seed), m, spl, n_sweeps;
                              init=init, n_adapts=n_adapts, discard_initial=n_burn)
    println("done in ", round((time() - t0) / 60, digits=1), " min")
    chn
end

function report(chn)
    println("\n=== Posterior summary ===")
    for name in (:tau, :lambda, :beta, :q, :c1, :a1, :b1, :a2, :b2)
        v = vec(chn[name])
        println(rpad(string(name), 8), " mean ", rpad(round(mean(v); digits=5), 10),
                " sd ", round(std(v); digits=5))
    end
    for name in (:thetas, :rhos, :phis, :etas)
        m = chn[name]
        println(rpad(string(name), 8), " means ", round.(vec(mean(reduce(hcat, m); dims=2)); digits=3))
    end
end

if get(ENV, "BADGER_RUN", "1") == "1"
    chn = run_badger_fit()
    report(chn)
    outdir = get(ENV, "BADGER_OUT", joinpath(@__DIR__, "outputs"))
    mkpath(outdir)
    path = joinpath(outdir, "badger-$(Dates.format(now(), "yyyymmdd-HHMMSS")).jld2")
    @save path chn
    println("\nsaved: ", path)
end
