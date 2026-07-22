# SILER-FIXED variant — the Siler survival model (run_base_exp.jl / Julia ref)
# WITH all of this session's correctness fixes. This is the fixed analogue of
# badger_fit_reststotal_hmc.jl (which is the original, un-fixed Siler model).
#
# Keeps SILER survival (c1, a1, b1, a2, b2 — the early-life a1/b1 term is present,
# unlike the Gompertz variant). Applies the four fixes verified this session:
#   * mean-time progression 1-exp(-1/tau) (was the inverted rate 1-exp(-tau));
#   * survival evaluated at the DESTINATION time t+1 (was source t);
#   * death forbidden before LAST capture via the OBSERVATION (filter), so
#     survival is pure Siler (no capture gate);
#   * ENTRY CONDITIONING — likelihood loops from FIRST capture, charging real
#     survival over [first, last] instead of crediting it for free.
#
# tau prior: CONFIGURABLE via BADGER_TAU_PRIOR (default 100, the C++ scale; set
# to 10 for run_base_exp.jl's Exp(10)). nu ~ Dirichlet(1,1,1). All else as the
# base model.
#
# Fitting the badger bovine-TB model with PracticalBayes, using:
#   * the `rest_contribution` power-user coupling term (badger_model_reststotal.jl,
#     the O(n_states)-per-lookup running-total shape — the package's fastest
#     iFFBS sweep as of 2026-07-19), and
#   * a FIXED-epsilon HMC kernel (AdvancedHMC's `HMC`, not `NUTS`) for the
#     continuous parameters, with per-parameter step sizes copied verbatim from
#     badger_ref/run_base_exp.jl's `model` tuple (4th element per entry), and
#     L=30 leapfrog steps — matching that reference run's `HMCSampler(L=30)`
#     exactly. No step-size/mass adaptation (the reference's HMCSampler has
#     none either); n_adapts=0.
#
#     AdvancedHMC's `HMC(ϵ, L)` only accepts a SCALAR ϵ, but per-parameter step
#     sizes are reproduced exactly via the mass matrix instead: the leapfrog
#     position update is `θ += ϵ .* (M⁻¹ .* r)` with momentum refreshed as
#     `r ~ N(0, M)`. Fixing the scalar `ϵ = 1` and setting the diagonal
#     `M⁻¹_i = eps_i^2` (`eps_i` = the reference's per-parameter step size)
#     reproduces the reference's unit-mass, per-parameter-ϵ leapfrog step for
#     step exactly (verified: identical trajectories under matched random
#     draws) — no monkey-patching of AdvancedHMC internals needed.
#
# Conjugate Gibbs (etas, nu) and iFFBS (X) blocks are unchanged from badger_fit.jl.
#
# AD backend: `AutoPolyesterForwardDiff()`, matching badger_ref/run_base_exp.jl's
# own `backend = AutoPolyesterForwardDiff(; chunksize=nothing, tag=nothing)` —
# threaded chunked ForwardDiff, rather than PracticalBayes's plain-ForwardDiff
# default.
#
# Run (detached): times a 100-sweep run, then immediately launches a 500-sweep
# run in the same process.
#   julia --project=examples examples/badger_fit_reststotal_hmc.jl
# Env:  BADGER_BURN / BADGER_SEED / BADGER_OUT

using EpidemicTrajectories
using PracticalBayes
using Distributions
using Random
using AdvancedHMC: HMC, Leapfrog, DiagEuclideanMetric
using ADTypes: AutoPolyesterForwardDiff
using PolyesterForwardDiff: PolyesterForwardDiff
import AbstractMCMC
using StableRNGs: StableRNG
using Statistics: mean, std, median
using Dates
using JLD2: @save

const ADTYPE = AutoPolyesterForwardDiff(; chunksize=nothing, tag=nothing)

include(joinpath(@__DIR__, "badger_model_reststotal.jl"))
include(joinpath(@__DIR__, "badger_model_obssplit.jl"))
include(joinpath(@__DIR__, "badger_model_gompertz.jl"))   # Gompertz-Makeham survival
include(joinpath(@__DIR__, "badger_progression_meantime.jl"))  # corrected E->I (mean-time tau)

# The badger CSVs (RData2). Defaults to the in-repo dev location, but override
# with BADGER_DATA_DIR when running somewhere the repo's `badger_ref/` isn't
# present (e.g. a server that has its own copy of RData2). Only RData2 is needed
# — `load_badger_data` reads nothing from WPbadgerData.
const DATA_DIR = get(ENV, "BADGER_DATA_DIR", joinpath(@__DIR__, "..", "badger_ref", "RData2"))

# The FACTORED observation process (capture x tests). Identical weights to
# badger_data_reststotal's — verified to 1.1e-16 over 186,850 cells — but split
# into two factors so the test factor alone can go into the likelihood while
# `etas` stays in its conjugate Gibbs block. See badger_model_obssplit.jl.
# The FILTER's observation process is `badger_observations_deathban`: the capture
# factor gives state D weight 0 for every t up to last_capture (forbidding death
# before last capture in the iFFBS trajectory), times the test factor. Survival is
# pure Gompertz; the death constraints live in the observation (filter, last
# capture) and the likelihood entry bound (first capture) instead.
b = badger_data_obssplit(DATA_DIR;
                         trans_mat=badger_transitions_meantime_siler_tp1(),
                         observation_process=badger_observations_deathban)
data, raw = b.data, b.raw
const G = raw.n_groups
const NT = raw.n_tests
const NS = raw.n_seasons
const NNU = raw.n_nu_times

println("Badger base model (reststotal coupling): ", data.n_individuals, " badgers x ",
        data.n_timepoints, " timepoints x ", G, " groups, ", NT, " tests")
println("Brock changepoint fixed at t=", BROCK_CHANGEPOINT, " (not inferred)")

# TWO likelihood terms, summed in @addlogprob!:
#
#  * `loglik`     — starting state + transitions (epidemic/survival parameters).
#  * `obs_loglik` — the TEST factor of the observation process, which is what
#    informs thetas/rhos/phis.
#
# The observation term was MISSING before 2026-07-20. Without it, halving
# thetas/rhos/phis/etas changed the log density by exactly 0.000e+00 — those
# parameters were sampled from their priors, since `observation_process` is
# otherwise used only by the iFFBS forward filter and never enters the
# differentiated density. That was a model bug, not a design choice.
#
# `observation_weight` supplies the ALLOCATION-FREE scalar path: the likelihood
# needs one entry of the weight vector, and going via the vector allocates ~18 MB
# per call (~187k arrays of Duals). Measured: gradient 0.226 -> 0.075 s (3.0x),
# i.e. this is what makes the correct model cost ~14% more than the wrong one
# rather than ~3x more. Verified exact vs the vector path (0.000e+00 across all
# 747,400 (i,t,s) entries).
#
# `etas` is NOT in this term — it stays a conjugate Gibbs block, and the test
# factor is provably independent of it (checked: 0.000e+00), so nothing is
# double-counted.
# ENTRY CONDITIONING (survival-only gate, matching posterior.jl:145-150). The
# likelihood loops over the WHOLE sampling window, but before FIRST capture it
# divides the SURVIVAL factor back out — scoring the infection (S->E) and
# progression (E->I) transitions (the epidemic was happening whether or not we
# were watching) while NOT charging survival (the badger is known alive at entry).
# From first capture on, full survival is charged. `survival=` must be the same
# function the transitions use. (An earlier version SKIPPED the pre-entry window
# entirely, dropping ~9452 S->E/E->I terms and biasing beta/tau/q — see
# progression_bug.md.)
loglik = epidemic_loglik(data; entry_time=raw.first_capture_time,
                               survival=siler_survival_tp1)
obs_loglik = epidemic_obs_loglik(data;
                                 observation_process=badger_obs_tests,
                                 observation_weight=badger_obs_tests_weight)
latent! = epidemic_latent_sampler(data)

## ---------------------------------------------------------------------------
## The latent trajectory
## ---------------------------------------------------------------------------

struct TrajectoryLatent <: Distributions.DiscreteMatrixDistribution
    n_time::Int
    n_ind::Int
end
Base.size(d::TrajectoryLatent) = (d.n_time, d.n_ind)
Distributions.logpdf(::TrajectoryLatent, X::AbstractMatrix) = 0.0
Distributions.rand(rng::AbstractRNG, d::TrajectoryLatent) = fill(1, d.n_time, d.n_ind)

_pars(c) = (; tau=c.values.tau, alpha=c.values.alpha, lambda=c.values.lambda,
            beta=c.values.beta, q=c.values.q,
            a1=c.values.a1, b1=c.values.b1, a2=c.values.a2, b2=c.values.b2, c1=c.values.c1,
            thetas=c.values.thetas, rhos=c.values.rhos, phis=c.values.phis,
            etas=c.values.etas, nu=c.values.nu)

# `latent!` is type-PARAMETERISED, not declared `::Function`. An abstract field
# type boxes the closure and makes every `k.latent!(...)` a dynamic dispatch;
# parameterising it keeps the call concrete. (The call happens once per sweep, so
# unlike the EtaKernel fix this is not a hot-loop win — but it costs nothing and
# removes an inference barrier at the entry to the 26%-of-sweep iFFBS block.)
struct iFFBSKernel{F} <: PracticalBayes.AbstractLatentKernel
    latent!::F
end
PracticalBayes.latent_step(rng, k::iFFBSKernel, block_names, c::PracticalBayes.ModelConditional) = begin
    X = copy(c.values.X)
    k.latent!(rng, _pars(c), X)
    (; X=X)
end

## ---------------------------------------------------------------------------
## Conjugate updates (identical to badger_fit.jl)
## ---------------------------------------------------------------------------

# The kernel CARRIES its data in a type-parameterised field rather than reading
# the module-level `data` binding. That binding (line ~57, `data, raw = b.data,
# b.raw`) is a non-const global, so every `data.season[t]` / `data.social_group[i,t]`
# inside the kernel body was a dynamic lookup returning `Any` — the whole
# ~1.5M-iteration count loop ran untyped.
#
# Measured (examples/bench_eta_overhead.jl), same loop, globals vs argument:
#     globals   : 0.6114 s, 164,109,040 bytes allocated
#     arguments : 0.0032 s,         224 bytes allocated   -> 189x
#
# This block was 11.7% of a sweep (0.541 s) purely from that. Exactly the trap
# CLAUDE.md's performance section describes — `data.season` reports the right
# type either way, so only a benchmark reveals it.
#
# The count loop is ALSO restructured: season is a function of `t` alone, so one
# pass over (i, t) can bucket straight into caught[season[t]] / available[season[t]],
# instead of running NS full passes that each discard ~75% of the grid via
# `season[t] == s || continue`. Verified bit-identical counts
# (caught=[1579,3466,4463,2599], available=[7856,8802,8667,8156]) — see
# examples/bench_eta.jl.
struct EtaKernel{D} <: PracticalBayes.AbstractLatentKernel
    a::Float64
    b::Float64
    data::D
end
EtaKernel(a, b) = EtaKernel(a, b, data)   # convenience: capture the current `data`

function PracticalBayes.latent_step(rng, k::EtaKernel, block_names, c::PracticalBayes.ModelConditional)
    X = c.values.X
    d = k.data                      # typed field, not the untyped global
    caught = zeros(Int, NS)
    available = zeros(Int, NS)
    @inbounds for i in 1:d.n_individuals
        for t in 1:d.n_timepoints
            g = d.social_group[i, t]
            (g > 0 && d.capt_effort[g, t] == 1) || continue
            X[t, i] == 4 && continue                  # dead: not available
            s = d.season[t]
            available[s] += 1
            caught[s] += d.capture[t, i] == 1
        end
    end
    etas = zeros(Float64, NS)
    for s in 1:NS
        etas[s] = rand(rng, Beta(k.a + caught[s], k.b + max(available[s] - caught[s], 0)))
    end
    (; etas=etas)
end

# Carries `data` in a typed field for the same reason as EtaKernel above — this
# loop is far smaller (nu_times x n_individuals, not NS x T x N), so it measured
# only 0.022 s/sweep, but the untyped-global defect is identical and the fix free.
struct NuKernel{D} <: PracticalBayes.AbstractLatentKernel
    hp::Vector{Float64}
    data::D
end
NuKernel(hp) = NuKernel(hp, data)

function PracticalBayes.latent_step(rng, k::NuKernel, block_names, c::PracticalBayes.ModelConditional)
    X = c.values.X
    d = k.data
    nu = Matrix{Float64}(undef, NNU, 2)     # columns: nuE, nuI
    for (idx, nt) in enumerate(d.nu_times)
        counts = zeros(Int, 3)
        @inbounds for i in 1:d.n_individuals
            start_time = d.sampling_period[i][1]
            (start_time == nt && d.birth_time[i] < start_time) || continue
            s = X[nt, i]
            s <= 3 && (counts[s] += 1)
        end
        p = rand(rng, Dirichlet(counts .+ k.hp))
        nu[idx, 1], nu[idx, 2] = p[2], p[3]
    end
    (; nu=nu)
end

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
## The model — priors matched to the C++ reference (classic rcpp)
## ---------------------------------------------------------------------------
#
# Prior audit vs the C++ (runmodel.R hp_* values + logPost_HMC.cpp forms; each
# C++ prior is a log-density written in unconstrained space, decoded here):
#
#   param   C++ prior                          ours (this file)
#   ------  ---------------------------------  -----------------
#   tau     Gamma(1, scale=1/0.01=100)         Exponential(100)   [was Exp(10)]
#   alpha   -exp(logPar)+logPar = Exp(rate 1)  Exponential(1)
#   lambda  hp=(1,1): -lambda+log = Exp(1)     Exponential(1)
#   beta    Gamma(1, scale=1) = Exp(1)         Exponential(1)
#   q       Beta(1,1) in logit space           Beta(1,1)
#   c1      Gamma(1, scale=1) = Exp(1)         Exponential(1)
#   a2      Gamma(1, scale=1) = Exp(1)         Exponential(1)
#   b2      Gamma(1, scale=1) = Exp(1)         Exponential(1)
#   a1,b1   ABSENT (Gompertz-Makeham, no       DROPPED
#           early-life term)
#   thetas  Beta(1,1)                          Beta(1,1)
#   rhos    Beta(1,1)                          Beta(1,1)
#   phis    Beta(1,1)                          Beta(1,1)
#   etas    Beta(1,1)                          Beta(1,1)
#   nu      Dirichlet(1,1,1)                   Dirichlet(1,1,1)   [was (8,1,1)]
#
# `Distributions.Exponential` takes the SCALE, so Exponential(100) has mean 100 —
# matching the C++'s Gamma(shape=1, scale=100). NuSimplex takes the Dirichlet
# concentration vector directly.

@model function badger_base(data, n_time, n_ind, n_groups, n_tests, n_seasons, n_nu, loglik_fn, obs_loglik_fn)
    tau ~ Exponential(TAU_PRIOR_SCALE)               # progression MEAN TIME; configurable (default 100)
    alpha ~ PracticalBayes.filldist(Exponential(1.0), n_groups)   # per-group external FOI
    lambda ~ Exponential(1.0)
    beta ~ Exponential(1.0)
    q ~ Beta(1, 1)                                   # density-dependence exponent

    c1 ~ Exponential(1.0)                            # Siler: constant hazard (Makeham)
    a1 ~ Exponential(1.0)                            # Siler: early-life scale
    b1 ~ Exponential(1.0)                            # Siler: early-life rate
    a2 ~ Exponential(1.0)                            # Siler: late-life scale
    b2 ~ Exponential(1.0)                            # Siler: late-life rate

    thetas ~ PracticalBayes.filldist(Beta(1, 1), n_tests)   # sensitivity
    rhos ~ PracticalBayes.filldist(Beta(1, 1), n_tests)     # exposed-state discount
    phis ~ PracticalBayes.filldist(Beta(1, 1), n_tests)     # specificity

    etas ~ PracticalBayes.filldist(Beta(1, 1), n_seasons)
    nu ~ NuSimplex(n_nu, [1.0, 1.0, 1.0])       # (nuE, nuI) per nu-time; C++ hp_nu=(1,1,1)

    X ~ TrajectoryLatent(n_time, n_ind)

    pars = (; tau=tau, alpha=alpha, lambda=lambda, beta=beta, q=q,
            a1=a1, b1=b1, a2=a2, b2=b2, c1=c1,
            thetas=thetas, rhos=rhos, phis=phis,
            etas=etas, nu=nu)
    # Transitions + observations. The second term is what informs
    # thetas/rhos/phis; without it they are sampled from their priors (verified:
    # halving them changed the density by exactly 0.000e+00). See the comment at
    # the top of this file.
    @addlogprob! loglik_fn(pars, data, X) + obs_loglik_fn(pars, data, X)
end

## ---------------------------------------------------------------------------
## Fixed-epsilon HMC block, matching badger_ref/run_base_exp.jl's model tuple
## ---------------------------------------------------------------------------
#
# run_base_exp.jl's `model` NamedTuple (4th element per entry = per-parameter
# epsilon, on the SAME log/logit-unconstrained scale PracticalBayes's flat
# vector uses — confirmed via badger_ref's get_transformations: log for
# positive support (Exponential priors), logit for unit support (Beta
# priors), i.e. exactly Bijectors' defaults):
#
#   progression_scale (tau) = 0.002        thetas = 0.005 (x NT)
#   alpha              = 0.2  (x G)        rhos   = 0.005 (x NT)
#   lambda             = 0.01              phis   = 0.005 (x NT)
#   beta               = 0.05
#   q                  = 0.05
#   c1                 = 0.005
#   a2 = b2            = 0.001    (a1/b1 dropped — Gompertz-Makeham has neither)
#
# These per-parameter step sizes are kept from run_base_exp.jl (only a1/b1
# removed), NOT switched to the C++ runmodel.R's epsilons, deliberately: changing
# the survival function AND the step sizes at once would confound any comparison.
# Step size is a sampler-efficiency knob, not part of the target posterior.
#
# Flat-vector order must match the block's `names` tuple below (PracticalBayes
# lays out the flat vector in model-declaration order, restricted to the
# block's own names): tau, alpha, lambda, beta, q, c1, a2, b2, thetas, rhos, phis.
# Trajectory length: FIXED at 15 — the reference's EXPECTED length, not its
# nominal L=30.
#
# badger_ref's HMC_.cpp:71 draws `int intL = ceil(runif(0,1) * L)` with L=30 —
# uniform on {1,...,30}, mean 15.5 — then does `intL - 1` gradient calls inside
# the loop plus one half-step either side, i.e. `intL + 1` ~= 16.5 gradients per
# HMC step. A FIXED L=30 does 30, i.e. 1.82x the reference's gradient work for no
# added fidelity. (Verified by direct simulation of `ceil(rand()*30)`.)
#
# We use a FIXED L = 15, the reference's expected trajectory length.
#
# A randomised-L kernel IS buildable from public API — `Trajectory`, `HMCKernel`,
# `FixedNSteps`, `EndPointTS` are all exported, and `HMCSampler(κ, metric,
# adaptor)` passes any kernel straight through (`make_kernel(spl::HMCSampler, _)
# = spl.κ`), so `HMC`/`NUTS` really are just conveniences over that. It was
# implemented and then REVERTED, for a reason worth recording:
#
#   `AdvancedHMC.nsteps(τ)` takes no RNG and is called TWICE per transition —
#   once for the actual trajectory (trajectory.jl:337) and once for the reported
#   `n_steps` stat (:288). Drawing independently in each would simulate a
#   different L than it reports. The workaround is a `Ref{Int}` inside the
#   termination criterion, redrawn from the momentum-refreshment hook (the one
#   place per transition that does receive the RNG). That works, but it puts
#   MUTABLE STATE in the sampler struct and depends on `refresh` being called
#   exactly once per transition — an ordering assumption about a package we do
#   not control. If a future version broke it, the sampler would silently
#   simulate a different L than it reports, with no error.
#
# For a 5000-sweep production run that fragility is not worth the difference
# between L~Uniform{1..30} and a fixed L=15: same expected gradient count, and
# the mean is what drives the cost. The tradeoff is that a fixed L can hit
# resonance in some geometries where a randomised one would not — if the chain
# ever shows that pathology, the randomised kernel is the fix, not a larger L.
const HMC_L = parse(Int, get(ENV, "BADGER_HMC_L", "15"))
# tau prior scale (mean E->I time). Configurable; default 100 (the C++ value).
# Set BADGER_TAU_PRIOR to override (e.g. 10 to match run_base_exp.jl's Exp(10)).
const TAU_PRIOR_SCALE = parse(Float64, get(ENV, "BADGER_TAU_PRIOR", "100.0"))
const HMC_EPS = vcat(
    0.002,                  # tau
    fill(0.2, G),           # alpha
    0.01,                   # lambda
    0.05,                   # beta
    0.05,                   # q
    0.005,                  # c1
    # a1/b1 (Siler early-life) were never in the original manuscript, so these
    # step sizes are a GUESS, not tuned. They are weakly identified (no age-1/2
    # death data — see progression_bug.md), and base_exp explores a large range
    # (a1~0.44, b1~2.36), so give them wider steps than a2/b2. Override via a mass
    # matrix once PracticalBayes can learn one.
    0.02,                   # a1  (wider — weakly identified, large posterior range)
    0.05,                   # b1  (wider — base_exp b1~2.4)
    0.001,                  # a2
    0.001,                  # b2
    fill(0.005, NT),         # thetas
    fill(0.005, NT),         # rhos
    fill(0.005, NT),         # phis
)

# AdvancedHMC's `HMC(ϵ, L)` only accepts a SCALAR ϵ. Per-parameter step sizes
# are instead reproduced through the (fixed, un-adapted) diagonal mass matrix:
# the leapfrog position update is `θ += ϵ .* (M⁻¹ .* r)` with momentum
# refreshed each step as `r ~ N(0, M)`. Fixing the scalar `ϵ = 1` and setting
# `M⁻¹_i = eps_i^2` reproduces the reference's unit-mass, per-parameter-ϵ
# leapfrog step exactly (verified empirically: identical trajectories under
# matched random draws). `HMC`'s `make_adaptor` returns `NoAdaptation()`
# unconditionally, so this mass matrix — like the reference's — is never
# touched after construction.
function make_hmc_block(eps_vec, L)
    HMC(L; integrator=Leapfrog(1.0), metric=DiagEuclideanMetric(eps_vec .^ 2))
end

## ---------------------------------------------------------------------------
## Run
## ---------------------------------------------------------------------------

function run_badger_fit(n_sweeps; n_burn=0, seed=13)
    m = badger_base(data, data.n_timepoints, data.n_individuals, G, NT, NS, NNU,
                    loglik, obs_loglik)

    hmc_kernel = make_hmc_block(HMC_EPS, HMC_L)

    # BLOCKING: ONE HMC block for all 61 continuous parameters, plus conjugate
    # Gibbs for etas/nu and iFFBS for X. Benchmarked (50 sweeps each,
    # examples/bench_obs_variants.jl, results in perf_gap_log.md):
    #
    #   this (1 HMC block, conjugate etas)     3.682 s/sweep
    #   everything in HMC incl. etas           3.663 s/sweep
    #   test params in a SECOND HMC block      4.527 s/sweep   <- 23% SLOWER
    #
    # Do NOT split the test parameters into their own block. It mirrors the C++
    # reference's grad_/gradThetasRhos split, but that only works there because
    # those are hand-written gradients over disjoint data. PracticalBayes
    # evaluates the WHOLE model body for every block, so two HMC blocks means two
    # full primal evaluations AND two independent L=30 leapfrog trajectories —
    # which costs far more than the saved AD partials (61 -> 43 + 18).
    #
    # Keeping etas/nu CONJUGATE rather than folding them into HMC is a wash on
    # wall-clock (3.682 vs 3.663 s/sweep, within noise) but should mix better: a
    # conjugate draw is an exact independent sample from the correct conditional,
    # whereas an HMC step is a correlated move with a hand-set, unadapted step
    # size. Wall-clock is the wrong metric for that choice; ESS per second would
    # be the right one, and has NOT been measured.
    spl = Gibbs(
        (:tau, :alpha, :lambda, :beta, :q, :c1, :a1, :b1, :a2, :b2,
         :thetas, :rhos, :phis) => hmc_kernel,
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

    reset_aggregates!(data)
    apply_derived_summaries!(init0, data, X0)

    # The latent trajectory X is a 161×2384 Int matrix — ~3 MB PER SWEEP. Keeping
    # every draw in the chain is what exhausted memory on the earlier 5000-sweep
    # run (~15 GB projected). `save_states` keeps X live for conditioning (so
    # iFFBS/the likelihood are unaffected) but controls what happens to the OUTPUT.
    # `BADGER_X_SAVE` selects the mode:
    #   "disk"   (default): stream X to `X_iters_x_to_y.jld2` chunks every
    #            `BADGER_X_FLUSH` sweeps; recover later with `read_states(x_path)`.
    #   "buffer"          : drop X from the output entirely (no chain, no disk) —
    #            the memory-lightest option, for when only the parameters matter.
    #   "chain"           : the old behaviour (retain every X in the chain) — will
    #            blow up memory at large n_sweeps; here only for A/B timing.
    outdir = get(ENV, "BADGER_OUT", joinpath(@__DIR__, "outputs"))
    mkpath(outdir)
    x_path = joinpath(outdir, "badger-siler-fixed-hmc-X-$(n_sweeps)iter.jld2")
    x_flush_every = parse(Int, get(ENV, "BADGER_X_FLUSH", "500"))
    x_save = get(ENV, "BADGER_X_SAVE", "disk")
    x_disp = x_save == "disk"   ? (x_path, x_flush_every) :
             x_save == "buffer" ? :buffer :
             x_save == "chain"  ? :chain :
             error("BADGER_X_SAVE must be \"disk\", \"buffer\", or \"chain\", got \"$x_save\"")

    println("\nGibbs: fixed-eps HMC(L=$HMC_L, Siler survival (fixed), configurable tau prior) ",
            "(13 continuous, Siler a1/b1) + conjugate(etas, nu) + iFFBS(X, reststotal coupling)")
    println("adtype=", ADTYPE)
    println("sweeps=$n_sweeps burn=$n_burn seed=$seed")
    println("X save mode: $x_save", x_save == "disk" ? " (every $x_flush_every sweeps -> $x_path)" : "")
    t0 = time()
    chn = AbstractMCMC.sample(StableRNG(seed), m, spl, n_sweeps;
                              init=init, adtype=ADTYPE, n_adapts=0, discard_initial=n_burn,
                              save_states=(X=x_disp,))
    elapsed = time() - t0
    println("done in ", round(elapsed / 60, digits=1), " min (", round(elapsed, digits=1), " s)")
    chn, elapsed
end

function report(chn)
    println("\n=== Posterior summary ===")
    for name in (:tau, :lambda, :beta, :q, :c1, :a1, :b1, :a2, :b2)
        v = vec(chn[name])
        println(rpad(string(name), 8), " mean ", rpad(round(mean(v); digits=5), 10),
                " sd ", round(std(v); digits=5))
    end
    for name in (:thetas, :rhos, :phis, :etas)
        mm = chn[name]
        println(rpad(string(name), 8), " means ", round.(vec(mean(reduce(hcat, mm); dims=2)); digits=3))
    end
    # alpha is a per-group vector (length G) sampled in the HMC block. Printing
    # all G means would flood the log, so summarise the posterior means across
    # groups and list the first few — the full per-group draws are always in the
    # chain (`chn[:alpha]`); X_SAVE only governs the latent X, not the parameters.
    # NOTE: alpha and lambda are non-identifiable up to a shared scale — the group
    # FOI is `lambda * alpha[g]`, so only the product enters the likelihood. If
    # lambda and the alpha means drift together across the chain, that is the
    # scale ridge, not poor mixing (the reference has the same structure).
    am = vec(mean(reduce(hcat, chn[:alpha]); dims=2))
    println(rpad("alpha", 8), " ", length(am), " groups; posterior means: ",
            "min ", round(minimum(am); digits=4), " median ", round(median(am); digits=4),
            " max ", round(maximum(am); digits=4))
    println(rpad("", 8), " first 5: ", round.(am[1:min(5, length(am))]; digits=4))
end

function save_run(chn, elapsed, n_sweeps; tag)
    outdir = get(ENV, "BADGER_OUT", joinpath(@__DIR__, "outputs"))
    mkpath(outdir)
    stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    path = joinpath(outdir, "badger-siler-fixed-hmc-$tag-$stamp.jld2")
    @save path chn elapsed
    timing_path = joinpath(outdir, "badger-siler-fixed-hmc-$tag-$stamp-timing.txt")
    open(timing_path, "w") do io
        println(io, "n_sweeps=", n_sweeps)
        println(io, "elapsed_seconds=", elapsed)
        println(io, "elapsed_minutes=", elapsed / 60)
        println(io, "elapsed_hours=", elapsed / 3600)
    end
    println("saved: ", path)
    println("timing: ", timing_path)
    path
end

if get(ENV, "BADGER_RUN", "1") == "1"
    n_burn = parse(Int, get(ENV, "BADGER_BURN", "0"))
    seed = parse(Int, get(ENV, "BADGER_SEED", "13"))
    compile_sweeps = parse(Int, get(ENV, "BADGER_COMPILE_SWEEPS", "1"))
    warmup_sweeps = parse(Int, get(ENV, "BADGER_WARMUP_SWEEPS", "100"))
    main_sweeps = parse(Int, get(ENV, "BADGER_MAIN_SWEEPS", "500"))

    # An untimed throwaway pass first: the FIRST sweep JIT-compiles the whole
    # pipeline (loglik, gradient, iFFBS, kernels), which would otherwise land
    # entirely inside the 100-sweep timing and inflate its per-sweep figure. We
    # discard this pass's result — it exists only to pay the compilation cost
    # before the clock starts. Set BADGER_COMPILE_SWEEPS=0 to skip it.
    if compile_sweeps > 0
        println("\n########## Compile pass (untimed): $compile_sweeps sweep(s) ##########")
        run_badger_fit(compile_sweeps; n_burn=0, seed=seed)
    end

    println("\n########## Timed run: $warmup_sweeps sweeps ##########")
    chn_w, elapsed_w = run_badger_fit(warmup_sweeps; n_burn=n_burn, seed=seed)
    report(chn_w)
    save_run(chn_w, elapsed_w, warmup_sweeps; tag="$(warmup_sweeps)iter")
    println("$warmup_sweeps-sweep timing: ", round(elapsed_w, digits=1), " s (",
            round(elapsed_w / 60, digits=2), " min); ",
            round(elapsed_w / warmup_sweeps, digits=3), " s/sweep")

    println("\n########## Main run: $main_sweeps sweeps ##########")
    chn_m, elapsed_m = run_badger_fit(main_sweeps; n_burn=n_burn, seed=seed)
    report(chn_m)
    save_run(chn_m, elapsed_m, main_sweeps; tag="$(main_sweeps)iter")
    println("$main_sweeps-sweep timing: ", round(elapsed_m, digits=1), " s (",
            round(elapsed_m / 60, digits=2), " min); ",
            round(elapsed_m / main_sweeps, digits=3), " s/sweep")
end
