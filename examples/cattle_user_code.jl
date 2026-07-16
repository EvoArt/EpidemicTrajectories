# The cattle E. coli model (Touloupou et al. 2019), written entirely as user code
# against EpidemicTrajectories. Everything below the imports is what a user writes.
#
# Two states, S and I, with recurrent S<->I: an animal is infected at a rate
# driven by how many of its penmates are infected, and recovers back to
# susceptible. Neither state is ever observed directly — only two imperfect
# diagnostic tests, on a subset of days.

using Random
using Distributions
using EpidemicTrajectories
using PracticalBayes
using AdvancedHMC: NUTS
import AbstractMCMC
using StableRNGs: StableRNG
using Statistics: mean, std

## ---------------------------------------------------------------------------
## The model
## ---------------------------------------------------------------------------

n_pens = 10
n_per_pen = 8
n_timepoints = 80
n_individuals = n_pens * n_per_pen
group = repeat(1:n_pens; inner=n_per_pen)
observed_days = collect(1:6:n_timepoints)

# Declaring the state space fixes the numbering, so the package and I agree on
# which state is which: S is 1, I is 2.
state_space = [:S, :I]

# The array I want tracked during the latent update, declared together with how
# it updates. The package allocates it and knows nothing about what it means; it
# just runs this update forwards and, when the sampler needs to, in reverse.
aggs = @aggregate state_space begin
    @array n_infected Int (n_pens, n_timepoints)
    n_infected[data.group[i], t] += (state == :I)
end

# The rates read that array. `n_infected` excludes the individual being resampled
# while it is being resampled, which is exactly the leave-one-out count the force
# of infection needs.
function infection_func(model, data, i, t)
    I_minus = data.aggregates[:n_infected][data.group[i], t]
    -expm1(-(model.α + model.β * I_minus))
end

recovery_func(model, data, i, t) = 1 / model.m

trans_mat = @transitions state_space begin
    S -> I = infection_func
    I -> S = recovery_func
end

starting_state = (model, data, X, i, t) -> [1 - model.ν, model.ν]

## ---------------------------------------------------------------------------
## Simulate data from known parameters
## ---------------------------------------------------------------------------

true_pars = (; α=0.01, β=0.02, m=6.0, ν=0.10, θʳ=0.8, θᶠ=0.5)

no_tests = [fill(-1, n_timepoints, n_individuals), fill(-1, n_timepoints, n_individuals)]
sim_data = epidemic_data(
    n_individuals=n_individuals,
    n_timepoints=n_timepoints,
    group=group,
    trans_mat=trans_mat,
    starting_state=starting_state,
    test_mats=no_tests,
    aggregates=aggs,
)

simulate = epidemic_simulator(sim_data)
rng = StableRNG(2024)
X_true = simulate(rng, true_pars)

function simulate_observations(rng, X; θʳ, θᶠ, observed_days)
    n_timepoints, n_individuals = size(X)
    R = fill(-1, n_timepoints, n_individuals)
    F = fill(-1, n_timepoints, n_individuals)
    for t in observed_days, i in 1:n_individuals
        infected = X[t, i] == 2
        R[t, i] = rand(rng) < (infected ? θʳ : 0.0) ? 1 : 0
        F[t, i] = rand(rng) < (infected ? θᶠ : 0.0) ? 1 : 0
    end
    R, F
end

Rmask, Fmask = simulate_observations(rng, X_true; θʳ=true_pars.θʳ, θᶠ=true_pars.θᶠ, observed_days=observed_days)

## ---------------------------------------------------------------------------
## Fit
## ---------------------------------------------------------------------------

fit_data = epidemic_data(
    n_individuals=n_individuals,
    n_timepoints=n_timepoints,
    group=group,
    trans_mat=trans_mat,
    starting_state=starting_state,
    test_mats=[Rmask, Fmask],
    aggregates=aggs,
)

loglik = epidemic_loglik(fit_data)
latent! = epidemic_latent_sampler(fit_data)

# The whole trajectory is one discrete latent block: PracticalBayes routes it to
# the value store, so it is held constant during the NUTS gradients and updated
# once per Gibbs sweep by the iFFBS kernel below.
struct TrajectoryLatent <: Distributions.DiscreteMatrixDistribution
    n_time::Int
    n_ind::Int
end
Base.size(d::TrajectoryLatent) = (d.n_time, d.n_ind)
Distributions.logpdf(::TrajectoryLatent, X::AbstractMatrix) = 0.0
Distributions.rand(rng::AbstractRNG, d::TrajectoryLatent) = fill(1, d.n_time, d.n_ind)

struct iFFBSKernel <: PracticalBayes.AbstractLatentKernel
    latent!::Function
end

PracticalBayes.latent_step(rng, k::iFFBSKernel, block_names, c::PracticalBayes.ModelConditional) = begin
    block_names == (:X,) || error("iFFBS kernel only handles :X")
    pars = (; α=c.values.α, β=c.values.β, m=c.values.m̃ + 1.0, ν=c.values.ν, θʳ=c.values.θʳ, θᶠ=c.values.θᶠ)
    X = copy(c.values.X)
    k.latent!(rng, pars, X)
    (; X=X)
end

# The initial infection frequency and the two test sensitivities have closed-form
# conditional posteriors, so they get conjugate kernels rather than NUTS.
struct NuKernel <: PracticalBayes.AbstractLatentKernel
    a::Float64
    b::Float64
end

PracticalBayes.latent_step(rng, k::NuKernel, block_names, c::PracticalBayes.ModelConditional) = begin
    X = c.values.X
    n_inf = count(==(2), @view X[1, :])
    (; ν=rand(rng, Beta(k.a + n_inf, k.b + size(X, 2) - n_inf)))
end

struct TestSensKernel <: PracticalBayes.AbstractLatentKernel
    a::Float64
    b::Float64
    Y::Matrix{Int}
    name::Symbol
end

PracticalBayes.latent_step(rng, k::TestSensKernel, block_names, c::PracticalBayes.ModelConditional) = begin
    X = c.values.X
    n_pos = 0
    n_tested_inf = 0
    for t in axes(k.Y, 1), i in axes(k.Y, 2)
        y = k.Y[t, i]
        y < 0 && continue
        if X[t, i] == 2
            n_tested_inf += 1
            y == 1 && (n_pos += 1)
        end
    end
    draw = rand(rng, Beta(k.a + n_pos, k.b + (n_tested_inf - n_pos)))
    NamedTuple{(k.name,)}((draw,))
end

@model function cattle_model(data, n_time, n_ind, loglik_fn)
    α ~ Gamma(1, 1)
    β ~ Gamma(1, 1)
    m̃ ~ Gamma(2, 4)
    m := m̃ + 1.0

    ν ~ Beta(1, 1)
    θʳ ~ Beta(1, 1)
    θᶠ ~ Beta(1, 1)

    X ~ TrajectoryLatent(n_time, n_ind)

    pars = (; α=α, β=β, m=m, ν=ν, θʳ=θʳ, θᶠ=θᶠ)
    @addlogprob! loglik_fn(pars, data, X)
end

function run_fit(; n_sweeps=parse(Int, get(ENV, "WALKTHROUGH_SWEEPS", "2000")),
                   n_burn=parse(Int, get(ENV, "WALKTHROUGH_BURN", "800")),
                   n_adapts=parse(Int, get(ENV, "WALKTHROUGH_ADAPTS", "400")),
                   seed=7)
    m = cattle_model(fit_data, n_timepoints, n_individuals, loglik)

    spl = Gibbs(
        (:α, :β, :m̃) => NUTS(0.8),
        :ν => NuKernel(1.0, 1.0),
        :θʳ => TestSensKernel(1.0, 1.0, Rmask, :θʳ),
        :θᶠ => TestSensKernel(1.0, 1.0, Fmask, :θᶠ),
        :X => iFFBSKernel(latent!),
    )

    X0 = copy(X_true)
    init = (; X=X0, α=0.05, β=0.05, m̃=4.0, ν=0.1, θʳ=0.7, θᶠ=0.4)

    # Make the aggregates agree with the starting X before the first likelihood
    # call; the sampler keeps them consistent from then on.
    init_pars = (; α=init.α, β=init.β, m=init.m̃ + 1.0, ν=init.ν, θʳ=init.θʳ, θᶠ=init.θᶠ)
    reset_aggregates!(fit_data)
    apply_derived_summaries!(init_pars, fit_data, X0)

    chn = AbstractMCMC.sample(StableRNG(seed), m, spl, n_sweeps;
        init=init, n_adapts=n_adapts, discard_initial=n_burn)

    (
        α=collect(vec(chn[:α])),
        β=collect(vec(chn[:β])),
        m=collect(vec(chn[:m̃])) .+ 1.0,
        ν=collect(vec(chn[:ν])),
        θʳ=collect(vec(chn[:θʳ])),
        θᶠ=collect(vec(chn[:θᶠ])),
    )
end

function report(draws)
    println("\n=== Posterior recovery ===")
    for name in (:α, :β, :m, :ν, :θʳ, :θᶠ)
        post = getfield(draws, name)
        truth = getfield(true_pars, name)
        mn, sd = mean(post), std(post)
        within = abs(mn - truth) < 3 * (sd + sd / sqrt(length(post)))
        println(rpad(string(name), 4), ": posterior mean = ", rpad(round(mn; digits=4), 8),
                " (sd ", round(sd; digits=4), ")   truth = ", truth, within ? "   ✓" : "   ✗")
    end
end

println("Simulated herd: $n_pens pens x $n_per_pen animals x $n_timepoints days")
println("True infection prevalence: ", round(mean(X_true .== 2); digits=3))

if get(ENV, "WALKTHROUGH_RUN", "1") == "1"
    report(run_fit())
end
