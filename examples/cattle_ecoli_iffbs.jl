# =============================================================================
# Cattle E. coli iFFBS example — recapturing the parameters of the two-state
# S/I recurrent Markov model from Touloupou et al. (2019), fit with
# EpidemicTrajectories.jl + PracticalBayes.jl.
#
# This is the flagship end-to-end example: it simulates capture-recapture data
# from a known set of parameters, then recovers them using
#
#   * NUTS (via PracticalBayes' Gibbs) for the transmission parameters α, β and
#     the mean infectious period m — differentiating the trajectory
#     log-likelihood `EpidemicTrajectories.trajectory_loglik`;
#   * an iFFBS latent kernel (`EpidemicTrajectories.ffbs_sweep!`) resampling the
#     entire hidden state matrix X once per Gibbs sweep;
#   * conjugate Gibbs updates for the initial infection frequency ν and the test
#     sensitivity θ.
#
# The SAME EpidemicTrajectories rate functions drive both the differentiated
# likelihood and the FFBS sampler — the single-source-of-truth property.
#
# Run with:  julia --project=examples examples/cattle_ecoli_iffbs.jl
# (the examples environment develops EpidemicTrajectories + PracticalBayes).
# =============================================================================

using EpidemicTrajectories
using PracticalBayes
using Distributions
using Random
using AdvancedHMC: NUTS
import AbstractMCMC
using StableRNGs: StableRNG
using Statistics: mean, std

const ET = EpidemicTrajectories

# -----------------------------------------------------------------------------
# 1. Simulate data from known parameters (same structure as the iFFBS paper's
#    simulate_data.jl, at a smaller scale so the example runs quickly).
# -----------------------------------------------------------------------------

true_pars = (; α=0.01, β=0.02, m=6.0)   # external FOI, within-pen FOI, mean inf. period
true_ν = 0.10                            # initial infection frequency
true_θ = 0.8                             # RAMS test sensitivity (specificity = 1)

n_pens = 10
n_per_pen = 8
n_times = 80
n_ind = n_pens * n_per_pen

# group vector: individuals 1:8 in pen 1, 9:16 in pen 2, ...
group = repeat(1:n_pens; inner=n_per_pen)

ss = SI
rates = TwoStateSI()
init_prob = [1 - true_ν, true_ν]

rng = StableRNG(2024)
# simulate the hidden trajectory for the whole herd (all pens share params)
states, data = simulate_trajectory(rng, ss, rates, true_pars, group, init_prob; n_times=n_times)

# imperfect diagnostic test (RAMS): sensitivity θ, specificity 1
rams = DiagnosticTest(; sensitivity=p -> p.θ, specificity=p -> 1.0, positive_code=1)
obs = simulate_observations(rng, (rams,), (; θ=true_θ), ss, states)
R = obs[1]  # n_ind x n_times observed test results (0/1)

# mask most days as unobserved (capture-recapture: animals tested on a subset of
# days). Keep every 6th day observed, rest set to -1 (missing).
observed_days = collect(1:6:n_times)
Rmask = fill(-1, size(R))
Rmask[:, observed_days] .= R[:, observed_days]

println("Simulated herd: $n_pens pens x $n_per_pen animals x $n_times days")
println("True parameters: α=$(true_pars.α), β=$(true_pars.β), m=$(true_pars.m), ν=$true_ν, θ=$true_θ")
println("Observed on days: ", observed_days)
println("Overall infection prevalence in truth: ", round(mean(states .== 1); digits=3))

# -----------------------------------------------------------------------------
# 2. The whole-trajectory latent `X`: a discrete matrix distribution so
#    PracticalBayes routes `X` to the value-store (an AD-constant), sampled by
#    the iFFBS kernel. (This inline definition is what the future
#    PracticalEpiBayes glue package will provide as a reusable type.)
# -----------------------------------------------------------------------------

struct TrajectoryLatent <: Distributions.DiscreteMatrixDistribution
    n_ind::Int
    n_times::Int
end
Base.size(d::TrajectoryLatent) = (d.n_ind, d.n_times)
# improper: the real trajectory density enters via @addlogprob! below, so this
# prior contributes nothing (avoids double-counting the transition probabilities).
Distributions.logpdf(::TrajectoryLatent, X::AbstractMatrix) = 0.0
# a valid initial draw (all susceptible) — the kernel overwrites it immediately.
Distributions.rand(rng::AbstractRNG, d::TrajectoryLatent) = zeros(Int, d.n_ind, d.n_times)

# -----------------------------------------------------------------------------
# 3. The iFFBS latent kernel: resample the whole X matrix once per Gibbs sweep.
# -----------------------------------------------------------------------------

struct iFFBS{RB<:RateBundle} <: PracticalBayes.AbstractLatentKernel
    ss::StateSpace
    rates::RB                # any RateBundle: TwoStateSI OR EpiTransitionMatrix
    group::Vector{Int}
    tests::Tuple{DiagnosticTest}
    results::Tuple{Matrix{Int}}
    init_prob_fn::Function   # ν -> [1-ν, ν]
end

function PracticalBayes.latent_step(rng, k::iFFBS, block_names, c::ModelConditional)
    block_names == (:X,) || error("iFFBS kernel only handles the :X block")
    # current continuous params from the other Gibbs blocks. `m` is the computed
    # `m̃ + 1` reparameterization (m̃ is the sampled variable).
    pars = (; α=c.values.α, β=c.values.β, m=c.values.m̃ + 1.0, θ=c.values.θ)
    # rebuild the `data` object around the CURRENT X (a fresh copy so the kernel
    # mutates its own working matrix, not the value store's)
    X = copy(c.values.X)
    data = ET.make_data(X, k.group)
    model = (; state_space=k.ss, rates=k.rates, pars=pars)
    ET.ffbs_sweep!(rng, model, data, k.tests, k.results;
                   initial_prob=k.init_prob_fn(c.values.ν), coupling=true)
    return (; X=data.states)
end

# -----------------------------------------------------------------------------
# 4. Conjugate Gibbs kernels for ν (initial infection freq) and θ (test
#    sensitivity) — closed-form Beta posteriors, exactly as in the paper.
# -----------------------------------------------------------------------------

# ν ~ Beta(a,b) prior; likelihood = product of Bernoulli(ν) over t=1 states.
struct NuKernel <: PracticalBayes.AbstractLatentKernel
    a::Float64
    b::Float64
end
function PracticalBayes.latent_step(rng, k::NuKernel, block_names, c::ModelConditional)
    X = c.values.X
    x1 = @view X[:, 1]
    n_inf = count(==(1), x1)
    n_sus = length(x1) - n_inf
    return (; ν=rand(rng, Beta(k.a + n_inf, k.b + n_sus)))
end

# θ ~ Beta(a,b) prior; likelihood over observed test results at truly-infected
# animal-times (specificity 1, so susceptibles never test positive and carry no
# info about θ).
struct ThetaKernel <: PracticalBayes.AbstractLatentKernel
    a::Float64
    b::Float64
    results::Matrix{Int}
end
function PracticalBayes.latent_step(rng, k::ThetaKernel, block_names, c::ModelConditional)
    X = c.values.X
    R = k.results
    n_pos = 0; n_tested_inf = 0
    for t in axes(R, 2), i in axes(R, 1)
        r = R[i, t]
        r < 0 && continue           # not tested
        if X[i, t] == 1             # truly infected
            n_tested_inf += 1
            r == 1 && (n_pos += 1)
        end
    end
    return (; θ=rand(rng, Beta(k.a + n_pos, k.b + (n_tested_inf - n_pos))))
end

# -----------------------------------------------------------------------------
# 5. The PracticalBayes @model. Continuous params via ~; X via the latent block;
#    trajectory log-likelihood via @addlogprob! reading the current X.
# -----------------------------------------------------------------------------

@model function cattle_model(Rmask, group, ss, rates, n_ind, n_times)
    α ~ Gamma(1, 1)
    β ~ Gamma(1, 1)
    # mean infectious period reparameterized as m = m̃ + 1 (m̃ > 0), so recovery
    # rate 1/m < 1 always — matches the iFFBS paper and keeps the recovery
    # transition probability valid without relying on the clamp.
    m̃ ~ Gamma(2, 4)
    m := m̃ + 1.0
    ν ~ Beta(1, 1)           # initial infection frequency (updated by conjugate kernel)
    θ ~ Beta(1, 1)           # test sensitivity (updated by conjugate kernel)

    X ~ TrajectoryLatent(n_ind, n_times)

    # trajectory log-likelihood: the SAME rate functions as the kernel, with X
    # read as an AD-constant (ValueSlot). This is what NUTS differentiates for
    # α, β, m.
    pars = (; α=α, β=β, m=m, θ=θ)
    data = EpidemicTrajectories.make_data(X, group)
    model = (; state_space=ss, rates=rates, pars=pars)
    @addlogprob! EpidemicTrajectories.trajectory_loglik(pars, model, data)
end

# -----------------------------------------------------------------------------
# 6. The transition-matrix style, for comparison. `EpiTransitionMatrix` (built
#    here with the @transitions modelling-language macro) is ITSELF a RateBundle
#    — it implements `transition_matrix_at` — so it drops into the SAME @model,
#    the SAME iFFBS kernel, the SAME likelihood, unchanged. The two styles are
#    interchangeable front-ends onto one core; we fit with each and check they
#    recover the same parameters.
#
#    Each transition's rate is the general per-individual signature
#    f(pars, model, data, i, t), so the S->I rate can compute the leave-one-out
#    infected count exactly as the functional TwoStateSI bundle does internally.
# -----------------------------------------------------------------------------

rates_matrix = @transitions :individual SI begin
    S -> I = (pars, model, data, i, t) -> begin
        g = data.group[i]
        I_minus = count(j -> j != i && data.states[j, t] == 1, data.members(data, g))
        -expm1(-(pars.α + pars.β * I_minus))     # 1 - exp(-(α + β·I₋))
    end
    I -> S = (pars, model, data, i, t) -> 1 / pars.m
end

# -----------------------------------------------------------------------------
# 7. Assemble the Gibbs sampler and run — parameterized over the rate bundle so
#    we can call it for both styles.
# -----------------------------------------------------------------------------

function run_fit(rates_bundle; n_sweeps=2000, n_burn=800, n_adapts=400, seed=7)
    m = cattle_model(Rmask, group, ss, rates_bundle, n_ind, n_times)
    kernel = iFFBS(ss, rates_bundle, group, (rams,), (Rmask,), ν -> [1 - ν, ν])
    spl = Gibbs(
        (:α, :β, :m̃) => NUTS(0.8),
        :ν => NuKernel(1.0, 1.0),
        :θ => ThetaKernel(1.0, 1.0, Rmask),
        :X => kernel,
    )
    # start X from a plausible simulation; the kernel refines it every sweep.
    X0, _ = simulate_trajectory(StableRNG(99), ss, rates, true_pars, group, init_prob; n_times=n_times)
    init = (; X=copy(X0), α=0.05, β=0.05, m̃=4.0, ν=0.1, θ=0.7)

    rng_fit = StableRNG(seed)
    draws = (α=Float64[], β=Float64[], m=Float64[], ν=Float64[], θ=Float64[])
    transition, state = AbstractMCMC.step(rng_fit, m, spl; init=init)
    for _ in 1:n_sweeps
        transition, state = AbstractMCMC.step(rng_fit, m, spl, state; n_adapts=n_adapts)
        push!(draws.α, transition.α)
        push!(draws.β, transition.β)
        push!(draws.m, transition.m̃ + 1.0)   # report m = m̃ + 1
        push!(draws.ν, transition.ν)
        push!(draws.θ, transition.θ)
    end
    # drop burn-in
    return map(v -> v[(n_burn + 1):end], draws)
end

function report(label, draws)
    println("\n=== Posterior recovery ($label) ===")
    truths = (α=true_pars.α, β=true_pars.β, m=true_pars.m, ν=true_ν, θ=true_θ)
    for name in (:α, :β, :m, :ν, :θ)
        post = getfield(draws, name); truth = getfield(truths, name)
        mn, sd = mean(post), std(post)
        within = abs(mn - truth) < 3 * (sd + sd / sqrt(length(post)))
        println(rpad(string(name), 4), ": posterior mean = ", rpad(round(mn; digits=4), 8),
                " (sd ", round(sd; digits=4), ")   truth = ", truth, within ? "   ✓" : "   ✗")
    end
end

println("\nRunning Gibbs (NUTS + iFFBS + conjugate) — FUNCTIONAL style (TwoStateSI) ...")
draws_functional = run_fit(rates)
report("functional style — TwoStateSI", draws_functional)

println("\nRunning Gibbs (NUTS + iFFBS + conjugate) — TRANSITION-MATRIX style (@transitions) ...")
draws_matrix = run_fit(rates_matrix)
report("transition-matrix style — EpiTransitionMatrix", draws_matrix)

println("\nBoth styles fit the same process via the same iFFBS kernel and likelihood — done.")
