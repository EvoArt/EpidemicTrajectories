# The badger bTB model, tuned.
#
# The SAME model as `badger_friendly.jl` — same states, same rates, same
# observation process, same posterior — with every performance seam the package
# offers turned on, plus post-hoc residuals. Read the friendly version first;
# this one is a diff against it.
#
# What changes, and why each is exact rather than an approximation:
#
#   1. `coupled_transitions = [(:S, :E)]`
#      The only way one badger changes another's fate is by contributing to its
#      force of infection. A neighbour whose realised move is not S->E has the
#      same probability whatever the focal does, so it cancels on normalisation
#      and can be skipped. Worth ~10x on the coupling.
#
#   2. `observation_weight` — the scalar observation path
#      The likelihood needs exactly one entry of the weight vector, `w[X[t,i]]`.
#      The vector form allocates a whole array per (individual, time) and throws
#      all but one element away: ~187k arrays of Duals per gradient call.
#
#   3. a custom `rest_contribution` — the O(n_states) coupling
#      The default recomputes, per candidate focal state, how probable every
#      affected neighbour's realised move is: O(n_states x |affected|) per cell.
#      Keeping per-(group, time) running totals of the S->E and S->S moves makes
#      the coupling weight a closed form in those totals, with no neighbour loop.
#
#   4. the `depends=` split — two likelihood terms instead of one sum
#      A Gibbs block skips a term its parameters cannot move. The epidemic
#      parameters never enter the observation likelihood and vice versa, so each
#      block differentiates roughly half of what it did before.
#
#   5. residuals — survival, exposure and latent period
#      The trajectory is streamed to disc during the fit and the residuals are
#      computed from it afterwards, so they can be recomputed, or new ones added,
#      without refitting.
#
# Run:  julia --project=examples examples/showcase/badger_power.jl
#       BADGER_SWEEPS=200 julia --project=examples ...   (default is 20)

using EpidemicTrajectories
using PracticalBayes
using PracticalEpiBayes
using Distributions
using Random
using StableRNGs: StableRNG
using Statistics: mean, std
using ADTypes: AutoPolyesterForwardDiff
using JLD2                       # backend for save_states / read_states
using PolyesterForwardDiff
import AbstractMCMC

include(joinpath(@__DIR__, "badger_data.jl"))

const STATES = [:S, :E, :I, :D]
const S_, E_, I_, D_ = 1, 2, 3, 4         # positions in STATES

raw = load_badger_data(DATA_DIR())
println("badger data: $(raw.n_individuals) individuals x $(raw.n_timepoints) quarters, " *
        "$(raw.n_groups) groups, $(raw.n_tests) tests")

# ---------------------------------------------------------------------------
# Aggregates — the two counts, plus two running totals for the coupling
# ---------------------------------------------------------------------------
# The friendly version declares two arrays with `@aggregate` and lets the package
# write the reversible updates. Here we want two MORE arrays whose contribution
# depends on where the individual goes NEXT (`X[t+1, i]`), which is outside what
# the macro expresses — so all four are written out by hand.
#
# The contract is the same either way: each summary must honour `reverse`, so the
# sampler can take an individual's contribution back out, refilter, and put it
# back. That reversibility is what makes iFFBS both correct and cheap.

function summary_n_infectious(model, data, X, s, i, t, reverse=false)
    g = data.social_group[i, t]; g > 0 || return nothing
    c = (s == I_)
    reverse ? (data.aggregates.n_infectious[g, t] -= c) :
              (data.aggregates.n_infectious[g, t] += c)
    nothing
end

function summary_n_alive(model, data, X, s, i, t, reverse=false)
    g = data.social_group[i, t]; g > 0 || return nothing
    c = (s != D_)
    reverse ? (data.aggregates.n_alive[g, t] -= c) :
              (data.aggregates.n_alive[g, t] += c)
    nothing
end

# How many S individuals in this group went on to E (and to S) at t -> t+1.
function summary_nSE(model, data, X, s, i, t, reverse=false)
    g = data.social_group[i, t]; g > 0 || return nothing
    t == data.n_timepoints && return nothing
    c = (s == S_) && (X[t + 1, i] == E_)
    reverse ? (data.aggregates.nSE[g, t] -= c) : (data.aggregates.nSE[g, t] += c)
    nothing
end

function summary_nSS(model, data, X, s, i, t, reverse=false)
    g = data.social_group[i, t]; g > 0 || return nothing
    t == data.n_timepoints && return nothing
    c = (s == S_) && (X[t + 1, i] == S_)
    reverse ? (data.aggregates.nSS[g, t] -= c) : (data.aggregates.nSS[g, t] += c)
    nothing
end

# ---------------------------------------------------------------------------
# Rates — identical to the friendly version
# ---------------------------------------------------------------------------

function infection(model, data, i, t)
    g = data.social_group[i, t]
    g == 0 && return 0.0
    infectious, alive = data.aggregates.n_infectious[g, t], data.aggregates.n_alive[g, t]
    alive == 0 && return 0.0
    foi = model.lambda * model.alpha[g] +
          model.beta * infectious / ((alive / data.K)^model.q)
    return -expm1(-foi)
end

function progression(model, data, i, t)
    x = data.k / model.tau
    s = term = 1.0
    for j in 1:(data.k - 1)
        term *= x / j
        s += term
    end
    return 1.0 - s * exp(-x)
end

function survival(model, data, i, t)
    tt = t + 1
    age = tt <= data.n_timepoints ? data.age[i, tt] :
          data.age[i, data.n_timepoints] + (tt - data.n_timepoints)
    age < 0 && return 1.0
    y1, y2 = model.b2 * (age - 1), model.b2 * age
    return exp(-model.c1 + (model.a2 / model.b2) * (-exp(y1) * expm1(y2 - y1)))
end

transitions = @transitions STATES begin
    @survival survival death=:D
    S -> E = infection
    E -> I = progression
end

function starting_state(model, data, X, i, t)
    p = zeros(Float64, data.n_states)
    start = data.sampling_period[i][1]
    nuE = nuI = 0.0
    if data.birth_time[i] < start
        idx = findfirst(==(start), data.nu_times)
        idx === nothing || ((nuE, nuI) = (model.nu[idx, 1], model.nu[idx, 2]))
    end
    p[1] = 1.0 - nuE - nuI; p[2] = nuE; p[3] = nuI
    return p
end

# ---------------------------------------------------------------------------
# 3. The O(n_states) coupling
# ---------------------------------------------------------------------------
# This stays USER code deliberately. The closed form assumes a single coupled
# transition (S->E) with a per-group frequency-dependent FOI — a modelling
# choice the generic core does not bake in. `rest_contribution=` is the seam.

const CASE_SorE, CASE_I, CASE_D = 1, 2, 3
@inline foi_case(s) = s == I_ ? CASE_I : (s == D_ ? CASE_D : CASE_SorE)

# The group FOI with the focal placed back in a candidate state. The focal was
# reversed out of the aggregates for its own resample, so its own contribution to
# the denominator has to be added back here.
@inline function group_foi(model, data, g, t, c, I_minus, M_minus)
    I, M = c == CASE_I ? (I_minus + 1, M_minus + 1) :
           c == CASE_D ? (I_minus, M_minus) : (I_minus, M_minus + 1)
    M == 0 && return 0.0
    foi = model.lambda * model.alpha[g] + model.beta * I / ((M / data.K)^model.q)
    return -expm1(-foi)
end

function rest_contribution(model, data, X, i, t, n_states, affected=nothing)
    t == data.n_timepoints && return ones(n_states)
    g = data.social_group[i, t]
    g == 0 && return ones(n_states)
    I_minus, M_minus = data.aggregates.n_infectious[g, t], data.aggregates.n_alive[g, t]
    nSE, nSS = data.aggregates.nSE[g, t], data.aggregates.nSS[g, t]
    logw = zeros(Float64, n_states)
    @inbounds for s in 1:n_states
        foi = group_foi(model, data, g, t, foi_case(s), I_minus, M_minus)
        logw[s] = nSE * log(max(foi, 1e-12)) + nSS * log(max(1.0 - foi, 1e-12))
    end
    logw .-= maximum(logw)          # stabilise before exponentiating
    return exp.(logw)
end

# ---------------------------------------------------------------------------
# 2. The observation process, in both forms
# ---------------------------------------------------------------------------
# The FILTER needs the whole weight vector (it must know a non-capture is
# informative, and that a badger seen alive later cannot be dead now). The
# LIKELIHOOD needs one entry.
#
# They also differ in CONTENT, not just shape. `etas` is drawn by an exact
# conjugate kernel, so the likelihood must score the TESTS factor only —
# scoring capture in both places would double-count it. Because the weights
# multiply, dropping a factor from the likelihood drops exactly its term.

obs_eltype(model) = promote_type(eltype(model.thetas), eltype(model.rhos),
                                 eltype(model.phis))

# capture x tests, for the filter.
function observation(model, data, X, i, t)
    T = promote_type(obs_eltype(model), eltype(model.etas))
    w = ones(T, data.n_states)
    eta = model.etas[data.season[t]]
    if data.capture[t, i] == 0
        w[S_] = w[E_] = w[I_] = 1 - eta
        w[D_] = t <= data.last_capture_time[i] ? zero(T) : one(T)
        return w
    end
    w[S_] = w[E_] = w[I_] = eta
    w[D_] = zero(T)
    for j in 1:size(data.tests, 3)
        nneg, npos = data.n_neg[t, i, j], data.n_pos[t, i, j]
        (nneg == 0 && npos == 0) && continue
        theta, rho, phi = model.thetas[j], model.rhos[j], model.phis[j]
        tr = theta * rho
        if npos > 0
            w[S_] *= (1 - phi)^npos; w[E_] *= tr^npos; w[I_] *= theta^npos
        end
        if nneg > 0
            w[S_] *= phi^nneg; w[E_] *= (1 - tr)^nneg; w[I_] *= (1 - theta)^nneg
        end
    end
    return w
end

# tests only, one state, for the likelihood. No array, no allocation.
@inline function tests_weight(model, data, X, i, t, s)
    T = obs_eltype(model)
    (data.capture[t, i] == 0 || s == D_) && return one(T)
    w = one(T)
    @inbounds for j in 1:size(data.tests, 3)
        nneg, npos = data.n_neg[t, i, j], data.n_pos[t, i, j]
        (nneg == 0 && npos == 0) && continue
        if s == S_
            phi = model.phis[j]
            npos > 0 && (w *= (1 - phi)^npos)
            nneg > 0 && (w *= phi^nneg)
        elseif s == E_
            tr = model.thetas[j] * model.rhos[j]
            npos > 0 && (w *= tr^npos)
            nneg > 0 && (w *= (1 - tr)^nneg)
        else
            theta = model.thetas[j]
            npos > 0 && (w *= theta^npos)
            nneg > 0 && (w *= (1 - theta)^nneg)
        end
    end
    return w
end

# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------

aggregates = (; n_infectious = zeros(Int, raw.n_groups, raw.n_timepoints),
                n_alive      = zeros(Int, raw.n_groups, raw.n_timepoints),
                nSE          = zeros(Int, raw.n_groups, raw.n_timepoints),
                nSS          = zeros(Int, raw.n_groups, raw.n_timepoints))

data = epidemic_data(;
    n_individuals   = raw.n_individuals,
    n_timepoints    = raw.n_timepoints,
    trans_mat       = transitions,
    starting_state  = starting_state,
    aggregates      = aggregates,
    derived_summaries = (summary_n_infectious, summary_n_alive,
                         summary_nSE, summary_nSS),
    observation_process = observation,        # the filter sees capture x tests
    rest_contribution   = rest_contribution,  # (3) O(n_states) coupling
    coupled_transitions = [(:S, :E)],         # (1) skip uncoupled neighbours
    sampling_period = raw.sampling_period,
    affected_individuals = badger_affected_lists(raw),
    social_group = raw.social_group, age = raw.age, capture = raw.capture,
    capt_effort = raw.capt_effort, tests = raw.tests,
    n_neg = raw.n_neg, n_pos = raw.n_pos, season = raw.season,
    birth_time = raw.birth_time, last_capture_time = raw.last_capture_time,
    nu_times = raw.nu_times, K = Float64(raw.K), k = raw.k)

loglik  = epidemic_loglik(data; entry_time = raw.first_capture_time,
                                survival   = survival)
# (2) the likelihood scores the tests factor only, via the scalar path.
obs_loglik = epidemic_obs_loglik(data; observation_weight = tests_weight)
latent!    = epidemic_latent_sampler(data)

# ---------------------------------------------------------------------------
# The model, with the depends= split
# ---------------------------------------------------------------------------

struct NuPrior <: Distributions.DiscreteMatrixDistribution
    n_nu::Int
end
Base.size(d::NuPrior) = (d.n_nu, 2)
Distributions.logpdf(::NuPrior, ::AbstractMatrix) = 0.0
Distributions.rand(::Random.AbstractRNG, d::NuPrior) = fill(0.05, d.n_nu, 2)

@model function badger(data, loglik_fn, obs_loglik_fn, n_groups, n_tests,
                       n_seasons, n_nu, n_time, n_ind)
    tau    ~ Exponential(100.0)
    alpha  ~ PracticalBayes.filldist(Exponential(1.0), n_groups)
    lambda ~ Exponential(1.0)
    beta   ~ Exponential(1.0)
    q      ~ Beta(1, 1)

    c1 ~ Exponential(1.0)
    a2 ~ Exponential(1.0)
    b2 ~ Exponential(1.0)

    thetas ~ PracticalBayes.filldist(Beta(1, 1), n_tests)
    rhos   ~ PracticalBayes.filldist(Beta(1, 1), n_tests)
    phis   ~ PracticalBayes.filldist(Beta(1, 1), n_tests)
    etas   ~ PracticalBayes.filldist(Beta(1, 1), n_seasons)
    nu     ~ NuPrior(n_nu)

    X ~ TrajectoryLatent(n_time, n_ind)

    pars = (; tau, alpha, lambda, beta, q, c1, a2, b2, thetas, rhos, phis, etas, nu)

    # (4) TWO terms, each declaring what it reads. A Gibbs block owning none of a
    # term's variables skips it entirely — exact, because the term is then a
    # constant w.r.t. that block and contributes zero to its gradient.
    #
    # `etas` appears in NEITHER: it reaches the filter through
    # `observation_process`, which is never differentiated, and the likelihood's
    # `tests_weight` does not read it.
    #
    # Under-declaring here is SILENT — a real gradient contribution vanishes with
    # no error. `check_depends` below verifies it.
    @addlogprob! loglik_fn(pars, data, X) depends=(:tau, :alpha, :lambda, :beta,
                                                   :q, :c1, :a2, :b2, :nu, :X)
    @addlogprob! obs_loglik_fn(pars, data, X) depends=(:thetas, :rhos, :phis, :X)
end

model = badger(data, loglik, obs_loglik, raw.n_groups, raw.n_tests,
               raw.n_seasons, raw.n_nu_times, raw.n_timepoints, raw.n_individuals)

# Hand-tuned step sizes, as NAMED pairs. The metric is one flat vector whose
# order must match the block's; `check_step_sizes` asserts that rather than
# leaving it to a comment.
const HMC_NAMES = (:tau, :alpha, :lambda, :beta, :q, :c1, :a2, :b2,
                   :thetas, :rhos, :phis)

eps = hmc_step_sizes(
    :tau => 0.002, :alpha => (0.2, raw.n_groups), :lambda => 0.01, :beta => 0.05,
    :q => 0.05, :c1 => 0.02, :a2 => 0.001, :b2 => 0.001,
    :thetas => (0.005, raw.n_tests), :rhos => (0.005, raw.n_tests),
    :phis => (0.005, raw.n_tests))
check_step_sizes(eps, HMC_NAMES)

sampler = Gibbs(
    # `n_steps = 15` matches the C++ reference's EXPECTED trajectory length (it
    # draws L uniform on 1..30, mean 15.5), not its nominal L = 30.
    HMC_NAMES => hmc_block(eps, 15),

    :etas => capture_prob_kernel(:etas;
        caught = raw.capture, effort = raw.capt_effort, group = raw.social_group,
        index  = raw.season, dead_state = D_, n = raw.n_seasons),

    :nu => initial_state_kernel(:nu;
        at       = raw.nu_times,
        eligible = (X, d, i, nt) -> d.sampling_period[i][1] == nt &&
                                    d.birth_time[i] < nt,
        states = (S_, E_, I_), prior = [1.0, 1.0, 1.0], n = raw.n_nu_times),

    :X => iffbs_kernel(latent!;
        params = (:tau, :alpha, :lambda, :beta, :q, :c1, :a2, :b2,
                  :thetas, :rhos, :phis, :etas, :nu)))

# Verify the depends= annotations before trusting a single gradient.
report = check_depends(model, sampler; verbose = false)
report.ok || error("depends= is under-declared:\n" * sprint(show, report))
println("depends= check passed")

# ---------------------------------------------------------------------------
# 5. Residuals — survival, exposure, latent period
# ---------------------------------------------------------------------------
# Each is a PIT residual: uniform on (0,1) if the model is right, so a departure
# from uniformity is evidence of misfit in that specific mechanism.

# How long a badger LIVES, against Gompertz-Makeham. Left-truncated because we
# only ever see badgers that survived to first capture: without conditioning on
# that, the residual is calibrated against the unconditional lifetime and reports
# the sampling design as model misfit.
# `origin` is clamped to 1: 48 badgers have a birth time at or before the start
# of monitoring (the earliest is -5), and the survival function indexes `data.age`,
# which only exists over 1:n_timepoints. Their age is still correct — `age[i, t]`
# is `t - birth_time[i]`, so a badger born before the window is simply older at
# t = 1 — but the product has to start inside the array.
survival_resid = LeftTruncatedSurvivalResidual(
    survival     = survival,
    origin       = i -> max(raw.birth_time[i], 1),
    condition_on = i -> raw.first_capture_time[i],
    censor_at    = i -> raw.last_capture_time[i],
    death        = :D,
    name         = :survival)

# How long a SUSCEPTIBLE badger waits before being infected. The clock starts
# when observation starts, not when it entered S — it was already susceptible
# when we began watching. Death is a competing risk, so it censors rather than
# drops: a badger that died before catching it has not falsified anything.
exposure_resid = WaitingTimeResidual(:S => :E;
    accumulate = :discrete_product,
    origin     = :window_start,
    censor_at  = (:window_end, :D),
    name       = :exposure)

# How long an EXPOSED badger takes to become infectious — the latent period, the
# residual for `tau`. The clock starts when it entered E, which is in X.
latent_resid = WaitingTimeResidual(:E => :I;
    accumulate = :discrete_product,
    origin     = :entry_to_from_state,
    censor_at  = (:window_end, :D),
    name       = :latent_period)

n_sweeps = parse(Int, get(ENV, "BADGER_SWEEPS", "20"))

# The residuals are computed POST HOC, from the trajectories streamed to disc
# during the fit (see `save_states` below). That keeps the fit a single `sample`
# call, and — more usefully — lets the residuals be recomputed, or new ones
# added, without refitting.
#
# `SummaryCollector` + `collect_summaries!` is the alternative: it streams during
# sampling and never stores X at all, at the cost of writing the sweep loop out
# by hand. Prefer it when the trajectories are too large to keep even on disc.

# ---------------------------------------------------------------------------
# Fit
# ---------------------------------------------------------------------------

# Named once, so the writer and the reader cannot drift apart.
const X_PATH = joinpath(@__DIR__, "badger_power_X.jld2")

X0 = copy(raw.X_init)
reset_aggregates!(data)
apply_derived_summaries!((;), data, X0)

const ADTYPE = AutoPolyesterForwardDiff(; chunksize = nothing, tag = nothing)

t0 = time()
chain = AbstractMCMC.sample(StableRNG(13), model, sampler, n_sweeps;
                            init        = (; X = X0),
                            adtype      = ADTYPE,
                            # X is 2384 x 161 Ints per sweep. Keep it live for
                            # conditioning, stream it to disc, keep it out of the
                            # chain object.
                            save_states = (X = (X_PATH, 100),))
elapsed = time() - t0

println("\n$n_sweeps sweeps in $(round(elapsed; digits=1)) s " *
        "($(round(elapsed / n_sweeps; digits=3)) s/sweep)")

println("\n=== posterior summary ===")
for name in (:tau, :lambda, :beta, :q, :c1, :a2, :b2)
    v = vec(chain[name])
    println(rpad(name, 8), " mean ", rpad(round(mean(v); digits = 4), 10),
            " sd ", round(std(v); digits = 4))
end

# ---------------------------------------------------------------------------
# Residuals, post hoc
# ---------------------------------------------------------------------------
# Pair each streamed trajectory with the parameter draw that produced it.

Xs = read_states(X_PATH)

draws = [((; tau = chain[:tau][d], alpha = chain[:alpha][d],
             lambda = chain[:lambda][d], beta = chain[:beta][d], q = chain[:q][d],
             c1 = chain[:c1][d], a2 = chain[:a2][d], b2 = chain[:b2][d],
             thetas = chain[:thetas][d], rhos = chain[:rhos][d],
             phis = chain[:phis][d], etas = chain[:etas][d], nu = chain[:nu][d]),
           Matrix{Int}(Xs[d]))
          for d in eachindex(Xs)]

# The force of infection reads the aggregates, so a spec-derived hazard depends
# on them agreeing with THIS draw's X — not with whatever X was current when the
# fit ended. `trajectory_summaries` deliberately does not rebuild them
# unconditionally, because for a model whose rates ignore the aggregates that is
# a wasted O(individuals x time) pass per draw.
R = trajectory_summaries((survival_resid, exposure_resid, latent_resid),
                         data, aggregate_synced_draws(data, draws);
                         rng = StableRNG(99))

println("\n=== residuals (PIT; Uniform(0,1) if the model is right) ===")
for name in (:survival, :exposure, :latent_period)
    v = collect(skipmissing(residual_values(R, name)))
    if isempty(v)
        println(rpad(name, 15), " no scoreable events")
    else
        println(rpad(name, 15), " n = ", rpad(length(v), 8),
                " mean ", rpad(round(mean(v); digits = 4), 8),
                " (0.5 if calibrated)")
    end
end
