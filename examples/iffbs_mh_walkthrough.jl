# iFFBS as a proposal, corrected by Metropolis-Hastings — a worked example.
#
# Runnable: `julia --project=examples examples/iffbs_mh_walkthrough.jl`
#
# Covers the three things the MH kernel is for, on one SEID model:
#
#   1. IS my sampler exact?         -> check_iffbs_exact
#   2. MHiFFBS                      -> drop the coupling from the FILTER, correct it
#   3. SM-iFFBS                     -> a semi-Markov target with a Markov proposal
#
# Reference: Touloupou, Finkenstadt & Spencer (2020), "Scalable Bayesian Inference
# for Coupled Hidden Markov and Semi-Markov Models", JCGS 29(2):238-249.

using EpidemicTrajectories
using Random
using StableRNGs: StableRNG
using Statistics: mean

# ---------------------------------------------------------------------------
# The model: S -> E -> I with mortality, coupled through a per-group force of
# infection. Nothing here is MH-specific; it is an ordinary package model.
# ---------------------------------------------------------------------------
const STATES = [:S, :E, :I, :D]
const N_T, N_GROUPS, N_PER = 40, 8, 15
const N_IND = N_GROUPS * N_PER
const GROUP = repeat(1:N_GROUPS; inner=N_PER)

aggs = @aggregate STATES begin
    @array n_infected Int (N_GROUPS, N_T)
    @array n_alive Int (N_GROUPS, N_T)
    n_infected[data.group[i], t] += (state == :I)
    n_alive[data.group[i], t] += (state != :D)
end

# Frequency-dependent FOI. `n_alive` includes the focal because the iFFBS filter
# re-inserts it (`focal_self_contribution`, on by default).
function foi(model, data, i, t)
    g = data.group[i]
    I = data.aggregates[:n_infected][g, t]
    M = max(data.aggregates[:n_alive][g, t], 1)
    -expm1(-(model.alpha + model.beta * I / M))
end
progression(model, data, i, t) = model.gamma
survival(model, data, i, t) = 1 - model.mu

spec = @transitions STATES begin
    @survival survival death = :D
    S -> E = foi
    E -> I = progression
end

# A noisy diagnostic test, applied on a fixed schedule.
#
# Supplied in BOTH forms, and that is not optional if you care about speed.
# `observation_process` returns the whole per-state weight VECTOR, which is what
# the forward filter needs. `observation_weight` returns ONE state's weight as a
# scalar, which is what every likelihood-shaped consumer needs --
# `epidemic_obs_loglik` AND `epidemic_conditional_loglik`. Give only the vector
# and both fall back to allocating an array per (i, t), which on this model made
# the MH sweep 4x slower than the exact one (see the timings below vs. the
# comment at the end of section 2).
function noisy_test(model, data, X, i, t)
    w = ones(Float64, data.n_states)
    y = data.results[t, i]
    y < 0 && return w
    if y == 1
        w[1] = 1 - model.spec_        # false positive from S
        w[2] = 1 - model.spec_        # E tests like S here
        w[3] = model.sens
        w[4] = 1e-9                   # the dead are not tested positive
    else
        w[1] = model.spec_
        w[2] = model.spec_
        w[3] = 1 - model.sens
        w[4] = 1.0
    end
    w
end

# The scalar twin of `noisy_test`. These two MUST agree entry for entry; a
# disagreement silently changes the posterior.
function noisy_test_weight(model, data, X, i, t, s)
    y = data.results[t, i]
    y < 0 && return 1.0
    if y == 1
        s == 3 && return model.sens
        s == 4 && return 1e-9
        return 1 - model.spec_
    else
        s == 3 && return 1 - model.sens
        s == 4 && return 1.0
        return model.spec_
    end
end

pars = (; alpha=0.015, beta=0.55, gamma=0.18, mu=0.008, nu=0.08,
          sens=0.75, spec_=0.9, prog_shape=2.2, prog_scale=3.0)

# Simulate, then build the observations from the simulated truth.
scaffold = epidemic_data(; n_individuals=N_IND, n_timepoints=N_T, group=GROUP,
                           trans_mat=spec, aggregates=aggs,
                           starting_state=(m, d, X, i, t) -> [1 - m.nu, m.nu, 0.0, 0.0])
X_true = epidemic_simulator(scaffold)(StableRNG(1), pars)

rng = StableRNG(2)
results = fill(-1, N_T, N_IND)
for t in 1:4:N_T, i in 1:N_IND
    X_true[t, i] == 4 && continue                     # dead animals are not tested
    p_pos = X_true[t, i] == 3 ? pars.sens : 1 - pars.spec_
    results[t, i] = rand(rng) < p_pos ? 1 : 0
end

data = epidemic_data(; n_individuals=N_IND, n_timepoints=N_T, group=GROUP,
                       trans_mat=spec, aggregates=aggs,
                       starting_state=(m, d, X, i, t) -> [1 - m.nu, m.nu, 0.0, 0.0],
                       observation_process=noisy_test,
                       observation_weight=noisy_test_weight,
                       coupled_transitions=[(:S, :E)],
                       results=results)

# Cheap insurance: the two observation forms must agree.
let Xc = fill(1, N_T, N_IND)
    worst = 0.0
    for t in 1:N_T, i in 1:N_IND, s in 1:4
        w = noisy_test(pars, data, Xc, i, t)[s]
        worst = max(worst, abs(w - noisy_test_weight(pars, data, Xc, i, t, s)))
    end
    worst > 1e-12 && error("observation_process and observation_weight disagree by $worst")
end

loglik = epidemic_loglik(data)
obs_loglik = epidemic_obs_loglik(data)

# Start from an all-susceptible X, which is deliberately wrong.
X = fill(1, N_T, N_IND)
sync!(X) = (reset_aggregates!(data); apply_derived_summaries!(pars, data, X); X)
sync!(X)

prevalence(X) = mean(==(3), X)

println("truth prevalence        : ", round(prevalence(X_true); digits=4))
println("starting prevalence     : ", round(prevalence(X); digits=4))

# ---------------------------------------------------------------------------
# 1. IS the plain sweep exact for this model?
#
# `check_iffbs_exact` runs the MH ratio with the proposal set equal to the target,
# where it must be 0. This is the answer to "can I use `iffbs!` here?", and it is
# a RUNTIME question: rate functions are opaque closures, and the mismatches that
# matter live in keyword arguments to other functions.
# ---------------------------------------------------------------------------
report = check_iffbs_exact(pars, data, copy(X); rng=StableRNG(3), n_sweeps=1)
println()
println("check_iffbs_exact")
println("  exact            : ", report.exact)
println("  max |log alpha|  : ", report.max_abs_logratio)
println("  decisions checked: ", report.n_checked, " (identical: ", report.n_identical, ")")
if !report.exact
    println("  worst individual : ", report.worst_individual)
    println("  offenders        : ", first(report.offenders, 3))
end

# ---------------------------------------------------------------------------
# 2. MHiFFBS: drop the coupling from the FILTER, correct with MH.
#
# The saving is bounded by arithmetic: the filter stops doing
# `n_states x |affected|` neighbour visits per timepoint, but the acceptance ratio
# still needs `2 x |affected|` of them (current path and candidate). So expect
# about `n_states / 2` on the coupling term, and only as much of your total sweep
# as the coupling was.
#
# It is bounded much harder by the ACCEPTANCE RATE, and this example is built to
# show that rather than to flatter the method. The paper: the choice between
# MHiFFBS and keeping the coupling "depends on how important the missing arrows
# were" (section 4.3). Below, the same code runs on a strongly coupled model
# (where the arrows matter a great deal) and a weakly coupled one.
#
# Note the cost asymmetry this exposes. An ACCEPT costs the same summary passes as
# a plain iFFBS sweep; a REJECT costs one extra reverse/apply round trip. So a low
# acceptance rate is not merely bad mixing -- it is also SLOWER per sweep, and the
# measured speedup drops below 1x.
# ---------------------------------------------------------------------------
target = epidemic_conditional_loglik(data)
cheap = uncorrected_proposal(data)

# Burn in with the EXACT sweep first. Comparing from the all-susceptible start
# would measure how fast each method climbs out of a silly initial state, not how
# they behave where the sampler actually lives.
X_burn = copy(X)
sync!(X_burn)
for _ in 1:30
    iffbs!(pars, data, X_burn, StableRNG(4))
end
println()
println("after 30 burn-in iffbs! sweeps: prevalence ",
        round(prevalence(X_burn); digits=4), " (truth ",
        round(prevalence(X_true); digits=4), ")")

# Timing a Gibbs sweep has two traps, and this example hit both while being
# written. Getting either wrong reversed the conclusion.
#
#   1. WARM UP. Without a warm-up call the timed loop for whichever method runs
#      second absorbs its own JIT compilation.
#   2. RESET `X` BETWEEN REPS. This is the subtle one. Left to run, the two
#      methods walk to DIFFERENT parts of the state space -- here `iffbs!` reached
#      57% infected while MHiFFBS was still at 41% -- and the cost of a sweep
#      depends on where you are. Timing them from wherever they happen to have
#      drifted compares two different workloads and reports nonsense (it said 0.6x
#      for something that is actually 1.5x).
#
# So: same start, every repetition, both methods, and take the minimum.
function timed_sweeps(f, ps, dat, Xstart; reps=8, sweeps=5)
    Xw = copy(Xstart)
    reset_aggregates!(dat); apply_derived_summaries!(ps, dat, Xw)
    f(Xw, StableRNG(1))                      # warm up
    best = Inf
    for k in 1:reps
        Xw .= Xstart
        reset_aggregates!(dat); apply_derived_summaries!(ps, dat, Xw)
        el = @elapsed begin
            for _ in 1:sweeps
                f(Xw, StableRNG(100 + k))
            end
        end
        best = min(best, el / sweeps)
    end
    best
end

function compare_from(label, ps, dat, Xstart, tgt, prop; n=20)
    st = MHStats(size(Xstart, 2))
    t_gibbs = timed_sweeps((Xw, r) -> iffbs!(ps, dat, Xw, r), ps, dat, Xstart)
    t_mh = timed_sweeps((Xw, r) -> iffbs_mh!(ps, dat, Xw, r; proposal=prop,
                                             target=tgt, stats=st), ps, dat, Xstart)
    # How much of the exact sweep IS the coupling term? That is the ceiling on
    # what dropping it from the filter can buy.
    t_nocoup = timed_sweeps((Xw, r) -> iffbs!(ps, dat, Xw, r; proposal=prop),
                            ps, dat, Xstart)

    # Where each method actually goes, run forward from the same start. Separate
    # from the timing on purpose.
    Xg = copy(Xstart)
    reset_aggregates!(dat); apply_derived_summaries!(ps, dat, Xg)
    for _ in 1:n
        iffbs!(ps, dat, Xg, StableRNG(11))
    end
    Xm = copy(Xstart)
    reset_aggregates!(dat); apply_derived_summaries!(ps, dat, Xm)
    for _ in 1:n
        iffbs_mh!(ps, dat, Xm, StableRNG(11); proposal=prop, target=tgt)
    end

    # The invariant the whole design rests on, and the one a REJECT branch is most
    # likely to break: the incrementally maintained aggregates must still equal a
    # from-scratch recompute.
    incremental = copy(dat.aggregates[:n_infected])
    reset_aggregates!(dat); apply_derived_summaries!(ps, dat, Xm)
    ok = incremental == dat.aggregates[:n_infected]

    println()
    println(label)
    println("  iffbs!      : ", round(1000 * t_gibbs; digits=2), " ms/sweep   ",
            "prevalence after $n sweeps ", round(prevalence(Xg); digits=4))
    println("  MHiFFBS     : ", round(1000 * t_mh; digits=2), " ms/sweep   ",
            "prevalence after $n sweeps ", round(prevalence(Xm); digits=4))
    println("  acceptance  : ", round(acceptance_rate(st); digits=3),
            "   identical: ", round(identical_rate(st); digits=3))
    println("  speedup     : ", round(t_gibbs / t_mh; digits=2), "x")
    println("  coupling is ", round(100 * (1 - t_nocoup / t_gibbs); digits=1),
            "% of the exact sweep (the ceiling on what MHiFFBS can save)")
    println("  aggregates consistent: ", ok)
end

compare_from("STRONGLY coupled (beta = $(pars.beta), prevalence ~$(round(prevalence(X_burn); digits=2)))",
             pars, data, X_burn, target, cheap)

# What the scalar observation weight is worth, measured. `epidemic_data` stored
# both forms above; building the conditional with `observation_weight=nothing`
# forces it back onto the vector-returning process, which allocates an array per
# (i, t) -- twice per MH decision.
let slow_target = epidemic_conditional_loglik(data; observation_weight=nothing)
    t_fast = timed_sweeps((Xw, r) -> iffbs_mh!(pars, data, Xw, r; proposal=cheap,
                                               target=target), pars, data, X_burn)
    t_slow = timed_sweeps((Xw, r) -> iffbs_mh!(pars, data, Xw, r; proposal=cheap,
                                               target=slow_target), pars, data, X_burn)
    # They must produce the same trajectory: the two observation forms are the
    # same function, one indexed and one not.
    Xa = copy(X_burn); Xb = copy(X_burn)
    reset_aggregates!(data); apply_derived_summaries!(pars, data, Xa)
    iffbs_mh!(pars, data, Xa, StableRNG(13); proposal=cheap, target=target)
    reset_aggregates!(data); apply_derived_summaries!(pars, data, Xb)
    iffbs_mh!(pars, data, Xb, StableRNG(13); proposal=cheap, target=slow_target)
    println()
    println("observation_weight (scalar) vs observation_process (vector) in the target")
    println("  scalar : ", round(1000 * t_fast; digits=2), " ms/sweep")
    println("  vector : ", round(1000 * t_slow; digits=2), " ms/sweep   (",
            round(t_slow / t_fast; digits=2), "x)")
    println("  same trajectory: ", Xa == Xb)
end

# The same comparison on a weakly coupled model: most of the force of infection is
# the external term `alpha`, so a neighbour's state says little about the focal and
# dropping the coupling from the filter costs little.
weak_pars = merge(pars, (; alpha=0.12, beta=0.03))
weak = epidemic_data(; n_individuals=N_IND, n_timepoints=N_T, group=GROUP,
                       trans_mat=spec, aggregates=aggs,
                       starting_state=(m, d, X, i, t) -> [1 - m.nu, m.nu, 0.0, 0.0],
                       observation_process=noisy_test,
                       observation_weight=noisy_test_weight,
                       coupled_transitions=[(:S, :E)],
                       results=results)
X_weak = fill(1, N_T, N_IND)
reset_aggregates!(weak); apply_derived_summaries!(weak_pars, weak, X_weak)
for _ in 1:30
    iffbs!(weak_pars, weak, X_weak, StableRNG(12))
end
compare_from("WEAKLY coupled (beta = $(weak_pars.beta), prevalence ~$(round(prevalence(X_weak); digits=2)))",
             weak_pars, weak, X_weak, epidemic_conditional_loglik(weak),
             uncorrected_proposal(weak))

# Restore the aggregates for the main model before section 3.
reset_aggregates!(data); apply_derived_summaries!(pars, data, X_burn)

# ---------------------------------------------------------------------------
# 3. SM-iFFBS: a semi-Markov target.
#
# The E -> I hazard now depends on how long the individual has been in E. That is
# a function of the PATH, so it cannot be written as a `@transitions` rate (those
# see only `(model, data, i, t)`) and no first-order filter can represent it.
#
# `step_logprob` is the seam: ONE function, receiving `X`, given to BOTH
# `epidemic_loglik` and `epidemic_conditional_loglik`. The filter keeps running
# the geometric chain, and MH corrects the difference.
# ---------------------------------------------------------------------------
function time_in_state(X, i, t)
    s = 0
    j = t
    @inbounds while j > 1 && X[j - 1, i] == X[t, i]
        s += 1
        j -= 1
    end
    s
end

function weibull_hazard(s, shape, scale)
    clamp(1 - exp(-(((s + 1) / scale)^shape - (s / scale)^shape)), 1e-12, 1 - 1e-12)
end

# Identical to `default_step_logprob` except for the E row. Note the survival
# factor is kept: `@survival` folded it into every live transition, so the
# semi-Markov version has to fold it in too or the two blocks disagree.
function sm_step(model, d, X, i, t)
    from, to = X[t, i], X[t + 1, i]
    if from == 2                                     # E
        p_surv = survival(model, d, i, t)
        h = weibull_hazard(time_in_state(X, i, t), model.prog_shape, model.prog_scale)
        to == 4 && return log(1 - p_surv + 1e-12)    # E -> D
        to == 3 && return log(p_surv * h + 1e-12)    # E -> I
        return log(p_surv * (1 - h) + 1e-12)         # E -> E
    end
    default_step_logprob(model, d, X, i, t)
end

sm_loglik = epidemic_loglik(data; step_logprob=sm_step)
sm_target = epidemic_conditional_loglik(data; step_logprob=sm_step)

# The proposal is the ORIGINAL geometric chain -- `iffbs_proposal(data)` -- which
# is now an approximation of the target rather than equal to it.
sm_stats = MHStats(N_IND)
t_sm = timed_sweeps((Xw, r) -> iffbs_mh!(pars, data, Xw, r;
                                         proposal=iffbs_proposal(data),
                                         target=sm_target, stats=sm_stats),
                    pars, data, X_burn)
Xs = copy(X_burn)
sync!(Xs)
for _ in 1:20
    iffbs_mh!(pars, data, Xs, StableRNG(5); proposal=iffbs_proposal(data),
              target=sm_target)
end

println()
println("SM-iFFBS: Weibull sojourn in E, geometric proposal")
println("  ", round(1000 * t_sm; digits=2), " ms/sweep   prevalence after 20 sweeps ",
        round(prevalence(Xs); digits=4))
println("  acceptance  : ", round(acceptance_rate(sm_stats); digits=3))

# The check that matters most, and the one worth copying into your own model's
# tests: the per-individual conditional must be the joint's restriction. If it is
# not, the latent block and the HMC block are sampling different posteriors.
sync!(Xs)
i = 7
joint0 = sm_loglik(pars, data, Xs) + obs_loglik(pars, data, Xs)
cond0 = sm_target(pars, data, Xs, i)
saved = copy(Xs[:, i])
for t in 1:N_T
    Xs[t, i] = rand(StableRNG(100 + t), 1:4)
end
sync!(Xs)
joint1 = sm_loglik(pars, data, Xs) + obs_loglik(pars, data, Xs)
cond1 = sm_target(pars, data, Xs, i)
Xs[:, i] = saved
println("  delta-consistency error: ",
        abs((cond1 - cond0) - (joint1 - joint0)))

# ---------------------------------------------------------------------------
# 4. As a PracticalBayes latent kernel.
#
# `epidemic_latent_sampler(; mh=true)` returns something callable as
# `(rng, model, X) -> X`, exactly like the plain sampler, but carrying `.stats`.
# ---------------------------------------------------------------------------
# Using the SM target here, since on this model that is the case where the
# correction earns its keep.
latent! = epidemic_latent_sampler(data; mh=true, proposal=iffbs_proposal(data),
                                        target=sm_target)
Xl = copy(X_burn)
sync!(Xl)
for _ in 1:5
    latent!(StableRNG(6), pars, Xl)
end
println()
println("latent! kernel (SM target): acceptance ",
        round(acceptance_rate(latent!.stats); digits=3))

# Passing a proposal WITHOUT asking for the correction is refused: that
# combination is the paper's "uncorrected-iFFBS", which targets the wrong
# conditional, and it must not be reachable by forgetting a keyword.
try
    epidemic_latent_sampler(data; proposal=cheap)
    println("  ERROR: the mh=false guard did not fire")
catch e
    println("  mh=false + proposal is refused, as intended")
end
