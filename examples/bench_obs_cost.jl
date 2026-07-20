# How expensive is the observation likelihood, and how much does the scalar
# (allocation-free) path save?
#
# THE PROBLEM. `epidemic_obs_loglik`'s default path calls
# `observation_process(model, data, X, i, t)` at EVERY (i,t) and reads ONE entry
# of the returned vector. Each call allocates a fresh n_states array — ~187k
# allocations per likelihood call on the badger model, each of Duals under AD.
#
# This is the same trap `epidemic_loglik` already avoids: its docstring records
# that building the full transition matrix per (i,t) was "~380k matrix
# allocations per call" and dominated the gradient, which is why it uses the
# scalar `transition_prob` instead. The observation term reintroduced the
# pattern; `observation_weight` is its `transition_prob`.
#
# Measures: primal cost + allocations for (a) the transition likelihood as a
# reference point, (b) obs via the vector path, (c) obs via the scalar path.
# Then the GRADIENT of each, since that is what the sweep actually pays.
#
# Run: JULIA_NUM_THREADS=8 julia --project=examples examples/bench_obs_cost.jl

ENV["BADGER_RUN"] = "0"
include(joinpath(@__DIR__, "badger_fit_reststotal_hmc.jl"))
include(joinpath(@__DIR__, "badger_model_obssplit.jl"))
using Printf, Statistics
using StableRNGs: StableRNG

bs = badger_data_obssplit(DATA_DIR)
DATA, RAW = bs.data, bs.raw

init0 = badger_initial_params(RAW; rng=StableRNG(13))
pars = (; tau=init0.tau, alpha=init0.alpha, lambda=init0.lambda, beta=init0.beta,
        q=init0.q, a1=init0.a1, b1=init0.b1, a2=init0.a2, b2=init0.b2, c1=init0.c1,
        thetas=init0.thetas, rhos=init0.rhos, phis=init0.phis,
        etas=init0.etas, nu=hcat(init0.nuE, init0.nuI))
X0 = copy(RAW.X_init)

reset_aggregates!(DATA)
apply_derived_summaries!(init0, DATA, X0)

trans_ll = epidemic_loglik(DATA)
obs_vec  = epidemic_obs_loglik(DATA; observation_process=badger_obs_tests)
obs_scal = epidemic_obs_loglik(DATA; observation_process=badger_obs_tests,
                               observation_weight=badger_obs_tests_weight)

# Correctness before speed — a faster wrong answer is worthless.
v, s = obs_vec(pars, DATA, X0), obs_scal(pars, DATA, X0)
@printf("vector path = %.10f\nscalar path = %.10f\ndifference  = %.3e\n\n", v, s, s - v)
abs(s - v) < 1e-8 || error("scalar and vector paths disagree — fix before benchmarking")

trans_ll(pars, DATA, X0); obs_vec(pars, DATA, X0); obs_scal(pars, DATA, X0)  # warm up

n = 20
t_trans = minimum(@elapsed(trans_ll(pars, DATA, X0)) for _ in 1:n)
t_vec   = minimum(@elapsed(obs_vec(pars, DATA, X0)) for _ in 1:n)
t_scal  = minimum(@elapsed(obs_scal(pars, DATA, X0)) for _ in 1:n)
a_trans = @allocated trans_ll(pars, DATA, X0)
a_vec   = @allocated obs_vec(pars, DATA, X0)
a_scal  = @allocated obs_scal(pars, DATA, X0)

println("=== PRIMAL, one call (min of $n) ===")
@printf("%-26s %9.4f s  %12d bytes\n", "epidemic_loglik (trans)", t_trans, a_trans)
@printf("%-26s %9.4f s  %12d bytes\n", "obs_loglik (vector path)", t_vec, a_vec)
@printf("%-26s %9.4f s  %12d bytes\n", "obs_loglik (scalar path)", t_scal, a_scal)
@printf("\nscalar vs vector: %.2fx faster, %.1fx fewer bytes\n",
        t_vec / t_scal, a_vec / max(a_scal, 1))

ncells = sum(min(DATA.sampling_period[i][2], DATA.n_timepoints) - DATA.sampling_period[i][1] + 1
             for i in 1:DATA.n_individuals)
@printf("(%d (i,t) cells; vector path allocates ~%.0f bytes/cell)\n", ncells, a_vec / ncells)

# --- The gradient is what the sweep actually pays ----------------------------
println("\n=== GRADIENT (what HMC calls L=30 times per sweep) ===")
using LogDensityProblems

@model function badger_withobs_local(data, n_time, n_ind, n_groups, n_tests, n_seasons, n_nu, ll_fn, obs_fn)
    tau ~ Exponential(10.0); alpha ~ PracticalBayes.filldist(Exponential(1.0), n_groups)
    lambda ~ Exponential(1.0); beta ~ Exponential(1.0); q ~ Beta(1, 1)
    c1 ~ Exponential(1.0); a1 ~ Exponential(1.0); b1 ~ Exponential(1.0)
    a2 ~ Exponential(1.0); b2 ~ Exponential(1.0)
    thetas ~ PracticalBayes.filldist(Beta(1, 1), n_tests)
    rhos ~ PracticalBayes.filldist(Beta(1, 1), n_tests)
    phis ~ PracticalBayes.filldist(Beta(1, 1), n_tests)
    etas ~ PracticalBayes.filldist(Beta(1, 1), n_seasons)
    nu ~ NuSimplex(n_nu, [8.0, 1.0, 1.0])
    X ~ TrajectoryLatent(n_time, n_ind)
    pars = (; tau, alpha, lambda, beta, q, a1, b1, a2, b2, c1, thetas, rhos, phis, etas, nu)
    @addlogprob! ll_fn(pars, data, X) + obs_fn(pars, data, X)
end

function grad_time(obs_fn; n=10)
    m = badger_withobs_local(DATA, DATA.n_timepoints, DATA.n_individuals, G, NT, NS, NNU,
                             trans_ll, obs_fn)
    init = (; X=X0, tau=init0.tau, alpha=init0.alpha, lambda=init0.lambda,
            beta=init0.beta, q=init0.q, c1=init0.c1, a1=init0.a1, b1=init0.b1,
            a2=init0.a2, b2=init0.b2, thetas=init0.thetas, rhos=init0.rhos,
            phis=init0.phis, etas=init0.etas, nu=hcat(init0.nuE, init0.nuI))
    names = (:tau, :alpha, :lambda, :beta, :q, :c1, :a1, :b1, :a2, :b2,
             :thetas, :rhos, :phis)
    layout, theta0, store0 = build_layout(m; flat=names, values=(:X, :etas, :nu), init=init)
    ldf = PracticalBayes.LogDensityFunction(m, layout, store0, ADTYPE; θ0=theta0)
    LogDensityProblems.logdensity_and_gradient(ldf, theta0)   # warm up
    minimum(@elapsed(LogDensityProblems.logdensity_and_gradient(ldf, theta0)) for _ in 1:n)
end

g_vec  = grad_time(obs_vec)
g_scal = grad_time(obs_scal)
@printf("gradient, obs vector path : %.4f s\n", g_vec)
@printf("gradient, obs scalar path : %.4f s\n", g_scal)
@printf("speedup                   : %.2fx\n", g_vec / g_scal)
@printf("\nper sweep (L=30): %.2f s -> %.2f s, saving %.2f s/sweep\n",
        30 * g_vec, 30 * g_scal, 30 * (g_vec - g_scal))
