# Verify the factored observation process before relying on it.
#
# Two claims must hold:
#
# 1. capture_factor .* test_factor == the original badger_observations, at EVERY
#    (i, t). If this fails, the split changes the iFFBS forward filter and hence
#    the posterior.
#
# 2. The log-likelihood really is additive in the two factors:
#       obs_loglik(full) == obs_loglik(capture) + obs_loglik(tests)
#    This is what licenses passing ONLY the test factor to epidemic_obs_loglik
#    while `etas` stays in its conjugate block — dropping the capture factor from
#    the likelihood must drop exactly its term and nothing else.
#
# Run: julia --project=examples examples/check_obs_split.jl

ENV["BADGER_RUN"] = "0"
include(joinpath(@__DIR__, "badger_fit_reststotal_hmc.jl"))
include(joinpath(@__DIR__, "badger_model_obssplit.jl"))
using Printf
using StableRNGs: StableRNG

init0 = badger_initial_params(raw; rng=StableRNG(13))
pars = (; tau=init0.tau, alpha=init0.alpha, lambda=init0.lambda, beta=init0.beta,
        q=init0.q, a1=init0.a1, b1=init0.b1, a2=init0.a2, b2=init0.b2, c1=init0.c1,
        thetas=init0.thetas, rhos=init0.rhos, phis=init0.phis,
        etas=init0.etas, nu=hcat(init0.nuE, init0.nuI))
X0 = copy(raw.X_init)

# --- Claim 1: the product reproduces the original, cell by cell --------------
println("=== Claim 1: capture .* tests == badger_observations ===")
maxdiff = 0.0
ncells = 0
for i in 1:data.n_individuals
    first_t, last_t = data.sampling_period[i]
    for t in first_t:min(last_t, data.n_timepoints)
        w_orig  = badger_observations(pars, data, X0, i, t)
        w_split = badger_observations_split(pars, data, X0, i, t)
        global maxdiff = max(maxdiff, maximum(abs.(w_orig .- w_split)))
        global ncells += 1
    end
end
@printf("checked %d (i,t) cells\nmax abs difference: %.3e\n", ncells, maxdiff)
println(maxdiff == 0.0 ? ">>> EXACT match.\n" :
        maxdiff < 1e-12 ? ">>> match to floating-point tolerance.\n" :
        ">>> MISMATCH — do not use the split.\n")

# --- Claim 2: the log-likelihood is additive in the factors ------------------
println("=== Claim 2: obs_loglik(full) == obs_loglik(capture) + obs_loglik(tests) ===")
ll_full    = epidemic_obs_loglik(data; observation_process=badger_observations_split)(pars, data, X0)
ll_capture = epidemic_obs_loglik(data; observation_process=badger_obs_capture)(pars, data, X0)
ll_tests   = epidemic_obs_loglik(data; observation_process=badger_obs_tests)(pars, data, X0)

@printf("obs_loglik(full)              = %.10f\n", ll_full)
@printf("obs_loglik(capture)           = %.10f\n", ll_capture)
@printf("obs_loglik(tests)             = %.10f\n", ll_tests)
@printf("capture + tests               = %.10f\n", ll_capture + ll_tests)
@printf("difference from full          = %.3e\n", (ll_capture + ll_tests) - ll_full)
println()

# --- The point of the whole exercise: do thetas/rhos now matter? ------------
println("=== Does the TEST-only likelihood actually see thetas/rhos/phis? ===")
obs_tests = epidemic_obs_loglik(data; observation_process=badger_obs_tests)
base = obs_tests(pars, data, X0)
halved = obs_tests(merge(pars, (; thetas=pars.thetas .* 0.5, rhos=pars.rhos .* 0.5,
                                 phis=pars.phis .* 0.5)), data, X0)
@printf("obs_loglik(tests), original   = %.10f\n", base)
@printf("obs_loglik(tests), 3x halved  = %.10f\n", halved)
@printf("difference                    = %.3e\n", halved - base)
println(abs(halved - base) > 1e-6 ?
        ">>> YES — thetas/rhos/phis now carry likelihood information." :
        ">>> NO — something is wrong, the term is still flat.")

# --- Claim 3: the SCALAR forms agree with the vector forms, entry for entry ---
# These are the allocation-free path used by epidemic_obs_loglik's
# `observation_weight` keyword. They must reproduce the vector versions exactly;
# a silent disagreement would change the posterior with no error.
println("\n=== Claim 3: scalar weights == vector weights, every (i,t,s) ===")
function check_scalar(vecfn, scalfn, label)
    maxd = 0.0; n = 0
    for i in 1:data.n_individuals
        first_t, last_t = data.sampling_period[i]
        for t in first_t:min(last_t, data.n_timepoints)
            w = vecfn(pars, data, X0, i, t)
            for s in 1:data.n_states
                maxd = max(maxd, abs(w[s] - scalfn(pars, data, X0, i, t, s)))
                n += 1
            end
        end
    end
    @printf("%-22s  %8d entries   max diff %.3e  %s\n", label, n, maxd,
            maxd == 0.0 ? "EXACT" : maxd < 1e-12 ? "ok (fp)" : "MISMATCH")
    maxd
end
d1 = check_scalar(badger_obs_tests,        badger_obs_tests_weight,   "tests")
d2 = check_scalar(badger_obs_capture,      badger_obs_capture_weight, "capture")
d3 = check_scalar(badger_observations_split, badger_obs_split_weight, "split (product)")
println(max(d1, d2, d3) < 1e-12 ?
        ">>> scalar path agrees with the vector path." :
        ">>> MISMATCH — the scalar path would change the posterior.")

# And the whole likelihood, computed both ways, must match.
ll_vec  = epidemic_obs_loglik(data; observation_process=badger_obs_tests)(pars, data, X0)
ll_scal = epidemic_obs_loglik(data; observation_process=badger_obs_tests,
                              observation_weight=badger_obs_tests_weight)(pars, data, X0)
@printf("\nobs_loglik via vector = %.10f\nobs_loglik via scalar = %.10f\ndifference            = %.3e\n",
        ll_vec, ll_scal, ll_scal - ll_vec)

println("\n=== And is it free of `etas` (so the conjugate block is safe)? ===")
etas_moved = obs_tests(merge(pars, (; etas=pars.etas .* 0.5)), data, X0)
@printf("obs_loglik(tests), etas halved= %.10f\n", etas_moved)
@printf("difference                    = %.3e\n", etas_moved - base)
println(etas_moved == base ?
        ">>> YES — the test factor does not depend on etas; no double-counting." :
        ">>> NO — etas leaks into the test factor; the conjugate block would double-count.")
