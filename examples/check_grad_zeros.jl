# Does the differentiated log-density actually depend on every parameter in the
# HMC block?
#
# `epidemic_loglik` (src/build.jl) sums the starting-state term and the
# transition probabilities. `observation_process` — the ONLY place thetas/rhos/
# phis/etas appear in the badger model — is used exclusively by the iFFBS forward
# filter (src/iffbs.jl), never by the likelihood. If that is right, the gradient
# entries for thetas/rhos/phis are pure prior (from the Beta(1,1) priors, whose
# log-density is flat, so the entries are the logit Jacobian term alone) and
# carry NO likelihood information.
#
# We are differentiating 61 parameters; if 18 of them (3 x 6 tests) contribute
# nothing from the likelihood, the AD is doing ~30% more chunk-passes than the
# problem requires.
#
# Run: julia --project=examples examples/check_grad_zeros.jl

ENV["BADGER_RUN"] = "0"
include(joinpath(@__DIR__, "badger_fit_reststotal_hmc.jl"))
using LogDensityProblems, Printf
using StableRNGs: StableRNG

init0 = badger_initial_params(raw; rng=StableRNG(13))
X0 = copy(raw.X_init)
init = (; X=X0, tau=init0.tau, alpha=init0.alpha, lambda=init0.lambda,
        beta=init0.beta, q=init0.q, c1=init0.c1, a1=init0.a1, b1=init0.b1,
        a2=init0.a2, b2=init0.b2, thetas=init0.thetas, rhos=init0.rhos,
        phis=init0.phis, etas=init0.etas, nu=hcat(init0.nuE, init0.nuI))
reset_aggregates!(data)
apply_derived_summaries!(init0, data, X0)

m = badger_base(data, data.n_timepoints, data.n_individuals, G, NT, NS, NNU, loglik, obs_loglik)
nuts_names = (:tau, :alpha, :lambda, :beta, :q, :c1, :a1, :b1, :a2, :b2,
              :thetas, :rhos, :phis)
layout, theta0, store0 = build_layout(m; flat=nuts_names, values=(:X, :etas, :nu), init=init)

ldf = PracticalBayes.LogDensityFunction(m, layout, store0, ADTYPE; θ0=theta0)
lp, g = LogDensityProblems.logdensity_and_gradient(ldf, theta0)

# Flat-vector order, per the fit script's comment block.
names = vcat("tau", ["alpha[$i]" for i in 1:G], "lambda", "beta", "q",
             "c1", "a1", "b1", "a2", "b2",
             ["thetas[$i]" for i in 1:NT], ["rhos[$i]" for i in 1:NT],
             ["phis[$i]" for i in 1:NT])

println("logdensity = ", lp, "   (", length(theta0), " parameters)\n")

# The decisive test: perturb ONLY the test parameters and see whether the
# LIKELIHOOD term moves. Comparing gradient entries alone can't separate
# "flat prior" from "absent from the likelihood", so call loglik directly.
pars0 = (; tau=init0.tau, alpha=init0.alpha, lambda=init0.lambda, beta=init0.beta,
         q=init0.q, a1=init0.a1, b1=init0.b1, a2=init0.a2, b2=init0.b2, c1=init0.c1,
         thetas=init0.thetas, rhos=init0.rhos, phis=init0.phis,
         etas=init0.etas, nu=hcat(init0.nuE, init0.nuI))
ll0 = loglik(pars0, data, X0)

pars1 = merge(pars0, (; thetas=init0.thetas .* 0.5, rhos=init0.rhos .* 0.5,
                       phis=init0.phis .* 0.5, etas=init0.etas .* 0.5))
ll1 = loglik(pars1, data, X0)

@printf("epidemic_loglik with original thetas/rhos/phis/etas : %.10f\n", ll0)
@printf("epidemic_loglik with all four HALVED               : %.10f\n", ll1)
@printf("difference                                          : %.3e\n", ll1 - ll0)
println()
if abs(ll1 - ll0) < 1e-10
    println(">>> `epidemic_loglik` alone does NOT depend on thetas/rhos/phis/etas.")
    println(">>> That is BY DESIGN — it covers the starting state and transitions only.")
    println(">>> The observation parameters are informed by `epidemic_obs_loglik`,")
    println(">>> which the model adds separately. Check the GRADIENT entries below:")
    println(">>>   before the 2026-07-20 fix they were O(0.1-1)  (prior/Jacobian only);")
    println(">>>   with the obs term present they are O(10-1000) (real likelihood info).")
else
    println(">>> `epidemic_loglik` DOES depend on them — unexpected for this model.")
end

println("\ngradient entries for the test parameters (prior/Jacobian only, if above holds):")
for k in (G+11):length(theta0)
    @printf("  %-12s % .6e\n", names[k], g[k])
end
println("\nfor contrast, the epidemic/survival entries:")
for k in vcat(1, (G+2):(G+10))
    @printf("  %-12s % .6e\n", names[k], g[k])
end
