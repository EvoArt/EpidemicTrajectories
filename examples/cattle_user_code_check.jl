# Correctness check against the refactored package + sugar: start iFFBS from a
# deliberately wrong all-susceptible X and confirm it rebuilds the epidemic, and
# that the incrementally-maintained aggregate matches a from-scratch recompute.
ENV["WALKTHROUGH_RUN"] = "0"
include(joinpath(@__DIR__, "cattle_user_code.jl"))
using Statistics: mean

X0 = fill(1, n_timepoints, n_individuals)   # everyone susceptible — wrong on purpose
reset_aggregates!(fit_data)
apply_derived_summaries!(true_pars, fit_data, X0)
println("start: prevalence = ", round(mean(X0 .== 2); digits=3))

rng = StableRNG(11)
for _ in 1:60
    latent!(rng, true_pars, X0)
end
println("after 60 sweeps at true params:")
println("  prevalence = ", round(mean(X0 .== 2); digits=3), "  (truth ", round(mean(X_true .== 2); digits=3), ")")
println("  agreement with truth = ", round(mean(X0 .== X_true); digits=3))
pos = findall(Rmask .== 1)
println("  P(infected) at test-positive cells = ", round(mean(X0[pos] .== 2); digits=3), " (should be 1.0)")

agg_live = copy(fit_data.aggregates[:n_infected])
reset_aggregates!(fit_data)
apply_derived_summaries!(true_pars, fit_data, X0)
println("  incremental aggregate == recompute: ", agg_live == fit_data.aggregates[:n_infected])
