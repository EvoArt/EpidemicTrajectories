# Start iFFBS from an ALL-SUSCEPTIBLE X (not X_true) to check the sampler
# genuinely reconstructs the latent trajectory rather than coasting on a
# correct initialisation.
ENV["WALKTHROUGH_RUN"] = "0"
include(joinpath(@__DIR__, "walkthrough_cattle.jl"))

using Statistics: mean

X0 = fill(1, n_timepoints, n_individuals)   # everyone susceptible — deliberately wrong
init_pars = (; α=0.05, β=0.05, m=5.0, ν=0.1, θʳ=0.7, θᶠ=0.4)
reset_aggregates!(fit_data)
apply_derived_summaries!(init_pars, fit_data, X0)

println("start: agreement with truth = ", round(mean(X0 .== X_true); digits=3),
        "  prevalence = ", round(mean(X0 .== 2); digits=3))

rng = StableRNG(11)
for sweep in 1:60
    latent!(rng, true_pars, X0)     # iFFBS at the TRUE params
end

println("after 60 iFFBS sweeps at true params:")
println("  agreement with truth = ", round(mean(X0 .== X_true); digits=3))
println("  prevalence = ", round(mean(X0 .== 2); digits=3), "  (truth ", round(mean(X_true .== 2); digits=3), ")")

# a test-positive cell must be called infected (specificity is 1 in this model)
pos = findall(Rmask .== 1)
println("  P(infected) at RAMS-positive cells = ", round(mean(X0[pos] .== 2); digits=3), " (should be 1.0)")

# aggregates must still agree with a from-scratch recompute
agg_live = copy(fit_data.aggregates[:n_infected_per_group])
reset_aggregates!(fit_data)
apply_derived_summaries!(true_pars, fit_data, X0)
println("  incremental aggregate == recompute: ", agg_live == fit_data.aggregates[:n_infected_per_group])
