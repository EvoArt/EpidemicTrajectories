# Correctness on a subsample: does the badger iFFBS reconstruct hidden states it
# was never shown? Simulate from known parameters with the badger structure, then
# start the sampler from an all-susceptible X and see if it rebuilds the truth.
include(joinpath(@__DIR__, "badger_model.jl"))
using StableRNGs, Statistics

n_groups, n_per, n_t, n_tests, n_seasons = 4, 8, 30, 2, 4
n_ind = n_groups * n_per
grp = repeat(1:n_groups; inner=n_per)

social_group = repeat(grp, 1, n_t)                 # fixed groups for this check
age = [t for i in 1:n_ind, t in 1:n_t]   # young badgers
season = make_season_vec(n_seasons, 1, n_t)
affected = Matrix{Vector{Int}}(undef, n_t, n_ind)
for t in 1:n_t, i in 1:n_ind
    affected[t, i] = [j for j in 1:n_ind if grp[j] == grp[i] && j != i]
end

function build(; capture, tests, last_capture)
    epidemic_data(;
        n_individuals=n_ind, n_timepoints=n_t,
        trans_mat=badger_transitions(),
        starting_state=badger_starting_state,
        observation_process=badger_observations,
        aggregates=badger_aggregates(n_groups, n_t),
        affected_individuals=affected, state_space=BADGER_STATES,
        social_group=social_group, age=age, capture=capture, tests=tests,
        season=season, birth_time=fill(0, n_ind), last_capture_time=last_capture,
        nu_times=[1], K=85.0, k=1, n_groups=n_groups,
    )
end

# Truth: strong transmission and progression so there is signal; survival ~1 so
# the check is about infection, not mortality.
truth = (; tau=1.0, alpha=fill(0.05, n_groups), lambda=1.0, beta=0.9, q=0.0,
         c1=1e-8, a1=1e-8, b1=1.0, a2=1e-12, b2=0.01,   # near-flat survival: tiny senescence
         thetas=fill(0.95, n_tests), rhos=fill(0.9, n_tests), phis=fill(0.99, n_tests),
         etas=fill(0.6, n_seasons), nuE=[0.1], nuI=[0.1])   # capture is imperfect: not seen != dead

no_tests = fill(-1, n_t, n_ind, n_tests)
sim_data = build(capture=fill(0, n_t, n_ind), tests=no_tests, last_capture=fill(0, n_ind))
X_true = epidemic_simulator(sim_data)(StableRNG(11), truth)
println("simulated: S=", round(mean(X_true.==1),digits=3), " E=", round(mean(X_true.==2),digits=3),
        " I=", round(mean(X_true.==3),digits=3), " D=", round(mean(X_true.==4),digits=3))

# Observe: capture everyone every 3rd step; a near-perfect test reveals I (and,
# via rho, partially E).
capture = fill(0, n_t, n_ind); capture[1:3:n_t, :] .= 1
tests = fill(-1, n_t, n_ind, n_tests)
for t in 1:3:n_t, i in 1:n_ind
    X_true[t,i] == 4 && continue
    tests[t,i,1] = X_true[t,i] == 3 ? 1 : (X_true[t,i] == 2 ? 1 : 0)
end
last_capture = [something(findlast(==(1), @view capture[:, i]), 0) for i in 1:n_ind]
fit_data = build(capture=capture, tests=tests, last_capture=last_capture)

X = fill(1, n_t, n_ind)                       # all susceptible: wrong on purpose
reset_aggregates!(fit_data); apply_derived_summaries!(truth, fit_data, X)
latent! = epidemic_latent_sampler(fit_data)
rng = StableRNG(12)
for s in 1:30; latent!(rng, truth, X); end

println("recovered: S=", round(mean(X.==1),digits=3), " E=", round(mean(X.==2),digits=3),
        " I=", round(mean(X.==3),digits=3), " D=", round(mean(X.==4),digits=3))
println("infected-state (E or I) agreement with truth: ",
        round(mean((X.>=2).&(X.<=3) .== (X_true.>=2).&(X_true.<=3)), digits=3))
pos = findall(tests[:,:,1] .== 1)
println("P(E or I) at test-positive cells: ", round(mean(2 .<= X[pos] .<= 3), digits=3), " (should be ~1)")
# the invariant must still hold
live = copy(fit_data.aggregates[:n_infectious])
reset_aggregates!(fit_data); apply_derived_summaries!(truth, fit_data, X)
println("incremental aggregate == recompute: ", live == fit_data.aggregates[:n_infectious])
