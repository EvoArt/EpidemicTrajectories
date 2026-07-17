# The latent sampler, on a small two-state S/I model: simulate, score, and
# resample the hidden trajectory.
#
# The gates that matter:
#  * the sampler reconstructs an epidemic from a deliberately wrong starting X,
#  * the incrementally-maintained aggregate stays exactly equal to a from-scratch
#    recompute (the property the reversible update exists to provide).

# A perfect test, observed on the days where `results` is not negative. Supplied
# by the caller as the model's own observation process — the package has no
# built-in notion of what is observed.
function _perfect_test(model, data, X, i, t)
    w = ones(Float64, data.n_states)
    y = data.results[t, i]
    y < 0 && return w                       # not tested: no information
    w[y == 1 ? 1 : 2] = 0.0                 # positive rules out S; negative rules out I
    w
end

function _si_setup(; n_groups=4, n_per_group=6, n_t=40, results=nothing)
    n_ind = n_groups * n_per_group
    group = repeat(1:n_groups; inner=n_per_group)
    state_space = [:S, :I]

    aggs = @aggregate state_space begin
        @array n_infected Int (n_groups, n_t)
        n_infected[data.group[i], t] += (state == :I)
    end

    infection(model, data, i, t) =
        -expm1(-(model.α + model.β * data.aggregates[:n_infected][data.group[i], t]))
    recovery(model, data, i, t) = 1 / model.m

    spec = @transitions state_space begin
        S -> I = infection
        I -> S = recovery
    end

    starting_state = (model, data, X, i, t) -> [1 - model.ν, model.ν]

    kw = results === nothing ? (;) :
         (; observation_process=_perfect_test, results=results)
    data = epidemic_data(;
        n_individuals=n_ind, n_timepoints=n_t, group=group,
        trans_mat=spec, starting_state=starting_state, aggregates=aggs, kw...,
    )
    (; data, n_ind, n_t, n_groups)
end

@testset "simulator: shape, states, and determinism" begin
    s = _si_setup()
    pars = (; α=0.02, β=0.05, m=5.0, ν=0.1, θʳ=0.8, θᶠ=0.5)
    simulate = epidemic_simulator(s.data)

    X = simulate(StableRNG(1), pars)
    @test size(X) == (s.n_t, s.n_ind)
    @test all(x -> x in (1, 2), X)

    X2 = simulate(StableRNG(1), pars)
    @test X == X2                       # same seed, same trajectory

    # The simulator must leave the aggregates agreeing with the whole of `X` —
    # including the final timepoint — since that invariant is what the likelihood
    # and the latent sampler assume on entry.
    live = copy(s.data.aggregates[:n_infected])
    reset_aggregates!(s.data)
    apply_derived_summaries!(pars, s.data, X)
    @test live == s.data.aggregates[:n_infected]
end

@testset "loglik: finite, negative, and responds to the parameters" begin
    s = _si_setup()
    pars = (; α=0.02, β=0.05, m=5.0, ν=0.1, θʳ=0.8, θᶠ=0.5)
    X = epidemic_simulator(s.data)(StableRNG(2), pars)

    reset_aggregates!(s.data)
    apply_derived_summaries!(pars, s.data, X)
    loglik = epidemic_loglik(s.data)

    ll = loglik(pars, s.data, X)
    @test isfinite(ll)
    @test ll < 0

    # the trajectory should be more likely under the parameters that generated it
    # than under a badly wrong recovery rate
    wrong = (; α=0.02, β=0.05, m=1.05, ν=0.1, θʳ=0.8, θᶠ=0.5)
    @test loglik(pars, s.data, X) > loglik(wrong, s.data, X)
end

@testset "iFFBS: reconstructs the trajectory from a wrong starting X" begin
    pars = (; α=0.05, β=0.08, m=5.0, ν=0.15, θʳ=0.9, θᶠ=0.9)

    # simulate a truth, then observe a perfect test on some days so the sampler
    # has real information to reconstruct from
    sim = _si_setup()
    X_true = epidemic_simulator(sim.data)(StableRNG(3), pars)
    R = fill(-1, sim.n_t, sim.n_ind)
    for t in 1:4:sim.n_t, i in 1:sim.n_ind
        R[t, i] = X_true[t, i] == 2 ? 1 : 0
    end

    s = _si_setup(results=R)
    X = fill(1, s.n_t, s.n_ind)         # everyone susceptible: wrong on purpose
    reset_aggregates!(s.data)
    apply_derived_summaries!(pars, s.data, X)
    @test mean(X .== 2) == 0.0

    latent! = epidemic_latent_sampler(s.data)
    rng = StableRNG(4)
    for _ in 1:40
        latent!(rng, pars, X)
    end

    # the epidemic is rebuilt, and every observed-positive cell is called infected
    @test mean(X .== 2) > 0.05
    pos = findall(R .== 1)
    @test all(X[pos] .== 2)
end

@testset "epidemic_data: user-supplied structure" begin
    # sampling_period, affected_individuals and the observation process are the
    # user's to decide; the package only supplies defaults.
    spec = @transitions [:S, :I] begin
        S -> I = 0.1
        I -> S = 0.2
    end
    aggs = @aggregate [:S, :I] begin
        @array a Int (1, 3)
        a[1, t] += (state == :I)
    end
    common = (; n_individuals=2, n_timepoints=3, trans_mat=spec,
              starting_state=(model, data, X, i, t) -> [0.9, 0.1], aggregates=aggs)

    # defaults: whole window for everyone, no observations
    d = epidemic_data(; common...)
    @test d.sampling_period == [(1, 3), (1, 3)]
    @test d.observation_process === no_observations

    # user-supplied per-individual windows
    d2 = epidemic_data(; common..., sampling_period=[(1, 2), (2, 3)])
    @test d2.sampling_period == [(1, 2), (2, 3)]
    @test_throws ArgumentError epidemic_data(; common..., sampling_period=[(1, 2)])

    # user-supplied coupling structure, indexed [t, i] so it may vary over time
    af = [Int[] for t in 1:3, i in 1:2]
    af[1, 1] = [2]                       # who individual 1 affects at t=1
    d3 = epidemic_data(; common..., affected_individuals=af)
    @test d3.affected_individuals[1, 1] == [2]
    @test_throws ArgumentError epidemic_data(; common...,
                                             affected_individuals=[Int[] for t in 1:2, i in 1:2])

    # extras are reachable by name, and the package never inspects them
    d4 = epidemic_data(; common..., my_covariate=[1.5, 2.5])
    @test d4.my_covariate == [1.5, 2.5]
    @test_throws ArgumentError d4.not_a_thing
end

@testset "iFFBS: the incremental aggregate equals a from-scratch recompute" begin
    # This is the property the reversible @aggregate update exists to provide: the
    # sampler never rebuilds the aggregates, so they must stay exactly consistent.
    s = _si_setup()
    pars = (; α=0.05, β=0.08, m=5.0, ν=0.15, θʳ=0.9, θᶠ=0.9)
    X = epidemic_simulator(s.data)(StableRNG(5), pars)

    reset_aggregates!(s.data)
    apply_derived_summaries!(pars, s.data, X)

    latent! = epidemic_latent_sampler(s.data)
    rng = StableRNG(6)
    for _ in 1:20
        latent!(rng, pars, X)
        incremental = copy(s.data.aggregates[:n_infected])

        reset_aggregates!(s.data)
        apply_derived_summaries!(pars, s.data, X)
        @test incremental == s.data.aggregates[:n_infected]
    end
end

@testset "epidemic_data: verbose fallback takes explicit summaries" begin
    # An aggregate @aggregate cannot express: supply the storage and a hand-written
    # reversible summary directly.
    n_ind, n_t = 4, 6
    group = [1, 1, 2, 2]
    spec = @transitions [:S, :I] begin
        S -> I = 0.1
        I -> S = 0.2
    end

    hand_written = function (model, data, X, s, i, t; reverse=false)
        if s == 2
            data.aggregates[:mine][data.group[i], t] += reverse ? -1 : 1
        end
        nothing
    end

    data = epidemic_data(
        n_individuals=n_ind, n_timepoints=n_t, group=group,
        trans_mat=spec,
        starting_state=(model, data, X, i, t) -> [0.9, 0.1],
        aggregates=Dict{Symbol,Any}(:mine => zeros(Int, 2, n_t)),
        derived_summaries=[hand_written],
    )
    @test data.aggregates[:mine] isa Matrix{Int}
    @test length(data.derived_summaries) == 1

    # and it round-trips like a generated one
    hand_written(nothing, data, nothing, 2, 1, 1)
    @test data.aggregates[:mine][1, 1] == 1
    hand_written(nothing, data, nothing, 2, 1, 1; reverse=true)
    @test data.aggregates[:mine][1, 1] == 0
end

@testset "epidemic_data: mixing the two aggregate styles errors clearly" begin
    spec = @transitions [:S, :I] begin
        S -> I = 0.1
    end
    aggs = @aggregate [:S, :I] begin
        @array a Int (1, 1)
        a[1, 1] += (state == :I)
    end
    common = (; n_individuals=2, n_timepoints=3, group=[1, 1], trans_mat=spec,
              starting_state=(model, data, X, i, t) -> [0.9, 0.1],
              )

    # an @aggregate declaration already carries its updates
    @test_throws ErrorException epidemic_data(; common..., aggregates=aggs,
                                             derived_summaries=[(a...; kw...) -> nothing])
    # a plain Dict needs them
    @test_throws ErrorException epidemic_data(; common..., aggregates=Dict{Symbol,Any}())
end
