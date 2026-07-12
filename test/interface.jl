# The high-level entry point in the (model, X, data) vocabulary:
#   model = parameters, X = latent trajectory, data = observations + structure.

using ForwardDiff

@testset "epidemic_model: loglik(model, X, data) matches trajectory_loglik" begin
    ss = SI
    rates = TwoStateSI()
    pars = (; α=0.05, β=0.05, m=5.0)
    group = ones(Int, 8)
    init = [0.9, 0.1]

    rng = StableRNG(1)
    states, d0 = simulate_trajectory(rng, ss, rates, pars, group, init; n_times=40)
    rams = DiagnosticTest(; sensitivity=p -> 0.8)
    R = simulate_observations(rng, (rams,), (; θ=0.8), ss, states)[1]

    em = epidemic_model(ss, rates; tests=(rams,))
    data = build_data(em, group; observations=(R,), initial_prob=init)
    @test data isa EpidemicData
    @test data.n_ind == 8
    @test data.n_times == 40

    # the closure's value equals the hand-assembled trajectory_loglik
    model = (; state_space=ss, rates=rates, pars=pars)
    @test em.loglik(pars, states, data) ≈ trajectory_loglik(pars, model, d0)
end

@testset "epidemic_model: loglik is autodiff-friendly in model (params)" begin
    ss = SI; rates = TwoStateSI(); group = ones(Int, 8); init = [0.9, 0.1]
    truep = (; α=0.05, β=0.05, m=5.0)
    states, _ = simulate_trajectory(StableRNG(2), ss, rates, truep, group, init; n_times=50)
    em = epidemic_model(ss, rates)
    data = build_data(em, group; n_times=50, initial_prob=init)

    g = ForwardDiff.gradient(v -> em.loglik((; α=v[1], β=v[2], m=v[3]), states, data),
                             [0.05, 0.05, 5.0])
    @test all(isfinite, g)
    @test !all(iszero, g)
end

@testset "epidemic_model: simulate(rng, model, data)" begin
    ss = SI; rates = TwoStateSI(); group = repeat(1:3; inner=4); init = [0.9, 0.1]
    rams = DiagnosticTest(; sensitivity=p -> p.θ)
    em = epidemic_model(ss, rates; tests=(rams,))
    data = build_data(em, group; initial_prob=init, n_times=30)

    out = em.simulate(StableRNG(3), (; α=0.1, β=0.1, m=4.0, θ=0.8), data)
    @test size(out.states) == (12, 30)
    @test all(s -> s in (0, 1), out.states)
    @test length(out.observations) == 1
    @test size(out.observations[1]) == (12, 30)
end

@testset "epidemic_model: latent!(rng, model, X, data) resamples X in place" begin
    ss = SI; rates = TwoStateSI(); group = repeat(1:3; inner=4); init = [0.9, 0.1]
    pars = (; α=0.1, β=0.1, m=4.0, θ=0.8)
    rams = DiagnosticTest(; sensitivity=p -> p.θ)

    states, _ = simulate_trajectory(StableRNG(4), ss, rates, pars, group, init; n_times=25)
    R = simulate_observations(StableRNG(4), (rams,), pars, ss, states)[1]
    em = epidemic_model(ss, rates; tests=(rams,))
    data = build_data(em, group; observations=(R,), initial_prob=init)

    X = zeros(Int, size(states))
    returned = em.latent!(StableRNG(5), pars, X, data)
    @test returned === X            # mutates and returns the same matrix
    @test size(X) == size(states)
    @test all(x -> x in (0, 1), X)
    # a test-positive cell (specificity 1) must be called infected
    pos = findall(R .== 1)
    @test all(X[pos] .== 1)
end

@testset "epidemic_model: errors clearly on missing initial_prob / n_times" begin
    ss = SI; rates = TwoStateSI(); group = ones(Int, 4)
    em = epidemic_model(ss, rates)
    # no observations and no n_times
    @test_throws ArgumentError build_data(em, group)
    # no initial_prob at build_data or call
    data = build_data(em, group; n_times=10)
    @test_throws ArgumentError em.simulate(StableRNG(1), (; α=0.1, β=0.1, m=4.0), data)
end
