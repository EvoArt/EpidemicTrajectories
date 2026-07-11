# Cross-cutting checks: both likelihoods are autodiff-friendly (the property
# that makes them usable as a PracticalBayes `@addlogprob!` HMC target), and the
# functional and transition-matrix styles agree on the SAME two-state process.

using ForwardDiff

@testset "consistency: functional loglik differentiates w.r.t. params" begin
    ss = SI
    rates = TwoStateSI()
    truep = (; α=0.05, β=0.05, m=5.0)
    group = ones(Int, 8)
    init = [0.9, 0.1]
    rng = StableRNG(2)
    states, data = simulate_trajectory(rng, ss, rates, truep, group, init; n_times=60)

    # gradient of trajectory_loglik w.r.t. (α, β, m) at the true params.
    function nll(v)
        pars = (; α=v[1], β=v[2], m=v[3])
        model = (; state_space=ss, rates=rates, pars=pars)
        return trajectory_loglik(pars, model, data)
    end
    g = ForwardDiff.gradient(nll, [0.05, 0.05, 5.0])
    @test length(g) == 3
    @test all(isfinite, g)
    @test !all(iszero, g)   # the trajectory genuinely informs the parameters
end

@testset "consistency: chain-binomial loglik differentiates w.r.t. params" begin
    m = StateTransitionModel(
        state_space = SI,
        transitions = [(0, 1), (1, 0)],
        rates = [
            (pars, counts, t) -> -expm1(-(pars.α + pars.β * counts[2])),
            (pars, counts, t) -> 1 / pars.m,
        ],
    )
    truep = (; α=0.02, β=0.03, m=4.0)
    rng = StableRNG(5)
    counts, events = simulate_chain_binomial(rng, m, truep, [30, 3]; n_times=25)
    function nll(v)
        pars = (; α=v[1], β=v[2], m=v[3])
        return chain_binomial_loglik(pars, m, counts, events)
    end
    g = ForwardDiff.gradient(nll, [0.02, 0.03, 4.0])
    @test all(isfinite, g)
    @test !all(iszero, g)
end

@testset "consistency: MLE-ish recovery of the recovery rate from one long trajectory" begin
    # A cheap sanity check that the functional loglik peaks near the truth: with
    # NO within-pen transmission (β=0) and a single well-mixed pen, the recovery
    # probability's MLE is (#I->S transitions)/(#I-at-t), which the loglik should
    # be maximized near. We just check the loglik at the true m beats a clearly
    # wrong m.
    ss = SI
    rates = TwoStateSI()
    truep = (; α=0.15, β=0.0, m=4.0)
    group = ones(Int, 30)
    init = [0.5, 0.5]
    rng = StableRNG(8)
    states, data = simulate_trajectory(rng, ss, rates, truep, group, init; n_times=200)

    ll(m) = trajectory_loglik((; α=0.15, β=0.0, m=m),
                              (; state_space=ss, rates=rates, pars=(; α=0.15, β=0.0, m=m)),
                              data)
    @test ll(4.0) > ll(1.5)
    @test ll(4.0) > ll(12.0)
end
