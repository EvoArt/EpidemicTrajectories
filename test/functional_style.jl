# Functional (iFFBS-paper) style: TwoStateSI rate bundle, transition matrix,
# simulate, likelihood, and the iFFBS latent sampler.

@testset "functional style: TwoStateSI transition matrix" begin
    ss = SI
    rates = TwoStateSI()
    pars = (; α=0.009, β=0.01, m=9.0)

    # one pen of 3 animals, all in group 1; at t=1 animals 2 and 3 infected.
    states = [0 0; 1 1; 1 1]  # 3 individuals x 2 times (only t=1 col used here)
    data = EpidemicTrajectories.make_data(states, [1, 1, 1])
    model = (; state_space=ss, rates=rates, pars=pars)

    # focal individual 1 (susceptible): sees 2 OTHER infected penmates at t=1.
    P = EpidemicTrajectories.transition_matrix_at(rates, pars, model, data, 1, 1)
    @test size(P) == (2, 2)
    # rows sum to 1
    @test all(≈(1.0), sum(P; dims=2))
    foi = 0.009 + 0.01 * 2
    @test P[1, 2] ≈ -expm1(-foi)          # S -> I
    @test P[1, 1] ≈ exp(-foi)             # S -> S
    @test P[2, 1] ≈ 1 / 9.0               # I -> S recovery
    @test P[2, 2] ≈ 1 - 1 / 9.0           # I -> I

    # leave-one-out: focal individual 2 (infected) sees only 1 OTHER infected.
    P2 = EpidemicTrajectories.transition_matrix_at(rates, pars, model, data, 2, 1)
    foi2 = 0.009 + 0.01 * 1
    @test P2[1, 2] ≈ -expm1(-foi2)
end

@testset "functional style: simulate + loglik shapes and determinism" begin
    ss = SI
    rates = TwoStateSI()
    pars = (; α=0.05, β=0.05, m=5.0)
    group = ones(Int, 8)
    init = [0.9, 0.1]

    rng = StableRNG(1)
    states, data = simulate_trajectory(rng, ss, rates, pars, group, init; n_times=50)
    @test size(states) == (8, 50)
    @test all(s -> s in (0, 1), states)

    model = (; state_space=ss, rates=rates, pars=pars)
    lp = trajectory_loglik(pars, model, data)
    @test isfinite(lp)
    @test lp ≤ 0

    # determinism: same seed -> same trajectory
    states2, _ = simulate_trajectory(StableRNG(1), ss, rates, pars, group, init; n_times=50)
    @test states == states2
end

@testset "functional style: observation likelihood" begin
    ss = SI
    pars = (; θ=0.8)
    test = DiagnosticTest(; sensitivity=p -> p.θ)  # specificity 1
    # positive result: impossible if S, prob θ if I
    @test observation_likelihood((test,), pars, ss, (1,)) ≈ [0.0, 0.8]
    # negative result: certain if S, prob 1-θ if I
    @test observation_likelihood((test,), pars, ss, (0,)) ≈ [1.0, 0.2]
    # missing (-1): no information
    @test observation_likelihood((test,), pars, ss, (-1,)) ≈ [1.0, 1.0]
end

@testset "functional style: iFFBS brute-force check on a tiny instance" begin
    # 2 individuals, 3 timesteps, one pen. Enumerate all 2^(2*3)=64 joint state
    # configurations, compute the exact posterior over individual 1's trajectory
    # given individual 2's fixed trajectory + observed tests, and check the FFBS
    # sampler's empirical distribution matches it. This is the correctness gate
    # for the coupling + observation + prediction machinery.
    ss = SI
    rates = TwoStateSI()
    pars = (; α=0.3, β=0.5, m=2.5)
    θ = 0.8
    test = DiagnosticTest(; sensitivity=p -> 0.8)

    n_ind, n_t = 2, 3
    init = [0.7, 0.3]
    initial_prob = init

    # Fix individual 2's trajectory and the observed test results.
    fixed2 = [0, 1, 1]                     # individual 2: S, I, I
    # observed results (individual x time), -1 = missing
    R = [ -1  1 -1;                        # individual 1 tests + at t=2
           0  1  1 ]                       # individual 2
    results = (R,)

    # Enumerate individual 1's 2^3 = 8 possible trajectories; for each, compute
    # the unnormalized joint density = P(X1 traj) * P(X2 traj | X1) *
    # P(obs | X1) * P(obs | X2). We build the full joint over BOTH individuals'
    # trajectories so the coupling (X1 affects X2's transitions and vice versa)
    # is handled exactly, then marginalize to individual 1's trajectory.
    states = zeros(Int, n_ind, n_t)
    data = EpidemicTrajectories.make_data(states, [1, 1])

    function joint_logdens(x1, x2)
        # set states
        data.states[1, :] .= x1
        data.states[2, :] .= x2
        model = (; state_space=ss, rates=rates, pars=pars)
        lp = 0.0
        # initial
        lp += log(initial_prob[state_index(ss, x1[1])])
        lp += log(initial_prob[state_index(ss, x2[1])])
        # transitions
        lp += trajectory_loglik(pars, model, data)
        # observations
        lp += EpidemicTrajectories.observation_loglik(pars, model, data, (test,), results)
        return lp
    end

    trajs = [[a, b, c] for a in 0:1 for b in 0:1 for c in 0:1]
    # exact posterior over x1 (marginalize x2 = fixed2, since we condition on it)
    logw = [joint_logdens(x1, fixed2) for x1 in trajs]
    w = exp.(logw .- maximum(logw)); w ./= sum(w)
    exact_post = Dict(trajs[k] => w[k] for k in eachindex(trajs))

    # FFBS: fix individual 2, resample individual 1 many times, tally.
    rng = StableRNG(42)
    model = (; state_space=ss, rates=rates, pars=pars)
    counts = Dict{Vector{Int},Int}()
    nsamp = 40_000
    for _ in 1:nsamp
        data.states[2, :] .= fixed2
        # random init for individual 1 (overwritten by the sampler)
        data.states[1, :] .= rand(rng, 0:1, n_t)
        EpidemicTrajectories.ffbs_individual!(rng, model, data, 1, (test,), results;
                                              initial_prob=initial_prob, coupling=true)
        x1 = copy(data.states[1, :])
        counts[x1] = get(counts, x1, 0) + 1
    end

    # compare empirical vs exact for each trajectory with non-negligible mass
    maxerr = 0.0
    for (traj, p) in exact_post
        emp = get(counts, traj, 0) / nsamp
        maxerr = max(maxerr, abs(emp - p))
    end
    @test maxerr < 0.02   # empirical within 2 percentage points of exact everywhere
end
