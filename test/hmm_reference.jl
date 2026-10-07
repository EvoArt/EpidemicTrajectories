# The joint-state reference for coupled models.
#
# On an independent model it must agree with the per-individual forward
# algorithm (itself pinned against path enumeration in hmm.jl). On a coupled
# one: the joint forecast over every possible outcome sums to one, the event
# decomposition log P(A) + log p(y | A) is exact when the data imply A, and the
# exact sampler reproduces the filtered distribution.

# N/P/D over 6 steps, everyone observed throughout. With `b > 0` a negative's
# conversion probability rises with the share of the others that are positive.
function _coupled(y; b = 0.0, group = collect(1:size(y, 2)), coupled = nothing)
    n_t, n = size(y)
    ss = [:N, :P, :D]
    aggs = @aggregate ss begin
        @array n_pos Int (1, n_t)
        @array n_alive Int (1, n_t)
        n_pos[1, t] += (state == :P)
        n_alive[1, t] += (state != :D)
    end
    spec = @transitions ss begin
        N -> P = (model, data, i, t) -> model.phi * (model.a + model.b *
                     data.aggregates.n_pos[1, t] / max(data.aggregates.n_alive[1, t] - 1, 1))
        N -> D = (model, data, i, t) -> 1 - model.phi
        P -> D = (model, data, i, t) -> 1 - model.phi
    end
    epidemic_data(; n_individuals = n, n_timepoints = n_t, trans_mat = spec,
                  aggregates = aggs, group = group, coupled_transitions = coupled,
                  starting_state = (model, data, X, i, t) -> [1 - model.nu, model.nu, 0.0],
                  observation_weight = (model, data, X, i, t, s) -> begin
                      r = data.y[t, i]            # 0 missed, 1 neg, 2 pos, -1 no survey
                      r == -1 && return 1.0
                      s == 3 && return r == 0 ? 1.0 : 0.0
                      r == 0 && return 1 - model.p
                      r == s ? model.p : 0.0
                  end,
                  y = y)
end

const _YC = [1  2  1;
             0  2  1;
             1  0  0;
             0  2  2;
             1  0  0;
             0  0  2]
const _TC = (; phi = 0.85, a = 0.1, b = 0.0, p = 0.6, nu = 0.3)

@testset "joint reference: independent model agrees with per-individual filtering" begin
    d = _coupled(copy(_YC))
    for (t_star, M) in ((3, 1), (3, 2), (2, 4))
        ref = joint_reference(_TC, d, t_star, M)
        per = hmm_forecast_logliks(_TC, d, t_star, M).logp
        @test ref.logp ≈ sum(values(per)) atol = 1e-10
        for i in 1:3
            @test ref.logp_individual[i] ≈ per[i] atol = 1e-10
        end
    end
end

@testset "joint reference: coupled model" begin
    th = (; _TC..., b = 0.6)
    y = copy(_YC)
    d = _coupled(y; group = ones(Int, 3))
    t_star, M = 3, 2

    # summed over every possible block of observations, the forecast is one
    total = 0.0
    for a in Iterators.product(ntuple(_ -> 0:2, 6)...)
        y[4:5, :] .= reshape(collect(a), 2, 3)
        total += exp(joint_reference(th, d, t_star, M).logp)
    end
    @test total ≈ 1 atol = 1e-10
    y .= _YC

    # individual 3 is caught at 4, so the window implies it was alive at 3;
    # individual 2 is caught at 4, likewise
    ref = joint_reference(th, d, t_star, M; alive = [2, 3])
    @test ref.log_pA < 0
    @test ref.log_pA + ref.logp_given_A ≈ ref.logp atol = 1e-10
    # coupling makes the individuals' forecasts dependent given the parameters
    @test abs(ref.logp - sum(values(ref.logp_individual))) > 1e-4

    # the exact sampler reproduces the filtered joint distribution at t_star
    rng = StableRNG(3)
    n = 20_000
    counts = Dict{Vector{Int},Int}()
    for _ in 1:n
        x = joint_sample(rng, th, d, t_star)[t_star, :]
        counts[x] = get(counts, x, 0) + 1
    end
    worst = maximum(abs(get(counts, x, 0) / n - p) / sqrt(max(p * (1 - p), 1e-12) / n)
                    for (x, p) in zip(ref.states, ref.filtered) if p > 1e-3)
    @test worst < 4.5
    # and conditioned on A, never places a member of A in D
    @test all(all(joint_sample(rng, th, d, t_star; alive = [2, 3])[t_star, [2, 3]] .!= 3)
              for _ in 1:200)
end

@testset "conditioning a draw on presence at the cutoff: exact weights" begin
    th = (; _TC..., b = 0.6)
    # 2 and 3 are positive and then go unseen for three and two steps before the
    # cutoff, while 1 is seen negative throughout and converts at step 4. Its
    # conversion probability depends on how many positives are alive, so whether
    # and when 2 and 3 died changes the likelihood of 1's path: their chances
    # of being alive at the cutoff depend on each other's trajectories.
    y = [1  2  2;
         1  0  2;
         1  0  0;
         2  0  0;
         0  2  0;
         2  0  2]
    d = _coupled(y; group = ones(Int, 3))
    t_star, M = 4, 2
    # everyone is caught in the window, so it implies all three were alive at 4
    ref = joint_reference(th, d, t_star, M; alive = [1, 2, 3])
    plan = truncation(keep = (:y,))
    train = truncate_data(d, plan, t_star)

    # exact posterior draws, so nothing but the weights is being tested
    rng = StableRNG(11)
    n = 20_000
    Xs = joint_sample(rng, th, d, t_star, n)
    w = zeros(n)
    hits = Dict{Vector{Int},Float64}()
    for s in 1:n
        X = copy(Xs[s])
        w[s] = exp(condition_present!(rng, th, train, X, [1, 2, 3]))
        @test all(X[t_star, :] .!= 3)
        x = X[t_star, :]
        hits[x] = get(hits, x, 0.0) + w[s]
    end
    # the weight depends on the draw
    @test std(w) > 0.1 * mean(w)
    # its mean is the event's posterior probability
    pA = exp(ref.log_pA)
    @test abs(mean(w) - pA) < 4.5 * std(w) / sqrt(n)
    # and the weighted draws carry the restricted, unnormalised distribution
    for (x, p) in zip(ref.states, ref.filtered_A)
        p > 0.02 || continue
        @test get(hits, x, 0.0) / n ≈ pA * p rtol = 0.08
    end
    # counting the draws already in the event estimates the same probability
    # from far fewer of them
    inA = [all(X[t_star, :] .!= 3) for X in Xs]
    @test abs(mean(inA) - pA) < 4.5 * sqrt(pA * (1 - pA) / n)
    @test std(w) < std(inA)

    # through the driver: the window's predictive density at fixed parameters
    kp = (i, t, last_t) -> any(>(0), @view d.y[t:last_t, i])
    con, sw = survival_constrained(kp)
    fit = (tr, cutoff) -> ([th for _ in 1:n], Xs)
    logobs = (m, data, X, i, t) -> log(data.observation_weight(m, data, X, i, t, X[t, i]))
    spec = LFOSpec(; fit, cell_logdensity = logobs, plan, constrain = con,
                   survival_weight = sw, condition_cutoff = true)
    res = lfo_cv(spec, d; L = t_star, M, cutoffs = [t_star], granularity = Joint(),
                 verbose = false)
    @test elpd(res, :joint) ≈ ref.logp atol = 0.03
    @test res.windows[1].n_finite[:joint] == n

    # the constraint alone is right too, but only on the draws already in the event
    plain = LFOSpec(; fit, cell_logdensity = logobs, plan, constrain = con,
                    survival_weight = sw)
    rp = lfo_cv(plain, d; L = t_star, M, cutoffs = [t_star], granularity = Joint(),
                verbose = false)
    @test elpd(rp, :joint) ≈ ref.logp atol = 0.1
    @test rp.windows[1].n_finite[:joint] == count(inA)
    @test count(inA) < n ÷ 2

    # it needs the constraint, and only the joint score is valid
    @test_throws ArgumentError lfo_cv(
        LFOSpec(; fit, cell_logdensity = logobs, plan, condition_cutoff = true),
        d; L = t_star, M, cutoffs = [t_star], granularity = Joint(), verbose = false)
    @test_throws ArgumentError lfo_cv(spec, d; L = t_star, M, cutoffs = [t_star],
                                      granularity = Pointwise(), verbose = false)
end
