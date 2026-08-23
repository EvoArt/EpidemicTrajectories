# Leave-future-out predictive scoring (examples/lfo_common.jl).
#
# The gates that matter, in order of what they protect:
#
#  1. EXACTNESS AT M=1. The forward recursion must equal a direct one-step
#     marginalisation, to floating point. This is the cheapest possible check
#     that `exact_window_loglik` computes what it claims; the reference study
#     never asserted it despite it being free.
#
#  2. THE M>1 GAP, MEASURED. At M>1 each individual is marginalised exactly but
#     the neighbours driving its FOI come from simulation, so individuals'
#     observations are scored independently when they are really correlated
#     through the shared epidemic. That gap has a size. Here it is measured
#     against brute-force enumeration of every latent path on a small instance,
#     rather than assumed small. The reference specified this check and never
#     built it.
#
#  3. NO -690 ARTEFACTS. The whole reason for marginalising: a scorer that
#     conditions on one sampled trajectory hits zero-probability observation
#     cells and floors them. The exact scorer must not.
#
#  4. AGGREGATE CONSISTENCY after forecasting, the invariant the whole package
#     rests on.

include(joinpath(@__DIR__, "..", "examples", "lfo_common.jl"))

# A two-state S/I model with a capture-and-test observation process, small
# enough that every latent path can be enumerated.
function _lfo_setup(; n_ind=4, n_t=8, captured=nothing, results=nothing)
    state_space = [:S, :I]
    group = ones(Int, n_ind)

    aggs = @aggregate state_space begin
        @array n_infected Int (1, n_t)
        n_infected[1, t] += (state == :I)
    end

    infection(model, data, i, t) =
        -expm1(-(model.α + model.β * data.aggregates[:n_infected][1, t]))
    recovery(model, data, i, t) = 1 / model.m

    spec = @transitions state_space begin
        S -> I = infection
        I -> S = recovery
    end

    starting_state = (model, data, X, i, t) -> [1 - model.ν, model.ν]

    # Weight of the observation at (i,t) given state s. Uncaptured cells carry no
    # information (weight 1 for every state), which is what makes the held-out
    # window scoreable at all.
    function obs_w(model, data, X, i, t, s)
        data.captured[t, i] == 1 || return 1.0
        y = data.results[t, i]
        y < 0 && return model.η                     # captured, not tested
        sens, spec_ = model.θ, model.φ
        if s == 2                                   # I
            return model.η * (y == 1 ? sens : 1 - sens)
        else                                        # S
            return model.η * (y == 1 ? 1 - spec_ : spec_)
        end
    end

    cap = captured === nothing ? ones(Int, n_t, n_ind) : captured
    res = results === nothing ? fill(-1, n_t, n_ind) : results

    data = epidemic_data(;
        n_individuals=n_ind, n_timepoints=n_t, group=group,
        trans_mat=spec, starting_state=starting_state, aggregates=aggs,
        observation_weight=obs_w, captured=cap, results=res,
    )
    (; data, n_ind, n_t)
end

const _LFO_PARS = (; α=0.05, β=0.08, m=4.0, ν=0.2, θ=0.8, φ=0.9, η=0.6)

@testset "exact scorer: M=1 equals direct one-step marginalisation" begin
    s = _lfo_setup(; n_ind=4, n_t=8)
    rng = StableRNG(11)
    X = epidemic_simulator(s.data)(StableRNG(3), _LFO_PARS)
    # Give the held-out step some actual test results to score.
    for i in 1:s.n_ind
        s.data.results[5, i] = i <= 2 ? 1 : 0
    end

    t_star = 4
    got = exact_window_loglik(_LFO_PARS, s.data, X, t_star, 1;
                              rng=StableRNG(11), n_sim=1, at_risk_only=false)

    # Direct one-step: for each individual, p(y) = sum_s P(x_t -> s) * w(y|s),
    # with the SAME neighbour forecast (same seed) driving the rates.
    Xf = copy(X)
    prepare_forecast!(Xf, _LFO_PARS, s.data, t_star)
    forecast_neighbours!(Xf, _LFO_PARS, s.data, t_star, 1, StableRNG(11))

    N = s.data.n_states
    want = 0.0
    P = zeros(Float64, N, N); rowsum = zeros(Float64, N)
    for i in 1:s.n_ind
        EpidemicTrajectories.transition_matrix_at!(P, rowsum, s.data.trans_mat,
                                                   _LFO_PARS, s.data, Xf, i, t_star)
        z = 0.0
        for st in 1:N
            z += P[X[t_star, i], st] *
                 s.data.observation_weight(_LFO_PARS, s.data, Xf, i, t_star + 1, st)
        end
        want += log(z)
    end

    @test got ≈ want atol = 1e-12
end

@testset "exact scorer: agrees with brute-force enumeration" begin
    # Small enough to enumerate: 3 individuals x 3 forecast steps x 2 states
    # = 2^9 = 512 joint latent paths.
    n_ind, n_t, t_star, M = 3, 6, 3, 3
    s = _lfo_setup(; n_ind=n_ind, n_t=n_t)
    X = epidemic_simulator(s.data)(StableRNG(5), _LFO_PARS)
    for i in 1:n_ind, t in (t_star + 1):(t_star + M)
        s.data.results[t, i] = (i + t) % 2
    end

    data, pars = s.data, _LFO_PARS
    N = data.n_states

    # --- brute force: enumerate every joint path over the M-step window -------
    # p(y) = sum over all paths of  prod_t [ P(x_{t-1} -> x_t) * w(y_t | x_t) ],
    # with the aggregates rebuilt for each path so the coupling is exact. This is
    # the quantity the scorer approximates; nothing here is factorised.
    Xb = copy(X)
    P = zeros(Float64, N, N); rowsum = zeros(Float64, N)

    function path_logprob(states)          # states[m][i] = state of i at t_star+m
        copyto!(Xb, X)
        prepare_forecast!(Xb, pars, data, t_star)
        lp = 0.0
        for m in 1:M
            t = t_star + m
            # score every individual's move into t, under the aggregates as they
            # stand at t-1, then commit the whole slice.
            for i in 1:n_ind
                EpidemicTrajectories.transition_matrix_at!(P, rowsum, data.trans_mat,
                                                           pars, data, Xb, i, t - 1)
                lp += log(P[Xb[t - 1, i], states[m][i]])
            end
            for i in 1:n_ind
                Xb[t, i] = states[m][i]
                apply_summaries!(data.derived_summaries, pars, data, Xb, states[m][i], i, t, false)
            end
            for i in 1:n_ind
                lp += log(data.observation_weight(pars, data, Xb, i, t, states[m][i]))
            end
        end
        lp
    end

    # all N^(n_ind*M) assignments
    total = N^(n_ind * M)
    terms = Float64[]
    for code in 0:(total - 1)
        c = code
        states = Vector{Vector{Int}}(undef, M)
        for m in 1:M
            row = Vector{Int}(undef, n_ind)
            for i in 1:n_ind
                row[i] = (c % N) + 1
                c ÷= N
            end
            states[m] = row
        end
        push!(terms, path_logprob(states))
    end
    brute = logsumexp_vec(terms)

    # --- the scorer, averaged over many neighbour replicates ------------------
    got = exact_window_loglik(pars, data, X, t_star, M;
                              rng=StableRNG(7), n_sim=4000, at_risk_only=false)

    @test isfinite(brute)
    @test isfinite(got)

    # THE MEASURED GAP. Reported so the number is on the record rather than
    # assumed. On this instance (3 individuals, one group, beta=0.08):
    #
    #     M=1   gap =  0.0        <- exact, to floating point
    #     M=2   gap = -0.219
    #     M=3   gap = -0.360
    #
    # It is zero at M=1 and grows with M, which is the signature of the
    # factorisation in this file's header: the focal's own chain is marginalised
    # exactly, but its neighbours are averaged replicate-by-replicate, so the
    # correlation between the focal's path and its groupmates' paths through the
    # shared epidemic is not carried. Verified NOT to be Monte-Carlo noise —
    # raising n_sim from 100 to 16000 shrinks the sd from 0.037 to 0.0008 while
    # the gap stays at -0.36.
    #
    # The bias is CONSERVATIVE (the scorer under-states the predictive density)
    # and, being a property of the estimator rather than of either model, it
    # applies equally to both sides of a model comparison. That is what makes it
    # tolerable here; it is not a licence to read single ELPD values as absolute.
    #
    # Group size is what drives it: with 3 individuals one groupmate moves
    # p(S->I) from 0.049 to 0.122. Real badger groups are larger and each
    # neighbour matters proportionally less.
    @info "M>1 factorisation gap" M brute got gap = got - brute
    @test got < brute                       # one-sided, as the argument predicts
    @test got ≈ brute atol = 0.5
end

@testset "exact scorer: EXACT at M=1, gap grows with M" begin
    # The sharpest statement available about this estimator: at M=1 there is no
    # neighbour uncertainty to factorise (the cutoff states are known), so the
    # scorer must equal brute force to floating point. Any nonzero gap here would
    # mean a coding error rather than the factorisation.
    n_ind, n_t, t_star = 3, 6, 3
    s = _lfo_setup(; n_ind=n_ind, n_t=n_t)
    X = epidemic_simulator(s.data)(StableRNG(5), _LFO_PARS)
    for i in 1:n_ind, t in (t_star + 1):(t_star + 3)
        s.data.results[t, i] = (i + t) % 2
    end
    data, pars, N = s.data, _LFO_PARS, s.data.n_states

    Xb = copy(X)
    P = zeros(Float64, N, N); rowsum = zeros(Float64, N)
    function brute_M(Mx)
        terms = Float64[]
        for code in 0:(N^(n_ind * Mx) - 1)
            c = code
            sts = [Vector{Int}(undef, n_ind) for _ in 1:Mx]
            for m in 1:Mx, i in 1:n_ind
                sts[m][i] = (c % N) + 1; c ÷= N
            end
            copyto!(Xb, X)
            prepare_forecast!(Xb, pars, data, t_star)
            lp = 0.0
            for m in 1:Mx
                t = t_star + m
                for i in 1:n_ind
                    EpidemicTrajectories.transition_matrix_at!(P, rowsum, data.trans_mat,
                                                               pars, data, Xb, i, t - 1)
                    lp += log(P[Xb[t - 1, i], sts[m][i]])
                end
                for i in 1:n_ind
                    Xb[t, i] = sts[m][i]
                    apply_summaries!(data.derived_summaries, pars, data, Xb, sts[m][i], i, t, false)
                end
                for i in 1:n_ind
                    lp += log(data.observation_weight(pars, data, Xb, i, t, sts[m][i]))
                end
            end
            push!(terms, lp)
        end
        logsumexp_vec(terms)
    end

    got1 = exact_window_loglik(pars, data, X, t_star, 1;
                               rng=StableRNG(21), n_sim=200, at_risk_only=false)
    @test got1 ≈ brute_M(1) atol = 1e-10        # EXACT: no factorisation at M=1

    gap2 = exact_window_loglik(pars, data, X, t_star, 2;
                               rng=StableRNG(21), n_sim=8000, at_risk_only=false) - brute_M(2)
    gap3 = exact_window_loglik(pars, data, X, t_star, 3;
                               rng=StableRNG(21), n_sim=8000, at_risk_only=false) - brute_M(3)
    @test abs(gap2) < abs(gap3)                 # the gap grows with the horizon
end

@testset "exact scorer: no floored zero-probability artefacts" begin
    # An individual observed (captured, test-positive) at every held-out step.
    # A single-trajectory scorer that happened to place it in S with a perfect
    # test would score log(0); the marginal scorer must stay comfortably finite.
    s = _lfo_setup(; n_ind=4, n_t=8)
    pars = merge(_LFO_PARS, (; θ=1.0, φ=1.0))       # perfect test: zeros are reachable
    X = epidemic_simulator(s.data)(StableRNG(9), pars)
    for i in 1:s.n_ind, t in 5:8
        s.data.results[t, i] = 1                    # all positive
    end

    ll = exact_window_loglik(pars, s.data, X, 4, 4;
                             rng=StableRNG(9), n_sim=8, at_risk_only=false)
    @test isfinite(ll)
    @test ll < 0
    # -690 per floored cell is the artefact this design exists to avoid; with 16
    # scored cells a flooring scorer would be far below -1000.
    @test ll > -500
end

@testset "forecast: aggregates stay consistent with X" begin
    s = _lfo_setup(; n_ind=6, n_t=10)
    X = epidemic_simulator(s.data)(StableRNG(4), _LFO_PARS)

    Xf = copy(X)
    prepare_forecast!(Xf, _LFO_PARS, s.data, 5)
    forecast_neighbours!(Xf, _LFO_PARS, s.data, 5, 4, StableRNG(4))

    live = copy(s.data.aggregates[:n_infected])
    reset_aggregates!(s.data)
    apply_derived_summaries!(_LFO_PARS, s.data, Xf)
    @test live == s.data.aggregates[:n_infected]
end

@testset "forecast: conditioning window is untouched and future is redrawn" begin
    s = _lfo_setup(; n_ind=5, n_t=10)
    X = epidemic_simulator(s.data)(StableRNG(6), _LFO_PARS)
    t_star = 6

    Xf = copy(X)
    prepare_forecast!(Xf, _LFO_PARS, s.data, t_star)
    forecast_neighbours!(Xf, _LFO_PARS, s.data, t_star, 4, StableRNG(6))

    # The past must survive the round trip exactly...
    @test Xf[1:t_star, :] == X[1:t_star, :]
    # ...and the future must be a genuine redraw, not a carry-forward of the
    # cutoff state (which is what prepare_forecast! leaves behind).
    Xg = copy(X)
    prepare_forecast!(Xg, _LFO_PARS, s.data, t_star)
    forecast_neighbours!(Xg, _LFO_PARS, s.data, t_star, 4, StableRNG(99))
    @test Xf[(t_star + 1):end, :] != Xg[(t_star + 1):end, :]
end

@testset "psis: weights normalise and k is finite on well-behaved ratios" begin
    rng = StableRNG(2)
    lr = randn(rng, 500) .* 0.3
    res = psis(copy(lr))
    @test isapprox(sum(exp.(res.log_weights)), 1.0; atol=1e-10)
    @test isfinite(res.k)
    @test res.k < 0.7                    # mild ratios: no refit should be triggered

    # Identical ratios are the degenerate case the GPD fit cannot handle; the
    # implementation must short-circuit to uniform weights rather than divide by
    # a zero-width tail.
    flat = psis(fill(1.23, 200))
    @test flat.k == 0.0
    @test all(≈(-log(200.0)), flat.log_weights)
end

@testset "elpd_window and elpd_se" begin
    # Equal-probability draws: the window ELPD is that common value.
    @test elpd_window(fill(log(0.25), 8)) ≈ log(0.25)
    # A weighted combination reduces to the weighted average in probability space.
    lt = [log(0.1), log(0.3)]
    lw = [log(0.25), log(0.75)]
    @test elpd_window(lt; log_weights=lw) ≈ log(0.25 * 0.1 + 0.75 * 0.3)

    @test isnan(elpd_se([1.0], 4))                   # too few windows
    @test isfinite(elpd_se(collect(1.0:20.0), 4))
end
