# The model spec and its sugar: @transitions, rate forms, grouped sources,
# state_space handling.

foi(model, data, i, t) = 0.1
rec(model, data, i, t) = 1 / model.m
surv(model, data, i, t) = 0.9

@testset "@transitions: states inferred from the transitions" begin
    spec = @transitions begin
        S -> I = foi
        I -> S = rec
    end
    @test spec isa TransitionSpec
    @test spec.states == [:S, :I]                      # order of first appearance
    @test spec.transitions == [(:S, :I), (:I, :S)]
    @test spec.auto_self == true                       # the default
end

@testset "@transitions: explicit state_space fixes the numbering" begin
    ss = [:I, :S]                                      # deliberately reversed
    spec = @transitions ss begin
        S -> I = foi
        I -> S = rec
    end
    @test spec.states == [:I, :S]                      # user's order wins

    # a state_space missing a state used in the transitions is an error
    @test_throws Exception @eval @transitions [:S] begin
        S -> I = $foi
    end
end

@testset "@transitions: rate sugar" begin
    model = (; m=5.0, α=0.01)
    data = nothing

    # bare function name is called for us
    s1 = @transitions begin
        S -> I = foi
    end
    @test s1.rate_fns[1](model, data, 1, 1) ≈ 0.1

    # arithmetic composition of bare names
    s2 = @transitions begin
        S -> I = foi * surv
    end
    @test s2.rate_fns[1](model, data, 1, 1) ≈ 0.1 * 0.9

    # bare expression referring to the parameters
    s3 = @transitions begin
        I -> S = 1 / model.m
    end
    @test s3.rate_fns[1](model, data, 1, 1) ≈ 1 / 5.0

    # explicit call still works
    s4 = @transitions begin
        S -> I = foi(model, data, i, t)
    end
    @test s4.rate_fns[1](model, data, 1, 1) ≈ 0.1

    # explicit lambda: the power-user fallback
    s5 = @transitions begin
        S -> I = (model, data, i, t) -> 0.25
    end
    @test s5.rate_fns[1](model, data, 1, 1) ≈ 0.25
end

@testset "@transitions: grouped sources expand to one transition each" begin
    spec = @transitions begin
        S -> I = foi
        (S, I) -> D = 0.01
    end
    @test (:S, :D) in spec.transitions
    @test (:I, :D) in spec.transitions
    @test length(spec.transitions) == 3
end

@testset "@transitions: auto_self opt-out" begin
    spec = @transitions :no_auto_self begin
        S -> I = foi
    end
    @test spec.auto_self == false
end

@testset "@survival: scales the live transitions and adds the deaths" begin
    # The badger structure: S -> E -> I -> D, every step conditional on surviving.
    spec = @transitions [:S, :E, :I, :D] begin
        @survival surv death=:D
        S -> E = foi
        E -> I = 0.2
    end
    model = (; m=5.0)

    # every live state can die — including I, which only ever appears as a
    # DESTINATION above and so has no declared transition out of it
    @test (:S, :D) in spec.transitions
    @test (:E, :D) in spec.transitions
    @test (:I, :D) in spec.transitions

    rate_of(from, to) = spec.rate_fns[findfirst(==((from, to)), spec.transitions)](model, nothing, 1, 1)

    # the live transitions are scaled by survival...
    @test rate_of(:S, :E) ≈ 0.9 * 0.1
    @test rate_of(:E, :I) ≈ 0.9 * 0.2
    # ...and death takes the leftover
    @test rate_of(:S, :D) ≈ 1 - 0.9
    @test rate_of(:I, :D) ≈ 1 - 0.9
end

@testset "@survival: needs a death state" begin
    @test_throws Exception @eval @transitions [:S, :D] begin
        @survival $surv
        S -> D = 0.1
    end
end

@testset "@transitions: bad tag errors" begin
    @test_throws Exception @eval @transitions :nonsense begin
        S -> I = $foi
    end
end

# ---------------------------------------------------------------------------
# Row closure when the declared transitions overflow.
#
# `auto_self` derives `P[a,a] = 1 - rowsum[a]`. The per-rate clamp bounds each
# declared rate but never their SUM, so a state with several declared moves out
# of it can produce a NEGATIVE self-transition. That crashed `epidemic_loglik`
# with a DomainError from `log(negative)` and silently corrupted the iFFBS
# filter. Reported from a model that split its susceptible state, where the extra
# declared transitions removed the slack a two-state model always had.
# ---------------------------------------------------------------------------

function _overflow_setup(; n_t=2)
    states = [:S, :A, :B, :C]
    aggs = @aggregate states begin
        @array n Int (1, n_t)
        n[1, t] += (state == :A)
    end
    ra(model, data, i, t) = model.a
    rb(model, data, i, t) = model.b
    rc(model, data, i, t) = model.c
    spec = @transitions states begin
        S -> A = ra
        S -> B = rb
        S -> C = rc
    end
    data = epidemic_data(; n_individuals=1, n_timepoints=n_t, trans_mat=spec,
                           starting_state=(m, d, X, i, t) -> [1.0, 0.0, 0.0, 0.0],
                           aggregates=aggs)
    X = fill(1, n_t, 1)
    reset_aggregates!(data)
    apply_derived_summaries!((; a=0.1, b=0.1, c=0.1), data, X)
    (; data, X)
end

@testset "transition matrix: rows stay stochastic when rates overflow" begin
    s = _overflow_setup()
    for (a, b, c) in ((0.1, 0.1, 0.1), (0.3, 0.3, 0.3), (0.4, 0.4, 0.4),
                      (0.5, 0.5, 0.5), (0.9, 0.9, 0.9), (1.0, 1.0, 1.0))
        model = (; a=a, b=b, c=c)
        P = transition_matrix_at(s.data.trans_mat, model, s.data, s.X, 1, 1)
        row = P[1, :]
        @test all(>=(0), row)              # never negative — the reported bug
        @test sum(row) ≈ 1 atol = 1e-12    # and still a probability distribution
    end
end

@testset "transition_prob agrees with the matrix in the overflow region" begin
    # The likelihood reads `transition_prob` and the filter reads
    # `transition_matrix_at!`. If the two close a row differently the sampler and
    # the target silently disagree, which is the failure `check_iffbs_exact` is
    # built to detect — cheaper to just test it here.
    s = _overflow_setup()
    for (a, b, c) in ((0.2, 0.2, 0.2), (0.4, 0.4, 0.4), (0.9, 0.9, 0.9))
        model = (; a=a, b=b, c=c)
        P = transition_matrix_at(s.data.trans_mat, model, s.data, s.X, 1, 1)
        for from in 1:4, to in 1:4
            @test transition_prob(s.data.trans_mat, model, s.data, s.X, 1, 1,
                                  from, to) ≈ P[from, to] atol = 1e-15
        end
    end
end

@testset "overflowing rates do not crash the likelihood" begin
    s = _overflow_setup()
    loglik = epidemic_loglik(s.data)
    for (a, b, c) in ((0.2, 0.2, 0.2), (0.4, 0.4, 0.4), (0.9, 0.9, 0.9), (5.0, 5.0, 5.0))
        ll = loglik((; a=a, b=b, c=c), s.data, s.X)
        @test isfinite(ll)                 # was a DomainError from log(negative)
    end
end

@testset "the gradient survives the overflow region" begin
    import ForwardDiff
    # The point of rescaling rather than flooring at zero: `max(0, 1 - rowsum)`
    # also stops the crash, but then `d(self)/d(rate) == 0` and HMC gets no signal
    # to come back down. Rescaling keeps every entry a smooth function of every
    # rate.
    s = _overflow_setup()
    f(x) = transition_prob(s.data.trans_mat, (; a=x[1], b=0.9, c=0.9),
                           s.data, s.X, 1, 1, 1, 2)
    # a = 0.9 puts rowsum at 2.7, well inside the rescaled region
    @test abs(ForwardDiff.gradient(f, [0.9])[1]) > 1e-3
    g(x) = transition_prob(s.data.trans_mat, (; a=x[1], b=0.9, c=0.9),
                           s.data, s.X, 1, 1, 1, 1)      # the self-transition
    @test isfinite(ForwardDiff.gradient(g, [0.9])[1])
end

@testset "row closure leaves the non-overflow case bit-identical" begin
    # A guard on a region the model should never occupy must not perturb the
    # region it does occupy.
    s = _overflow_setup()
    for (a, b, c) in ((0.01, 0.02, 0.03), (0.1, 0.2, 0.3), (0.3, 0.3, 0.3))
        model = (; a=a, b=b, c=c)
        P = transition_matrix_at(s.data.trans_mat, model, s.data, s.X, 1, 1)
        @test P[1, 1] === 1 - (a + b + c)   # the literal old expression
        @test P[1, 2] === a
    end
end
