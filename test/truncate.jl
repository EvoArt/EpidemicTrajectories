# Truncating an EpidemicData to a training window.
#
# The gates that matter:
#  * each declared rule does exactly what it says and nothing else,
#  * a time-shaped extra with NO rule is an ERROR under strict mode — the silent
#    leak is the failure this whole mechanism exists to prevent,
#  * `copy` really breaks aliasing, so an in-place mutation on one side is not
#    visible on the other.

# Minimal two-state model, mirroring test/iffbs.jl's setup but with the kinds of
# extras a real model attaches: a per-individual "last seen" vector, an event
# list of (individual, time) pairs, an in-place-mutated matrix, and a covariate.
function _trunc_setup(; n_ind=6, n_t=10, sampling_period=nothing)
    group = repeat(1:2; inner=n_ind ÷ 2)
    state_space = [:S, :I]

    aggs = @aggregate state_space begin
        @array n_infected Int (2, n_t)
        n_infected[data.group[i], t] += (state == :I)
    end

    infection(model, data, i, t) = 0.1
    recovery(model, data, i, t) = 0.2
    spec = @transitions state_space begin
        S -> I = infection
        I -> S = recovery
    end
    starting_state = (model, data, X, i, t) -> [0.9, 0.1]

    data = epidemic_data(;
        n_individuals=n_ind, n_timepoints=n_t, group=group,
        trans_mat=spec, starting_state=starting_state, aggregates=aggs,
        sampling_period=sampling_period,
        # extras of each shape the rules have to cope with
        last_seen = [3, 5, 7, 9, 10, 10],                  # clamp
        events = [(1, 2), (2, 6), (3, 9), (4, 10)],        # filter
        tests = fill(1, n_t, n_ind),                       # copy (mutated in place)
        sex = [1, 2, 1, 2, 1, 2],                          # keep (a covariate)
    )
    (; data, n_ind, n_t)
end

@testset "truncation: rules do what they say" begin
    s = _trunc_setup()
    plan = truncation(clamp=(:last_seen,), filter=(:events,),
                      copy=(:tests,), keep=(:sex,))
    train = truncate_data(s.data, plan, 6)

    # clamp: nothing may point past the cutoff, values below it are untouched
    @test train.last_seen == [3, 5, 6, 6, 6, 6]
    @test all(<=(6), train.last_seen)

    # filter: post-cutoff events are dropped entirely, not clamped
    @test train.events == [(1, 2), (2, 6)]

    # keep: byte-identical, and deliberately still aliased (nothing mutates it)
    @test train.sex == s.data.sex

    # sampling_period is clamped for every individual, and never inverted
    @test all(l <= 6 for (_, l) in train.sampling_period)
    @test all(f <= l for (f, l) in train.sampling_period)

    # the untruncated data is untouched — truncation returns a copy
    @test s.data.last_seen == [3, 5, 7, 9, 10, 10]
    @test length(s.data.events) == 4
end

@testset "truncation: copy breaks aliasing" begin
    # This is the leak that is invisible to every value-based test: a sampler
    # that mutates `tests` in place (a changepoint kernel swapping test
    # labellings, say) would otherwise write through to the untruncated data
    # that the scorer reads.
    s = _trunc_setup()
    plan = truncation(clamp=(:last_seen,), filter=(:events,),
                      copy=(:tests,), keep=(:sex,))
    train = truncate_data(s.data, plan, 6)

    @test train.tests == s.data.tests          # same values ...
    @test train.tests !== s.data.tests         # ... different array
    train.tests[1, 1] = 99
    @test s.data.tests[1, 1] == 1              # mutation did not leak

    # and `keep` does NOT copy — asserted so the distinction stays deliberate
    @test train.sex === s.data.sex
end

@testset "truncation: undeclared time-shaped extras are an error" begin
    s = _trunc_setup()

    # `events` and `last_seen` left undeclared: both look time-indexed
    bad = truncation(copy=(:tests,), keep=(:sex,))
    err = try
        truncate_data(s.data, bad, 6); nothing
    catch e
        e
    end
    @test err !== nothing
    msg = sprint(showerror, err)
    @test occursin("last_seen", msg)
    @test occursin("events", msg)
    # the message must say what to DO, not merely that something is wrong
    @test occursin("clamp", msg) && occursin("filter", msg) && occursin("keep", msg)

    # strict=false is the documented escape hatch
    lax = truncation(copy=(:tests,), keep=(:sex,), strict=false)
    train = truncate_data(s.data, lax, 6)
    @test train.last_seen == s.data.last_seen      # passed through untouched
end

@testset "truncation: plan validation" begin
    # the same extra declared twice is a mistake, not a merge
    @test_throws ErrorException truncation(clamp=(:a,), keep=(:a,))

    s = _trunc_setup()
    plan = truncation(clamp=(:last_seen,), filter=(:events,),
                      copy=(:tests,), keep=(:sex,))
    @test_throws ArgumentError truncate_data(s.data, plan, 0)
    @test_throws ArgumentError truncate_data(s.data, plan, s.n_t + 1)
end

@testset "truncation: custom rule" begin
    s = _trunc_setup()
    # An arbitrary transformation, on purpose: the package must not care what a
    # custom rule means, only that the user declared one.
    plan = truncation(clamp=(:last_seen,), filter=(:events,), copy=(:tests,),
                      custom=(:sex => (v, c) -> v .* 10,))
    train = truncate_data(s.data, plan, 6)
    @test train.sex == s.data.sex .* 10

    # declaring the same extra under two rules is a mistake, not a merge
    @test_throws ErrorException truncation(keep=(:sex,), custom=(:sex => (v, c) -> v,))
end

@testset "lfo_cutoffs" begin
    s = _trunc_setup(n_t=10)
    @test lfo_cutoffs(s.data; L=5, M=2) == collect(5:8)
    @test lfo_cutoffs(s.data; L=5, M=2, stride=2) == [5, 7]
    # every cutoff must leave M steps to score
    @test all(c + 2 <= s.n_t for c in lfo_cutoffs(s.data; L=5, M=2))
    # a window that cannot fit is an error, not an empty sweep
    @test_throws ArgumentError lfo_cutoffs(s.data; L=9, M=2)
    @test_throws ArgumentError lfo_cutoffs(s.data; L=0, M=2)
end

@testset "truncation: aggregates are zeroed, not carried" begin
    # Aggregates are a function of X and are rebuilt by the sampler. Carrying the
    # full-series values in would hand the training fit counts computed from
    # post-cutoff states.
    s = _trunc_setup()
    s.data.aggregates[:n_infected] .= 7
    plan = truncation(clamp=(:last_seen,), filter=(:events,),
                      copy=(:tests,), keep=(:sex,))
    train = truncate_data(s.data, plan, 6)
    @test all(==(0), train.aggregates[:n_infected])
    @test all(==(7), s.data.aggregates[:n_infected])   # source untouched
end

@testset "truncation: future entrants are absent from the training fit" begin
    # Individuals 3 and 4 first enter after the cutoff. Their truncated periods
    # must be empty, not one-cell windows at their future first capture.
    periods = [(1, 10), (3, 9), (7, 10), (8, 10), (1, 5), (2, 10)]
    s = _trunc_setup(sampling_period=periods)
    plan = truncation(clamp=(:last_seen,), filter=(:events,),
                      copy=(:tests,), keep=(:sex,))
    train = truncate_data(s.data, plan, 6)

    @test train.sampling_period ==
        [(1, 6), (3, 6), (7, 6), (8, 6), (1, 5), (2, 6)]
    @test all(isempty(f:l) for (f, l) in train.sampling_period[3:4])

    # Empty-period trajectories contribute neither latent nor observation
    # likelihood and are not touched by the latent sampler.
    X = fill(1, s.n_t, s.n_ind)
    ll = epidemic_loglik(train)
    oll = epidemic_obs_loglik(train)
    before = ll((;), train, X) + oll((;), train, X)
    X[7:10, 3:4] .= 2
    @test ll((;), train, X) + oll((;), train, X) == before
    held = copy(X[:, 3:4])
    iffbs_individual!((;), train, X, 3, StableRNG(71))
    iffbs_individual!((;), train, X, 4, StableRNG(72))
    @test X[:, 3:4] == held

    proposal = iffbs_proposal(train)
    target = epidemic_conditional_loglik(train)
    stats = MHStats(train.n_individuals)
    @test iffbs_mh_individual!((;), train, X, 3, StableRNG(73),
                               proposal, target, stats) === nothing
    @test sum(stats.proposed) == 0
    @test sum(stats.identical) == 0
    @test X[:, 3:4] == held
end

@testset "truncation: an individual not yet known is absent, even if present" begin
    # Individual 2's sampling period opens at 1 (it was alive, like a cub born
    # into a watched group) but the study only learns of it at 8. A fit at
    # cutoff 6 must not contain it; one at 8 must.
    s = _trunc_setup()
    first_seen = [1, 8, 1, 3, 1, 1]
    data = epidemic_data(;
        n_individuals=s.n_ind, n_timepoints=s.n_t, group=s.data.group,
        trans_mat=s.data.trans_mat, starting_state=s.data.starting_state,
        aggregates=Dict(k => zero(v) for (k, v) in pairs(s.data.aggregates)),
        derived_summaries=s.data.derived_summaries,
        last_seen=s.data.last_seen, first_seen=first_seen)
    plan = truncation(clamp=(:last_seen,), keep=(:first_seen,),
                      known_from=:first_seen)
    @test known_times(data, plan) == first_seen
    @test known_times(data, truncation(clamp=(:last_seen,), keep=(:first_seen,))) ==
          fill(1, s.n_ind)

    train = truncate_data(data, plan, 6)
    @test isempty(range(train.sampling_period[2]...))
    @test train.sampling_period[4] == (1, 6)     # known at 3, so present
    @test train.sampling_period[1] == (1, 6)

    X = fill(1, s.n_t, s.n_ind)
    ll = epidemic_loglik(train)
    before = ll((;), train, X)
    X[:, 2] .= 2
    @test ll((;), train, X) == before
    held = copy(X[:, 2])
    iffbs_individual!((;), train, X, 2, StableRNG(81))
    @test X[:, 2] == held

    @test train.sampling_period[2] != data.sampling_period[2]
    @test truncate_data(data, plan, 8).sampling_period[2] == (1, 8)
end
