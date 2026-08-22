# iFFBS as a proposal, corrected by Metropolis-Hastings.
#
# The tests are ordered by how much each one buys, and several exist to pin down a
# specific way of getting this wrong:
#
#   0. the proposal refactor changed NOTHING about `iffbs!`  (bit-identical)
#   1. the conditional target really is the joint's restriction (delta-consistency)
#   2. proposal == target  =>  every log-ratio is 0            (the keystone)
#   3. the aggregates survive BOTH branches, especially reject
#   4. the kernel targets the right distribution               (exact enumeration)
#   5. detailed balance, spot-checked
#   6. the MH sweep reduces to `iffbs!` draw for draw
#   7. semi-Markov via `step_logprob`
#   8. acceptance rates are sane, and the plain sweep is provably biased where MH is not
#   9. degenerate ratios (+/-Inf, NaN) do the right thing
#
# Shared fixtures live at the top; `_si_setup` from `test/iffbs.jl` is deliberately
# NOT reused, because these tests need to vary the observation model, the windows
# and the state space independently.

using EpidemicTrajectories: _param_eltype
using Logging: Logging

# --- fixtures ---------------------------------------------------------------

# A two-state S/I model with recurrent S<->I, group coupling through the number of
# infected, and an optional noisy test. Small enough that the whole latent space
# can be enumerated when `n_ind * n_t` is kept tiny.
function _mh_setup(; n_groups=2, n_per_group=3, n_t=6, results=nothing,
                     sampling_period=nothing, coupled_transitions=nothing,
                     observation_process=nothing, extras...)
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

    kw = (;)
    if observation_process !== nothing
        kw = (; observation_process=observation_process)
    elseif results !== nothing
        kw = (; observation_process=_noisy_test)
    end
    results !== nothing && (kw = merge(kw, (; results=results)))
    sampling_period !== nothing && (kw = merge(kw, (; sampling_period=sampling_period)))
    coupled_transitions !== nothing &&
        (kw = merge(kw, (; coupled_transitions=coupled_transitions)))

    data = epidemic_data(;
        n_individuals=n_ind, n_timepoints=n_t, group=group,
        trans_mat=spec, starting_state=starting_state, aggregates=aggs,
        kw..., extras...)
    (; data, n_ind, n_t, n_groups, state_space)
end

# A NOISY test — unlike `test/iffbs.jl`'s perfect one, this keeps every state
# reachable, so the filter never produces a zero weight and the conditional never
# hits `-Inf`. That matters: a perfect test makes most proposals identical to the
# current path, which would leave the MH tests with almost nothing to decide.
function _noisy_test(model, data, X, i, t)
    w = ones(_param_eltype(model), data.n_states)
    y = data.results[t, i]
    y < 0 && return w
    if y == 1                      # positive
        w[1] = 1 - model.θᶠ        # false positive rate for S
        w[2] = model.θʳ            # sensitivity for I
    else                           # negative
        w[1] = model.θᶠ
        w[2] = 1 - model.θʳ
    end
    w
end

const _MH_PARS = (; α=0.05, β=0.08, m=4.0, ν=0.2, θʳ=0.75, θᶠ=0.7)

# Rebuild the aggregates from scratch, so nothing incremental is trusted.
function _sync!(model, data, X)
    reset_aggregates!(data)
    apply_derived_summaries!(model, data, X)
    X
end

_agg_copy(data) = deepcopy(data.aggregates)

function _random_results(rng, n_t, n_ind; p_tested=0.5)
    r = fill(-1, n_t, n_ind)
    for t in 1:n_t, i in 1:n_ind
        rand(rng) < p_tested && (r[t, i] = rand(rng, 0:1))
    end
    r
end

# ============================================================================
# 0. The proposal refactor changed nothing
# ============================================================================

@testset "iffbs!: threading a proposal through changes nothing" begin
    s = _mh_setup(; n_t=12, results=_random_results(StableRNG(7), 12, 6))
    simulate = epidemic_simulator(s.data)

    X1 = simulate(StableRNG(1), _MH_PARS)
    _sync!(_MH_PARS, s.data, X1)
    iffbs!(_MH_PARS, s.data, X1, StableRNG(99))
    agg1 = _agg_copy(s.data)

    X2 = simulate(StableRNG(1), _MH_PARS)
    _sync!(_MH_PARS, s.data, X2)
    # The explicit default proposal must be the same object graph the implicit one
    # builds, so this is bit-identical rather than merely close.
    iffbs!(_MH_PARS, s.data, X2, StableRNG(99); proposal=iffbs_proposal(s.data))

    @test X1 == X2
    @test agg1[:n_infected] == s.data.aggregates[:n_infected]
end

@testset "iffbs_proposal: defaults are the data's own fields" begin
    s = _mh_setup()
    p = iffbs_proposal(s.data)
    @test p.trans_mat === s.data.trans_mat
    @test p.starting_state === s.data.starting_state
    @test p.observation_process === s.data.observation_process
    @test p.rest_contribution === s.data.rest_contribution

    u = uncorrected_proposal(s.data)
    @test u.rest_contribution === no_rest_contribution
    @test u.trans_mat === s.data.trans_mat        # only the coupling is dropped
end

@testset "EpidemicData: every field of a built instance is concrete" begin
    # CLAUDE.md's third-instance trap: a field annotated with a parametric type
    # must bind EVERY parameter or it is silently abstract. `neighbor_logprob` and
    # `coupled_mask` are new fields, so re-run the check on the whole struct.
    s = _mh_setup(; coupled_transitions=[(:S, :I)])
    D = typeof(s.data)
    for f in fieldnames(EpidemicData)
        T = fieldtype(D, f)
        # `coupled_mask` and `affected_individuals` are deliberate small Unions.
        f in (:coupled_mask, :affected_individuals) && continue
        @test isconcretetype(T)
    end
    @test s.data.neighbor_logprob !== nothing
    @test s.data.coupled_mask isa Matrix{Bool}
end

# ============================================================================
# 1. Delta-consistency: the conditional IS the joint, restricted
# ============================================================================

# The single most valuable test of the target. If the conditional is missing the
# neighbour block, or the observation term, or the entry gate, this fails and
# nothing else in the file necessarily does.
function _delta_consistency(s; entry_time=nothing, survival=nothing,
                              step_logprob=nothing, n_trials=25, seed=11,
                              obs_kw=(;), atol=1e-9)
    data = s.data
    loglik = epidemic_loglik(data; entry_time=entry_time, survival=survival,
                             step_logprob=step_logprob)
    obsll = epidemic_obs_loglik(data)
    cond = epidemic_conditional_loglik(data; entry_time=entry_time, survival=survival,
                                       step_logprob=step_logprob, obs_kw...)

    rng = StableRNG(seed)
    simulate = epidemic_simulator(data)
    X = simulate(rng, _MH_PARS)

    worst = 0.0
    for _ in 1:n_trials
        i = rand(rng, 1:data.n_individuals)
        first_t, last_t = data.sampling_period[i]

        _sync!(_MH_PARS, data, X)
        joint_cur = loglik(_MH_PARS, data, X) + obsll(_MH_PARS, data, X)
        cond_cur = cond(_MH_PARS, data, X, i)

        saved = copy(X[first_t:last_t, i])
        for t in first_t:last_t
            X[t, i] = rand(rng, 1:data.n_states)
        end

        _sync!(_MH_PARS, data, X)
        joint_can = loglik(_MH_PARS, data, X) + obsll(_MH_PARS, data, X)
        cond_can = cond(_MH_PARS, data, X, i)

        worst = max(worst, abs((cond_can - cond_cur) - (joint_can - joint_cur)))

        # keep the candidate half the time, so later trials see a moving X
        rand(rng, Bool) && (X[first_t:last_t, i] = saved)
    end
    worst
end

@testset "conditional target: delta equals the joint's delta" begin
    @testset "plain" begin
        s = _mh_setup(; n_t=10)
        @test _delta_consistency(s) < 1e-9
    end

    @testset "with observations" begin
        s = _mh_setup(; n_t=10, results=_random_results(StableRNG(3), 10, 6))
        @test _delta_consistency(s) < 1e-9
    end

    @testset "with coupled_transitions declared" begin
        # The mask must skip only moves whose probability cannot depend on the
        # focal. If it skipped a coupled one the delta would be short a term.
        s = _mh_setup(; n_t=10, results=_random_results(StableRNG(4), 10, 6),
                        coupled_transitions=[(:S, :I)])
        @test _delta_consistency(s) < 1e-9
    end

    @testset "with heterogeneous sampling windows" begin
        n_t = 10
        windows = [(1, 10), (2, 9), (1, 6), (4, 10), (3, 8), (1, 10)]
        s = _mh_setup(; n_t=n_t, results=_random_results(StableRNG(5), n_t, 6),
                        sampling_period=windows)
        # `:likelihood` (the default) restricts neighbour steps to the neighbour's
        # OWN window, which is exactly the set `epidemic_loglik` scores.
        @test _delta_consistency(s) < 1e-9
        # `:filter` matches what the forward filter does instead, and is therefore
        # NOT a restriction of the joint when windows differ. Assert the direction
        # of the discrepancy rather than pretending it does not exist.
        @test _delta_consistency(s; obs_kw=(; neighbor_window=:filter)) > 1e-6
    end

    @testset "with an entry gate" begin
        n_t = 10
        surv(model, data, i, t) = 0.97
        s = _mh_setup(; n_t=n_t, results=_random_results(StableRNG(6), n_t, 6))
        entry = [1, 3, 2, 4, 1, 5]
        @test _delta_consistency(s; entry_time=entry, survival=surv) < 1e-9
    end
end

@testset "conditional target: the neighbour block is actually present" begin
    # A guard against the test above passing for the wrong reason. Build the same
    # conditional with the coupling neutered and check the delta test now FAILS —
    # otherwise the model has no coupling and test 1 proves nothing about it.
    s = _mh_setup(; n_t=10, results=_random_results(StableRNG(3), 10, 6))
    data = s.data
    loglik = epidemic_loglik(data)
    obsll = epidemic_obs_loglik(data)
    no_neigh = epidemic_conditional_loglik(data;
        neighbor_logprob=(model, d, X, j, t, upd) -> 0.0)

    rng = StableRNG(21)
    X = epidemic_simulator(data)(rng, _MH_PARS)
    i = 2
    first_t, last_t = data.sampling_period[i]

    _sync!(_MH_PARS, data, X)
    j0 = loglik(_MH_PARS, data, X) + obsll(_MH_PARS, data, X)
    c0 = no_neigh(_MH_PARS, data, X, i)
    for t in first_t:last_t
        X[t, i] = X[t, i] == 1 ? 2 : 1
    end
    _sync!(_MH_PARS, data, X)
    j1 = loglik(_MH_PARS, data, X) + obsll(_MH_PARS, data, X)
    c1 = no_neigh(_MH_PARS, data, X, i)

    @test abs((c1 - c0) - (j1 - j0)) > 1e-6
end

# ============================================================================
# 2. The keystone: proposal == target  =>  log-ratio identically 0
# ============================================================================

@testset "iffbs_mh!: proposal == target gives log-ratio 0 (exactness)" begin
    for (name, s) in (
        ("no observations",   _mh_setup(; n_t=10)),
        ("noisy test",        _mh_setup(; n_t=10, results=_random_results(StableRNG(8), 10, 6))),
        ("coupled_transitions", _mh_setup(; n_t=10, results=_random_results(StableRNG(9), 10, 6),
                                            coupled_transitions=[(:S, :I)])),
        ("late-starting windows", _mh_setup(; n_t=10, sampling_period=[(3, 10) for _ in 1:6],
                                              results=_random_results(StableRNG(10), 10, 6))),
    )
        @testset "$name" begin
            data = s.data
            X = epidemic_simulator(data)(StableRNG(2), _MH_PARS)
            _sync!(_MH_PARS, data, X)
            rep = check_iffbs_exact(_MH_PARS, data, X; rng=StableRNG(3), n_sweeps=3)
            @test rep.n_checked > 5          # a check that decided nothing is no check
            @test rep.exact
            @test rep.max_abs_logratio < 1e-8
            @test isempty(rep.offenders)
        end
    end
end

@testset "check_iffbs_exact: a window ending before n_timepoints is NOT exact" begin
    # A REAL, PRE-EXISTING mismatch in `iffbs!`, found by this check and left
    # unfixed here on purpose (fixing it changes the sampler, which is a separate
    # measured change — see CLAUDE.md "measure each change alone").
    #
    # `forward_filter!` calls `rest_contribution` at every t in the focal's window,
    # INCLUDING its last one. At `t == last_t` the coupling term scores each
    # neighbour's `t -> t+1` move. When that neighbour's own window also ends at
    # `last_t`, the joint contains no such factor — `epidemic_loglik` stops at
    # `last_t - 1` — and `X[t+1, j]` is a cell the sampler never writes. So the
    # filter is scoring a term that is not in the target, off stale data.
    #
    # It cannot bite when every window ends at `n_timepoints`, because
    # `rest_contribution` returns all-ones at `t == n_timepoints`. It CAN bite any
    # model with per-individual end times (the badger model has them).
    s = _mh_setup(; n_t=10, sampling_period=[(2, 9) for _ in 1:6],
                    results=_random_results(StableRNG(30), 10, 6))
    data = s.data
    X = epidemic_simulator(data)(StableRNG(31), _MH_PARS)
    _sync!(_MH_PARS, data, X)

    rep = check_iffbs_exact(_MH_PARS, data, X; rng=StableRNG(32), n_sweeps=2)
    @test !rep.exact
    @test rep.max_abs_logratio > 1e-3

    # ... and `neighbor_window=:filter` reproduces what the filter actually does,
    # so the ratio goes back to 1. That is a diagnosis, not a fix: with `:filter`
    # the conditional is no longer a restriction of the joint (asserted in the
    # delta-consistency testset above), so the MH step would be correcting towards
    # the wrong target.
    filter_target = epidemic_conditional_loglik(data; neighbor_window=:filter)
    rep2 = check_iffbs_exact(_MH_PARS, data, X; rng=StableRNG(32), n_sweeps=2,
                             target=filter_target)
    @test rep2.exact
end

@testset "check_iffbs_exact: reports a real mismatch" begin
    # A `coupling_trans_mat` whose rates have drifted from the target's is the
    # silent failure `check_iffbs_exact` exists to make loud.
    n_t = 10
    results = _random_results(StableRNG(12), n_t, 6)
    state_space = [:S, :I]
    n_groups, n_per_group = 2, 3
    group = repeat(1:n_groups; inner=n_per_group)

    aggs = @aggregate state_space begin
        @array n_infected Int (n_groups, n_t)
        n_infected[data.group[i], t] += (state == :I)
    end
    infection(model, data, i, t) =
        -expm1(-(model.α + model.β * data.aggregates[:n_infected][data.group[i], t]))
    # deliberately WRONG coupling rate: twice the transmission
    infection_wrong(model, data, i, t) =
        -expm1(-(model.α + 2 * model.β * data.aggregates[:n_infected][data.group[i], t]))
    recovery(model, data, i, t) = 1 / model.m

    spec = @transitions state_space begin
        S -> I = infection
        I -> S = recovery
    end
    wrong = @transitions state_space begin
        S -> I = infection_wrong
        I -> S = recovery
    end

    data = epidemic_data(; n_individuals=6, n_timepoints=n_t, group=group,
                           trans_mat=spec, coupling_trans_mat=wrong,
                           starting_state=(model, data, X, i, t) -> [1 - model.ν, model.ν],
                           aggregates=aggs, observation_process=_noisy_test,
                           results=results)

    X = epidemic_simulator(data)(StableRNG(2), _MH_PARS)
    _sync!(_MH_PARS, data, X)
    rep = check_iffbs_exact(_MH_PARS, data, X; rng=StableRNG(3), n_sweeps=2)
    @test !rep.exact
    @test rep.max_abs_logratio > 1e-3
    @test !isempty(rep.offenders)
    @test rep.worst_individual in 1:6
end

# ============================================================================
# 3. The aggregates survive both branches
# ============================================================================

@testset "iffbs_mh!: aggregates stay consistent on accept AND reject" begin
    for force in (:accept, :reject, :none)
        @testset "force=$force" begin
            s = _mh_setup(; n_t=12, results=_random_results(StableRNG(13), 12, 6))
            data = s.data
            X = epidemic_simulator(data)(StableRNG(4), _MH_PARS)
            _sync!(_MH_PARS, data, X)

            # A mismatched proposal, so accept and reject are both reachable.
            prop = uncorrected_proposal(data)
            target = epidemic_conditional_loglik(data)
            stats = MHStats(data.n_individuals)
            rng = StableRNG(5)
            for _ in 1:8
                iffbs_mh!(_MH_PARS, data, X, rng; proposal=prop, target=target,
                          stats=stats, force=force)
                incremental = copy(data.aggregates[:n_infected])
                _sync!(_MH_PARS, data, X)
                @test incremental == data.aggregates[:n_infected]
            end

            if force === :reject
                @test sum(stats.accepted) == 0
                @test sum(stats.proposed) > 0
            elseif force === :accept
                @test sum(stats.accepted) == sum(stats.proposed)
            end
        end
    end
end

@testset "iffbs_mh!: a forced reject leaves X untouched" begin
    s = _mh_setup(; n_t=12, results=_random_results(StableRNG(14), 12, 6))
    data = s.data
    X = epidemic_simulator(data)(StableRNG(6), _MH_PARS)
    _sync!(_MH_PARS, data, X)
    before = copy(X)
    agg_before = copy(data.aggregates[:n_infected])

    iffbs_mh!(_MH_PARS, data, X, StableRNG(7); proposal=uncorrected_proposal(data),
              target=epidemic_conditional_loglik(data), force=:reject)

    @test X == before
    @test data.aggregates[:n_infected] == agg_before
end

# ============================================================================
# 4. Exact enumeration: the kernel targets the right distribution
# ============================================================================

# Enumerate every trajectory of a tiny model and compare the MH kernel's empirical
# distribution against the exact posterior. This is the only test here that proves
# the CORRECTION works; everything else proves internal consistency.
function _enumerate_posterior(model, data, loglik, obsll)
    n_ind, n_t, N = data.n_individuals, data.n_timepoints, data.n_states
    ncell = n_ind * n_t
    total = N^ncell
    logps = Vector{Float64}(undef, total)
    states = Vector{Matrix{Int}}(undef, total)
    X = Matrix{Int}(undef, n_t, n_ind)
    for k in 0:(total - 1)
        v = k
        for c in 1:ncell
            X[c] = (v % N) + 1
            v ÷= N
        end
        _sync!(model, data, X)
        logps[k + 1] = loglik(model, data, X) + obsll(model, data, X)
        states[k + 1] = copy(X)
    end
    m = maximum(logps)
    p = exp.(logps .- m)
    p ./= sum(p)
    (; states, probs=p)
end

_state_key(X) = Tuple(vec(X))

function _empirical_distribution(sweeps!, model, data, X0, n_draws, keys_index)
    counts = zeros(Int, length(keys_index))
    X = copy(X0)
    _sync!(model, data, X)
    for _ in 1:n_draws
        sweeps!(X)
        counts[keys_index[_state_key(X)]] += 1
    end
    counts ./ n_draws
end

@testset "iffbs_mh!: hits the exact posterior of a tiny enumerable model" begin
    # 2 individuals x 3 timepoints x 2 states = 64 trajectories.
    n_t, n_ind = 3, 2
    state_space = [:S, :I]
    results = [1 -1; -1 0; 0 1]                 # n_t x n_ind

    aggs = @aggregate state_space begin
        @array n_infected Int (1, n_t)
        n_infected[1, t] += (state == :I)
    end
    # Strong coupling, so the between-individual term genuinely matters and a
    # proposal that drops it is genuinely wrong.
    infection(model, data, i, t) =
        -expm1(-(model.α + model.β * data.aggregates[:n_infected][1, t]))
    recovery(model, data, i, t) = 1 / model.m
    spec = @transitions state_space begin
        S -> I = infection
        I -> S = recovery
    end

    pars = (; α=0.15, β=0.9, m=2.5, ν=0.35, θʳ=0.8, θᶠ=0.75)
    data = epidemic_data(; n_individuals=n_ind, n_timepoints=n_t,
                           group=ones(Int, n_ind), trans_mat=spec,
                           starting_state=(model, d, X, i, t) -> [1 - model.ν, model.ν],
                           aggregates=aggs, observation_process=_noisy_test,
                           results=results)

    loglik = epidemic_loglik(data)
    obsll = epidemic_obs_loglik(data)
    exact = _enumerate_posterior(pars, data, loglik, obsll)
    keys_index = Dict(_state_key(exact.states[k]) => k for k in eachindex(exact.states))
    @test length(keys_index) == 2^(n_ind * n_t)
    @test isapprox(sum(exact.probs), 1.0; atol=1e-12)

    X0 = fill(1, n_t, n_ind)
    n_draws = 200_000

    @testset "MH with a coupling-free proposal" begin
        prop = uncorrected_proposal(data)
        target = epidemic_conditional_loglik(data)
        stats = MHStats(n_ind)
        rng = StableRNG(1234)
        emp = _empirical_distribution(
            X -> iffbs_mh!(pars, data, X, rng; proposal=prop, target=target, stats=stats),
            pars, data, X0, n_draws, keys_index)
        tv = 0.5 * sum(abs, emp .- exact.probs)
        @test tv < 0.01
        @test 0.0 < acceptance_rate(stats) <= 1.0
    end

    @testset "MH with a wrong-rate proposal" begin
        # A proposal chain with the transmission halved: still a valid proposal,
        # and the correction must still land on the exact posterior.
        infection_half(model, d, i, t) =
            -expm1(-(model.α + 0.5 * model.β * d.aggregates[:n_infected][1, t]))
        wrong = @transitions state_space begin
            S -> I = infection_half
            I -> S = recovery
        end
        prop = markov_proposal(data, wrong)
        target = epidemic_conditional_loglik(data)
        rng = StableRNG(4321)
        emp = _empirical_distribution(
            X -> iffbs_mh!(pars, data, X, rng; proposal=prop, target=target),
            pars, data, X0, n_draws, keys_index)
        @test 0.5 * sum(abs, emp .- exact.probs) < 0.01
    end

    @testset "the uncorrected sweep is measurably biased" begin
        # Without this, the two tests above cannot distinguish "MH is correct" from
        # "the mismatch did not matter". `iffbs!` run with the coupling-free filter
        # is the paper's uncorrected-iFFBS, and it must MISS the exact posterior.
        prop = uncorrected_proposal(data)
        rng = StableRNG(999)
        emp = _empirical_distribution(
            X -> iffbs!(pars, data, X, rng; proposal=prop),
            pars, data, X0, n_draws, keys_index)
        @test 0.5 * sum(abs, emp .- exact.probs) > 0.02
    end

    @testset "plain iffbs! (exact filter) also hits it" begin
        rng = StableRNG(777)
        emp = _empirical_distribution(
            X -> iffbs!(pars, data, X, rng),
            pars, data, X0, n_draws, keys_index)
        @test 0.5 * sum(abs, emp .- exact.probs) < 0.01
    end
end

# ============================================================================
# 5. Detailed balance, spot check
# ============================================================================

@testset "iffbs_mh!: detailed balance holds for the single-individual kernel" begin
    # For an independence proposal q and target pi, the MH kernel satisfies
    #   pi(x) q(x') min(1, r)  ==  pi(x') q(x) min(1, 1/r),  r = pi(x')q(x)/pi(x)q(x')
    # which is an identity given exact q and pi. Checking it numerically catches a
    # sign error in the ratio that test 4 would only show as a slow drift.
    n_t, n_ind = 4, 2
    state_space = [:S, :I]
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
    pars = (; α=0.12, β=0.7, m=3.0, ν=0.3, θʳ=0.8, θᶠ=0.7)
    results = [1 -1; -1 0; 0 1; 1 1]
    data = epidemic_data(; n_individuals=n_ind, n_timepoints=n_t,
                           group=ones(Int, n_ind), trans_mat=spec,
                           starting_state=(model, d, X, i, t) -> [1 - model.ν, model.ν],
                           aggregates=aggs, observation_process=_noisy_test,
                           results=results)

    prop = uncorrected_proposal(data)
    target = epidemic_conditional_loglik(data)
    i = 1

    # Enumerate all 2^4 paths for individual i, with individual 2 held fixed.
    X = fill(1, n_t, n_ind)
    X[:, 2] = [1, 2, 2, 1]
    paths = [[((k >> (j - 1)) & 1) + 1 for j in 1:n_t] for k in 0:(2^n_t - 1)]

    # log q and log pi for every path, both evaluated the way the kernel does.
    logq = Dict{Vector{Int},Float64}()
    logpi = Dict{Vector{Int},Float64}()

    # The filter is leave-one-out, hence identical for every candidate path — this
    # is exactly why the proposal is an INDEPENDENCE proposal.
    X[:, i] .= paths[1]
    _sync!(pars, data, X)
    for t in 1:n_t
        apply_summaries!(data.derived_summaries, pars, data, X, X[t, i], i, t, true)
    end
    data._focal[] = i
    probs, trans_cache = forward_filter(view(X, 1:n_t, i), 1, n_t, pars, data, X, i, prop)
    probs = copy(probs); trans_cache = copy(trans_cache)
    data._focal[] = -1

    for p in paths
        logq[p] = backward_logq(probs, trans_cache, p, data, n_t)
        X[:, i] .= p
        _sync!(pars, data, X)
        logpi[p] = target(pars, data, X, i)
    end

    # q must be a proper distribution over the 2^n_t paths.
    @test isapprox(sum(exp(logq[p]) for p in paths), 1.0; atol=1e-10)

    worst = 0.0
    for a in paths, b in paths
        a == b && continue
        lr_ab = (logpi[b] - logpi[a]) - (logq[b] - logq[a])
        lr_ba = -lr_ab
        # flow a -> b against flow b -> a
        f_ab = exp(logpi[a] + logq[b] + min(0.0, lr_ab))
        f_ba = exp(logpi[b] + logq[a] + min(0.0, lr_ba))
        worst = max(worst, abs(f_ab - f_ba) / max(f_ab, f_ba, 1e-300))
    end
    @test worst < 1e-10
end

# ============================================================================
# 6. RNG-stream equivalence
# ============================================================================

@testset "iffbs_mh!: with proposal == target it reproduces iffbs! exactly" begin
    # Requires the candidate to be drawn with the same categorical calls in the
    # same order, and scoring the current path to consume no randomness at all.
    #
    # `force=:accept`, NOT `force=:none`. With the proposal equal to the target the
    # exact ratio is 0, but the COMPUTED ratio is a difference of ~1e-14 sums whose
    # sign is arbitrary; a ratio of -1e-16 takes the `log(rand(rng)) < logratio`
    # branch, consumes a variate (and then accepts anyway), and the streams part
    # company. Forcing accept isolates the property actually being tested — that
    # the proposal machinery draws exactly what `iffbs!` draws — from floating-point
    # noise in a comparison against zero. The no-`rand()` rule gets its own test
    # below, and the size of the ratio gets test 2.
    s = _mh_setup(; n_t=14, results=_random_results(StableRNG(15), 14, 6))
    data = s.data

    X1 = epidemic_simulator(data)(StableRNG(8), _MH_PARS)
    _sync!(_MH_PARS, data, X1)
    for _ in 1:5
        iffbs!(_MH_PARS, data, X1, StableRNG(1000))
    end
    agg1 = copy(data.aggregates[:n_infected])

    X2 = epidemic_simulator(data)(StableRNG(8), _MH_PARS)
    _sync!(_MH_PARS, data, X2)
    target = epidemic_conditional_loglik(data)
    prop = iffbs_proposal(data)
    for _ in 1:5
        iffbs_mh!(_MH_PARS, data, X2, StableRNG(1000); proposal=prop, target=target,
                  force=:accept)
    end

    @test X1 == X2
    @test agg1 == data.aggregates[:n_infected]
end

@testset "iffbs_mh!: log alpha >= 0 consumes no randomness" begin
    # The rule the reference records ("when acc >= 1, avoid drawing RNG so
    # forced-accept runs reproduce classic iFFBS random stream"). Tested directly
    # by comparing the RNG STATE after a sweep that always accepts on its own
    # merits against one that was forced to.
    s = _mh_setup(; n_t=12, results=_random_results(StableRNG(23), 12, 6))
    data = s.data
    prop = uncorrected_proposal(data)

    # A target that GUARANTEES log alpha >= 0, which a merely "large" one does not:
    # the kernel evaluates the target twice per decision, current first and
    # candidate second, so a strictly increasing counter makes the candidate's
    # value larger by 1e9 every time and swamps any proposal-density difference.
    # (A path-dependent bonus like `1e6 * sum(X[:, i])` is NOT enough — its sign
    # follows whichever path has more infected cells, so roughly half the ratios
    # come out negative and the test measures nothing.)
    function increasing_target()
        counter = Ref(0.0)
        (model, d, XX, i) -> (counter[] += 1e9; counter[])
    end

    function after(force)
        X = epidemic_simulator(data)(StableRNG(24), _MH_PARS)
        _sync!(_MH_PARS, data, X)
        rng = StableRNG(2000)
        iffbs_mh!(_MH_PARS, data, X, rng; proposal=prop, target=increasing_target(),
                  force=force)
        (copy(X), rand(rng))
    end
    Xa, ra = after(:none)
    Xb, rb = after(:accept)
    @test Xa == Xb
    @test ra == rb          # identical RNG state => no `rand()` was consumed
end

@testset "backward_logq: scores exactly what backward_sample_logq! drew" begin
    s = _mh_setup(; n_t=9, results=_random_results(StableRNG(16), 9, 6))
    data = s.data
    X = epidemic_simulator(data)(StableRNG(9), _MH_PARS)
    _sync!(_MH_PARS, data, X)
    i = 3
    first_t, last_t = data.sampling_period[i]
    n_t = last_t - first_t + 1

    for t in first_t:last_t
        apply_summaries!(data.derived_summaries, _MH_PARS, data, X, X[t, i], i, t, true)
    end
    data._focal[] = i
    probs, trans_cache = forward_filter(view(X, first_t:last_t, i), first_t, last_t,
                                        _MH_PARS, data, X, i)
    dest = zeros(Int, n_t)
    lq_drawn = backward_sample_logq!(dest, probs, trans_cache, data, n_t, StableRNG(31))
    lq_scored = backward_logq(probs, trans_cache, dest, data, n_t)
    data._focal[] = -1
    for t in first_t:last_t
        apply_summaries!(data.derived_summaries, _MH_PARS, data, X, X[t, i], i, t, false)
    end

    @test lq_drawn ≈ lq_scored atol = 1e-12
    @test all(1 .<= dest .<= data.n_states)

    # `backward_sample_logq!` must draw exactly what `backward_sample!` would.
    xᵢ = zeros(Int, n_t)
    dest2 = zeros(Int, n_t)
    for t in first_t:last_t
        apply_summaries!(data.derived_summaries, _MH_PARS, data, X, X[t, i], i, t, true)
    end
    data._focal[] = i
    probs, trans_cache = forward_filter(view(X, first_t:last_t, i), first_t, last_t,
                                        _MH_PARS, data, X, i)
    backward_sample!(probs, trans_cache, xᵢ, first_t, last_t, _MH_PARS, data, X, i, StableRNG(31))
    backward_sample_logq!(dest2, probs, trans_cache, data, n_t, StableRNG(31))
    data._focal[] = -1
    for t in first_t:last_t
        apply_summaries!(data.derived_summaries, _MH_PARS, data, X, X[t, i], i, t, false)
    end
    @test xᵢ == dest2
end

# ============================================================================
# 7. Semi-Markov via `step_logprob`
# ============================================================================

# A three-state S -> I -> R chain whose I -> R hazard depends on time since
# infection (a discrete Weibull), i.e. genuinely semi-Markov. The filter runs the
# geometric approximation; MH corrects.
function _time_in_state(X, i, t)
    s = 0
    j = t
    @inbounds while j > 1 && X[j - 1, i] == X[t, i]
        s += 1
        j -= 1
    end
    s
end

_weibull_hazard(s, shape, scale) =
    clamp(1 - exp(-(((s + 1) / scale)^shape - (s / scale)^shape)), 1e-12, 1 - 1e-12)

@testset "semi-Markov target: step_logprob seam" begin
    n_t, n_ind = 4, 2
    state_space = [:S, :I, :R]
    results = fill(-1, n_t, n_ind)

    aggs = @aggregate state_space begin
        @array n_infected Int (1, n_t)
        n_infected[1, t] += (state == :I)
    end
    infection(model, data, i, t) =
        -expm1(-(model.α + model.β * data.aggregates[:n_infected][1, t]))
    # the PROPOSAL's recovery: memoryless, matched to the Weibull's mean-ish
    recovery_geom(model, data, i, t) = model.p_rec
    spec_geom = @transitions state_space begin
        S -> I = infection
        I -> R = recovery_geom
    end

    pars = (; α=0.2, β=0.6, ν=0.3, p_rec=0.35, w_shape=2.5, w_scale=2.0)
    data = epidemic_data(; n_individuals=n_ind, n_timepoints=n_t,
                           group=ones(Int, n_ind), trans_mat=spec_geom,
                           starting_state=(model, d, X, i, t) -> [1 - model.ν, model.ν, 0.0],
                           aggregates=aggs)

    # The TARGET step: identical to the default except that I -> {I,R} uses the
    # sojourn-dependent hazard. S -> {S,I} is untouched, so the coupling (which
    # only ever influences S -> I) stays Markov — which is why one seam suffices.
    function sm_step(model, d, X, i, t)
        from, to = X[t, i], X[t + 1, i]
        if from == 2                                   # I
            h = _weibull_hazard(_time_in_state(X, i, t), model.w_shape, model.w_scale)
            return to == 3 ? log(h) : (to == 2 ? log1p(-h) : -Inf)
        end
        default_step_logprob(model, d, X, i, t)
    end

    loglik = epidemic_loglik(data; step_logprob=sm_step)
    obsll = epidemic_obs_loglik(data)
    target = epidemic_conditional_loglik(data; step_logprob=sm_step)

    @testset "delta-consistency with the semi-Markov step" begin
        rng = StableRNG(41)
        X = epidemic_simulator(data)(rng, pars)
        worst = 0.0
        for _ in 1:40
            i = rand(rng, 1:n_ind)
            _sync!(pars, data, X)
            j0 = loglik(pars, data, X) + obsll(pars, data, X)
            c0 = target(pars, data, X, i)
            for t in 1:n_t
                X[t, i] = rand(rng, 1:3)
            end
            _sync!(pars, data, X)
            j1 = loglik(pars, data, X) + obsll(pars, data, X)
            c1 = target(pars, data, X, i)
            isfinite(j0) && isfinite(j1) &&
                (worst = max(worst, abs((c1 - c0) - (j1 - j0))))
        end
        @test worst < 1e-9
    end

    @testset "SM-iFFBS hits the exact posterior" begin
        exact = _enumerate_posterior(pars, data, loglik, obsll)
        keys_index = Dict(_state_key(exact.states[k]) => k for k in eachindex(exact.states))
        X0 = fill(1, n_t, n_ind)

        prop = iffbs_proposal(data)              # the geometric chain
        stats = MHStats(n_ind)
        rng = StableRNG(4242)
        emp = _empirical_distribution(
            X -> iffbs_mh!(pars, data, X, rng; proposal=prop, target=target, stats=stats),
            pars, data, X0, 300_000, keys_index)
        @test 0.5 * sum(abs, emp .- exact.probs) < 0.012
        @test acceptance_rate(stats) > 0.3

        # And the UNCORRECTED geometric sweep must miss it — otherwise the semi-
        # Markov structure was not doing anything and this test proves nothing.
        rng2 = StableRNG(515)
        emp2 = _empirical_distribution(X -> iffbs!(pars, data, X, rng2),
                                       pars, data, X0, 300_000, keys_index)
        @test 0.5 * sum(abs, emp2 .- exact.probs) > 0.02
    end
end

# ============================================================================
# 8. Acceptance statistics
# ============================================================================

@testset "MHStats: bookkeeping" begin
    s = _mh_setup(; n_t=12, results=_random_results(StableRNG(17), 12, 6))
    data = s.data
    X = epidemic_simulator(data)(StableRNG(10), _MH_PARS)
    _sync!(_MH_PARS, data, X)

    stats = MHStats(data.n_individuals)
    prop = uncorrected_proposal(data)
    target = epidemic_conditional_loglik(data)
    rng = StableRNG(11)
    for _ in 1:30
        iffbs_mh!(_MH_PARS, data, X, rng; proposal=prop, target=target, stats=stats)
    end

    @test sum(stats.proposed) + sum(stats.identical) == 30 * data.n_individuals
    @test all(stats.accepted .<= stats.proposed)
    @test 0 <= acceptance_rate(stats) <= 1
    @test 0 <= identical_rate(stats) <= 1
    @test acceptance_rate(stats) > 0.2       # a usable proposal on this model
    @test stats.max_abs_logratio[] > 0       # the proposal really is mismatched

    reset_stats!(stats)
    @test sum(stats.proposed) == 0
    @test isnan(acceptance_rate(stats))
    @test stats.max_abs_logratio[] == 0.0
end

@testset "epidemic_latent_sampler: mh switch and its guards" begin
    s = _mh_setup(; n_t=10, results=_random_results(StableRNG(18), 10, 6))
    data = s.data

    plain = epidemic_latent_sampler(data)
    mhs = epidemic_latent_sampler(data; mh=true, proposal=uncorrected_proposal(data))
    @test mhs isa IFFBSMHSampler
    @test mhs.stats isa MHStats

    X = epidemic_simulator(data)(StableRNG(12), _MH_PARS)
    _sync!(_MH_PARS, data, X)
    @test plain(StableRNG(13), _MH_PARS, X) === X

    _sync!(_MH_PARS, data, X)
    @test mhs(StableRNG(14), _MH_PARS, X) === X
    @test sum(mhs.stats.proposed) + sum(mhs.stats.identical) == data.n_individuals

    # The guards. Passing a proposal without asking for the correction is the
    # paper's uncorrected-iFFBS by accident, and must not be reachable silently.
    @test_throws ErrorException epidemic_latent_sampler(data; proposal=uncorrected_proposal(data))
    @test_throws ErrorException epidemic_latent_sampler(data;
        target=epidemic_conditional_loglik(data))
    @test_throws ErrorException epidemic_latent_sampler(data; stats=MHStats(6))
end

@testset "iffbs_mh!: refuses focal_self_contribution=false" begin
    # The construction warning is expected here; silence it so the test output
    # stays readable.
    s = Logging.with_logger(Logging.NullLogger()) do
        _mh_setup(; n_t=8, focal_self_contribution=false)
    end
    X = epidemic_simulator(s.data)(StableRNG(15), _MH_PARS)
    _sync!(_MH_PARS, s.data, X)
    @test_throws ErrorException iffbs_mh!(_MH_PARS, s.data, X, StableRNG(16))
    @test_throws ErrorException check_iffbs_exact(_MH_PARS, s.data, X; rng=StableRNG(16))
    @test_throws ErrorException epidemic_latent_sampler(s.data; mh=true)
end

# ============================================================================
# 9. Degenerate ratios
# ============================================================================

@testset "iffbs_mh_individual!: non-finite ratios" begin
    s = _mh_setup(; n_t=8, results=_random_results(StableRNG(19), 8, 6))
    data = s.data
    X = epidemic_simulator(data)(StableRNG(17), _MH_PARS)
    _sync!(_MH_PARS, data, X)
    prop = iffbs_proposal(data)

    # A target that is -Inf for everything must reject every non-identical
    # proposal and leave X exactly where it was.
    minus_inf = (model, d, XX, i) -> -Inf
    before = copy(X)
    agg_before = copy(data.aggregates[:n_infected])
    iffbs_mh!(_MH_PARS, data, X, StableRNG(18); proposal=prop, target=minus_inf)
    @test X == before
    @test data.aggregates[:n_infected] == agg_before

    # A NaN ratio rejects by default and errors on request.
    nan_target = (model, d, XX, i) -> NaN
    _sync!(_MH_PARS, data, X)
    before = copy(X)
    iffbs_mh!(_MH_PARS, data, X, StableRNG(19); proposal=prop, target=nan_target)
    @test X == before
    @test_throws ErrorException iffbs_mh!(_MH_PARS, data, X, StableRNG(20);
                                          proposal=prop, target=nan_target,
                                          on_nonfinite=:error)

    # +Inf ratios accept. The kernel scores the current path first and the
    # candidate second, so a strictly increasing target makes every log alpha
    # +1e9 and every decision an accept.
    _sync!(_MH_PARS, data, X)
    counter = Ref(0.0)
    increasing = (model, d, XX, i) -> (counter[] += 1e9; counter[])
    stats = MHStats(data.n_individuals)
    iffbs_mh!(_MH_PARS, data, X, StableRNG(21); proposal=prop, target=increasing, stats=stats)
    @test sum(stats.accepted) == sum(stats.proposed)
    @test sum(stats.proposed) > 0
end

@testset "iffbs_mh_individual!: rejects an unknown force" begin
    s = _mh_setup(; n_t=8)
    data = s.data
    X = epidemic_simulator(data)(StableRNG(22), _MH_PARS)
    _sync!(_MH_PARS, data, X)
    @test_throws ErrorException iffbs_mh!(_MH_PARS, data, X, StableRNG(23);
                                          proposal=uncorrected_proposal(data),
                                          force=:sometimes)
end

# ============================================================================
# Allocation guard
# ============================================================================

@testset "iffbs_mh!: the sweep does not allocate per timepoint" begin
    # Not a benchmark — a guard that the path buffers really are reused. A per-
    # timepoint allocation would scale with n_t; this compares two window lengths.
    function sweep_alloc(n_t)
        s = _mh_setup(; n_t=n_t, results=_random_results(StableRNG(20), n_t, 6))
        data = s.data
        X = epidemic_simulator(data)(StableRNG(21), _MH_PARS)
        _sync!(_MH_PARS, data, X)
        prop = uncorrected_proposal(data)
        target = epidemic_conditional_loglik(data)
        rng = StableRNG(22)
        iffbs_mh!(_MH_PARS, data, X, rng; proposal=prop, target=target)   # warm up
        @allocated iffbs_mh!(_MH_PARS, data, X, rng; proposal=prop, target=target)
    end
    a20 = sweep_alloc(20)
    a80 = sweep_alloc(80)
    # `starting_state` and the observation process allocate per (i, t) by design
    # (that is the documented slow path), so this cannot assert zero — only that
    # the MH machinery itself adds no new per-timepoint allocation on top.
    @test a80 < 6 * a20
end
