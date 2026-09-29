# Scoring one leave-future-out window: forward-simulate the population from
# the cutoff, then score the observations that follow cell by cell.
#
# The user supplies `cell_logdensity` and, optionally, `is_informative`. The
# second is worth supplying: on the badger data only ~2% of cells depended on
# the latent state at all, so the granularity axis had almost nothing to
# redistribute and every arm agreed to within a few nats.

"""
    forward_simulate(rng, model, data, X, t_star, M; constrain=nothing) -> Matrix

One forward trajectory for the whole population over `t_star+1 : t_star+M`,
starting from the states in `X` at `t_star`.

The population is simulated jointly, not individual by individual: each step's
transition probabilities are computed from that trajectory's own composition, so
the coupling between individuals is preserved. Simulating one individual at a
time against a frozen background would silently drop it.

Returns a copy; `X` is not modified.

`constrain` is an optional `(i, t, last_t) -> Bool` (or `(i, t) -> Bool`) saying
"individual `i` is known to be present at `t`, so forbid the absorbing state";
`last_t` is the end of the window being simulated. See
[`survival_constrained`](@ref) for why that needs a matching weight and what
happens if the two disagree.

`known` gives, per individual, when the study first knew of it (see
[`known_times`](@ref)); by default the start of its sampling period. An
individual whose sampling period has begun but who is not yet known at `t_star`
was never part of the fit, so it has no sampled state to carry forward: it is
left out of the forecast and out of the population summaries.

`entrants` says what to do with individuals whose sampling period starts after
`t_star`. `true` (the default) draws each one from the starting state when it
enters, so that it can affect the cohort's dynamics. That uses its entry time,
which is only known from later data; `false` fixes the forecast population at
the cutoff, leaving later entrants out of the forecast and out of the
population summaries. The choice matters only when individuals are coupled.
"""
function forward_simulate(rng::AbstractRNG, model, data::EpidemicData, X,
                          t_star::Int, M::Int; constrain = nothing,
                          aggregates_ready::Bool = false, known = nothing,
                          entrants::Bool = true, guide::Bool = false,
                          logw = nothing)
    guide && constrain !== nothing && throw(ArgumentError(
        "use the observation-guided proposal or the survival constraint, not both"))
    guide && logw === nothing && throw(ArgumentError(
        "the observation-guided proposal needs `logw` to record its weights"))
    known === nothing && (known = first.(data.sampling_period))
    Xf = copy(X)
    # Forward transitions may depend on user-declared population summaries
    # (infectious counts, numbers alive, and so on). Rebuild them from this
    # posterior trajectory, then keep each simulated future row in sync as it
    # replaces the stored post-cutoff states.
    if !aggregates_ready
        reset_aggregates!(data)
        apply_derived_summaries!(model, data, Xf)
        _drop_unknown!(model, data, Xf, t_star, known; entrants)
    end
    last_t = min(t_star + M, data.n_timepoints)
    N = data.n_states
    P = zeros(Float64, N, N)
    rowsum = zeros(Float64, N)
    row = zeros(Float64, N)
    absorbing = _absorbing_state(data)
    constrain === nothing || (constrain = _windowed(constrain))
    if guide
        beta, bnext, gw = zeros(N), zeros(N), zeros(N)
        P2, rowsum2 = zeros(Float64, N, N), zeros(Float64, N)
    end

    for t in (t_star + 1):last_t
        prev = copy(view(Xf, t - 1, :))
        for i in 1:data.n_individuals
            f_i, l_i = data.sampling_period[i]
            (f_i <= t <= min(l_i, data.n_timepoints)) || continue
            f_i <= t_star < known[i] && continue
            !entrants && f_i > t_star && continue
            # An individual entering after the cutoff has no valid state at
            # `t-1`. Draw its state at entry from the declared initial
            # distribution; advancing the stale pre-entry cell would leak an
            # arbitrary placeholder into the forecast.
            if f_i > t_star && t == f_i
                p0 = data.starting_state(model, data, Xf, i, t)
                k = _sample_categorical(rng, p0)
                old = Xf[t, i]
                apply_summaries!(data.derived_summaries, model, data, Xf,
                                 old, i, t, true)
                Xf[t, i] = k
                apply_summaries!(data.derived_summaries, model, data, Xf,
                                 k, i, t, false)
                continue
            end
            # An absorbing state stays absorbing: it must not be re-randomised,
            # and the constraint below must not resurrect it.
            if absorbing !== nothing && prev[i] == absorbing
                old = Xf[t, i]
                apply_summaries!(data.derived_summaries, model, data, Xf,
                                 old, i, t, true)
                Xf[t, i] = absorbing
                apply_summaries!(data.derived_summaries, model, data, Xf,
                                 absorbing, i, t, false)
                continue
            end
            transition_matrix_at!(P, rowsum, data.trans_mat, model, data, Xf, i, t - 1)
            @inbounds for s in 1:N
                row[s] = Float64(P[prev[i], s])
            end
            if known[i] <= t_star && constrain !== nothing && absorbing !== nothing &&
                    constrain(i, t, last_t)
                p_surv = 1.0 - row[absorbing]
                p_surv = max(p_surv, 1e-12)
                @inbounds for s in 1:N
                    row[s] = s == absorbing ? 0.0 : row[s] / p_surv
                end
            end
            lw = 0.0
            if guide && known[i] <= t_star
                # Draw in proportion to the move's probability times how likely
                # the individual's own remaining observations in the window are
                # from each state, and record p/q = Z / beta[k].
                _lookahead!(beta, bnext, gw, P2, rowsum2, model, data, Xf, i, t,
                            min(last_t, l_i, data.n_timepoints))
                Z = 0.0
                @inbounds for s in 1:N
                    Z += row[s] * beta[s]
                end
                if Z > 0
                    @inbounds for s in 1:N
                        row[s] = row[s] * beta[s] / Z
                    end
                else
                    lw = -Inf        # no state explains the window: h = 0
                end
            end
            u = rand(rng); c = 0.0; k = N
            @inbounds for s in 1:N
                c += row[s]
                if u <= c; k = s; break; end
            end
            if guide && known[i] <= t_star
                logw[t, i] = isfinite(lw) ? log(Z) - log(beta[k]) : -Inf
            end
            old = Xf[t, i]
            apply_summaries!(data.derived_summaries, model, data, Xf,
                             old, i, t, true)
            Xf[t, i] = k
            apply_summaries!(data.derived_summaries, model, data, Xf,
                             k, i, t, false)
        end
    end
    Xf
end

# Backward message over hi, hi-1, ..., t for individual i: beta[s] is
# proportional to the probability of its observations at t..hi given it is in s
# at t. Future rates are evaluated with whatever the aggregates currently hold
# for those times, which the simulation has not reached yet; that makes this an
# approximation, but only of the proposal -- the weight p/q is exact.
function _lookahead!(beta, bnext, w, P, rowsum, model, data::EpidemicData, X, i, t, hi)
    K = length(beta)
    _obs_weights!(w, model, data, X, i, hi)
    copyto!(beta, w)
    for u in (hi - 1):-1:t
        transition_matrix_at!(P, rowsum, data.trans_mat, model, data, X, i, u)
        _obs_weights!(w, model, data, X, i, u)
        z = 0.0
        @inbounds for a in 1:K
            acc = 0.0
            for b in 1:K
                acc += Float64(P[a, b]) * beta[b]
            end
            bnext[a] = w[a] * acc
            z += bnext[a]
        end
        if z > 0
            @inbounds for a in 1:K
                beta[a] = bnext[a] / z
            end
        else
            fill!(beta, 0.0)
            return beta
        end
    end
    beta
end

"""
    _absorbing_state(data) -> Union{Int,Nothing}

The index of the model's absorbing state, or `nothing` if it has none.

DERIVED from the declared transitions -- a state that is the source of no
transition to any other state -- rather than assumed to be the last one. A model
without mortality has no absorbing state and must not be given one; hard-coding
"the last state is death" would silently freeze a real state in any model whose
state space happens to end elsewhere.

Returns `nothing` if more than one state qualifies: that is a genuinely ambiguous
model, and guessing would be worse than declining.
"""
function _absorbing_state(data::EpidemicData)
    spec = data.trans_mat
    states = spec.states
    found = Int[]
    for (si, s) in enumerate(states)
        any(tr -> tr[1] === s && tr[2] !== s, spec.transitions) && continue
        push!(found, si)
    end
    length(found) == 1 ? found[1] : nothing
end

"""
    _drop_unknown!(model, data, X, t_star, known; entrants=true)

Take out of the population summaries every individual whose sampling period has
begun by `t_star` but who is not yet known then. The fit never sampled their
states, so whatever `X` holds for them is a placeholder. With
`entrants = false`, later entrants are taken out too.
"""
function _drop_unknown!(model, data::EpidemicData, X, t_star::Int, known;
                        entrants::Bool = true)
    for i in 1:data.n_individuals
        f_i, l_i = data.sampling_period[i]
        (f_i <= t_star < known[i] || (!entrants && f_i > t_star)) || continue
        for t in f_i:min(l_i, data.n_timepoints)
            apply_summaries!(data.derived_summaries, model, data, X, X[t, i], i, t, true)
        end
    end
    data
end

"""
    score_window(model, data, X, t_star, M, granularity;
                 cell_logdensity, is_informative=nothing,
                 n_sim=1, rng, constrain=nothing, survival_weight=nothing)
        -> (cell_lp::Dict, n_informative::Int)

Per-cell log densities for one draw over one window, averaged over `n_sim`
forward trajectories.

The prediction cohort is fixed at the cutoff: observations are scored only for
individuals known by `t_star` (`known`, by default the start of each sampling
period). Individuals entering later in the window are still simulated, because
their states may affect the forecast dynamics of that fixed cohort, unless
`entrants = false` (see [`forward_simulate`](@ref)); individuals already present
but not yet known are neither simulated nor scored.

``guide = true` uses the observation-guided proposal (see
[`forward_simulate`](@ref)) and adds its weights to each cell. Under coupling a
cell's weight alone does not correct its density -- another individual's
guided moves change its dynamics -- so for a coupled model only the `Joint()`
cell is a valid estimate; [`lfo_cv`](@ref) refuses the rest.

`-Inf` is charged to the offending individual's own cell and the loop continues.
Returning early on the first inadmissible individual would silently re-impose
joint scoring no matter which granularity was asked for: the single easiest way
to get this wrong.
"""
function score_window(model, data::EpidemicData, X, t_star::Int, M::Int,
                      g::Granularity;
                      cell_logdensity,
                      is_informative = nothing,
                      n_sim::Int = 1,
                      rng::AbstractRNG = Random.default_rng(),
                      constrain = nothing,
                      survival_weight = nothing,
                      known = nothing,
                      entrants::Bool = true,
                      guide::Bool = false)
    known === nothing && (known = first.(data.sampling_period))
    last_t = min(t_star + M, data.n_timepoints)
    reps = [Dict{Any,Float64}() for _ in 1:n_sim]
    n_inf = 0

    # Building arbitrary user summaries can cost O(NT). Do it once for this
    # posterior draw, then restore the concrete arrays before each independent
    # forward path. `forward_simulate` keeps the restored copy in sync.
    reset_aggregates!(data)
    apply_derived_summaries!(model, data, X)
    _drop_unknown!(model, data, X, t_star, known; entrants)
    aggregate_base = [copy(v) for v in values(data.aggregates)]

    for r in 1:n_sim
        for (v, base) in zip(values(data.aggregates), aggregate_base)
            copyto!(v, base)
        end
        logw = guide ? zeros(Float64, size(X)) : nothing
        Xf = forward_simulate(rng, model, data, X, t_star, M;
                              constrain, aggregates_ready = true, known, entrants,
                              guide, logw)
        acc = reps[r]
        for t in (t_star + 1):last_t, i in 1:data.n_individuals
            f_i, l_i = data.sampling_period[i]
            # The prediction cohort is fixed at the cutoff. Future entrants are
            # simulated because they may affect the existing animals, but their
            # own observations are not part of this LFO window.
            (known[i] <= t_star && f_i <= t <= min(l_i, data.n_timepoints)) || continue
            key = cell_of(g, i, t - t_star)
            lp = cell_logdensity(model, data, Xf, i, t)
            logw === nothing || (lp += logw[t, i])
            acc[key] = get(acc, key, 0.0) + lp
            if r == 1 && is_informative !== nothing && is_informative(data, i, t)
                n_inf += 1
            end
        end
        if survival_weight !== nothing
            # The weight converting the constrained proposal back to the true
            # kernel. It must cover exactly the individuals the constraint
            # covered -- see `survival_constrained`.
            for (key, w) in survival_weight(model, data, Xf, t_star, M, g; known)
                acc[key] = get(acc, key, 0.0) + w
            end
        end
    end

    # average over replicates within each cell
    out = Dict{Any,Float64}()
    allkeys = Set{Any}()
    for d in reps, k in keys(d); push!(allkeys, k); end
    buf = Vector{Float64}(undef, n_sim)
    for k in allkeys
        @inbounds for r in 1:n_sim
            buf[r] = get(reps[r], k, -Inf)
        end
        out[k] = n_sim == 1 ? buf[1] : logsumexp(buf) - log(n_sim)
    end
    (out, n_inf)
end

"""
    survival_constrained(known_present) -> (constrain, survival_weight)

A proposal that never kills an individual known to be present, plus the
importance weight that converts back to the true kernel.

`known_present(i, t) -> Bool` says individual `i` is known to be present at `t`
(e.g. it was physically observed later). Forward-simulating without this wastes
draws: measured on one study, 52% of naive trajectories killed someone the data
prove was alive.

# The rule that must not be broken

**The constraint and its weight must be gated on exactly the same condition.**
Constraining an individual without accumulating its weight deletes that
individual's death branch with nothing to correct it: and a death changes the
group's composition, hence every other individual's transition probabilities.
That is a BIAS, not a variance effect: it does not shrink with `n_sim`. Measured
against exact enumeration on a small model: **-0.474 nats at n=3, -1.14 nats at
n=8**, versus ~1e-3 once both halves were gated together.

It was invisible to earlier checks because those compared two estimators to each
other with nothing pinning either to the truth. **A-versus-B agreement is not a
correctness test**, at least one test must pin an absolute value.

This constructor returns both halves together precisely so they cannot drift.

# `known_present` must not look past the scored window

Bound it at `t_star + M`. Within the window the constraint is free: a branch in
which the individual dies is contradicted by the very capture that made
`known_present` true, so it scores zero regardless and excluding it costs
nothing. The weight then returns the true predictive exactly.

A capture AFTER the window is different. Nothing inside the block contradicts
the death branch, and at a step whose observation is "no capture" being dead
explains that perfectly, so the excluded branch carries real mass and no
reweighting restores it. Enumerated on one individual over `M = 3` with
`mu = 0.15`, `pc = 0.5` and no capture inside the window: the true density is
0.3176, constraining on a later capture gives 0.0768. That is a
different estimand, not a variance effect.

`known_present` may take the window's last timepoint as a third argument,
`known_present(i, t, last_t)`, and the scorer passes it, so the bound is simply
`t:last_t` -- whether or not windows overlap. A two-argument
`known_present(i, t)` still works, but then has to recover the window end from
`t` alone, which is only possible when windows do not overlap.
"""
function survival_constrained(known_present)
    kp = _windowed(known_present)
    constrain = (i, t, last_t) -> kp(i, t, last_t)

    function survival_weight(model, data::EpidemicData, Xf, t_star::Int, M::Int,
                             g::Granularity; known = nothing)
        known === nothing && (known = first.(data.sampling_period))
        absorbing = _absorbing_state(data)
        absorbing === nothing && return Dict{Any,Float64}()
        last_t = min(t_star + M, data.n_timepoints)
        N = data.n_states
        P = zeros(Float64, N, N)
        rowsum = zeros(Float64, N)
        w = Dict{Any,Float64}()
        for t in (t_star + 1):last_t, i in 1:data.n_individuals
            f_i, l_i = data.sampling_period[i]
            (known[i] <= t_star && f_i <= t <= min(l_i, data.n_timepoints)) || continue
            # exactly the condition `constrain` used -- co-gated by construction.
            kp(i, t, last_t) || continue
            transition_matrix_at!(P, rowsum, data.trans_mat, model, data, Xf, i, t - 1)
            p_surv = max(1.0 - Float64(P[Xf[t-1, i], absorbing]), 1e-12)
            key = cell_of(g, i, t - t_star)
            w[key] = get(w, key, 0.0) + log(p_surv)
        end
        w
    end

    (constrain, survival_weight)
end

# A constraint as a function of (i, t, last_t), whichever form it was written
# in. Decided once, not per call: this sits in the simulation's inner loop.
_windowed(f) = hasmethod(f, Tuple{Int,Int,Int}) ? f : ((i, t, last_t) -> f(i, t))
