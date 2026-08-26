# Scoring one leave-future-out window: forward-simulate the population from the
# cutoff and score the observations that follow, cell by cell.
#
# WHAT THE USER SUPPLIES. Two functions, in keeping with the package's central
# rule that it never assumes what the user's arrays mean:
#
#   cell_logdensity(model, data, X, i, t) -> Float64
#       the log density of individual `i`'s observation at time `t`, given the
#       (simulated) state in `X`. `-Inf` means "this trajectory is inadmissible
#       for this individual" and is charged to that individual's cell ONLY.
#
#   is_informative(data, i, t) -> Bool  (optional)
#       whether that cell's density depends on the latent state at all. On the
#       badger data only ~2% of cells did -- the rest contributed a constant
#       identical under every granularity -- which is why all three arms agreed
#       to within a few nats. Reported so the question is asked BEFORE a margin
#       is interpreted, not after.

"""
    forward_simulate(rng, model, data, X, t_star, M; constrain=nothing) -> Matrix

One forward trajectory for the WHOLE population over `t_star+1 : t_star+M`,
starting from the states in `X` at `t_star`.

The population is simulated jointly, not individual by individual: each step's
transition probabilities are computed from that trajectory's own composition, so
the coupling between individuals is preserved. Simulating one individual at a
time against a frozen background would silently drop it.

Returns a copy; `X` is not modified.

`constrain` is an optional `(i, t) -> Bool` saying "individual `i` is known to be
present at `t`, so forbid the absorbing state". See [`survival_constrained`](@ref)
for why that needs a matching weight and what happens if the two disagree.
"""
function forward_simulate(rng::AbstractRNG, model, data::EpidemicData, X,
                          t_star::Int, M::Int; constrain = nothing)
    Xf = copy(X)
    last_t = min(t_star + M, data.n_timepoints)
    N = data.n_states
    P = zeros(Float64, N, N)
    rowsum = zeros(Float64, N)
    row = zeros(Float64, N)
    absorbing = _absorbing_state(data)

    for t in (t_star + 1):last_t
        prev = copy(view(Xf, t - 1, :))
        for i in 1:data.n_individuals
            f_i, l_i = data.sampling_period[i]
            (f_i <= t <= min(l_i, data.n_timepoints)) || continue
            # An absorbing state stays absorbing: it must not be re-randomised,
            # and the constraint below must not resurrect it.
            if absorbing !== nothing && prev[i] == absorbing
                Xf[t, i] = absorbing
                continue
            end
            transition_matrix_at!(P, rowsum, data.trans_mat, model, data, Xf, i, t - 1)
            @inbounds for s in 1:N
                row[s] = Float64(P[prev[i], s])
            end
            if constrain !== nothing && absorbing !== nothing && constrain(i, t)
                p_surv = 1.0 - row[absorbing]
                p_surv = max(p_surv, 1e-12)
                @inbounds for s in 1:N
                    row[s] = s == absorbing ? 0.0 : row[s] / p_surv
                end
            end
            u = rand(rng); c = 0.0; k = N
            @inbounds for s in 1:N
                c += row[s]
                if u <= c; k = s; break; end
            end
            Xf[t, i] = k
        end
    end
    Xf
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
    score_window(model, data, X, t_star, M, granularity;
                 cell_logdensity, is_informative=nothing,
                 n_sim=1, rng, constrain=nothing, survival_weight=nothing)
        -> (cell_lp::Dict, n_informative::Int)

Per-cell log densities for one draw over one window, averaged over `n_sim`
forward trajectories.

`-Inf` is charged to the offending individual's OWN cell and the loop continues.
Returning early on the first inadmissible individual would silently re-impose
joint scoring no matter which granularity was asked for — the single easiest way
to get this wrong.
"""
function score_window(model, data::EpidemicData, X, t_star::Int, M::Int,
                      g::Granularity;
                      cell_logdensity,
                      is_informative = nothing,
                      n_sim::Int = 1,
                      rng::AbstractRNG = Random.default_rng(),
                      constrain = nothing,
                      survival_weight = nothing)
    last_t = min(t_star + M, data.n_timepoints)
    reps = [Dict{Any,Float64}() for _ in 1:n_sim]
    n_inf = 0

    for r in 1:n_sim
        Xf = forward_simulate(rng, model, data, X, t_star, M; constrain)
        acc = reps[r]
        for t in (t_star + 1):last_t, i in 1:data.n_individuals
            f_i, l_i = data.sampling_period[i]
            (f_i <= t <= min(l_i, data.n_timepoints)) || continue
            key = cell_of(g, i, t - t_star)
            lp = cell_logdensity(model, data, Xf, i, t)
            acc[key] = get(acc, key, 0.0) + lp
            if r == 1 && is_informative !== nothing && is_informative(data, i, t)
                n_inf += 1
            end
        end
        if survival_weight !== nothing
            # Eq-14: the weight converting the constrained proposal back to the
            # true kernel. It MUST cover exactly the individuals the constraint
            # covered -- see `survival_constrained`.
            for (key, w) in survival_weight(model, data, Xf, t_star, M, g)
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
individual's death branch with nothing to correct it — and a death changes the
group's composition, hence every other individual's transition probabilities.
That is a BIAS, not a variance effect: it does not shrink with `n_sim`. Measured
against exact enumeration on a small model: **-0.474 nats at n=3, -1.14 nats at
n=8**, versus ~1e-3 once both halves were gated together.

It was invisible to earlier checks because those compared two estimators to each
other with nothing pinning either to the truth. **A-versus-B agreement is not a
correctness test** — at least one test must pin an absolute value.

This constructor returns both halves together precisely so they cannot drift.

# What this targets

Not the unconditional predictive `p(y | y_{1:t})`, but the conditional
`p(y | y_{1:t}, present through t_c)`. At a step where an individual was not
observed but is known present later, being absent explains a non-observation
perfectly, so the excluded branch carries real mass that the weight does not
restore. Conditioning on it is usually what you want — `t_c` is observed data —
but any rival estimator must be made to condition identically or the two are not
estimating the same thing.
"""
function survival_constrained(known_present)
    constrain = (i, t) -> known_present(i, t)

    function survival_weight(model, data::EpidemicData, Xf, t_star::Int, M::Int,
                             g::Granularity)
        absorbing = _absorbing_state(data)
        absorbing === nothing && return Dict{Any,Float64}()
        last_t = min(t_star + M, data.n_timepoints)
        N = data.n_states
        P = zeros(Float64, N, N)
        rowsum = zeros(Float64, N)
        w = Dict{Any,Float64}()
        for t in (t_star + 1):last_t, i in 1:data.n_individuals
            f_i, l_i = data.sampling_period[i]
            (f_i <= t <= min(l_i, data.n_timepoints)) || continue
            # EXACTLY the condition `constrain` used -- co-gated by construction.
            known_present(i, t) || continue
            transition_matrix_at!(P, rowsum, data.trans_mat, model, data, Xf, i, t - 1)
            p_surv = max(1.0 - Float64(P[Xf[t-1, i], absorbing]), 1e-12)
            key = cell_of(g, i, t - t_star)
            w[key] = get(w, key, 0.0) + log(p_surv)
        end
        w
    end

    (constrain, survival_weight)
end
