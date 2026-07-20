# Assembling a per-individual transition matrix from the spec, and the coupling
# term ("rest contribution") that accounts for how one individual's state affects
# the others.

"""
    transition_matrix_at(trans_mat, model, data, X, i, t) -> Matrix

The transition matrix for individual `i` from time `t` to `t+1`: `P[a, b]` is the
probability of moving from state `a` to state `b`. Each declared transition's rate
is evaluated and placed at its `(from, to)` entry; the leftover mass in each row
fills that state's self-transition.

Rates are clamped strictly inside `(0, 1)`: a sampler exploring parameter space
can momentarily propose values that would otherwise send `log(P[a,b])` in the
likelihood to `log(0)`, crashing the chain. This regularizes only the tails.

The element type follows the parameters, so this differentiates cleanly under AD.

Allocates a fresh `N × N` matrix (and an `N`-length scratch vector); a caller that
needs many of these in a tight loop — the iFFBS forward filter is the package's own
example — should use [`transition_matrix_at!`](@ref) instead to reuse buffers
across calls.
"""
function transition_matrix_at(trans_mat::TransitionSpec, model, data::EpidemicData, X, i, t)
    N = data.n_states
    T = _param_eltype(model)
    P = zeros(T, N, N)
    rowsum = zeros(T, N)
    transition_matrix_at!(P, rowsum, trans_mat, model, data, X, i, t)
end

"""
    transition_matrix_at!(P, rowsum, trans_mat, model, data, X, i, t) -> P

In-place [`transition_matrix_at`](@ref): fills the caller-provided `N × N` matrix
`P` and `N`-length scratch vector `rowsum` (both overwritten completely, so neither
needs zeroing first) instead of allocating new ones. `rowsum` is pure scratch —
nothing outside this call reads it — so one buffer may be reused across every
`(i, t)` in a sweep; `P` is the whole return value and must NOT be reused before
the caller is done with it (the iFFBS forward filter, which needs every
timepoint's matrix alive for the backward pass, allocates one `P` per timepoint
but shares one `rowsum` across all of them — see `forward_filter`).
"""
function transition_matrix_at!(P, rowsum, trans_mat::TransitionSpec, model, data::EpidemicData, X, i, t)
    N = data.n_states
    fill!(P, zero(eltype(P)))
    fill!(rowsum, zero(eltype(rowsum)))

    # `rate_fns` is a Tuple of DIFFERENT concrete function types, so iterating it
    # with a plain loop would infer the element as a union/Any and dispatch on
    # every call — this is the hub of the package, so that cost lands everywhere.
    # `_fill_rates!` recurses over the tuple instead, specialising on one rate at a
    # time, which keeps each call concrete.
    _fill_rates!(P, rowsum, trans_mat.rate_fns, trans_mat.transitions, 1, model, data, i, t)

    for a in 1:N
        P[a, a] += (1 - rowsum[a])
    end
    P
end

"""
    transition_prob(trans_mat, model, data, X, i, t, from, to) -> probability

The probability of one specific transition `from -> to` for individual `i` at time
`t` — the single entry of [`transition_matrix_at`](@ref)'s matrix, without
building the matrix.

This exists because the likelihood only ever wants one entry per `(i, t)`: the
move the individual actually made. Going through the full matrix there allocates a
fresh `n_states × n_states` array and evaluates every rate, for every individual at
every timepoint — on the badger model that is ~380k matrix allocations per
likelihood call, each of Dual numbers under AD. The sampler still uses the full
matrix, because it genuinely needs every entry.

Respects `auto_self`: a self-transition takes whatever probability mass the
declared transitions out of that state leave behind.
"""
function transition_prob(trans_mat::TransitionSpec, model, data::EpidemicData, X, i, t, from::Int, to::Int)
    T = _param_eltype(model)
    p_to = zero(T)          # mass on the requested move
    rowsum = zero(T)        # total mass leaving `from` via declared transitions
    _accum_row(trans_mat.rate_fns, trans_mat.transitions, 1, from, to, model, data, i, t,
               p_to, rowsum, T)
end

# Walk the rate tuple once, accumulating only what the `from` row needs: the
# requested entry, and the row's total (for the self-transition's leftover).
# Recursive for the same reason as `_fill_rates!` — one concrete rate type per step.
@inline function _accum_row(::Tuple{}, transitions, k, from, to, model, data, i, t, p_to, rowsum, ::Type{T}) where {T}
    # A self-transition takes the mass the declared transitions leave behind.
    return from == to ? p_to + (one(T) - rowsum) : p_to
end
@inline function _accum_row(rates::Tuple, transitions, k, from, to, model, data, i, t, p_to, rowsum, ::Type{T}) where {T}
    @inbounds f, s = transitions[k]
    if _state_index(data, f) == from
        p = clamp(first(rates)(model, data, i, t), 1e-12, 1 - 1e-12)
        rowsum += p
        _state_index(data, s) == to && (p_to += p)
    end
    return _accum_row(Base.tail(rates), transitions, k + 1, from, to, model, data, i, t,
                      p_to, rowsum, T)
end

# Recurse over the rate tuple one element at a time. Each step sees a concrete
# function type, so the call devirtualises and the arithmetic stays unboxed.
@inline _fill_rates!(P, rowsum, ::Tuple{}, transitions, k, model, data, i, t) = nothing
@inline function _fill_rates!(P, rowsum, rates::Tuple, transitions, k, model, data, i, t)
    @inbounds begin
        from_sym, to_sym = transitions[k]
        a = _state_index(data, from_sym)
        b = _state_index(data, to_sym)
        p = clamp(first(rates)(model, data, i, t), 1e-12, 1 - 1e-12)
        P[a, b] += p
        rowsum[a] += p
    end
    return _fill_rates!(P, rowsum, Base.tail(rates), transitions, k + 1, model, data, i, t)
end

# The working number type: whatever number type the parameters are made of. Under
# AD this is the backend's dual/tracked type, so the transition matrix follows
# automatically and gradients flow.
#
# A parameter set mixes scalars and containers — `(; beta=0.1, alpha=[...])` — so
# this reaches through arrays to the numbers inside rather than promoting the
# container types (which would land on `Any`, and `zeros(Any, ...)` fails).
# Non-numeric entries are ignored: a model may carry an integer index or a flag
# that has nothing to do with the working precision.
_number_type(x::Number) = typeof(x)
_number_type(x::AbstractArray) = eltype(x)
_number_type(::Any) = Union{}

function _param_eltype(model)
    T = mapreduce(_number_type, promote_type, Tuple(values(model)); init=Union{})
    # `Union{}` means nothing numeric was found; `Bool` promotes badly for
    # arithmetic. Fall back to Float64 in both cases.
    return (T === Union{} || T === Bool) ? Float64 : T
end

@inline function _sample_categorical(rng, probs)
    u = rand(rng)
    c = 0.0
    @inbounds for k in eachindex(probs)
        c += probs[k]
        if u <= c
            return k
        end
    end
    return lastindex(probs)
end

"""
    no_rest_contribution(model, data, X, i, t, n_states)

A coupling term for models where an individual's state does not enter anyone
else's transition rates: contributes nothing (all ones).
"""
no_rest_contribution(model, data, X, i, t, n_states) = ones(n_states)

"""
    make_neighbor_logprob_from_transitions(trans_mat; eps_prob=1e-12)

Build the "how likely was this neighbour's realized move" term used by the
coupling: for individual `j` at time `t`, the log-probability of the transition
`j` actually makes under the current transition matrix.

Uses [`transition_prob`](@ref), not [`transition_matrix_at`](@ref): only one
entry — the move `j` actually made — is ever read here, and this is called once
per affected individual per candidate state inside [`make_rest_contribution`](@ref)
(`n_states × |affected|` times per `(i, t)` in the iFFBS forward filter). Building
the whole matrix just to read one entry was, before this, the dominant cost of a
badger-model iFFBS sweep — confirmed by direct per-phase timing, not just
profiler self-time percentages (see the repro log): the earlier fixes to
`forward_filter`'s OWN matrix build (buffer reuse) and to `derived_summaries`'
type stability were both real but left this call, one level further into the
coupling term, still allocating a fresh matrix on every one of the
`n_states × |affected|` calls this makes per timepoint.
"""
function make_neighbor_logprob_from_transitions(trans_mat::TransitionSpec; eps_prob=1e-12)
    return function (model, data::EpidemicData, X, j, t, updated_id)
        from_state = X[t, j]
        to_state = X[t + 1, j]
        log(max(transition_prob(trans_mat, model, data, X, j, t, from_state, to_state), eps_prob))
    end
end

"""
    coupled_transition_mask(state_space, coupled_transitions) -> Matrix{Bool}

Which of a neighbour's moves the focal individual can influence, as an
`n_states × n_states` mask over `(from, to)` pairs.

`coupled_transitions` names the transitions whose RATE depends on the focal — e.g.
`[(:S, :E)]` for a model where the only effect one individual has on another is
contributing to its force of infection. Names or indices both work.

The mask returned is **wider than what you pass**: it marks every transition out of
any state that has a coupled transition out of it. That is not a safety margin, it
is required. Probabilities out of a state sum to one, so if the focal raises a
neighbour's `S -> E`, it necessarily lowers that neighbour's `S -> S` (and any
other `S -> *`) by the same amount. Those moves are coupled too, even though their
own rate functions never look at the focal. Verified empirically: a neighbour's
`S -> S` log-probability really does move with the focal's state, and masking it
out changes the sampler's weights by up to 0.25.

So declaring `[(:S, :E)]` in the badger model buys the skip only for neighbours
currently in `E`, `I` or `D` — every `S -> *` move stays in. That is still most of
the population once mortality bites, and it is exact.

The saving is exact, not an approximation: a neighbour whose realised move is
outside the mask has the same probability whatever the focal does, so it
contributes an identical constant to every candidate and cancels when the weights
are normalised.
"""
function coupled_transition_mask(state_space, coupled_transitions)
    n = length(state_space)
    idx(s::Integer) = Int(s)
    function idx(s::Symbol)
        k = findfirst(==(s), state_space)
        k === nothing && error("State :$s is not in the state space $(collect(state_space))")
        k
    end

    # Any state with a coupled transition out of it has ALL of its outgoing
    # transitions coupled: the rates out of a state must sum to one, so changing
    # one changes the others. Missing this makes the mask silently wrong.
    coupled_sources = Set{Int}(idx(from) for (from, _) in coupled_transitions)
    mask = falses(n, n)
    for a in coupled_sources, b in 1:n
        mask[a, b] = true
    end
    return Matrix(mask)
end

"""
    make_rest_contribution(; affected_ids, neighbor_logprob, normalize=true,
                             min_logprob=-1e12)

Build the coupling term for the latent sampler.

Because individual `i`'s state at time `t` feeds into the transition rates of the
individuals it affects, their realized moves carry information about `i`'s state.
For each candidate state of `i`, this evaluates how probable the affected
individuals' actual transitions would be if `i` were in that state, and weights
accordingly. Without this term the sampler targets the wrong conditional.

The candidate state is applied by running the user's derived summaries forward and
then reversing them — so the aggregates the neighbours' rates read reflect the
hypothesis, and are restored exactly afterwards. This is why the aggregate updates
must be reversible.

- `affected_ids`: `(data, t, i) -> individuals affected by i at time t`.
- `neighbor_logprob`: `(model, data, X, j, t, updated_id) -> log-probability of j's
  realized move` — see [`make_neighbor_logprob_from_transitions`](@ref).
"""
function make_rest_contribution(; normalize=true, min_logprob=-1e12, affected_ids,
                                  neighbor_logprob, coupled_mask=nothing)
    function rest_contribution(model, data::EpidemicData, X, i, t, n_states, affected_override=nothing)
        # At the final timepoint there is no t -> t+1 move to be informative about.
        t == data.n_timepoints && return ones(n_states)
        ids = affected_override === nothing ? affected_ids(data, t, i) : affected_override

        current_state = X[t, i]
        logw = zeros(Float64, n_states)

        for s in 1:n_states
            X[t, i] = s
            apply_summaries!(data.derived_summaries, model, data, X, s, i, t, false)

            acc = 0.0
            for j in ids
                # A neighbour whose realised move is not one the focal can
                # influence has the same probability under every candidate state,
                # so it contributes an identical constant that cancels on
                # normalisation. Skipping it is exact, not an approximation — and
                # it saves building that neighbour's whole transition matrix.
                if coupled_mask !== nothing
                    @inbounds coupled_mask[X[t, j], X[t + 1, j]] || continue
                end
                acc += max(neighbor_logprob(model, data, X, j, t, i), min_logprob)
            end

            apply_summaries!(data.derived_summaries, model, data, X, s, i, t, true)
            logw[s] = acc
        end

        X[t, i] = current_state
        if normalize
            logw .-= maximum(logw)
        end
        exp.(logw)
    end

    rest_contribution
end
