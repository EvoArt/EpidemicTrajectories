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
"""
function transition_matrix_at(trans_mat::TransitionSpec, model, data::EpidemicData, X, i, t)
    N = data.n_states
    T = _param_eltype(model)
    P = zeros(T, N, N)
    rowsum = zeros(T, N)

    for (k, (from_sym, to_sym)) in enumerate(trans_mat.transitions)
        a = _state_index(data, from_sym)
        b = _state_index(data, to_sym)
        p = trans_mat.rate_fns[k](model, data, i, t)
        p = clamp(p, 1e-12, 1 - 1e-12)
        P[a, b] += p
        rowsum[a] += p
    end

    for a in 1:N
        P[a, a] += (1 - rowsum[a])
    end
    P
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
"""
function make_neighbor_logprob_from_transitions(trans_mat::TransitionSpec; eps_prob=1e-12)
    return function (model, data::EpidemicData, X, j, t, updated_id)
        Pj = transition_matrix_at(trans_mat, model, data, X, j, t)
        from_state = X[t, j]
        to_state = X[t + 1, j]
        log(max(Pj[from_state, to_state], eps_prob))
    end
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
function make_rest_contribution(; normalize=true, min_logprob=-1e12, affected_ids, neighbor_logprob)
    function rest_contribution(model, data::EpidemicData, X, i, t, n_states, affected_override=nothing)
        # At the final timepoint there is no t -> t+1 move to be informative about.
        t == data.n_timepoints && return ones(n_states)
        ids = affected_override === nothing ? affected_ids(data, t, i) : affected_override

        current_state = X[t, i]
        logw = zeros(Float64, n_states)

        for s in 1:n_states
            X[t, i] = s
            for ds in data.derived_summaries
                ds(model, data, X, s, i, t)
            end

            acc = 0.0
            for j in ids
                acc += max(neighbor_logprob(model, data, X, j, t, i), min_logprob)
            end

            for ds in data.derived_summaries
                ds(model, data, X, s, i, t; reverse=true)
            end
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
