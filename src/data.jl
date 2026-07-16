# The `data` object: everything that isn't the parameters (`model`) or the latent
# trajectory (`X`) — the fixed structure, the observations, the user's derived
# summaries, and the user's aggregates container.

"""
    EpidemicData

Holds a model's fixed structure and the user's tracked state. Built by
[`epidemic_data`](@ref). The three-part vocabulary is `model` (parameters), `X`
(the latent trajectory, indexed `X[t, i]`), and `data` (this object).

Fields:
- `n_individuals`, `n_timepoints`, `n_states`
- `state_space` — state names; state `k` in `X` means `state_space[k]`
- `group`, `members_by_group` — group membership
- `sampling_period` — `(first, last)` timepoint per individual
- `trans_mat` — the [`TransitionSpec`](@ref)
- `starting_state` — `(model, data, X, i, t) -> probability vector` at `t = 1`
- `derived_summaries` — the user's reversible aggregate updates
- `rest_contribution` — the coupling term (see [`make_rest_contribution`](@ref))
- `affected_individuals` — who each individual's state at each time affects
- `test_mats` — observation matrices, indexed `[t, i]` (`-1` = not observed)
- `aggregates` — the user's arrays. The package attaches no meaning to these.
"""
mutable struct EpidemicData
    n_individuals::Int
    n_timepoints::Int
    n_states::Int
    state_space::Vector{Symbol}
    group::Vector{Int}
    members_by_group::Dict{Int,Vector{Int}}
    sampling_period::Vector{Tuple{Int,Int}}
    trans_mat::TransitionSpec
    starting_state::Function
    derived_summaries::Vector{Function}
    rest_contribution::Function
    affected_individuals::Union{Nothing,Matrix{Vector{Int}}}
    test_mats::Vector{Matrix{Int}}
    aggregates::Dict{Symbol,Any}
end

"""
    members(data, g)

The individuals in group `g`.
"""
members(data::EpidemicData, g) = data.members_by_group[g]

@inline function _state_index(data::EpidemicData, s::Symbol)
    idx = findfirst(==(s), data.state_space)
    idx === nothing && error("State $s not found in state_space")
    idx
end

"""
    build_affected_individuals_from_groups(group, n_timepoints; include_self=false)

Who each individual's state affects at each time, derived from group membership:
individual `i` at time `t` affects its groupmates. Returns a
`n_timepoints × n_individuals` matrix of index vectors.

This is the default for a group-structured model. For a network or spatial model,
build the equivalent structure yourself and pass it as `affected_individuals`.
"""
function build_affected_individuals_from_groups(group::Vector{Int}, n_timepoints::Int; include_self=false)
    n_individuals = length(group)
    out = Matrix{Vector{Int}}(undef, n_timepoints, n_individuals)
    for t in 1:n_timepoints
        for i in 1:n_individuals
            out[t, i] = [j for j in 1:n_individuals if group[j] == group[i] && (include_self || j != i)]
        end
    end
    out
end

"""
    epidemic_data(; n_individuals, n_timepoints, group, trans_mat, starting_state,
                    test_mats, aggregates,
                    state_space=trans_mat.states, derived_summaries=nothing)

Build the [`EpidemicData`](@ref) for a model.

- `trans_mat`: a [`TransitionSpec`](@ref) from [`@transitions`](@ref).
- `starting_state`: `(model, data, X, i, t) -> probability vector` over states at
  `t = 1`.
- `test_mats`: observation matrices indexed `[t, i]`, `-1` meaning not observed.
- `aggregates`: normally an [`@aggregate`](@ref) declaration — the package
  allocates the arrays and takes the reversible updates from it.
- `state_space`: state names, in the order that fixes their encoding in `X`.
  Defaults to the transitions' own state list.
- `derived_summaries`: only needed with the verbose fallback (see below).

The package attaches no meaning to the aggregates — it only allocates them and
passes them to your own summaries and rate functions.

**Verbose fallback** (for an aggregate `@aggregate` cannot express): pass
`aggregates` as a plain `Dict{Symbol,Any}` of your own arrays and
`derived_summaries` as a vector of functions
`(model, data, X, s, i, t; reverse=false)` that honour `reverse` themselves.

Remember to establish the aggregates-agree-with-`X` invariant before the first
likelihood call; see [`apply_derived_summaries!`](@ref).
"""
function epidemic_data(; n_individuals, n_timepoints, group, trans_mat,
                         starting_state, test_mats, aggregates,
                         state_space=trans_mat.states, derived_summaries=nothing)
    # An @aggregate declaration carries both the storage to allocate and the
    # updates; the verbose fallback supplies a Dict plus explicit summaries.
    if aggregates isa AggregateDeclaration
        derived_summaries === nothing ||
            error("pass `derived_summaries` only with the verbose fallback; an @aggregate " *
                  "declaration already carries its own updates")
        derived_summaries = aggregates.summaries
        aggregates = allocate_aggregates(aggregates)
    else
        derived_summaries === nothing &&
            error("`derived_summaries` is required when `aggregates` is a plain Dict " *
                  "(the verbose fallback); use an @aggregate declaration to avoid this")
    end
    return _epidemic_data(n_individuals, n_timepoints, group, state_space, trans_mat,
                          starting_state, test_mats, derived_summaries, aggregates)
end

function _epidemic_data(n_individuals, n_timepoints, group, state_space, trans_mat,
                        starting_state, test_mats, derived_summaries, aggregates)
    n_groups = maximum(group)
    members_by_group = Dict(g => findall(==(g), group) for g in 1:n_groups)
    sampling_period = [(1, n_timepoints) for _ in 1:n_individuals]

    affected_individuals = build_affected_individuals_from_groups(collect(group), n_timepoints)
    affected_ids = (data, t, i) -> affected_individuals[t, i]
    neighbor_logprob = make_neighbor_logprob_from_transitions(trans_mat)
    rest_contribution = make_rest_contribution(affected_ids=affected_ids, neighbor_logprob=neighbor_logprob)

    EpidemicData(
        n_individuals,
        n_timepoints,
        length(state_space),
        collect(state_space),
        collect(group),
        members_by_group,
        sampling_period,
        trans_mat,
        starting_state,
        Vector{Function}(collect(derived_summaries)),
        rest_contribution,
        affected_individuals,
        test_mats,
        Dict{Symbol,Any}(aggregates),
    )
end
