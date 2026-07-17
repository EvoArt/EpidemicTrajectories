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
- `group`, `members_by_group` — group membership, for the fixed-group convenience
  path. Nothing in the package requires groups to be fixed, or to exist at all:
  the coupling structure is `affected_individuals`, and a model with time-varying
  membership indexes its own structure off `data` however it likes.
- `sampling_period` — `(first, last)` timepoint per individual
- `trans_mat` — the [`TransitionSpec`](@ref)
- `starting_state` — `(model, data, X, i, t) -> probability vector` at the
  individual's first timepoint
- `observation_process` — `(model, data, X, i, t) -> per-state weight vector`
- `derived_summaries` — the user's reversible aggregate updates
- `rest_contribution` — the coupling term (see [`make_rest_contribution`](@ref))
- `affected_individuals` — who each individual's state affects, indexed `[t, i]`,
  so it may vary over time
- `extras` — anything else the user's own functions need (covariates, test
  matrices, capture histories, ...). The package never looks inside.
- `aggregates` — the user's arrays. The package attaches no meaning to these.

`extras` and `aggregates` are `NamedTuple`s, and `EpidemicData` is parameterised
on their types. This matters for speed rather than style: a rate function reads
`data.age[i, t]` on every one of millions of evaluations, and a `Dict{Symbol,Any}`
would return `Any` — so the arithmetic on top of it would dispatch at runtime and
box, which measured ~18x slower with ~500k allocations per 20k evaluations. With a
NamedTuple the element type is known, and the same code allocates nothing. It costs
the user nothing: they still put whatever they like in, and the package still never
looks inside.
"""
struct EpidemicData{SS,OP,RC,EX<:NamedTuple,AG<:NamedTuple,RF<:Tuple}
    n_individuals::Int
    n_timepoints::Int
    n_states::Int
    state_space::Vector{Symbol}
    group::Vector{Int}
    members_by_group::Dict{Int,Vector{Int}}
    sampling_period::Vector{Tuple{Int,Int}}
    trans_mat::TransitionSpec{RF}
    starting_state::SS
    observation_process::OP
    derived_summaries::Vector{Function}
    rest_contribution::RC
    affected_individuals::Union{Nothing,Matrix{Vector{Int}}}
    extras::EX
    aggregates::AG
end

# `data.whatever` reaches into `extras`, so a user's own functions can read their
# covariates/test matrices/capture histories off `data` by name, without the
# package knowing any of them exist.
#
# The `Val(s)` dispatch is not decoration. The obvious spelling —
# `s in fieldnames(EpidemicData) ? getfield(d, s) : ...` — is a runtime search over
# a tuple of symbols on EVERY property access, and it allocates: measured 620x
# slower than this version (11.7 ms / 20k allocations vs 18.9 µs / 0) on nothing
# but repeated `data.age[i, t]`. Dispatching on `Val` decides the branch at compile
# time, so the whole accessor disappears.
#
# Note that inference alone does not catch this: both spellings report
# `data.age::Matrix{Int64}`. Only benchmarking does.
@inline Base.getproperty(d::EpidemicData, s::Symbol) = _getfield_or_extra(d, Val(s))

# One method per real field, so each resolves at compile time...
for f in fieldnames(EpidemicData)
    @eval @inline _getfield_or_extra(d::EpidemicData, ::Val{$(QuoteNode(f))}) = getfield(d, $(QuoteNode(f)))
end
# ...and anything else is an extra.
@inline function _getfield_or_extra(d::EpidemicData, ::Val{s}) where {s}
    extras = getfield(d, :extras)
    hasproperty(extras, s) ||
        throw(ArgumentError("EpidemicData has no field or extra `$s`; " *
                            "pass it to `epidemic_data` as `$s = ...`"))
    return getproperty(extras, s)
end

Base.propertynames(d::EpidemicData) = (fieldnames(EpidemicData)..., propertynames(getfield(d, :extras))...)

"""
    no_observations(model, data, X, i, t)

The default observation process: every state equally likely, i.e. no observations
at all. Supply your own `observation_process` to `epidemic_data` for a model with
data — the package has no idea what you observe or how it relates to the states.
"""
no_observations(model, data, X, i, t) = ones(Float64, data.n_states)

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
    epidemic_data(; n_individuals, n_timepoints, trans_mat, starting_state, aggregates,
                    group=ones(Int, n_individuals),
                    observation_process=no_observations,
                    sampling_period=nothing,
                    affected_individuals=nothing,
                    state_space=trans_mat.states,
                    derived_summaries=nothing,
                    extras...)

Build the [`EpidemicData`](@ref) for a model.

- `trans_mat`: a [`TransitionSpec`](@ref) from [`@transitions`](@ref).
- `starting_state`: `(model, data, X, i, t) -> probability vector` over states at
  the individual's first timepoint.
- `aggregates`: normally an [`@aggregate`](@ref) declaration — the package
  allocates the arrays and takes the reversible updates from it.
- `observation_process`: `(model, data, X, i, t) -> per-state weight vector`, the
  likelihood of individual `i`'s observations at `t` under each state. Defaults
  to [`no_observations`](@ref). The package has no idea what you observe — supply
  this for any model with data.
- `sampling_period`: `(first, last)` timepoint per individual. Defaults to
  `(1, n_timepoints)` for everyone.
- `affected_individuals`: who each individual's state affects, indexed `[t, i]`,
  so it may vary over time. Defaults to groupmates under a fixed `group` (see
  [`build_affected_individuals_from_groups`](@ref)). Pass your own for a network,
  spatial, or time-varying-membership model.
- `coupled_transitions`: WHICH of a neighbour's transitions this individual can
  influence, e.g. `[(:S, :E)]` when the only effect one individual has on another
  is contributing to its force of infection. Purely an optimisation, and often a
  large one: a neighbour whose realised move is not in this list has the same
  probability whatever the focal does, so it can be skipped exactly. Defaults to
  `nothing`, meaning "assume every transition is coupled" — correct, but it makes
  the sampler do the full work. See [`coupled_transition_mask`](@ref).
- `group`: group index per individual, used only by the default
  `affected_individuals` and by whatever your own functions read off it. Ignore it
  entirely if your model has no fixed groups.
- `state_space`: state names, in the order that fixes their encoding in `X`.
  Defaults to the transitions' own state list.
- `derived_summaries`: only needed with the verbose fallback (see below).
- `extras...`: anything else your functions need — covariates, test matrices,
  capture histories, time-varying group membership. Reachable as `data.name`. The
  package never looks inside.

**Verbose fallback** (for an aggregate `@aggregate` cannot express): pass
`aggregates` as a plain `Dict{Symbol,Any}` of your own arrays and
`derived_summaries` as a vector of functions
`(model, data, X, s, i, t; reverse=false)` that honour `reverse` themselves.

Remember to establish the aggregates-agree-with-`X` invariant before the first
likelihood call; see [`apply_derived_summaries!`](@ref).
"""
function epidemic_data(; n_individuals, n_timepoints, trans_mat,
                         starting_state, aggregates,
                         group=ones(Int, n_individuals),
                         observation_process=no_observations,
                         sampling_period=nothing,
                         affected_individuals=nothing,
                         coupled_transitions=nothing,
                         state_space=trans_mat.states, derived_summaries=nothing,
                         extras...)
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
    group = collect(Int, group)
    n_groups = maximum(group)
    members_by_group = Dict(g => findall(==(g), group) for g in 1:n_groups)

    # Every individual is sampled over the whole window unless told otherwise.
    if sampling_period === nothing
        sampling_period = [(1, n_timepoints) for _ in 1:n_individuals]
    else
        sampling_period = [(Int(first(p)), Int(last(p))) for p in sampling_period]
        length(sampling_period) == n_individuals ||
            throw(ArgumentError("`sampling_period` needs one (first, last) per individual"))
    end

    # The coupling structure is indexed [t, i], so it may vary over time. The
    # fixed-group default is a convenience, not an assumption: pass your own for a
    # network, spatial, or time-varying-membership model.
    if affected_individuals === nothing
        affected_individuals = build_affected_individuals_from_groups(group, n_timepoints)
    else
        size(affected_individuals) == (n_timepoints, n_individuals) ||
            throw(ArgumentError("`affected_individuals` must be a (n_timepoints, n_individuals) " *
                                "matrix of index vectors, got size $(size(affected_individuals))"))
    end

    # Which of a neighbour's moves this individual can influence. Declaring it is
    # a pure optimisation — see `coupled_transition_mask`.
    coupled_mask = coupled_transitions === nothing ? nothing :
        coupled_transition_mask(state_space, coupled_transitions)

    affected_ids = (data, t, i) -> affected_individuals[t, i]
    neighbor_logprob = make_neighbor_logprob_from_transitions(trans_mat)
    rest_contribution = make_rest_contribution(affected_ids=affected_ids,
                                               neighbor_logprob=neighbor_logprob,
                                               coupled_mask=coupled_mask)

    EpidemicData(
        n_individuals,
        n_timepoints,
        length(state_space),
        collect(state_space),
        group,
        members_by_group,
        sampling_period,
        trans_mat,
        starting_state,
        observation_process,
        Vector{Function}(collect(derived_summaries)),
        rest_contribution,
        affected_individuals,
        NamedTuple(extras),
        aggregates isa NamedTuple ? aggregates : NamedTuple(aggregates),
    )
end
