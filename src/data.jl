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

`derived_summaries` is likewise parameterised (`DS<:Tuple`), for the same reason as
`trans_mat`'s `RF`: each summary is a distinct closure, and storing them behind an
abstract `Vector{Function}` field erases that at every read, forcing `for ds in
data.derived_summaries` (the iFFBS sweep, the simulator, and the coupling term in
`make_rest_contribution`) through runtime dispatch on every call. Confirmed via
profiling: this alone accounted for the bulk of a ~6x gap between the badger
model's iFFBS sweep and its reference-implementation counterpart, matching the
same pattern found earlier for `trans_mat` — see the devlog/repro log for both.
"""
struct EpidemicData{SS,OP,RC,EX<:NamedTuple,AG<:NamedTuple,RF<:Tuple,DS<:Tuple}
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
    derived_summaries::DS
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
- `coupling_trans_mat`: the [`TransitionSpec`](@ref) used for the COUPLING term
  only — how likely a neighbour's realised move was, summed over the focal's
  candidate states (see [`make_rest_contribution`](@ref)). Defaults to
  `trans_mat`, which is almost always what you want. Supply a different one only
  to give the coupling a cheaper equivalent of a rate — see the worked example in
  its own section below, and read the warning there first.
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
`derived_summaries` as a collection of functions
`(model, data, X, s, i, t; reverse=false)` that honour `reverse` themselves — a
`Tuple` is preferred (each summary keeps its own concrete type all the way
through, same reasoning as `TransitionSpec`'s `rate_fns`), but any iterable
works: it is converted with `Tuple(...)`.

Remember to establish the aggregates-agree-with-`X` invariant before the first
likelihood call; see [`apply_derived_summaries!`](@ref).

**A separate `coupling_trans_mat`** (power users; skip unless the coupling term is
your bottleneck). The three consumers of the rates are not equal:

| consumer | spec used | differentiated? |
|---|---|---|
| [`epidemic_loglik`](@ref) | `trans_mat` | **yes** — this is the HMC/NUTS gradient |
| [`forward_filter`](@ref) (iFFBS) | `trans_mat` | no |
| the coupling term ([`make_rest_contribution`](@ref)) | `coupling_trans_mat` | no |

The coupling term evaluates a rate `n_states × |affected_individuals|` times per
`(i, t)` — by far the most rate calls in a sweep — and it is never differentiated.
So it is the one place where a rate that CACHES a parameter-dependent quantity is
both worthwhile and safe. Give it a spec whose rates read a value your derived
summaries maintain, and leave `trans_mat` computing that value honestly:

```julia
# maintained by a derived summary, O(1) per individual, plain Float64
cached_infection(model, data, i, t) = data.aggregates.foi[data.social_group[i, t], t]

epidemic_data(;
    trans_mat          = @transitions(STATES, begin S -> E = infection end),        # honest
    coupling_trans_mat = @transitions(STATES, begin S -> E = cached_infection end), # cached
    ...)
```

!!! warning "Never give a cached rate to `trans_mat`"
    `trans_mat` is differentiated. A rate that returns a cached `Float64` computed
    from the parameters *outside* the AD call is a CONSTANT to the AD backend, so
    the gradient with respect to whatever fed that cache silently collapses to the
    prior's gradient — while the log-density stays bit-identical, so nothing warns
    you and the fit simply never moves those parameters. `coupling_trans_mat` is
    safe from this by construction: the gradient never reaches it. Aggregates that
    are pure functions of `X` (counts, say) are fine in `trans_mat` — they are
    genuinely constant with respect to the parameters. It is caching a
    *parameter-derived* quantity that breaks.

The two specs must agree mathematically; the package cannot check that for you
(and does not try), it only takes the two you hand it.
"""
function epidemic_data(; n_individuals, n_timepoints, trans_mat,
                         coupling_trans_mat=trans_mat,
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
    # The coupling term is the only rate consumer that is never differentiated, so
    # it is the only one that may safely use a spec with cached, parameter-derived
    # rates. Defaults to `trans_mat` — see this function's docstring.
    neighbor_logprob = make_neighbor_logprob_from_transitions(coupling_trans_mat)
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
        Tuple(derived_summaries),
        rest_contribution,
        affected_individuals,
        NamedTuple(extras),
        aggregates isa NamedTuple ? aggregates : NamedTuple(aggregates),
    )
end
