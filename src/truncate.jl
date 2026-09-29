# Truncating an `EpidemicData` to a training window, for leave-future-out CV.
#
# Shortening `sampling_period` is not sufficient. Any extra the user attached
# may carry information from beyond the cutoff, and the package cannot know
# which ones do. Two real leaks, both measured on the badger model and both
# invisible until looked for: a `last_capture_time` death-ban still banning
# death past the cutoff, and `captures_after_monit` feeding post-cutoff
# survival evidence into the parameters the comparison is about.
#
# So the user declares what to do with each extra, and one that looks
# time-indexed but was never declared is an error. A silent leak inflates the
# score of whichever model exploits it, and nothing about the result looks
# wrong.

"""
    TruncationRule

How one entry of `EpidemicData.extras` is carried into a truncated copy.

Constructed by [`truncation`](@ref) rather than directly. The four rules cover
every case seen so far; `Custom` is the escape hatch.
"""
abstract type TruncationRule end

"Keep the value unchanged — it carries no time-indexed information."
struct Keep <: TruncationRule end

"""
Clamp every element to the cutoff: `min.(v, cutoff)`.

For "last time this individual was known present" vectors, where a value beyond
the cutoff is exactly the leak.
"""
struct Clamp <: TruncationRule end

"""
Keep only entries at or before the cutoff.

`by` extracts the time from an element (default `last`, which suits the
`(individual, time)` tuples that event lists usually take).
"""
struct Filter{F} <: TruncationRule
    by::F
end
Filter() = Filter(last)

"""
Copy the array, unchanged in value.

For arrays a sampler MUTATES IN PLACE. Sharing them between the truncated and
untruncated data lets one window's mutation leak into the other's view: e.g. a
changepoint kernel that swaps test labellings in place.
"""
struct CopyArray <: TruncationRule end

"Apply `f(value, cutoff)` and use the result."
struct Custom{F} <: TruncationRule
    f::F
end

# Base-qualified: `truncation`'s keyword arguments are deliberately named
# `filter` and `copy` for readability at the call site, which shadows the Base
# functions inside that method. Qualifying here keeps these independent of it.
_apply(::Keep, v, cutoff) = v
_apply(::Clamp, v, cutoff) = min.(v, cutoff)
_apply(r::Filter, v, cutoff) = Base.filter(x -> r.by(x) <= cutoff, v)
_apply(::CopyArray, v, cutoff) = Base.copy(v)
_apply(r::Custom, v, cutoff) = r.f(v, cutoff)

"""
    TruncationPlan

A declared, checked recipe for truncating an `EpidemicData`. Build with
[`truncation`](@ref) and apply with [`truncate_data`](@ref).
"""
struct TruncationPlan{R<:NamedTuple}
    rules::R
    strict::Bool
    known_from::Union{Nothing,Symbol}
end
TruncationPlan(rules::NamedTuple, strict::Bool) = TruncationPlan(rules, strict, nothing)

"""
    truncation(; clamp=(), filter=(), copy=(), keep=(), custom=(), strict=true,
                 known_from=nothing)

Declare how each entry of `EpidemicData.extras` behaves under truncation, for
leave-future-out cross-validation.

Every extra must be accounted for. With `strict=true` (the default) an undeclared
extra whose value is time-shaped, a `Vector`/`Matrix` long enough to be indexed
by time, or a collection of `(id, time)` tuples — is an **error**, because the
common failure mode is a silent leak that quietly inflates a model's score.
Declare it `keep` if it genuinely carries no future information; that is a
one-word statement that you checked.

# Rules
- `clamp`  , `min.(v, cutoff)`; for "last time known present" vectors.
- `filter` : drop entries after the cutoff; for event lists of `(id, time)`.
- `copy`   , same values, fresh array; for anything a sampler mutates in place.
- `keep`   , carried through untouched; an explicit "no future information here".
- `custom`  — `name => (v, cutoff) -> new_v` for anything else.

# When an individual becomes known
`known_from` names an extra holding, per individual, the first time the study
knew it existed. By default that is the start of its `sampling_period`. It must
be given separately when the sampling period opens before the individual was
first observed: a badger's opens at birth, but nobody knows about a cub until
it is caught. Without it, a fit at cutoff `t` would include an animal whose
existence only a later capture revealed, and so would the forecast from `t`.

An individual not yet known at the cutoff gets an empty sampling period in the
training copy; [`score_window`](@ref) neither scores nor simulates it.

# Example
```julia
plan = truncation(
    clamp  = (:last_capture_time,),      # death-ban must not see past the cutoff
    filter = (:captures_after_monit,),   # post-window survival evidence
    copy   = (:tests, :n_neg, :n_pos),   # mutated in place by the changepoint kernel
    keep   = (:age, :sex, :season),      # covariates, no future information
)
train = truncate_data(data, plan, 128)
```
"""
function truncation(; clamp = (), filter = (), copy = (), keep = (),
                      custom = (), strict::Bool = true,
                      known_from::Union{Nothing,Symbol} = nothing)
    pairs = Pair{Symbol,TruncationRule}[]
    for n in clamp;  push!(pairs, n => Clamp()); end
    for n in filter; push!(pairs, n isa Pair ? (n.first => Filter(n.second)) :
                                               (n => Filter())); end
    for n in copy;   push!(pairs, n => CopyArray()); end
    for n in keep;   push!(pairs, n => Keep()); end
    for n in custom; push!(pairs, n.first => Custom(n.second)); end

    names = first.(pairs)
    if length(unique(names)) != length(names)
        dup = [n for n in unique(names) if count(==(n), names) > 1]
        error("truncation: extra(s) declared more than once: " * join(dup, ", "))
    end
    TruncationPlan(NamedTuple{Tuple(names)}(Tuple(last.(pairs))), strict, known_from)
end

"""
    known_times(data, plan) -> Vector{Int}

When each individual became known to the study: the start of its sampling
period, or later if the plan names a `known_from` extra.
"""
function known_times(data::EpidemicData, plan::TruncationPlan)
    f = first.(data.sampling_period)
    plan.known_from === nothing && return f
    v = getproperty(getfield(data, :extras), plan.known_from)
    length(v) == data.n_individuals ||
        throw(ArgumentError("known_from extra `$(plan.known_from)` needs one time " *
                            "per individual, got $(length(v))"))
    max.(f, Int.(v))
end

"""
    _looks_time_indexed(v, n_timepoints) -> Bool

Heuristic used only to decide whether an undeclared extra should raise under
`strict`. Deliberately over-eager: a false positive costs one word (`keep`), a
false negative costs a silent leak.
"""
function _looks_time_indexed(v, n_timepoints::Int)
    v isa AbstractArray || return false
    eltype(v) <: Union{Tuple,Pair} && return true          # event list, (id, time)
    eltype(v) <: Integer || return false                    # times are integers
    any(==(n_timepoints), size(v)) && return true           # indexed by time
    # A per-individual vector of times: values that reach into the series.
    ndims(v) == 1 && !isempty(v) && maximum(v) > 1 && return true
    return false
end

"""
    truncate_data(data, plan, cutoff) -> EpidemicData

A copy of `data` containing no information after `cutoff`.

`sampling_period` is truncated on the right for every individual. An individual
not yet known at the cutoff (see `known_from` in [`truncation`](@ref)) gets an
empty period; this is deliberate, because clamping it to `(first, first)` would
leak that future entrant and its first observation into the training fit. Aggregates are
zeroed (they are rebuilt by the sampler), and each extra is handled by its
declared rule.
Undeclared time-shaped extras raise unless the plan was built with
`strict=false`.

The cutoff is the strict reading of "train on `1:t`". Clamping to `t + M`
instead would remove some downstream `-Inf`s, but only by letting the fit see
which individuals are observed during the scoring window: future information,
even if it is design rather than outcome.
"""
function truncate_data(data::EpidemicData, plan::TruncationPlan, cutoff::Int)
    1 <= cutoff <= data.n_timepoints ||
        throw(ArgumentError("cutoff $cutoff outside 1:$(data.n_timepoints)"))

    extras = getfield(data, :extras)
    if plan.strict
        undeclared = Symbol[]
        for n in propertynames(extras)
            haskey(plan.rules, n) && continue
            _looks_time_indexed(getproperty(extras, n), data.n_timepoints) &&
                push!(undeclared, n)
        end
        isempty(undeclared) || error("""
            truncate_data: these extras look time-indexed but have no rule:
                $(join(undeclared, ", "))
            Any of them could carry information from beyond the cutoff into the
            training fit, which silently inflates the score of whichever model
            exploits it. Declare each one:
                clamp , min.(v, cutoff), for "last time known present"
                filter: drop entries after the cutoff, for (id, time) events
                copy  : same values, fresh array, for in-place-mutated arrays
                keep   — carried through untouched (an explicit "I checked")
            or build the plan with `strict=false` to silence this check.""")
    end

    new_extras = NamedTuple(
        n => (haskey(plan.rules, n) ?
              _apply(getproperty(plan.rules, n), getproperty(extras, n), cutoff) :
              getproperty(extras, n))
        for n in propertynames(extras))

    known = known_times(data, plan)
    sp = [known[i] > cutoff ? (f, min(f - 1, cutoff)) : (f, min(l, cutoff))
          for (i, (f, l)) in enumerate(data.sampling_period)]

    return epidemic_data(;
        n_individuals = data.n_individuals,
        n_timepoints = data.n_timepoints,
        trans_mat = data.trans_mat,
        starting_state = data.starting_state,
        observation_process = getfield(data, :observation_process),
        observation_weight = getfield(data, :observation_weight),
        aggregates = Dict(k => zero(v) for (k, v) in pairs(data.aggregates)),
        derived_summaries = data.derived_summaries,
        rest_contribution = getfield(data, :rest_contribution),
        sampling_period = sp,
        affected_individuals = data.affected_individuals,
        # Carry the coupling declaration through. The mask is already the
        # closure over source states, so listing its true entries rebuilds it
        # exactly; an all-false mask (a declared uncoupled model) stays
        # all-false rather than reverting to "assume everything is coupled".
        coupled_transitions = data.coupled_mask === nothing ? nothing :
            [(data.state_space[a], data.state_space[b])
             for a in axes(data.coupled_mask, 1), b in axes(data.coupled_mask, 2)
             if data.coupled_mask[a, b]],
        state_space = data.state_space,
        group = data.group,
        focal_self_contribution = getfield(data, :focal_self_contribution),
        new_extras...)
end

"""
    lfo_cutoffs(data; L, M, stride=1) -> Vector{Int}

The cutoffs a leave-future-out sweep visits: `L:stride:(T - M)`.

`L` is the minimum training window. Choose it from when the data first identify
the model's parameters, not for roundness, on the badger model the natural
choice was set by when the diagnostic tests came into use, and a smaller `L` left
most observation parameters at their priors while the fit scored held-out data.
"""
function lfo_cutoffs(data::EpidemicData; L::Int, M::Int, stride::Int = 1)
    L >= 1 || throw(ArgumentError("L must be >= 1"))
    M >= 1 || throw(ArgumentError("M must be >= 1"))
    last_cutoff = data.n_timepoints - M
    L <= last_cutoff ||
        throw(ArgumentError("L=$L leaves no windows: n_timepoints=$(data.n_timepoints), M=$M"))
    collect(L:stride:last_cutoff)
end
