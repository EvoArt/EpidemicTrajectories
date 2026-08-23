# Source attribution: which of several competing sources caused an infection.
#
# The three quantities here all rest on one idea — that a force of infection
# DECOMPOSES into named components (background/environmental, within-group
# transmission, an imported case, ...) — and they read that decomposition three
# different ways:
#
#   SourceAttributionResidual  a randomized-PIT RESIDUAL on the attributed source
#                              (Lau et al. 2014 §2.2.2, the "infection-link
#                              residual"). Diagnoses whether the model's
#                              TRANSMISSION STRUCTURE is right, which neither a
#                              waiting-time nor a survival residual can see.
#   FOIRatioSummary            the share of cumulative hazard from one component.
#                              A :raw scientific output — the endogenous/exogenous
#                              split, not a diagnostic.
#   case_reproduction_numbers  R_i, expected secondary infections caused by i.
#                              Also :raw, and POPULATION-level (see below).
#
# ## Why the decomposition is DECLARED, not derived
#
# The package cannot split a force of infection by itself: `S -> E` is one rate
# function returning one number, and nothing in it says which part is background
# and which is transmission. The reference implementation recovers the split by
# MUTATING SHARED STATE — `foi_primary` zeroes `data.totalNumInfec[g,t]`,
# re-evaluates the group FOI, and restores the count. That is not thread-safe, it
# corrupts `data` if anything throws between the zero and the restore, and it can
# only ever produce a TWO-way split.
#
# So the user declares the components as ordinary pure rate functions, exactly as
# they declare aggregates and transitions. The package sums them, normalises, and
# never needs to know what any of them means. This also generalises for free:
# Lau's eq 2.10 is defined for k sources, and per-neighbour attribution ("which
# badger infected this one") is then just a longer component list.

"""
    SourceAttributionResidual(from => to; components, source, name=:source_attribution,
                              window=:sampling_period, require_start_state=nothing,
                              origin=:window_start)

The **infection-link residual** (Lau et al. 2014, §2.2.2, eqs 2.6–2.7 and 2.10): a
randomized-PIT residual on WHICH SOURCE caused each infection.

This is the third member of the Lau family, alongside the exposure-time residual
(§2.2.1) and the progression residual — and it asks a question neither of those
can. A waiting-time residual checks WHEN an event happened; this checks WHERE the
hazard came from. A model can get every waiting time right and still attribute
infections to the wrong sources, and only this residual sees that.

!!! danger "`source` is required, and the residual is VACUOUS without a real one"
    The source must be **inferred from the data** — a latent variable your sampler
    draws, stored in `data` — and must NOT be drawn from the same component rates
    this residual then scores.

    This is not a stylistic preference, it is the difference between a diagnostic
    and a random number generator. If the source is drawn from probabilities `p`
    and the PIT interval is then built from those same `p`, the result is
    **Uniform(0,1) by construction for any `p` whatsoever** — verified directly:
    with two components the residual is uniform at shares of 0.5, 0.9, 0.99 and
    0.01 alike. Such a residual passes calibration perfectly and has exactly zero
    power; it can never detect a wrong transmission structure, which is the only
    thing it exists to detect.

    In Lau et al. the infection-link sequence "determines the particular
    infectious–susceptible pair responsible for each infection event" and is part
    of the MCMC state, explored jointly with the parameters. The residual then
    scores that INFERRED source against the model's predicted probabilities. That
    comparison — inferred source vs predicted probability — is where all the
    diagnostic content lives.

## How it works

At the step on which individual `i` became infected, each declared component
contributes a rate; normalising gives the model's predicted probability for each
source. `source(model, data, X, i, t)` supplies the source actually inferred. The
components are ranked by predicted probability and the PIT interval is the
inferred source's slice of the cumulative sorted mass (eqs 2.7, 2.10).

The intuition: if the model says 90% background and the inferred sources say
otherwise across many infections, the residuals pile up and the test fires.

## Arguments

- `from => to` — the infection transition, e.g. `:S => :E`.
- `components` — **required**. Named pure rate functions, as a `NamedTuple` or a
  tuple of pairs:

  ```julia
  components = (; background   = (model, data, X, i, t) -> model.alpha,
                  transmission = (model, data, X, i, t) -> model.beta * n_infected(data, i, t))
  ```

  Each is `(model, data, X, i, t) -> rate`. **They must sum to the model's total
  force of infection** — the package cannot check this and does not try, exactly
  as it does not check that `coupling_trans_mat` agrees with `trans_mat`. Two or
  more components; with one there is nothing to attribute.

  Declaring them is also what avoids the reference's save/zero/restore trick
  (`foi_primary` zeroes `data.totalNumInfec[g,t]`, evaluates, restores): these are
  pure functions, so nothing is mutated and the residual is safe under threading.

- `source` — **required**. `(model, data, X, i, t) -> k`, the index (or name) of
  the component inferred to have caused `i`'s infection at step `t`. Read it from
  wherever your sampler stores it, typically `data.infection_source[i]`. Return
  `nothing` or `missing` for an individual whose source was not inferred.
- `origin` — where the infection-time scan starts. `:window_start` (default) or
  `:entry_to_from_state`.
- `require_start_state` — restrict to individuals in this state at their window
  start, e.g. `:S`. The reference does this unconditionally; here it is opt-in.

An individual never infected returns `missing` — no source to attribute. Coverage
is therefore the infected-and-attributed fraction, worth watching.

# Example
```julia
# `data.infection_source[i]` is maintained by the user's own sampler: 1 = background,
# 2 = within-group transmission.
ilr = SourceAttributionResidual(:S => :E;
    components = (; background = foi_background, transmission = foi_transmission),
    source = (model, data, X, i, t) -> data.infection_source[i])
```

!!! note "No inferred source in your model?"
    Then this residual is not available to you, and that is the honest answer
    rather than a limitation to work around. Use [`FOIRatioSummary`](@ref) for the
    endogenous/exogenous split, which needs no latent source and is a `:raw`
    scientific output rather than a diagnostic.
"""
function SourceAttributionResidual(pair::Pair;
                                   components=nothing,
                                   source=nothing,
                                   name::Symbol=:source_attribution,
                                   window=:sampling_period,
                                   origin::Symbol=:window_start,
                                   require_start_state=nothing)
    components === nothing && error(
        "SourceAttributionResidual: `components` is required. Declare the force of " *
        "infection's named parts as pure rate functions, e.g. " *
        "`components = (; background = f_bg, transmission = f_tr)`, each " *
        "`(model, data, X, i, t) -> rate`. The package cannot split one rate " *
        "function into sources by itself, and will not mutate your data to try.")
    # The guard that keeps this residual from being a random number generator.
    source === nothing && error(
        "SourceAttributionResidual: `source` is required and has no default. It must " *
        "return the INFERRED source of individual i's infection — a latent variable " *
        "your sampler draws — as `(model, data, X, i, t) -> component index or name`.\n\n" *
        "There is deliberately no fallback that samples the source from the " *
        "component rates: doing so makes the residual Uniform(0,1) BY CONSTRUCTION " *
        "for any rates at all (verified at shares 0.5/0.9/0.99/0.01), so it would " *
        "pass every calibration check while having exactly zero power to detect the " *
        "wrong transmission structure it exists to detect.\n\n" *
        "If your model has no inferred infection source, use FOIRatioSummary for " *
        "the endogenous/exogenous split instead.")

    comps = _normalize_components(components)
    length(comps) >= 2 || error(
        "SourceAttributionResidual: need at least 2 components to attribute between, " *
        "got $(length(comps)). With one source there is nothing to diagnose.")

    from_sym, to_sym = first(pair), last(pair)
    comp_fns = Tuple(last(c) for c in comps)
    comp_names = Tuple(first(c) for c in comps)
    req = require_start_state
    src = source

    f = function (model, data::EpidemicData, X, i, rng)
        from = state_code(data, from_sym)
        to = state_code(data, to_sym)
        t_start, t_end = _window(data, i, window)
        t_start > t_end && return missing

        if req !== nothing
            @inbounds X[t_start, i] == state_code(data, req) || return missing
        end

        t0 = if origin === :entry_to_from_state
            e = first_entry(X, i, from, t_start, t_end)
            e === nothing && return missing
            e
        else
            t_start
        end

        # The infection event. The rates are evaluated at `t_inf`, the step ACROSS
        # which the move happened — matching the exposure residual's convention,
        # and the reference's `t_inf = tE - 1`.
        t_event = first_entry(X, i, to, t0, t_end)
        t_event === nothing && return missing      # never infected: nothing to attribute
        t_inf = t_event - 1
        t_inf < t_start && return missing

        k = _source_index(src(model, data, X, i, t_inf), comp_names)
        k === nothing && return missing            # source not inferred for this one

        return _attribution_pit(comp_fns, k, model, data, X, i, t_inf, rng)
    end

    return TrajectorySummary(name, f, :pit)
end

# The inferred source, as an index into the declared components. Accepts an index
# or a component name; `nothing`/`missing` means "not inferred", which is a
# `missing` residual rather than an error.
@inline _source_index(::Nothing, names) = nothing
@inline _source_index(::Missing, names) = nothing
@inline function _source_index(k::Integer, names)
    1 <= k <= length(names) || error(
        "SourceAttributionResidual: `source` returned index $k, but there are " *
        "$(length(names)) components $(names)")
    return Int(k)
end
@inline function _source_index(s::Symbol, names)
    k = findfirst(==(s), names)
    k === nothing && error(
        "SourceAttributionResidual: `source` returned :$s, which is not one of the " *
        "declared components $(names)")
    return k
end

# Accept a NamedTuple, a tuple of `:name => f` pairs, or a Dict, and normalise to
# a vector of pairs so the order (which fixes nothing statistically, but must be
# deterministic) is stable.
_normalize_components(nt::NamedTuple) = [k => getfield(nt, k) for k in keys(nt)]
_normalize_components(t::Tuple) = [Symbol(first(p)) => last(p) for p in t]
_normalize_components(v::AbstractVector) = [Symbol(first(p)) => last(p) for p in v]
_normalize_components(d::AbstractDict) = [Symbol(k) => v for (k, v) in sort!(collect(d), by=first)]

# Lau et al. eqs 2.7 and 2.10, for ANY number of components.
#
# 2.7  sort the model's predicted source probabilities ascending, cumulative sum;
# 2.10 the PIT interval is the INFERRED source's slice of that cumulative mass.
#
# `inferred` is the source the sampler attributed, passed in from `data` — NOT
# drawn here. Drawing it from `probs` would make the returned value uniform for
# any `probs` at all, i.e. a residual with no power; see the constructor's
# docstring. Everything diagnostic about this quantity comes from `inferred` and
# `probs` being independently determined.
#
# The reference hard-codes exactly two sources and needs a coin-flip branch when
# their probabilities tie (it locates the source by MATCHING its probability
# value, which is ambiguous when two are equal). Resolving by INDEX instead makes
# ties fall out correctly with no special case: equal shares get adjacent,
# equal-width slices, which is exactly right.
@inline function _attribution_pit(comp_fns::Tuple, inferred::Int, model, data, X, i, t, rng)
    n = length(comp_fns)
    rates = _component_rates(comp_fns, model, data, X, i, t)

    probs = ntuple(k -> max(float(rates[k]), 0.0), n)
    psum = sum(probs)
    (isfinite(psum) && psum > 0) || return missing

    # 2.7 + 2.10: rank ascending by predicted probability, then take the inferred
    # source's slice of the cumulative sorted mass.
    order = sortperm(collect(probs))
    lb = 0.0
    ub = 0.0
    cum = 0.0
    for k in 1:n
        prev = cum
        cum += probs[order[k]]
        if order[k] == inferred
            lb = prev
            ub = cum
            break
        end
    end

    return randomized_pit(lb / psum, ub / psum, rng)
end

# Evaluate the component rates. Recursion over the tuple for the usual reason:
# the components are distinct closure types, and a `for` loop would infer their
# union and dispatch at run time on every call.
@inline _component_rates(fns::Tuple, model, data, X, i, t) =
    _component_rates_impl(fns, model, data, X, i, t)
@inline _component_rates_impl(::Tuple{}, model, data, X, i, t) = ()
@inline function _component_rates_impl(fns::Tuple, model, data, X, i, t)
    r = float(first(fns)(model, data, X, i, t))
    return (r, _component_rates_impl(Base.tail(fns), model, data, X, i, t)...)
end

"""
    FOIRatioSummary(from => to; components, numerator, name=:foi_ratio,
                    window=:sampling_period, require_start_state=nothing)

The share of an individual's cumulative force of infection contributed by one
named component — the **endogenous/exogenous split**.

A `:raw` summary, not a residual: it is a ratio in `[0,1]` with no uniformity
claim attached, so the calibration checks do not apply to it and
[`uniformity_test`](@ref) will refuse it. It is a scientific output — "what
fraction of infection pressure was background rather than within-group
transmission" — and it shares only the per-individual, per-draw SHAPE with the
residuals, which is why it lives on the same machinery.

Accumulation runs from the window start to the step before infection (or to
censoring, for individuals never infected), matching the exposure residual's
window exactly so the two are directly comparable.

- `components` — as in [`SourceAttributionResidual`](@ref); must sum to the total
  force of infection.
- `numerator` — which component's share to report, by name.

Returns `missing` when the cumulative total is zero — no hazard accumulated means
no meaningful share, and a `NaN` would poison a downstream mean.

# Example
```julia
# what fraction of infection pressure was environmental rather than from badgers?
ratio = FOIRatioSummary(:S => :E;
    components = (; background = foi_background, transmission = foi_transmission),
    numerator = :background)
```
"""
function FOIRatioSummary(pair::Pair;
                         components=nothing,
                         numerator::Symbol,
                         name::Symbol=:foi_ratio,
                         window=:sampling_period,
                         censor_at=(:window_end,),
                         require_start_state=nothing)
    components === nothing && error(
        "FOIRatioSummary: `components` is required — see SourceAttributionResidual.")
    comps = _normalize_components(components)
    idx = findfirst(c -> first(c) == numerator, comps)
    idx === nothing && error(
        "FOIRatioSummary: `numerator` :$numerator is not among the components " *
        "$(first.(comps))")

    from_sym, to_sym = first(pair), last(pair)
    comp_fns = Tuple(last(c) for c in comps)
    num_i = idx
    censors = Tuple(censor_at)
    req = require_start_state

    f = function (model, data::EpidemicData, X, i, rng)
        from = state_code(data, from_sym)
        to = state_code(data, to_sym)
        t_start, t_end = _window(data, i, window)
        t_start > t_end && return missing

        if req !== nothing
            @inbounds X[t_start, i] == state_code(data, req) || return missing
        end

        # Accumulate to the step before infection, or to censoring if never
        # infected — the same window the exposure residual uses.
        t_event = first_entry(X, i, to, t_start, t_end)
        t_stop = if t_event !== nothing
            t_event - 1
        else
            tc = t_end
            for c in censors
                cc = if c === :window_end
                    t_end
                else
                    e = first_entry(X, i, state_code(data, c), t_start, t_end)
                    e === nothing ? nothing : e
                end
                cc !== nothing && cc < tc && (tc = cc)
            end
            tc - 1
        end
        t_stop < t_start && return missing

        cum_num = 0.0
        cum_tot = 0.0
        for t in t_start:t_stop
            rates = _component_rates(comp_fns, model, data, X, i, t)
            cum_num += max(rates[num_i], 0.0)
            cum_tot += sum(max(r, 0.0) for r in rates)
        end

        (isfinite(cum_tot) && cum_tot > 0) || return missing
        return cum_num / cum_tot
    end

    return TrajectorySummary(name, f, :raw)
end

"""
    case_reproduction_numbers(model, data, X; components, infection=:S => :E,
                              infectious_state=:I, secondary,
                              group=(data, i, t) -> data.group[i],
                              weight=(model, data, X, i, t) -> 1.0)

R_i: the expected number of secondary infections caused by each individual —
returned as a `Vector{Union{Float64,Missing}}`, one entry per individual.

**This is a function, not a `TrajectorySummary`, and that is not an oversight.**
Every other quantity in this package is computed for one individual from that
individual's own trajectory. R_i is not: attributing an infection requires knowing
every infective present in the group at that moment, so it needs a pass over the
WHOLE population before any single individual's value is known. Forcing it into
the `(model, data, X, i, rng)` contract would make each individual re-scan the
entire population — `O(m² T)` instead of `O(m T)` — so it gets its own entry point.

Use [`summarize_population`](@ref) to run it across draws and get the same
`SummaryResult` everything else produces.

## How attribution works

For each infection event, the transmission component of the victim's force of
infection is shared among the infectives present in its group, in proportion to
their `weight`. An individual's R_i is the total it accrues across every event
occurring in its group while it was infectious. With uniform weights this is equal
attribution; with weights encoding relative infectiousness (a sex-specific
transmission coefficient, say) it splits proportionally.

- `components` / `secondary` — the decomposition and which component is
  transmission (the part attributable to other individuals). Background infections
  are attributed to nobody, which is the point of the split.
- `infection` — the transition marking a new infection, default `:S => :E`.
- `infectious_state` — the state from which an individual can infect others.
- `group` — `(data, i, t) -> group id`; defaults to `data.group[i]`. Pass your own
  for time-varying membership.
- `weight` — relative infectiousness, default uniform.

An individual that is never infectious gets `missing` — it had no opportunity to
infect anyone, which is different from having had the opportunity and infected
nobody (that is a genuine `0.0`).
"""
function case_reproduction_numbers(model, data::EpidemicData, X;
                                   components,
                                   secondary::Symbol,
                                   infection::Pair=(:S => :E),
                                   infectious_state=:I,
                                   group=nothing,
                                   weight=nothing)
    comps = _normalize_components(components)
    sec_i = findfirst(c -> first(c) == secondary, comps)
    sec_i === nothing && error(
        "case_reproduction_numbers: `secondary` :$secondary is not among the " *
        "components $(first.(comps))")
    comp_fns = Tuple(last(c) for c in comps)

    m = data.n_individuals
    T = data.n_timepoints
    from = state_code(data, first(infection))
    to = state_code(data, last(infection))
    inf_state = state_code(data, infectious_state)
    grp = group === nothing ? (d, i, t) -> d.group[i] : group
    wt = weight === nothing ? (mo, d, x, i, t) -> 1.0 : weight

    # Pass 1: every infection event, as (victim, group, time).
    events = Tuple{Int,Int,Int}[]
    @inbounds for j in 1:m
        t_start, t_end = data.sampling_period[j]
        for t in max(t_start, 1):min(t_end, T) - 1
            if X[t, j] == from && X[t + 1, j] == to
                push!(events, (j, grp(data, j, t), t))
                break                       # each individual is infected once
            end
        end
    end

    # Pass 2: per-event attribution per unit of infectiousness weight. The
    # transmission component's share of the victim's total FOI, divided by the
    # total weight of the infectives that could have caused it.
    attrib = zeros(Float64, length(events))
    @inbounds for (ei, (j, g, t)) in enumerate(events)
        rates = _component_rates(comp_fns, model, data, X, j, t)
        total = sum(max(r, 0.0) for r in rates)
        total > 0 || continue
        sec = max(rates[sec_i], 0.0)
        sec > 0 || continue

        W = 0.0
        for k in 1:m
            if X[t, k] == inf_state && grp(data, k, t) == g
                W += float(wt(model, data, X, k, t))
            end
        end
        W > 0 && (attrib[ei] = sec / (W * total))
    end

    # Pass 3: sum each individual's share over the events it could have caused.
    R = Vector{Union{Float64,Missing}}(missing, m)
    @inbounds for i in 1:m
        ever_infectious = false
        for t in 1:T
            if X[t, i] == inf_state
                ever_infectious = true
                break
            end
        end
        # Never infectious is `missing` — no opportunity to infect anyone. That is
        # a different statement from an infectious individual who happened to
        # infect nobody, which is a genuine 0.0 and must stay in the sample.
        ever_infectious || continue

        Ri = 0.0
        for (ei, (j, g, t)) in enumerate(events)
            attrib[ei] == 0.0 && continue
            if X[t, i] == inf_state && grp(data, i, t) == g && i != j
                Ri += attrib[ei] * float(wt(model, data, X, i, t))
            end
        end
        R[i] = Ri
    end

    return R
end

"""
    summarize_population(f, name, data, draws; kind=:raw) -> SummaryResult

Run a POPULATION-level summary over draws, giving the same
[`SummaryResult`](@ref) the per-individual driver produces.

`f(model, data, X) -> Vector{Union{Float64,Missing}}` of length
`data.n_individuals`. This is the seam for a quantity that cannot be computed one
individual at a time — [`case_reproduction_numbers`](@ref) is the motivating case,
since attributing an infection needs the whole population's state at that moment.

Because the result is a `SummaryResult`, everything downstream works unchanged:
coverage, plots, and the archiving and collector paths.

# Example
```julia
R = summarize_population(:R_i, data, draws) do model, data, X
    case_reproduction_numbers(model, data, X;
        components = (; background = f_bg, transmission = f_tr),
        secondary = :transmission)
end
```
"""
function summarize_population(f, name::Symbol, data::EpidemicData, draws; kind::Symbol=:raw)
    dlist = Base.IteratorSize(typeof(draws)) === Base.HasLength() ? draws : collect(draws)
    n_draws = length(dlist)
    n_draws == 0 && error("summarize_population: `draws` is empty")

    m = data.n_individuals
    M = Matrix{Union{Float64,Missing}}(missing, m, n_draws)

    for (k, (model, X)) in enumerate(dlist)
        size(X, 2) == m || error(
            "summarize_population: draw $k has $(size(X, 2)) individuals but `data` has $m")
        v = f(model, data, X)
        length(v) == m || error(
            "summarize_population: `f` returned $(length(v)) values, expected $m")
        @inbounds for i in 1:m
            M[i, k] = v[i]
        end
    end

    values = Dict{Symbol,Matrix{Union{Float64,Missing}}}(name => M)
    kinds = Dict{Symbol,Symbol}(name => kind)
    coverage = Dict{Symbol,Float64}(name => count(!ismissing, M) / length(M))
    return SummaryResult(values, kinds, coverage, [name], n_draws)
end
