# Trajectory summaries: post-hoc, per-individual, per-draw quantities computed
# from a SAMPLED trajectory `X` — residuals for diagnosing misspecification, and
# derived epidemiological quantities that share the same computational shape.
#
# This is the FOURTH artefact generated from one model spec, alongside the
# simulator, the likelihood and the latent sampler. The first three say what the
# model IS; this one says whether the fit was any good.
#
# ## The central design rule applies here too
#
# The package never assumes what a residual measures. Two waiting-time residuals
# differ on three axes that the transition spec does NOT determine:
#
#   * the ACCUMULATOR   — `1 - ∏(1-p_s)` (discrete per-step probabilities) or
#                         `1 - exp(-Σλ_s)` (continuous cumulative hazard);
#   * the CLOCK ORIGIN  — read from `X` (time of entry to a state), from the
#                         sampling window, or from the user's own `data` (a birth
#                         time the trajectory knows nothing about);
#   * the CONDITIONING  — none, or left-truncated by a normaliser.
#
# The two accumulators agree ONLY when `p_s = 1 - exp(-λ_s)` exactly, and diverge
# precisely at large per-step hazard — which is the regime a residual exists to
# report on. So `accumulate` has NO DEFAULT: omitting it is an error, never a
# silent choice between two answers that differ where it matters most.
#
# What the package CAN derive from the spec is the event-extraction and censoring
# skeleton (scan `X` for the entry to a state, for a competing risk, for the
# window end) and the per-step hazard (`transition_prob`). That is what
# `WaitingTimeResidual` supplies. Everything else is declared.
#
# ## Not reversible, and deliberately so
#
# `@residual` looks superficially like `@aggregate` and must NOT be built like it.
# Aggregates need reversible forward/reverse updates because they are maintained
# INCREMENTALLY inside iFFBS, under an invariant that they always agree with `X`.
# A summary is computed ONCE per draw, read-only, on a COMPLETE `X`. There is no
# invariant to preserve and nothing to reverse; copying that machinery would
# double the surface area and buy nothing.
#
# ## Never on the AD path
#
# Summaries consume `Int` states and return `Float64`, computed on a sampled `X`
# outside every gradient call. They are post-hoc by default, so the fit's hot loop
# is untouched: residuals cost exactly zero inside iFFBS and HMC because they do
# not run there.

# =============================================================================
# Tier 0 — the generic core
# =============================================================================

"""
    TrajectorySummary(name, f; kind=:pit)

One per-individual, per-draw quantity computed from a sampled trajectory. This is
the whole contract of the residuals layer; everything else in this file builds one
of these.

- `name` — a `Symbol`, used to key the result table.
- `f` — `(model, data, X, i, rng) -> Union{Float64,Missing}`.
- `kind` — `:pit` for a randomized probability-integral-transform residual (which
  is Uniform(0,1) under a correctly specified model, so the uniformity tests
  apply), `:raw` for anything else (an R_i, a lifetime, a rate ratio). The kind is
  what tells a downstream check whether "is this uniform?" is even a meaningful
  question to ask of the column.

**`missing` is a first-class return value**, meaning "this individual does not
contribute to this summary" — it was never in the origin state, its window is
empty, a normaliser underflowed. The reference implementation expresses the same
thing with a bare `continue`, which makes it invisible; returning `missing`
explicitly is what lets [`trajectory_summaries`](@ref) report COVERAGE. A change
that silently drops half the population shows up as a coverage fraction that
moved, and nothing else catches that.

Build one with [`@residual`](@ref) (sugar), [`WaitingTimeResidual`](@ref)
(auto-derived skeleton), or directly from a closure.
"""
struct TrajectorySummary{F}
    name::Symbol
    f::F
    kind::Symbol
end
TrajectorySummary(name::Symbol, f; kind::Symbol=:pit) = TrajectorySummary(name, f, kind)

"""
    randomized_pit(lb, ub, rng) -> Float64

Draw a randomized probability-integral-transform residual uniformly on `[lb, ub]`.

Discrete time makes the exact PIT unusable: an event observed at step `L` tells us
only that the underlying continuous waiting time fell somewhere in
`[F(L-1), F(L)]`, so the PIT is an INTERVAL, not a point. Drawing uniformly within
it restores exact Uniform(0,1) calibration under a correct model — which is what
makes the uniformity tests meaningful in the first place.

A right-censored individual gives `[F(c), 1]`; one that carries no information at
all gives `[0, 1]`, i.e. a plain uniform draw.

The bounds are clamped into `[0,1]` and ordered, so an accumulator that overshoots
by a rounding error cannot produce a residual outside the unit interval.
"""
@inline function randomized_pit(lb, ub, rng)
    l = clamp(float(lb), 0.0, 1.0)
    u = clamp(float(ub), l, 1.0)
    return l + (u - l) * rand(rng)
end

"""
    discrete_product_cdf(hazard, t_start, n) -> Float64

`F(n) = 1 - ∏_{s=0}^{n-1} (1 - p_s)`, where `p_s = hazard(t_start + s)`.

The DISCRETE accumulator: the probability that an event with per-step probability
`p_s` has happened within `n` steps of `t_start`. Use it when the model's rates
are genuinely per-step probabilities — which is what [`@transitions`](@ref) rates
are.

Returns `0.0` for `n <= 0` (no steps, no chance of the event).

See [`cumulative_hazard_cdf`](@ref) for the continuous counterpart, and read the
warning there before choosing between them.
"""
@inline function discrete_product_cdf(hazard, t_start, n)
    n <= 0 && return 0.0
    surv = 1.0
    for s in 0:(n - 1)
        surv *= (1.0 - float(hazard(t_start + s)))
    end
    return 1.0 - surv
end

"""
    cumulative_hazard_cdf(rate, t_start, n) -> Float64

`F(n) = 1 - exp(-Σ_{s=0}^{n-1} λ_s)`, where `λ_s = rate(t_start + s)`.

The CONTINUOUS accumulator: the Sellke-threshold form, treating the per-step
values as RATES accumulating into a cumulative hazard rather than as
probabilities. Use it when the quantity being summed is a force of infection.

!!! warning "This is not interchangeable with `discrete_product_cdf`"
    The two agree only when `p_s = 1 - exp(-λ_s)` exactly, and they diverge as the
    per-step value grows — which is exactly the regime where the residual is
    trying to tell you something. Picking the wrong one gives a plausible,
    silently miscalibrated number. This is why `accumulate` has no default.
"""
@inline function cumulative_hazard_cdf(rate, t_start, n)
    n <= 0 && return 0.0
    cum = 0.0
    for s in 0:(n - 1)
        cum += float(rate(t_start + s))
    end
    return 1.0 - exp(-cum)
end

# Dispatch a named accumulator to its implementation. `Val` so the branch resolves
# at compile time inside the per-individual loop, rather than as a symbol
# comparison on every one of the m calls per draw.
@inline _accumulate_cdf(::Val{:discrete_product}, hazard, t_start, n) =
    discrete_product_cdf(hazard, t_start, n)
@inline _accumulate_cdf(::Val{:cumulative_hazard}, hazard, t_start, n) =
    cumulative_hazard_cdf(hazard, t_start, n)

const _ACCUMULATORS = (:discrete_product, :cumulative_hazard)

# =============================================================================
# Event extraction — the skeleton the spec CAN derive
# =============================================================================

"""
    first_entry(X, i, state, t_from, t_to) -> Union{Int,Nothing}

The first time in `t_from:t_to` at which individual `i` is in `state`, or
`nothing` if it never is.

This is the clock-origin scanner: "when did this individual enter E?" is a
question about `X` alone, so the package can answer it without knowing what E
means. `state` is an integer state code — see [`state_code`](@ref) to get one from
a name. The range is clipped to `X`'s bounds, so an out-of-range window is empty
rather than an error.
"""
@inline function first_entry(X, i, state::Int, t_from::Int, t_to::Int)
    @inbounds for t in max(t_from, 1):min(t_to, size(X, 1))
        X[t, i] == state && return t
    end
    return nothing
end

"""
    state_code(data, s) -> Int

The integer code of state `s` in `data.state_space`. An integer passes through, so
a summary may be declared with either `:I` or its index — declaring a
`state_space` is what makes the two agree.
"""
state_code(data::EpidemicData, s::Integer) = Int(s)
state_code(data::EpidemicData, s::Symbol) = _state_index(data, s)

# =============================================================================
# The hazard seam — where the model spec earns its keep
# =============================================================================

"""
    transition_hazard(data, from, to; spec=:coupling) -> hazard

Build `hazard(model, data, X, i, t) -> Float64`, the per-step probability of the
`from -> to` move for individual `i` at time `t`, straight from the model spec.

This is the whole reason residuals belong in this package rather than in a script.
The reference implementation hand-rolls a `progression_probability` with a
`hasproperty(data, :likelihood_progression_fn)` fallback chain and probes for user
functions with `Base.applicable`/`@isdefined` at run time — duck-typed
rediscovery of a spec that `@transitions` has already declared. All of it collapses
to [`transition_prob`](@ref).

## `spec=:coupling` is the default, and it matters

`@survival` scales every live transition by the survival probability and adds
`* -> death` rows, so `data.trans_mat`'s rate for `E -> I` computes
`survival × progression`. A progression residual wants `P(E -> I | still alive)` —
the bare progression hazard. Accumulating the survival-scaled rate instead folds
mortality into a progression residual, and it fails SILENTLY: the number is still
in `[0,1]`, still looks like a residual, and is simply calibrated against the
wrong distribution.

`TransitionSpec.coupling` is the survival-free view, stashed by `@survival` for
exactly this shape of need. So:

- `:coupling` (default) — the survival-free view when there is one, else
  `trans_mat` (which IS survival-free when there is no `@survival`).
- `:full` — `data.trans_mat` as-is, survival folded in. Correct when the event of
  interest IS the survival-scaled move, e.g. a residual for `* -> D`.

Pass a [`TransitionSpec`](@ref) directly to use one the package did not choose.
"""
function transition_hazard(data::EpidemicData, from, to; spec=:coupling)
    tm = _resolve_spec(data, spec)
    a = state_code(data, from)
    b = state_code(data, to)
    return (model, data, X, i, t) -> transition_prob(tm, model, data, X, i, t, a, b)
end

_resolve_spec(data::EpidemicData, spec::TransitionSpec) = spec
function _resolve_spec(data::EpidemicData, spec::Symbol)
    tm = data.trans_mat
    spec === :full && return tm
    spec === :coupling && return tm.coupling === nothing ? tm : tm.coupling
    error("transition_hazard: `spec` must be :coupling, :full, or a TransitionSpec, got :$spec")
end

# =============================================================================
# Tier 1 — auto-derived skeleton, DECLARED accumulator
# =============================================================================

"""
    WaitingTimeResidual(from => to; accumulate, name, origin=:entry_to_from_state,
                        censor_at=(:window_end,), spec=:coupling, hazard=nothing,
                        window=:sampling_period, require_start_state=nothing)

A randomized-PIT residual for the waiting time between entering state `from` and
moving to state `to`, with the event scan and censoring derived from `X` and the
per-step hazard derived from the model spec.

`accumulate` is REQUIRED — `:discrete_product` or `:cumulative_hazard`. There is
no default and there must never be one: the two agree only when
`p_s = 1 - exp(-λ_s)` exactly, and diverge precisely where the residual carries
information. See [`discrete_product_cdf`](@ref).

## What is derived, and what you must declare

Derived from `X` and the spec:
- the clock origin, when it is `:entry_to_from_state` — scan `X` for `i`'s first
  time in `from` within its window;
- the event time — `i`'s first time in `to` at or after the origin;
- the censoring time — the earliest of the declared `censor_at` events;
- the per-step hazard — [`transition_hazard`](@ref) on the `from => to` pair.

Declared by you, because the spec cannot know them:
- `accumulate` (no default, see above);
- `origin` — `:entry_to_from_state` (from `X`) or `:window_start` (the
  individual's own `sampling_period` start). An exposure-time residual wants the
  latter: its clock starts when observation starts, not when the individual
  entered `S`, since it was already susceptible when we began watching.
- `censor_at` — a tuple naming what ends observation. `:window_end` is the
  individual's own `sampling_period` end; any other symbol is read as a STATE
  (typically the death state, a competing risk), censoring at the last step before
  the individual entered it. The earliest applicable one wins.

## Censoring, and why a competing risk is censored rather than dropped

An individual that dies before making the `from -> to` move has not falsified the
model — it simply stopped being observable. Dropping it biases the residual
towards individuals who lived long enough to progress. So a competing risk enters
as a right-censoring time, giving `[F(c), 1]`, exactly as running off the end of
the window does.

## Other keywords

- `name` — the result key. Defaults to `Symbol(from, :_to_, to)`.
- `hazard` — supply your own `(model, data, X, i, t) -> Float64` to override the
  spec-derived one. For a semi-Markov model whose progression hazard depends on
  time since infection, this is the seam.
- `spec` — which transition view the hazard comes from; see
  [`transition_hazard`](@ref). The default is the survival-free one.
- `window` — `:sampling_period` (default) or an explicit `(first, last)` applied
  to everyone.
- `require_start_state` — if given, individuals not in this state at their window
  start return `missing`. The reference does this unconditionally (skipping anyone
  not susceptible at `t_start`), which is a modelling choice about who is at risk,
  not a fact about the transition — so here it is opt-in.

Returns a [`TrajectorySummary`](@ref) of kind `:pit`.

# Example
```julia
progression = WaitingTimeResidual(:E => :I;
    accumulate = :discrete_product,   # @transitions rates are per-step probabilities
    origin     = :entry_to_from_state,
    censor_at  = (:D, :window_end))
```

!!! note "Survival does not fit here"
    A survival residual measures from a birth time the trajectory does not know,
    conditioned on being alive at first capture. Those are model-specific facts
    the spec cannot supply. It gets [`LeftTruncatedSurvivalResidual`](@ref), or
    drops to [`@residual`](@ref) — do NOT stretch this constructor to cover it.
    Widening Tier 1 is how a residual comes to be silently wrong.
"""
function WaitingTimeResidual(pair::Pair;
                             accumulate=nothing,
                             name::Symbol=Symbol(first(pair), :_to_, last(pair)),
                             origin::Symbol=:entry_to_from_state,
                             censor_at=(:window_end,),
                             spec=:coupling,
                             hazard=nothing,
                             window=:sampling_period,
                             require_start_state=nothing)
    # The one argument with no default, and the reason for it.
    accumulate === nothing && error(
        "WaitingTimeResidual: `accumulate` is required and has no default. Pass " *
        ":discrete_product (per-step probabilities, which is what @transitions " *
        "rates are) or :cumulative_hazard (rates accumulating into a cumulative " *
        "hazard, the Sellke form). They agree only when p_s = 1 - exp(-λ_s) " *
        "exactly and diverge at large per-step hazard — precisely where the " *
        "residual is informative — so the package will not guess.")
    accumulate in _ACCUMULATORS || error(
        "WaitingTimeResidual: unknown `accumulate` :$accumulate (expected one of $_ACCUMULATORS)")
    origin in (:entry_to_from_state, :window_start) || error(
        "WaitingTimeResidual: `origin` must be :entry_to_from_state or :window_start, got :$origin")

    from_sym, to_sym = first(pair), last(pair)
    acc = Val(accumulate)
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

        # The clock origin: either scanned out of `X`, or the window start.
        t0 = if origin === :entry_to_from_state
            e = first_entry(X, i, from, t_start, t_end)
            e === nothing && return missing      # never at risk of this transition
            e
        else
            t_start
        end

        # The event: the first time `i` is in `to` at or after the origin.
        t_event = first_entry(X, i, to, t0, t_end)

        # The censoring time: the earliest of the declared censor events. A state
        # name censors at the last step the individual was still observable, i.e.
        # one before it entered that state; `:window_end` at the window end.
        t_censor = t_end
        for c in censors
            tc = if c === :window_end
                t_end
            else
                e = first_entry(X, i, state_code(data, c), t0, t_end)
                e === nothing ? nothing : e - 1
            end
            tc !== nothing && tc < t_censor && (t_censor = tc)
        end

        hz = hazard === nothing ? transition_hazard(data, from_sym, to_sym; spec=spec) : hazard
        h = t -> hz(model, data, X, i, t)

        if t_event !== nothing
            # Event observed at step L after the origin. The underlying continuous
            # waiting time lies in [F(L-1), F(L)] — an interval, not a point.
            L = t_event - t0
            L <= 0 && return missing
            lb = _accumulate_cdf(acc, h, t0, L - 1)
            ub = _accumulate_cdf(acc, h, t0, L)
            return randomized_pit(lb, ub, rng)
        else
            # Right-censored at `t_censor`: all we know is that the waiting time
            # exceeded `c` steps, so the PIT lies in [F(c), 1].
            c = t_censor - t0
            c <= 0 && return randomized_pit(0.0, 1.0, rng)   # no information at all
            lb = _accumulate_cdf(acc, h, t0, c)
            return randomized_pit(lb, 1.0, rng)
        end
    end

    return TrajectorySummary(name, f, :pit)
end

@inline _window(data::EpidemicData, i, ::Symbol) = @inbounds data.sampling_period[i]
@inline _window(data::EpidemicData, i, w::Tuple) = (Int(first(w)), Int(last(w)))

"""
    LeftTruncatedSurvivalResidual(; survival, origin, condition_on, censor_at,
                                  death=:D, name=:survival)

A randomized-PIT residual for a lifetime measured from an EXTERNAL clock origin
and conditioned on having survived to a later time.

This is deliberately a separate constructor from [`WaitingTimeResidual`](@ref).
A survival residual differs on all three of the axes that constructor cannot
derive: its clock starts at a birth time the trajectory knows nothing about, it is
LEFT-TRUNCATED (we only ever see individuals that survived to first capture, so
the CDF must be renormalised by `S(origin -> condition_on)`), and it censors at a
capture that may fall after the monitoring window. Stretching the waiting-time
constructor to cover it is exactly how a residual comes to be silently wrong.

- `survival` — `(model, data, i, t) -> P(individual i survives the t -> t+1 step)`.
  The SAME survival the transitions use.
- `origin` — `i -> t0`, the clock origin. Typically `i -> data.birth_time[i]`.
- `condition_on` — `i -> t_c`, a time the individual is KNOWN to have been alive;
  the CDF is renormalised by `S(t0 -> t_c)`. Typically first capture.
- `censor_at` — `i -> t_censor` for an individual that does not die within the
  trajectory. Typically the last capture, which may fall after monitoring ends.
- `death` — the death state, scanned for in `X` (name or index).
- `name` — the result key.

An individual whose normaliser underflows (`S(t0 -> t_c) <= 0`, or non-finite)
returns `missing` rather than a `NaN`. It carries no information, and letting a
`NaN` through would poison a downstream uniformity test instead of showing up
honestly as a coverage drop.

Returns a [`TrajectorySummary`](@ref) of kind `:pit`.

!!! note "`condition_on` is a modelling choice, so it is yours to make"
    The reference conditions on FIRST capture and carries a commented-out
    alternative conditioning on LAST capture — the question was under active
    investigation when it was written. First capture is the defensible default (it
    is the earliest time the individual is known alive, so it truncates the least
    and discards the least information), but the package will not choose for you:
    pass the one your observation design implies.
"""
function LeftTruncatedSurvivalResidual(; survival, origin, condition_on, censor_at,
                                       death=:D, name::Symbol=:survival)
    f = function (model, data::EpidemicData, X, i, rng)
        d_state = state_code(data, death)
        t0 = origin(i)
        t_c = condition_on(i)

        # Cumulative survival from the clock origin: a plain product of per-step
        # survival probabilities, matching the reference's `cumulativeSurvivalProb`.
        S = function (n)
            n <= t0 && return 1.0
            p = 1.0
            for t in t0:(n - 1)
                p *= float(survival(model, data, i, t))
            end
            p
        end

        # Left truncation: we only ever observe individuals that survived to `t_c`,
        # so every probability below is conditional on that. Without this the
        # residual is calibrated against the WRONG distribution — the unconditional
        # lifetime — and reports the sampling design as model misfit.
        norm = S(t_c)
        (isfinite(norm) && norm > 0.0) || return missing

        t_death = first_entry(X, i, d_state, t0, size(X, 1))

        if t_death !== nothing
            # Died at `t_death`: the lifetime lies in the interval between the
            # conditional CDF one step before and at the death time.
            f_before = 1.0 - S(t_death - 1) / norm
            f_at = 1.0 - S(t_death) / norm
            (isfinite(f_before) && isfinite(f_at)) || return missing
            return randomized_pit(min(f_before, f_at), max(f_before, f_at), rng)
        else
            # Alive when last seen: right-censored, so the PIT lies in [F(c), 1].
            f_c = 1.0 - S(censor_at(i)) / norm
            isfinite(f_c) || return missing
            return randomized_pit(f_c, 1.0, rng)
        end
    end

    return TrajectorySummary(name, f, :pit)
end

# =============================================================================
# Tier 2 — the user's own summary
# =============================================================================

"""
    @residual name(model, data, X, i, rng) = body
    @residual :raw name(model, data, X, i, rng) = body

Declare a [`TrajectorySummary`](@ref) from an expression — the escape hatch for
everything the auto-derived constructors cannot know.

The body has `model`, `data`, `X`, `i` and `rng` in scope, and returns a `Float64`
or `missing`. Anything reachable as `data.name` (the `extras` convention) is
available, so a user's own birth times, capture histories or covariates need no
new plumbing. The package supplies the helpers — [`randomized_pit`](@ref),
[`discrete_product_cdf`](@ref), [`cumulative_hazard_cdf`](@ref),
[`first_entry`](@ref), [`transition_hazard`](@ref) — and calls your function; it
never assumes what your clock or conditioning is. This is exactly the `aggregates`
posture.

An optional leading `:pit` or `:raw` sets the kind (default `:pit`). Use `:raw`
for a quantity that is not a PIT residual — an R_i, a lifetime — so the uniformity
checks know not to apply.

**Unlike [`@aggregate`](@ref), no reverse is needed.** A summary is computed once
per draw on a complete `X`; there is no incremental invariant to preserve. Do not
reach for the reversibility machinery here.

# Example
```julia
@residual :raw infection_time(model, data, X, i, rng) = begin
    tI = first_entry(X, i, state_code(data, :I), 1, data.n_timepoints)
    tI === nothing ? missing : Float64(tI)
end
```
"""
macro residual(args...)
    rest = collect(args)
    kind = QuoteNode(:pit)
    if length(rest) == 2
        rest[1] isa QuoteNode || error("@residual kind must be :pit or :raw")
        kind = rest[1]
        rest = rest[2:end]
    end
    length(rest) == 1 || error("@residual expects [kind] name(model, data, X, i, rng) = body")
    ex = rest[1]
    (ex isa Expr && ex.head in (:(=), :function)) ||
        error("@residual expects `name(model, data, X, i, rng) = body`")
    sig = ex.args[1]
    (sig isa Expr && sig.head === :call) ||
        error("@residual expects a call signature on the left of `=`")
    name = sig.args[1]
    argnames = sig.args[2:end]
    length(argnames) == 5 || error(
        "@residual signature must take exactly (model, data, X, i, rng), got $(length(argnames))")
    body = ex.args[2]

    return esc(:($name = $TrajectorySummary($(QuoteNode(name)),
                                            ($(argnames...),) -> $body,
                                            $kind)))
end

# =============================================================================
# The driver
# =============================================================================

"""
    SummaryResult

What [`trajectory_summaries`](@ref) returns: one `n_individuals × n_draws` matrix
of `Union{Float64,Missing}` per summary, plus the coverage each achieved.

- `result[name]` — the matrix for summary `name`.
- `result.kinds[name]` — `:pit` or `:raw`.
- `result.coverage[name]` — the fraction of `(individual, draw)` cells that are
  NOT `missing`.

**Coverage is not decoration.** A residual silently applying to half the
population it should is invisible in the residual values themselves — every one of
them is a perfectly good number in `[0,1]`. It shows up here, and nowhere else.
Assert on it in a test.

[`residual_values`](@ref) gives the non-`missing` values pooled across draws (what
a histogram wants); [`draw_values`](@ref) gives one draw's (what a uniformity test
wants).
"""
struct SummaryResult
    values::Dict{Symbol,Matrix{Union{Float64,Missing}}}
    kinds::Dict{Symbol,Symbol}
    coverage::Dict{Symbol,Float64}
    names::Vector{Symbol}
    n_draws::Int
end

Base.getindex(r::SummaryResult, name::Symbol) = r.values[name]
Base.keys(r::SummaryResult) = r.names
Base.haskey(r::SummaryResult, name::Symbol) = haskey(r.values, name)
Base.length(r::SummaryResult) = length(r.names)

"""
    residual_values(result, name) -> Vector{Float64}

The non-`missing` values of one summary, pooled across individuals and draws.

This is the form a histogram or a QQ plot wants. For a TEST, use
[`draw_values`](@ref) instead: residuals from the same individual at different
draws are correlated, so pooling inflates the effective sample size and makes any
uniformity test anticonservative — it will reject a correct model.
"""
function residual_values(r::SummaryResult, name::Symbol)
    M = r.values[name]
    out = Float64[]
    sizehint!(out, length(M))
    @inbounds for x in M
        x === missing || push!(out, x)
    end
    return out
end

"""
    draw_values(result, name, draw) -> Vector{Float64}

The non-`missing` values of one summary for ONE draw — the per-draw sample a
uniformity test should be applied to.

Testing per draw and looking at the DISTRIBUTION of p-values across draws is the
correct treatment (see `pvalue_distribution` in the `HypothesisTests` extension),
because within one draw the residuals are one per individual and independent under
the model, which is what the test assumes.
"""
function draw_values(r::SummaryResult, name::Symbol, draw::Int)
    M = r.values[name]
    out = Float64[]
    @inbounds for i in axes(M, 1)
        x = M[i, draw]
        x === missing || push!(out, x)
    end
    return out
end

function Base.show(io::IO, r::SummaryResult)
    print(io, "SummaryResult: $(length(r.names)) summar$(length(r.names) == 1 ? "y" : "ies") ",
              "over $(r.n_draws) draw$(r.n_draws == 1 ? "" : "s")")
    for n in r.names
        cov = round(100 * r.coverage[n]; digits=1)
        print(io, "\n  :$n  ($(r.kinds[n]), $(cov)% coverage)")
    end
end

"""
    trajectory_summaries(summaries, data, draws; rng=Random.default_rng())

Compute every summary over every draw. `draws` is **any iterable of `(model, X)`
pairs** — an archive of thinned posterior draws, a generator reading them off
disk, or a single `[(model, X)]` for a one-off check.

Typing it that way is what keeps the package PPL-agnostic: the map-over-draws loop
lives here with no dependency on any chain type, and an adapter for a particular
PPL is a one-line generator on the caller's side.

```julia
R = trajectory_summaries((progression, exposure), data, draws; rng = StableRNG(1))
residual_values(R, :E_to_I)      # pooled, for a histogram
draw_values(R, :E_to_I, 1)       # one draw, for a uniformity test
```

Pass `summaries` as a `Tuple` — each summary is a distinct closure type, and a
`Vector{TrajectorySummary}` erases that, putting a runtime dispatch on every one
of the `m × n_draws × n_summaries` calls. (A `Vector` is accepted and converted,
but the conversion cannot recover what the container already erased at the point
it was built.)

## The aggregates are NOT rebuilt per draw

If your rate functions read `data.aggregates`, establish the invariant for each
draw's `X` yourself before handing it over — the same
`reset_aggregates!` + [`apply_derived_summaries!`](@ref) pair the likelihood needs.
The package does not do it for you because it cannot know whether your summaries
read the aggregates at all, and rebuilding them unconditionally would put an
`O(m × T)` pass on every draw for every model that does not need one.
[`aggregate_synced_draws`](@ref) wraps an iterator to do exactly this when you do.

## Cost

One `O(m × T)` pass per summary per draw — the same order as a single likelihood
evaluation, against the thousands of gradient evaluations the fit already spent
per draw. Post-hoc means the fit's hot loop is untouched entirely: residuals cost
zero inside iFFBS and HMC because they do not run there.
"""
function trajectory_summaries(summaries, data::EpidemicData, draws;
                              rng::AbstractRNG=Random.default_rng())
    specs = summaries isa Tuple ? summaries : Tuple(summaries)
    isempty(specs) && error("trajectory_summaries: no summaries given")

    names = Symbol[s.name for s in specs]
    length(unique(names)) == length(names) || error(
        "trajectory_summaries: summary names must be unique, got $names")

    # An iterator with no length cannot size the output matrices up front, so
    # materialise only in that case.
    dlist = Base.IteratorSize(typeof(draws)) === Base.HasLength() ? draws : collect(draws)
    n_draws = length(dlist)
    n_draws == 0 && error("trajectory_summaries: `draws` is empty")

    m = data.n_individuals
    values = Dict{Symbol,Matrix{Union{Float64,Missing}}}(
        s.name => Matrix{Union{Float64,Missing}}(missing, m, n_draws) for s in specs)
    kinds = Dict{Symbol,Symbol}(s.name => s.kind for s in specs)

    for (k, draw) in enumerate(dlist)
        model, X = draw
        size(X, 2) == m || error(
            "trajectory_summaries: draw $k has $(size(X, 2)) individuals but `data` has $m")
        _fill_draw!(values, specs, model, data, X, m, k, rng)
    end

    coverage = Dict{Symbol,Float64}(n => count(!ismissing, values[n]) / length(values[n])
                                    for n in names)

    return SummaryResult(values, kinds, coverage, names, n_draws)
end

# One draw, one summary at a time. Recursion over the tuple rather than a `for`
# loop, for the same reason `apply_summaries!` and `_fill_rates!` do it: the
# summaries are DISTINCT closure types, so a plain loop infers the loop variable
# as their union and dispatches at run time on every one of the m calls.
@inline _fill_draw!(values, ::Tuple{}, model, data, X, m, k, rng) = nothing
@inline function _fill_draw!(values, specs::Tuple, model, data, X, m, k, rng)
    s = first(specs)
    _fill_column!(values[s.name], s.f, model, data, X, m, k, rng)
    return _fill_draw!(values, Base.tail(specs), model, data, X, m, k, rng)
end

# Split out so the per-individual loop specialises on ONE concrete summary
# function, rather than on the whole tuple's tail type.
function _fill_column!(M, f, model, data, X, m, k, rng)
    @inbounds for i in 1:m
        M[i, k] = f(model, data, X, i, rng)
    end
    return nothing
end

# =============================================================================
# Archiving and draw adapters
# =============================================================================

"""
    archive_draw(X) -> Matrix{Int8}

A compact copy of a trajectory for a post-hoc residual archive.

Trajectories are the whole cost of computing residuals after the fact, and that
cost is storage, not compute. A state space has a handful of states, so `Int8`
holds one exactly — and on the badger model (2391 individuals × 161 timepoints)
that is 385 KB per draw instead of 3.1 MB, i.e. 192 MB for 500 draws instead of
1.5 GB.

Thin aggressively: store every `n_sweeps ÷ 500`-th draw, not every sweep. 500
draws far exceeds what a uniformity test needs, and writing 385 KB every few
hundred sweeps is invisible next to a sweep that runs iFFBS over the whole
population.

Errors if a state code will not fit in an `Int8`, rather than wrapping silently.
"""
function archive_draw(X::AbstractMatrix{<:Integer})
    isempty(X) && return Matrix{Int8}(undef, size(X)...)
    mn, mx = extrema(X)
    (mn >= typemin(Int8) && mx <= typemax(Int8)) || error(
        "archive_draw: state codes span [$mn, $mx], which does not fit in Int8")
    return Int8.(X)
end

"""
    post_hoc_draws(models, archive) -> iterator

Pair an iterable of parameter draws with an archive of trajectories, giving the
`(model, X)` iterator [`trajectory_summaries`](@ref) consumes.

The trajectories are widened back to `Int` on the way through. `X` is indexed and
compared against state codes throughout, and leaving it `Int8` propagates that
narrow type into arithmetic where it is a liability rather than a saving. `Int8`
is a STORAGE format; this is where it stops being one.
"""
function post_hoc_draws(models, archive)
    length(models) == length(archive) || error(
        "post_hoc_draws: $(length(models)) parameter draws but $(length(archive)) trajectories")
    return ((m, Matrix{Int}(X)) for (m, X) in zip(models, archive))
end

"""
    aggregate_synced_draws(data, draws) -> iterator

Wrap a `(model, X)` iterator so that each draw's aggregates are rebuilt before the
summaries see it — `reset_aggregates!` then [`apply_derived_summaries!`](@ref), the
same invariant the likelihood needs.

Use this when your rate functions read `data.aggregates` (a frequency-dependent
force of infection reading a per-group infected count, say), because a
spec-derived hazard then depends on the aggregates agreeing with THIS draw's `X`,
not with whatever `X` was current when the fit ended.

Skip it when they do not: it costs one `O(m × T)` pass per draw, and
[`trajectory_summaries`](@ref) deliberately does not do it unconditionally for
that reason.
"""
function aggregate_synced_draws(data::EpidemicData, draws)
    return Base.Generator(draws) do (model, X)
        reset_aggregates!(data)
        apply_derived_summaries!(model, data, X)
        (model, X)
    end
end

# =============================================================================
# ONLINE MODE — compute residuals during the fit, never storing X
# =============================================================================

"""
    SummaryCollector(summaries, data, n_stored; rng, thin=1)

Collect summaries DURING a fit, storing only the residuals and never the
trajectories.

This is the mode for a model whose `X` cannot be archived. Storage here is
`m × n_stored` `Float64` per summary — for 2391 individuals and 500 stored draws
that is 9.5 MB per summary, against 192 MB for even an `Int8` archive of the same
draws and 1.5 GB for the raw `Int` ones. Nothing about `X` is kept.

## When you need this rather than keeping `X` in memory

Keeping the trajectories in the chain object is perfectly workable at small scale
— the cattle E. coli model is a few hundred individuals over a few hundred
timepoints, so the whole latent history is megabytes and none of this machinery is
warranted. Reach for a bounded-memory route when the product
`n_individuals × n_timepoints × n_draws` stops fitting.

The badger model is the worked case where it does: 2391 × 161 is 385 KB per draw
even as `Int8`, so a serious number of MCMC samples cannot be held in memory, and
putting them in a chain object would exhaust it long before the sampler finished.
This collector and [`BatchedArchive`](@ref) are the two routes that bound the
cost, and they differ only in what they give up (see below).

**What you give up** is the post-hoc iteration loop: inventing a new residual
later costs a full refit, because the trajectories it would have been computed
from are gone. That is the whole trade. Decide the diagnostics you want BEFORE the
run, or archive to disc instead — [`BatchedArchive`](@ref) keeps the iteration
loop at a bounded memory cost.

- `n_stored` — how many draws to keep. 500 far exceeds what a uniformity test
  needs.
- `thin` — compute every `thin`-th call, matching the reference's `iterSub`. Set
  it to `n_sweeps ÷ n_stored`; the diagnostic does not need every sweep.
- `rng` — its own stream, so residual randomization never perturbs the sampler's.

Feed it with [`collect_summaries!`](@ref) once per sweep and read the result with
[`finish`](@ref), which returns the same [`SummaryResult`](@ref) post-hoc gives.

# Example
```julia
coll = SummaryCollector((progression, exposure), data, 500;
                        rng = StableRNG(1), thin = n_sweeps ÷ 500)

for sweep in 1:n_sweeps
    # ... one Gibbs sweep, updating `model` and `X` ...
    collect_summaries!(coll, model, X)      # a no-op except every `thin`-th call
end

R = finish(coll)
```
"""
mutable struct SummaryCollector{S<:Tuple,D<:EpidemicData,R<:AbstractRNG}
    summaries::S
    data::D
    # Preallocated once, written in place: no allocation enters the sweep. This is
    # what the reference does with its `(m × n_stored)` matrices, and it is what
    # makes online mode cheap enough to leave switched on.
    values::Dict{Symbol,Matrix{Union{Float64,Missing}}}
    names::Vector{Symbol}
    n_stored::Int
    thin::Int
    rng::R
    calls::Int      # how many times `collect_summaries!` has been called
    stored::Int     # how many draws have actually been written
end

function SummaryCollector(summaries, data::EpidemicData, n_stored::Int;
                          rng::AbstractRNG=Random.default_rng(), thin::Int=1)
    specs = summaries isa Tuple ? summaries : Tuple(summaries)
    isempty(specs) && error("SummaryCollector: no summaries given")
    thin >= 1 || error("SummaryCollector: `thin` must be at least 1, got $thin")
    n_stored >= 1 || error("SummaryCollector: `n_stored` must be at least 1, got $n_stored")

    names = Symbol[s.name for s in specs]
    length(unique(names)) == length(names) || error(
        "SummaryCollector: summary names must be unique, got $names")

    values = Dict{Symbol,Matrix{Union{Float64,Missing}}}(
        n => Matrix{Union{Float64,Missing}}(missing, data.n_individuals, n_stored)
        for n in names)

    return SummaryCollector(specs, data, values, names, n_stored, thin, rng, 0, 0)
end

"""
    collect_summaries!(collector, model, X) -> Bool

Offer one sweep's `(model, X)` to a [`SummaryCollector`](@ref). Returns whether
this call actually stored a draw.

A no-op except on every `thin`-th call, and once `n_stored` draws are in it stops
entirely — so leaving this in the sweep loop past the end of the schedule costs
one integer increment and a comparison.

The trajectory is read and discarded; nothing about it is retained, which is the
whole point of this mode.
"""
function collect_summaries!(c::SummaryCollector, model, X)
    c.calls += 1
    (c.calls % c.thin == 0) || return false
    c.stored >= c.n_stored && return false

    m = c.data.n_individuals
    size(X, 2) == m || error(
        "collect_summaries!: X has $(size(X, 2)) individuals but `data` has $m")
    c.stored += 1
    _fill_draw!(c.values, c.summaries, model, c.data, X, m, c.stored, c.rng)
    return true
end

"""
    finish(collector) -> SummaryResult

Close a [`SummaryCollector`](@ref) and return its [`SummaryResult`](@ref) — the
same type post-hoc computation gives, so everything downstream (plots, uniformity
tests, coverage) is identical whichever mode produced it.

Trims to the number of draws actually stored, so a run that ended early does not
report a block of unfilled columns as genuine non-coverage. That distinction
matters: coverage is a diagnostic in its own right, and padding it would make an
aborted run look like a residual that silently dropped half its population.
"""
function finish(c::SummaryCollector)
    n = c.stored
    n == 0 && error("finish: the collector stored no draws (was `collect_summaries!` called?)")

    values = Dict{Symbol,Matrix{Union{Float64,Missing}}}(
        name => c.values[name][:, 1:n] for name in c.names)
    kinds = Dict{Symbol,Symbol}(s.name => s.kind for s in c.summaries)
    coverage = Dict{Symbol,Float64}(
        name => count(!ismissing, values[name]) / length(values[name]) for name in c.names)

    return SummaryResult(values, kinds, coverage, copy(c.names), n)
end

# =============================================================================
# DISC ARCHIVING — bounded memory, and the post-hoc iteration loop preserved
# =============================================================================

"""
    BatchedArchive(dir, n_individuals; batch_size=50, prefix="draws")

Archive trajectories to disc in batches, so neither the fit nor the residual
computation ever holds more than one batch in memory.

This is the route for a model whose `X` is too big to keep in RAM — at small
scale (the cattle model, say) just holding the draws is simpler and fine. Batching
bounds the resident cost at `batch_size × per-draw size` — 19 MB at the default 50
for the badger model — regardless of how many draws are archived in total, and
regardless of how many are later read back.

**Why archive at all, rather than [`SummaryCollector`](@ref)?** Because it keeps
the post-hoc iteration loop. A new diagnostic idea then costs one pass over the
archive instead of a full refit — which, for work whose stated goal is diagnosing
misspecification via residuals, is the loop worth protecting. The collector is
cheaper in storage; the archive is cheaper in researcher time. Both consume the
identical [`TrajectorySummary`](@ref), so this is an operational choice, not a
modelling one.

States are stored as `Int8` (see [`archive_draw`](@ref)) through Julia's built-in
serializer, so there is no dependency to add. Each batch is one file
`<prefix>-<index>.jls` in `dir`, which is created if it does not exist.

Feed it with [`archive_push!`](@ref) and close it with [`archive_close!`](@ref) —
**the close is not optional**, or the final partial batch is never written. Read
it back lazily with [`archived_draws`](@ref).

# Example
```julia
arc = BatchedArchive("badger_archive", n_individuals; batch_size=50)
for sweep in 1:n_sweeps
    # ... one Gibbs sweep ...
    sweep % thin == 0 && archive_push!(arc, model, X)
end
archive_close!(arc)

# later, in a fresh session — one batch resident at a time
R = trajectory_summaries((progression,), data, archived_draws("badger_archive"))
```
"""
mutable struct BatchedArchive
    dir::String
    prefix::String
    batch_size::Int
    n_individuals::Int
    # The resident buffer: at most `batch_size` draws, flushed and cleared. This is
    # the only place trajectories live in memory, and its size is fixed.
    buffer::Vector{Tuple{Any,Matrix{Int8}}}
    n_batches::Int
    n_draws::Int
    closed::Bool
end

function BatchedArchive(dir::AbstractString, n_individuals::Int;
                        batch_size::Int=50, prefix::AbstractString="draws")
    batch_size >= 1 || error("BatchedArchive: `batch_size` must be at least 1, got $batch_size")
    mkpath(dir)
    return BatchedArchive(String(dir), String(prefix), batch_size, n_individuals,
                          Tuple{Any,Matrix{Int8}}[], 0, 0, false)
end

# Zero-padded so a plain filename sort is also the write order.
_batch_path(a::BatchedArchive, k::Int) = joinpath(a.dir, string(a.prefix, "-", lpad(k, 5, "0"), ".jls"))

"""
    archive_push!(archive, model, X) -> Int

Add one draw to a [`BatchedArchive`](@ref), flushing to disc when the batch fills.
Returns the total number of draws archived so far.

`X` is converted to `Int8` here (see [`archive_draw`](@ref)), so the caller's
matrix is neither retained nor modified — the sampler may go on mutating it in
place, as iFFBS does.

Thin BEFORE calling this, not inside it: the archive stores what it is given.
"""
function archive_push!(a::BatchedArchive, model, X)
    a.closed && error("archive_push!: the archive is closed")
    size(X, 2) == a.n_individuals || error(
        "archive_push!: X has $(size(X, 2)) individuals, expected $(a.n_individuals)")

    push!(a.buffer, (model, archive_draw(X)))
    a.n_draws += 1
    length(a.buffer) >= a.batch_size && _flush_batch!(a)
    return a.n_draws
end

function _flush_batch!(a::BatchedArchive)
    isempty(a.buffer) && return a
    a.n_batches += 1
    open(_batch_path(a, a.n_batches), "w") do io
        Serialization.serialize(io, a.buffer)
    end
    empty!(a.buffer)          # the whole point: nothing accumulates in memory
    return a
end

"""
    archive_close!(archive) -> archive

Flush the final partial batch and close a [`BatchedArchive`](@ref).

**Call this.** Without it, up to `batch_size` draws are left sitting in the buffer
and never written — a silent partial loss at the end of every run. Safe to call
more than once.
"""
function archive_close!(a::BatchedArchive)
    a.closed && return a
    _flush_batch!(a)
    a.closed = true
    return a
end

Base.length(a::BatchedArchive) = a.n_draws

function Base.show(io::IO, a::BatchedArchive)
    print(io, "BatchedArchive(\"", a.dir, "\"): ", a.n_draws, " draws in ", a.n_batches,
              " batch", a.n_batches == 1 ? "" : "es",
              a.closed ? "" : " (OPEN — call archive_close!)")
end

"""
    archived_draws(dir; prefix="draws") -> iterator

Lazily read a [`BatchedArchive`](@ref) back as the `(model, X)` iterator
[`trajectory_summaries`](@ref) consumes.

**One batch is resident at a time.** The iterator deserialises a batch, yields its
draws, and drops it before opening the next — so a 10 GB archive is traversable in
`batch_size × 385 KB` of memory. This is what makes post-hoc residuals possible at
all on a model whose trajectories cannot all be held at once.

Trajectories are widened from `Int8` back to `Int` on the way out; `Int8` is a
storage format, and this is where it stops being one.

Batches are read in filename order, which is the order they were written — the
batch index is zero-padded for exactly that reason.
"""
function archived_draws(dir::AbstractString; prefix::AbstractString="draws")
    isdir(dir) || error("archived_draws: no such directory: $dir")
    files = sort!(filter(f -> startswith(f, prefix * "-") && endswith(f, ".jls"),
                         readdir(dir)))
    isempty(files) && error(
        "archived_draws: no batches matching \"$prefix-*.jls\" in $dir " *
        "(was `archive_close!` called?)")

    # A flattening generator: each batch is deserialised, yielded from, then
    # dropped. Nothing holds a reference to a batch once its draws are consumed,
    # which is what bounds the memory.
    return Iterators.flatten(
        Iterators.map(files) do f
            batch = open(Serialization.deserialize, joinpath(dir, f))
            [(model, Matrix{Int}(X)) for (model, X) in batch]
        end)
end
