# The model specification: which states exist, which transitions are allowed,
# and the rate of each.
#
# A rate is `(model, data, i, t) -> probability`. Anything it needs beyond the
# parameters it reads off `data`, so the package places no constraint on what
# a rate may depend on.

"""
    TransitionSpec(states, transitions, rate_fns, auto_self)

The allowed state transitions of a model and their rates. `states` lists the state
names in the order that fixes their integer encoding (state `k` in a trajectory
`X` means `states[k]`). `transitions[k]` is the `(from, to)` pair for
`rate_fns[k]`, each rate a function `(model, data, i, t) -> probability`.

`auto_self`: whether each state's self-transition is filled in automatically with
the leftover probability mass, so the user only declares the transitions that
actually move an individual.

`rate_fns` is a `Tuple`, and the type is parameterised on it, so each rate's own
concrete function type is known. That is a performance decision, not a style one:
as a `Vector{Function}` every rate call in `transition_matrix_at` — the hub of the
whole package, called for every individual at every timepoint, would be a runtime
dispatch returning `Any`, and the arithmetic on the result would box.

## The `coupling` view (survival-free rates)

`coupling` holds a second, survival-free view of the transitions, used only by the
coupling term (`make_rest_contribution`). It exists to fix a subtle bug: `@survival`
scales every live transition by `survival` and adds `* -> death` rows, so the main
`rate_fns` compute `survival * infection`. But the coupling asks "how does the
FOCAL change a neighbour's move probability", and the focal influences only the
INFECTION factor, never the neighbour's own survival. Scoring the neighbour through
the survival-scaled probability lets a small `survival` destroy the infection signal
— when `survival * infection` underflows to `0.0`, every focal candidate is scored
identically and the coupling goes flat, silently degenerating the chain.

So `@survival` stashes the UN-scaled infection/progression transitions (no death
rows) here, and the coupling evaluates neighbours against these bare rates. Survival
never enters the coupling, so it cannot annihilate the signal. When there is no
`@survival`, `coupling === nothing` and the coupling falls back to the full spec
(correct, since there is no survival factor to strip).

## Shared step values

Several rates out of one `(i, t)` often need the same quantities: a death
probability that scales every live move, a hazard built from covariates. Written
rate by rate, each is recomputed once per rate that uses it, and under
forward-mode AD every repeat carries the full set of partials.

`shared`, when given, is a function `(model, data, i, t) -> value` evaluated
once per `(i, t)`; every rate is then called as `rate(model, data, i, t, shared)`
and reads what it needs from that value (typically a `NamedTuple`). Like a rate,
it may read the parameters, the data and `(i, t)`. Where the package re-adds the
focal individual to the aggregates before evaluating a rate (the latent
sampler's coupling term), `shared` is recomputed after that adjustment, so one
that reads an aggregate is still exact there.

Build one with [`@transitions`](@ref) rather than calling this directly.
"""
struct TransitionSpec{RF<:Tuple,C,S}
    states::Vector{Symbol}
    transitions::Vector{Tuple{Symbol,Symbol}}
    rate_fns::RF
    auto_self::Bool
    coupling::C            # a survival-free TransitionSpec, or `nothing`
    shared::S              # `(model, data, i, t) -> values` for every rate, or `nothing`
end
# Backward-compatible constructors: no separate coupling view (coupling === nothing
# means "use the full spec", which is correct whenever there is no survival
# factor), and no shared step values.
TransitionSpec(states, transitions, rate_fns, auto_self) =
    TransitionSpec(states, transitions, rate_fns, auto_self, nothing, nothing)
TransitionSpec(states, transitions, rate_fns, auto_self, coupling) =
    TransitionSpec(states, transitions, rate_fns, auto_self, coupling, nothing)

# Rates may be written at three levels of sugar, in decreasing order of brevity:
#
#   S -> I = infection_func                      # a bare name: called for you
#   S -> I = infection_func * survival_func      # composed with arithmetic
#   S -> I = 1 / model.m                         # a bare expression
#   S -> I = (model, data, i, t) -> ...          # an explicit lambda (the fallback)
#
# All four end up as a function `(model, data, i, t) -> probability`. The bare-name
# and composition forms work by rewriting every bare rate-function name in the
# expression into a call, so `f * g` becomes
# `(model,data,i,t) -> f(model,data,i,t) * g(model,data,i,t)`.
#
# A name is treated as a rate function to call if it is not a local of the rate
# signature (`model`, `data`, `i`, `t`), not a known arithmetic operator, and not
# a dotted access such as `model.m`. This is a heuristic; a user whose model
# defeats it can always drop to the explicit-lambda fallback.

const _RATE_ARGS = (:model, :data, :i, :t)
# With a `@shared` line, every rate also receives the step's shared values.
const _SHARED_RATE_ARGS = (:model, :data, :i, :t, :shared)

# Operators and functions that must not be rewritten into rate calls.
const _RATE_SAFE_CALLS = Set{Symbol}([
    :+, :-, :*, :/, :^, :\,
    :exp, :log, :log1p, :expm1, :sqrt, :abs, :inv,
    :min, :max, :clamp, :ifelse,
    :zero, :one, :float, :Float64,
])

# Rewrite bare rate-function names into calls with the rate signature `args`.
_sugar_rate(x, args = _RATE_ARGS) = x
function _sugar_rate(x::Symbol, args = _RATE_ARGS)
    (x in args || x in _RATE_SAFE_CALLS) && return x
    # A bare name in rate position: call it with the rate signature.
    return :($x($(args...)))
end
function _sugar_rate(ex::Expr, args = _RATE_ARGS)
    # `model.m`, `data.aggregates[...]` etc: leave the field/index path alone.
    ex.head === :. && return ex
    sub(x) = _sugar_rate(x, args)
    if ex.head === :call
        f = ex.args[1]
        # An explicit call like `infection_func(model, data, i, t)` stays as written;
        # only its ARGUMENTS get the sugar treatment (they may contain bare names).
        if f isa Symbol && !(f in _RATE_SAFE_CALLS)
            return ex
        end
        # An arithmetic/known call: sugar the arguments, keep the operator.
        return Expr(:call, f, map(sub, ex.args[2:end])...)
    end
    if ex.head === :ref
        # `arr[idx]`: sugar the indices but not the array name itself.
        return Expr(:ref, ex.args[1], map(sub, ex.args[2:end])...)
    end
    return Expr(ex.head, map(sub, ex.args)...)
end

# Wrap a rate into the full signature. An explicit lambda is used exactly as
# written (the power-user fallback); anything else is sugared.
function _wrap_rate_expr(rate, args = _RATE_ARGS)
    (rate isa Expr && rate.head == :->) && return rate
    return :(($(args...),) -> ($(_sugar_rate(rate, args))))
end

# The source of a transition line is either one state (`S -> I`) or several
# (`(S,E,I) -> D`), the latter expanding to one transition per source.
_source_states(from::Symbol) = (from,)
function _source_states(from::Expr)
    from.head === :tuple || error("Transition source must be a state or a tuple of states, got: $from")
    all(x -> x isa Symbol, from.args) || error("Grouped transition source must be bare state names, got: $from")
    return Tuple(from.args)
end

# Parse the body of `@transitions`. Each line is `A -> B = rate`, which Julia
# reads as `A -> (B = rate)`: an arrow whose body block holds the assignment. An
# `@survival` line is pulled out and applied to every transition (see below).
function _parse_transition_block(block)
    lines = [x for x in block.args if !(x isa LineNumberNode)]
    transitions = Tuple{Symbol,Symbol,Any}[]
    survival_expr = nothing
    death_state = nothing

    for line in lines
        if line isa Expr && line.head == :macrocall && line.args[1] == Symbol("@survival")
            length(line.args) >= 3 || error("@survival needs at least a survival expression")
            survival_expr = line.args[3]
            if length(line.args) >= 4
                kw = line.args[4]
                if kw isa Expr && kw.head == :(=) && kw.args[1] == :death
                    # `death=:D` gives a QuoteNode; `death=D` a bare Symbol. Accept
                    # both and carry the state name as a plain Symbol.
                    death_state = kw.args[2] isa QuoteNode ? kw.args[2].value : kw.args[2]
                end
            end
            (death_state isa Symbol) ||
                error("@survival requires `death=:State` naming the death state, got: $(death_state)")
            continue
        end

        (line isa Expr && line.head == :(->)) || error("Transition line must look like `A -> B = rate`")
        from = line.args[1]
        rhs = line.args[2]
        assign = rhs isa Expr && rhs.head == :block ? first(filter(x -> !(x isa LineNumberNode), rhs.args)) : rhs
        (assign isa Expr && assign.head == :(=)) || error("Transition line must include `= rate`")
        to = assign.args[1]
        rate = assign.args[2]
        # A grouped source `(S,E,I) -> D = rate` expands to one transition per
        # source state, each with the same rate.
        for f in _source_states(from)
            push!(transitions, (f, to, rate))
        end
    end

    # The bare (survival-free) transitions: the infection/progression moves exactly
    # as written, before any survival scaling and before the death rows are added.
    # This is what the coupling term uses (see `TransitionSpec`'s `coupling` field).
    # `nothing` when there is no `@survival` (then the full spec IS survival-free).
    bare_transitions = survival_expr === nothing ? nothing : copy(transitions)

    # `@survival p death=:D` means every non-death transition is conditional on
    # surviving the step (its rate is scaled by `p`), and every live state gains a
    # transition to the death state with the leftover probability `1 - p`.
    if survival_expr !== nothing
        transitions = map(transitions) do (from, to, rate)
            to == death_state ? (from, to, rate) : (from, to, :(($survival_expr) * ($rate)))
        end
        # Every live state can die, not just the ones that happen to appear as a
        # source above, an absorbing state like `I` (which only ever appears as a
        # destination) still needs its `I -> D`. So collect the live states from
        # both sides of every transition.
        live_states = Symbol[]
        for (from, to, _) in transitions
            for s in (from, to)
                s == death_state && continue
                s in live_states || push!(live_states, s)
            end
        end
        for s in live_states
            if !any(t -> t[1] == s && t[2] == death_state, transitions)
                push!(transitions, (s, death_state, :(1 - ($survival_expr))))
            end
        end
    end

    return transitions, bare_transitions
end

"""
    @survival expr death=:State

Declare, inside a [`@transitions`](@ref) block, that every step is conditional on
survival: each non-death transition's rate is scaled by `expr`, and **every** live
state gains a transition to `death` with the remaining probability `1 - expr` —
including states that never appear as the source of a declared transition.

`expr` is a rate, so it takes the usual sugar: a bare function name, a composition,
or an explicit `(model, data, i, t)` lambda.

Only meaningful inside `@transitions`, which consumes it while parsing; it expands
to nothing on its own.

# Example
```julia
# S -> E -> I, each step conditional on surviving, and any live state may die
spec = @transitions [:S, :E, :I, :D] begin
    @survival siler_survival death=:D
    S -> E = infection
    E -> I = progression
end
# gives (S,E), (E,I), (S,D), (E,D), (I,D): note the I -> D you never wrote
```
"""
macro survival(args...)
    return nothing
end

"""
    @transitions :individual [:auto_self] begin
        A -> B = rate
        ...
    end

Declare a model's states, transitions and rates, returning a
[`TransitionSpec`](@ref). Each line reads `A -> B = rate`, where `rate` is either
a bare expression referring to `model`, `data`, `i`, `t` by name, or an explicit
`(model, data, i, t) -> ...` lambda.

`:auto_self` fills each state's self-transition with the leftover probability
mass, so only the transitions that move an individual need declaring. The state
encoding is the order the state names first appear (state `k` in `X` means
`spec.states[k]`).

# Example
```julia
spec = @transitions :individual :auto_self begin
    S -> I = infection_func(model, data, i, t)
    I -> S = recovery_func(model, data, i, t)
end
```

A `@shared f` line computes `f(model, data, i, t)` once per `(i, t)` and hands
it to every rate as a fifth argument, `shared` (see [`TransitionSpec`](@ref)).
Use it when several rates need the same quantities:

```julia
step(model, data, i, t) = (; die = 1 - exp(-model.mu * data.age[t, i]))
spec = @transitions [:S, :I, :D] begin
    @shared step
    S -> I = (model, data, i, t, shared) -> (1 - shared.die) * model.beta
    S -> D = (model, data, i, t, shared) -> shared.die
    I -> D = (model, data, i, t, shared) -> shared.die
end
```

A bare rate name `r` is then called as `r(model, data, i, t, shared)`.
"""
macro transitions(args...)
    rest = collect(args)
    auto_self = true          # the common case; opt out with :no_auto_self

    while !isempty(rest) && rest[1] isa QuoteNode
        tag = rest[1].value
        if tag == :individual
            # The only supported style; accepted for explicitness.
        elseif tag == :auto_self
            auto_self = true
        elseif tag == :no_auto_self
            auto_self = false
        else
            error("Unsupported @transitions tag: $tag (expected :individual, :auto_self or :no_auto_self)")
        end
        rest = rest[2:end]
    end

    # An optional leading state_space fixes the state NUMBERING (state k in `X`
    # means state_space[k]). Without it the order states first appear in the
    # transitions is used.
    state_space_expr = nothing
    if length(rest) == 2
        state_space_expr = rest[1]
        rest = rest[2:end]
    end
    length(rest) == 1 || error("@transitions expects [tags] [state_space] begin ... end")
    block = rest[1]
    block isa Expr && block.head == :block || error("@transitions body must be begin...end")

    block, shared_expr = _take_shared_line(block)
    rate_args = shared_expr === nothing ? _RATE_ARGS : _SHARED_RATE_ARGS
    shared_val = shared_expr === nothing ? :nothing : esc(shared_expr)

    trs, bare = _parse_transition_block(block)
    seen_states = unique(vcat(Symbol[t[1] for t in trs], Symbol[t[2] for t in trs]))
    trans_pairs = [:(($(QuoteNode(t[1])), $(QuoteNode(t[2])))) for t in trs]
    rates = [_wrap_rate_expr(t[3], rate_args) for t in trs]

    states_expr = state_space_expr === nothing ?
        :(Symbol[$(map(QuoteNode, seen_states)...)]) :
        :($(_check_state_space)(collect(Symbol, $(esc(state_space_expr))),
                                Symbol[$(map(QuoteNode, seen_states)...)]))

    # The survival-free coupling view, built over the same state encoding so its
    # indices line up. Only the bare infection/progression moves; `auto_self` gives
    # the leftover mass as the "stay" probability (e.g. S->S = 1 - foi), which is
    # exactly what the coupling scores a non-moving neighbour against. `nothing`
    # when there was no `@survival`.
    coupling_expr = if bare === nothing
        :nothing
    else
        bare_pairs = [:(($(QuoteNode(t[1])), $(QuoteNode(t[2])))) for t in bare]
        bare_rates = [_wrap_rate_expr(t[3], rate_args) for t in bare]
        :(TransitionSpec($states_expr,
                         Tuple{Symbol,Symbol}[$(bare_pairs...)],
                         ($(map(esc, bare_rates)...),),
                         true, nothing, $shared_val))
    end

    quote
        TransitionSpec(
            $states_expr,
            Tuple{Symbol,Symbol}[$(trans_pairs...)],
            ($(map(esc, rates)...),),
            $(auto_self),
            $coupling_expr,
            $shared_val,
        )
    end
end

# Pull an `@shared f` line out of a `@transitions` body, returning the remaining
# block and `f` (or `nothing`).
function _take_shared_line(block)
    shared = nothing
    kept = Any[]
    for x in block.args
        if x isa Expr && x.head == :macrocall && x.args[1] == Symbol("@shared")
            shared === nothing || error("@transitions: more than one @shared line")
            vals = [a for a in x.args[2:end] if !(a isa LineNumberNode)]
            length(vals) == 1 || error("@shared takes one function, `@shared f`")
            shared = vals[1]
        else
            push!(kept, x)
        end
    end
    Expr(:block, kept...), shared
end

# A user-supplied state_space must cover every state the transitions mention; it
# may legitimately contain extra states (e.g. an absorbing state with no declared
# transitions out of it).
function _check_state_space(declared::Vector{Symbol}, seen::Vector{Symbol})
    missing_states = setdiff(seen, declared)
    isempty(missing_states) || error(
        "state_space $(declared) is missing state(s) $(missing_states) used in the transitions")
    return declared
end
