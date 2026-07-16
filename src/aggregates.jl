# User-declared aggregates and their REVERSIBLE updates.
#
# This file is where the package's central design rule lives: it never assumes
# what arrays (if any) the user wants tracked, or what they mean. The user
# declares whatever arrays they like and, for each, an update that can be run
# FORWARDS or IN REVERSE. The package only ever allocates the storage and calls
# the user's functions.
#
# Reversibility is what makes the latent sampler both correct and cheap. To
# resample individual `i`, iFFBS reverses `i`'s own contribution out of the
# aggregates, runs the forward filter / backward sample (so `i` sees leave-one-out
# statistics — the counts exclude itself), then re-applies `i`'s new contribution.
# The aggregates therefore stay exactly consistent with `X` at all times, without
# ever being rebuilt from scratch.
#
# A derived summary has the signature
#
#     (model, data, X, s, i, t; reverse=false) -> nothing
#
# where `s` is the state being applied (or reversed) for individual `i` at time
# `t`. It mutates `data.aggregates` in place. Write one with `@aggregate` (sugar),
# or by hand — by hand you must supply the reverse yourself and honour `reverse`.

"""
    AggregateSpec(name, eltype, dims_expr, summary)

One user-declared aggregate: the array's `name`, its element type and dimensions
(so the package can allocate it), and the reversible `summary` that updates it.

Built by [`@aggregate`](@ref); you should not need to construct one directly.
"""
struct AggregateSpec{F}
    name::Symbol
    eltype::Type
    dims::Tuple
    summary::F
end

# Inside an aggregate expression the user writes `state` to mean "the state being
# applied"; the generated lambda binds it as `s`.
_replace_state_sym(x) = x
_replace_state_sym(x::Symbol) = x === :state ? :s : x
_replace_state_sym(ex::Expr) = Expr(ex.head, map(_replace_state_sym, ex.args)...)

# The inverse of each supported update operator. This is what lets one written
# expression generate both the forward and the reverse update.
_invert_op(op) = op === :(+=) ? :(-=) :
                 op === :(-=) ? :(+=) :
                 op === :(*=) ? :(/=) :
                 op === :(/=) ? :(*=) :
                 error("Unsupported operator in a derived summary: $op (use +=, -=, *= or /=)")

# Rewrite a bare aggregate name `n_infected[...]` into `data.aggregates[:n_infected][...]`,
# so the user never writes the container lookup. Only names declared in the same
# `@aggregate` block are rewritten.
_sugar_agg(x, names) = x
function _sugar_agg(ex::Expr, names)
    if ex.head === :ref && ex.args[1] isa Symbol && ex.args[1] in names
        arr = ex.args[1]
        idx = map(a -> _sugar_agg(a, names), ex.args[2:end])
        return Expr(:ref, :(data.aggregates[$(QuoteNode(arr))]), idx...)
    end
    return Expr(ex.head, map(a -> _sugar_agg(a, names), ex.args)...)
end

# Rewrite `state == :I` into `state == <index of :I>`, so the user may compare
# states by NAME. Comparing by number still works: declaring a state_space is what
# lets user and package agree on the numbering, and the symbol form is sugar over
# exactly that numbering.
_sugar_state_syms(x, ss) = x
function _sugar_state_syms(ex::Expr, ss)
    if ex.head === :call && length(ex.args) == 3 && ex.args[1] in (:(==), :(!=))
        lhs, rhs = ex.args[2], ex.args[3]
        rhs isa QuoteNode && (rhs = :($(EpidemicTrajectories._state_code)($ss, $(rhs))))
        lhs isa QuoteNode && (lhs = :($(EpidemicTrajectories._state_code)($ss, $(lhs))))
        return Expr(:call, ex.args[1], _sugar_state_syms(lhs, ss), _sugar_state_syms(rhs, ss))
    end
    if ex.head === :call && ex.args[1] === :in
        # `state in (:E, :I)`
        return Expr(:call, :in, _sugar_state_syms(ex.args[2], ss),
                    Expr(:tuple, map(a -> a isa QuoteNode ? :(_state_code($ss, $a)) : a,
                                     ex.args[3] isa Expr && ex.args[3].head === :tuple ? ex.args[3].args : [ex.args[3]])...))
    end
    return Expr(ex.head, map(a -> _sugar_state_syms(a, ss), ex.args)...)
end

"""
    _state_code(state_space, s)

The integer code of state `s` in `state_space`. A `Symbol` is looked up by name; an
integer is returned as-is (so `state == :I` and `state == 2` both work — declaring
a `state_space` is what makes the two agree).
"""
_state_code(ss, s::Integer) = s
function _state_code(ss, s::Symbol)
    idx = findfirst(==(s), ss)
    idx === nothing && error("State :$s is not in the state space $(collect(ss))")
    idx
end
_state_code(ss, s::QuoteNode) = _state_code(ss, s.value)

# Turn one update line into a reversible derived-summary lambda. Accepts:
#
#   arr[...] += contribution                  # a bare update
#   if cond; arr[...] += 1; end               # a guarded update
#   count(cond, arr[...])                     # count how many satisfy cond
#
# Two forms are deliberately NOT accepted, because Julia's parser gets to them
# first: `arr[...] += 1 if cond` (postfix `if` is not Julia syntax) and
# `arr[...] += 1, cond` (which parses as `arr[...] += (1, cond)`). Guard with a
# real `if` block, or fold the condition into the contribution
# (`arr[...] += (state == :I)`).
function _aggregate_line_to_lambda(line, names, ss_expr)
    cond = :(true)
    upd = line
    if line isa Expr && line.head == :if
        # `if cond; update; end` — the guard is the first argument, the body second.
        cond = line.args[1]
        body = line.args[2]
        stmts = body isa Expr && body.head === :block ?
            filter(x -> !(x isa LineNumberNode), body.args) : [body]
        length(stmts) == 1 ||
            error("a guarded @aggregate line must contain exactly one update, got: $line")
        upd = stmts[1]
    elseif line isa Expr && line.head == :call && line.args[1] == :count
        if length(line.args) == 3
            cond = line.args[2]
            target = line.args[3]
            upd = Expr(:(+=), target, 1)
        else
            error("Unsupported count(...) form in @aggregate: expected count(cond, arr[...])")
        end
    end

    (upd isa Expr && upd.head in (:(+=), :(-=), :(*=), :(/=))) ||
        error("@aggregate update must be `arr[...] op= value` (op one of +=, -=, *=, /=), got: $upd")

    upd = _sugar_agg(_replace_state_sym(upd), names)
    cond = _sugar_agg(_replace_state_sym(cond), names)
    upd = _sugar_state_syms(upd, ss_expr)
    cond = _sugar_state_syms(cond, ss_expr)

    op = upd.head
    inv_op = _invert_op(op)
    rev = Expr(inv_op, upd.args[1], upd.args[2])

    return :((model, data, X, s, i, t; reverse=false) -> begin
        if $cond
            if reverse
                $rev
            else
                $upd
            end
        end
        nothing
    end)
end

"""
    @aggregate [state_space] begin
        @array name Type (dims...)
        name[idx...] += contribution
        ...
    end

Declare the arrays to track during the latent update, together with their
**reversible** updates. Returns a vector of [`AggregateSpec`](@ref)s to pass to
[`epidemic_data`](@ref) as `aggregates`; the package allocates the storage, so the
array and its update are declared in one place.

- `@array name Type (dims...)` declares an array; refer to it by bare name in the
  updates and in your rate functions (`data.aggregates[:name]`).
- Each update line generates both a forward update and its reverse (`+=` ↔ `-=`,
  `*=` ↔ `/=`), which is what lets the sampler remove and re-apply an individual's
  contribution.
- Inside a line you may refer to `model`, `data`, `X`, `i`, `t`, and `state` (the
  state being applied or reversed). Compare states by name (`state == :I`) when a
  `state_space` is given, or by number (`state == 2`) always.
- A line may be a bare update, a guarded update (`if cond; update; end`), or
  `count(cond, arr[...])`. Often the condition folds naturally into the
  contribution instead: `n_infected[...] += (state == :I)`.

The package attaches no meaning to these arrays — declare whatever your rates
need.

# Example
```julia
aggs = @aggregate state_space begin
    @array n_infected Int (n_pens, n_timepoints)
    n_infected[data.group[i], t] += (state == :I)
end
```

For an aggregate the macro cannot express, write the summary by hand: any function
`(model, data, X, s, i, t; reverse=false)` that honours `reverse` will do — pass it
as an [`AggregateSpec`](@ref) with your own storage.
"""
macro aggregate(args...)
    rest = collect(args)
    ss_expr = length(rest) == 2 ? rest[1] : :(Symbol[])
    block = rest[end]
    block isa Expr && block.head == :block || error("@aggregate body must be begin...end")

    lines = [x for x in block.args if !(x isa LineNumberNode)]

    # First pass: the `@array` declarations tell us the names (so updates can use
    # bare names) and the storage to allocate.
    decls = Tuple{Symbol,Any,Any}[]
    updates = Any[]
    for line in lines
        if line isa Expr && line.head === :macrocall && line.args[1] === Symbol("@array")
            parts = filter(x -> !(x isa LineNumberNode), line.args[2:end])
            length(parts) == 3 || error("@array needs: @array name Type (dims...)")
            push!(decls, (parts[1], parts[2], parts[3]))
        else
            push!(updates, line)
        end
    end
    isempty(decls) && error("@aggregate needs at least one `@array name Type (dims...)` declaration")

    names = Set(Symbol[d[1] for d in decls])
    lams = [_aggregate_line_to_lambda(u, names, ss_expr) for u in updates]

    # One AggregateSpec per declared array. Every update runs for every array (a
    # guard inside the update decides whether it applies), which keeps the common
    # one-array-one-update case trivial and multi-array blocks correct.
    specs = [:(AggregateSpec($(QuoteNode(d[1])), $(esc(d[2])), Tuple($(esc(d[3]))), nothing)) for d in decls]

    quote
        local _summaries = Function[$(map(esc, lams)...)]
        local _specs = AggregateSpec[$(specs...)]
        AggregateDeclaration(_specs, _summaries)
    end
end

"""
    AggregateDeclaration(specs, summaries)

What [`@aggregate`](@ref) returns: the arrays to allocate and the reversible
updates that maintain them. Pass it to [`epidemic_data`](@ref) as `aggregates`.
"""
struct AggregateDeclaration
    specs::Vector{AggregateSpec}
    summaries::Vector{Function}
end

"""
    allocate_aggregates(decl::AggregateDeclaration) -> Dict{Symbol,Any}

Allocate the storage declared by [`@aggregate`](@ref).
"""
function allocate_aggregates(decl::AggregateDeclaration)
    d = Dict{Symbol,Any}()
    for s in decl.specs
        d[s.name] = zeros(s.eltype, s.dims...)
    end
    d
end

"""
    reset_aggregates!(data)

Zero every aggregate array. The package does not know what the aggregates
represent; it just clears their storage before a fresh sum-up.

Use together with [`apply_derived_summaries!`](@ref) to establish the invariant
that the aggregates agree with a given `X` — see that function's docstring.
"""
function reset_aggregates!(data)
    for (_, v) in data.aggregates
        v isa AbstractArray && fill!(v, zero(eltype(v)))
    end
    nothing
end

"""
    apply_derived_summaries!(model, data, X)

Forward-apply every derived summary over the whole population and window, filling
the aggregates from `X`.

**The invariant**: the aggregates must always agree with the current `X`. Establish
it once, on the initial `X`, with

```julia
reset_aggregates!(data)
apply_derived_summaries!(model, data, X)
```

before the first likelihood evaluation. Thereafter the latent sampler maintains it
incrementally (reversing and re-applying each individual's contribution), so
neither this function nor a rebuild is needed again — the likelihood and the rate
functions only ever READ the aggregates.
"""
function apply_derived_summaries!(model, data, X)
    for i in 1:data.n_individuals
        start_sampling, end_sampling = data.sampling_period[i]
        for t in start_sampling:end_sampling
            for ds in data.derived_summaries
                ds(model, data, X, X[t, i], i, t)
            end
        end
    end
    nothing
end
