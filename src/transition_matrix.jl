# Transition-matrix model specification.
#
# Where the functional `RateBundle` describes per-INDIVIDUAL transitions, the
# transition-matrix style describes the state-to-state transitions of a
# compartmental system directly: a list of allowed `(from, to)` transitions, each
# with a rate. An incidence/stoichiometry matrix plus a rate per transition gives
# both a simulator and a log-likelihood.
#
# TWO levels of generality (per the package's naming convention):
#
#   * `SimpleEpiTransitionMatrix` — each transition's rate is a per-capita
#     per-step PROBABILITY given as a function of `(pars, counts, t)` only. This
#     is the population-level, exchangeable, count-sufficient case: the
#     likelihood is a product of binomials (the "chain binomial"). Good for
#     aggregate compartmental models.
#
#   * `EpiTransitionMatrix` — each transition's rate is the FULL functional-style
#     signature `f(pars, model, data, i, t)`, so a transition probability can
#     depend on individual identity, covariates, network/spatial structure, or
#     history. This is as general as the functional `RateBundle` style; the two
#     are different front-ends onto the same per-individual transition-matrix
#     machinery. A `SimpleEpiTransitionMatrix` is the special case where every
#     rate ignores `i` and reads only aggregate `counts`.
#
# Both can be built with the `@transitions` modelling-language macro:
#
#     @transitions SI begin
#         S -> I = (pars, counts, t) -> -expm1(-(pars.α + pars.β * counts[2]))
#         I -> S = (pars, counts, t) -> 1 / pars.m
#     end

# ---------------------------------------------------------------------------
# SimpleEpiTransitionMatrix — pure count-based rates, chain-binomial likelihood.
# ---------------------------------------------------------------------------

"""
    SimpleEpiTransitionMatrix(; state_space, transitions, rates)

A compartmental discrete-time state-transition model whose transition rates are
pure functions of the compartment counts: each element of
`rates` is `rate(pars, counts, t) -> per-capita per-step transition PROBABILITY`,
where `counts` is the current dense-state count vector and `t` the time step.
`transitions` is a vector of `(from_code, to_code)` pairs; `state_space` is the
[`StateSpace`](@ref).

This is the exchangeable, count-sufficient case — use it for aggregate
compartmental (chain-binomial) models. For transition probabilities that depend
on individual identity/covariates/network structure, use
[`EpiTransitionMatrix`](@ref) instead.

Use with [`simulate_chain_binomial`](@ref) and [`chain_binomial_loglik`](@ref).
Build directly or via the [`@transitions`](@ref) macro.
"""
Base.@kwdef struct SimpleEpiTransitionMatrix{SS<:StateSpace,Tr,Rt}
    state_space::SS
    transitions::Tr   # Vector of (from_code, to_code)
    rates::Rt         # Vector of rate functions f(pars, counts, t)
end

# Back-compat alias: the original name.
const StateTransitionModel = SimpleEpiTransitionMatrix

# ---------------------------------------------------------------------------
# EpiTransitionMatrix — general per-individual transition rates.
# ---------------------------------------------------------------------------

"""
    EpiTransitionMatrix(; state_space, transitions, rates)

A discrete-time state-transition model whose transition rates are the FULL
functional-style signature: each element of `rates` is `rate(pars, model, data,
i, t) -> per-step transition PROBABILITY` for individual `i` at time `t`, so a
transition probability can depend on individual identity, per-individual
covariates, network/spatial structure, or history — everything the functional
[`RateBundle`](@ref) style can express.

`transitions` is a vector of `(from_code, to_code)` pairs; `state_space` is the
[`StateSpace`](@ref). Because the rates carry the individual-level signature, an
`EpiTransitionMatrix` is itself a `RateBundle`: it implements
[`transition_matrix_at`](@ref), so it works directly with
[`trajectory_loglik`](@ref), [`ffbs_sweep!`](@ref), and the simulator — the same
machinery as any hand-written functional bundle. Build directly or via the
[`@transitions`](@ref) macro.
"""
Base.@kwdef struct EpiTransitionMatrix{SS<:StateSpace,Tr,Rt} <: RateBundle
    state_space::SS
    transitions::Tr   # Vector of (from_code, to_code)
    rates::Rt         # Vector of rate functions f(pars, model, data, i, t)
end

# An EpiTransitionMatrix assembles a per-individual transition matrix by placing
# each transition's probability into P[from, to] and giving each source state's
# self-transition the remaining mass (so rows sum to 1). This is the bridge that
# makes the transition-matrix front-end drive the functional machinery.
function transition_matrix_at(m::EpiTransitionMatrix, pars, model, data, i, t)
    ss = m.state_space::StateSpace
    N = nstates(ss)
    T = eltype(pars)
    P = zeros(T, N, N)
    rowsum = zeros(T, N)
    for (r, (from, to)) in enumerate(m.transitions)
        a = state_index(ss, from)
        b = state_index(ss, to)
        p = T(m.rates[r](pars, model, data, i, t))
        @inbounds P[a, b] += p
        @inbounds rowsum[a] += p
    end
    # remaining mass stays in the source state (self-transition)
    @inbounds for a in 1:N
        P[a, a] += one(T) - rowsum[a]
    end
    return P
end

# ---------------------------------------------------------------------------
# @transitions modelling-language macro.
# ---------------------------------------------------------------------------

"""
    @transitions state_space begin
        S -> I = f
        I -> S = g
        ...
    end

Modelling-language front-end for building a transition-matrix model. Each line
`A -> B = rate` declares a transition from state `A` to state `B`. The state
names `A`, `B` must be the `names` of the supplied `state_space` (a
[`StateSpace`](@ref) whose `names` are set); they are resolved to state codes
automatically.

The `rate` is written as a bare expression referring to the rate-function
arguments by name — no `-> ` lambda needed. Those arguments are `pars`,
`counts`, `t` for the default count-based style, and `pars`, `model`, `data`,
`i`, `t` for the per-individual style. An explicit lambda is still accepted for
rates that need a multi-line body.

By default this builds a [`SimpleEpiTransitionMatrix`](@ref). Pass `:individual`
as a leading argument to build an [`EpiTransitionMatrix`](@ref) instead, whose
rates can depend on individual identity, covariates, or history.

# Examples
```julia
# count-based (chain binomial): rate expressions see `pars`, `counts`, `t`
si = @transitions SI begin
    S -> I = -expm1(-(pars.α + pars.β * counts[2]))
    I -> S = 1 / pars.m
end

# per-individual: rate expressions see `pars`, `model`, `data`, `i`, `t`
si2 = @transitions :individual SI begin
    S -> I = -expm1(-(pars.α + pars.β * count_infected_penmates(data, i, t)))
    I -> S = 1 / pars.m
end
```
"""
macro transitions(args...)
    # Optional leading `:individual` / `:simple` style selector.
    style = :simple
    rest = collect(args)
    if length(rest) ≥ 1 && rest[1] isa QuoteNode
        style = rest[1].value
        rest = rest[2:end]
    end
    length(rest) == 2 || error("@transitions expects: [:style] state_space begin ... end")
    ss_expr, block = rest
    block isa Expr && block.head == :block || error("@transitions body must be a begin...end block")

    froms = Symbol[]
    tos = Symbol[]
    rate_exprs = Any[]
    for line in block.args
        line isa LineNumberNode && continue
        # `A -> B = rate` parses as Expr(:->, :A, block) where the block holds
        # `Expr(:(=), :B, rate)` — i.e. Julia reads it as `A -> (B = rate)`.
        (line isa Expr && line.head == :(->)) ||
            error("@transitions line must be `A -> B = rate`, got: $line")
        from = line.args[1]
        from isa Symbol || error("@transitions source state must be a bare name, got: $from")
        inner = line.args[2]
        # find the `B = rate` assignment inside the arrow body block
        assign = nothing
        if inner isa Expr && inner.head == :block
            for a in inner.args
                a isa LineNumberNode && continue
                assign = a
                break
            end
        else
            assign = inner
        end
        (assign isa Expr && assign.head == :(=)) ||
            error("@transitions line must be `A -> B = rate` (missing `= rate`?), got: $line")
        to = assign.args[1]
        to isa Symbol || error("@transitions destination state must be a bare name, got: $to")
        rate = assign.args[2]
        push!(froms, from)
        push!(tos, to)
        # Auto-wrap a bare rate expression in a rate function with the standard
        # argument names in scope. `I -> S = 1 / pars.m` becomes a full
        # `(pars, counts, t) -> 1 / pars.m` (simple style) or
        # `(pars, model, data, i, t) -> 1 / pars.m` (individual style). If the
        # user already wrote an explicit lambda (`= (pars, ...) -> ...`), it is
        # used unchanged, so full control is still available for rates that need
        # a multi-line body or unusual argument handling.
        push!(rate_exprs, _wrap_rate(rate, style))
    end

    quote
        local ss = $(esc(ss_expr))
        local names = ss.names
        names === nothing && error("@transitions requires a StateSpace with `names` set")
        local name_to_code = Dict(names[k] => ss.codes[k] for k in eachindex(names))
        local transitions = Tuple{Int,Int}[]
        $(map(eachindex(froms)) do k
            :(push!(transitions, (name_to_code[$(QuoteNode(froms[k]))], name_to_code[$(QuoteNode(tos[k]))])))
        end...)
        local rates = Any[$(map(esc, rate_exprs)...)]
        $(style == :individual ? :(EpiTransitionMatrix(; state_space=ss, transitions=transitions, rates=rates)) :
                                 :(SimpleEpiTransitionMatrix(; state_space=ss, transitions=transitions, rates=rates)))
    end
end

# Wrap a rate expression in a rate function unless it is already an explicit
# lambda. The injected argument names are the ones the transition-matrix
# machinery calls each rate with: `(pars, counts, t)` for the count-based simple
# style, `(pars, model, data, i, t)` for the per-individual style. So a bare
# `1 / pars.m` (referencing whichever of these it needs) becomes a complete rate
# function, while an explicit `(pars, ...) -> ...` is passed through untouched.
function _wrap_rate(rate, style::Symbol)
    # already a lambda (`args -> body`): use as written.
    rate isa Expr && rate.head == :-> && return rate
    argtuple = style == :individual ?
        Expr(:tuple, :pars, :model, :data, :i, :t) :
        Expr(:tuple, :pars, :counts, :t)
    return Expr(:->, argtuple, rate)
end

# ---------------------------------------------------------------------------
# Incidence matrix, simulation, chain-binomial likelihood.
# These operate on the `transitions`/`state_space` fields shared by BOTH the
# Simple and general transition-matrix types.
# ---------------------------------------------------------------------------

"""
    incidence_matrix(m) -> Matrix{Int}

The `nstates x ntransitions` incidence (stoichiometry) matrix `A`: column `r` has
`-1` in the row of transition `r`'s source state and `+1` in the row of its
destination state (dense-state ordering). Advancing counts by a per-transition
event vector is `counts_next = counts + A * events`. Works for any model with
`state_space` and `transitions` fields ([`SimpleEpiTransitionMatrix`](@ref) or
[`EpiTransitionMatrix`](@ref)).
"""
function incidence_matrix(m)
    ss = m.state_space
    N = nstates(ss)
    R = length(m.transitions)
    A = zeros(Int, N, R)
    for (r, (from, to)) in enumerate(m.transitions)
        A[state_index(ss, from), r] = -1
        A[state_index(ss, to), r] = 1
    end
    return A
end

"""
    simulate_chain_binomial(rng, m::SimpleEpiTransitionMatrix, pars, counts0; n_times)

Simulate a compartmental count trajectory forward `n_times` steps from an initial
count vector `counts0` (dense-state order). At each step, for each transition the
number of individuals making it is `Binomial(n_available_in_source, rate)`. When
several transitions leave the same source compartment they compete (the source is
depleted in transition order).

Returns `(counts, events)`: an `nstates x n_times` count matrix and an
`ntransitions x (n_times-1)` realized-event-count matrix.
"""
function simulate_chain_binomial(rng::AbstractRNG, m::SimpleEpiTransitionMatrix, pars, counts0; n_times)
    ss = m.state_space
    N = nstates(ss)
    A = incidence_matrix(m)
    R = length(m.transitions)
    counts = Matrix{Int}(undef, N, n_times)
    events = Matrix{Int}(undef, R, n_times - 1)
    counts[:, 1] .= counts0
    for t in 1:(n_times - 1)
        avail = copy(counts[:, t])
        ev = zeros(Int, R)
        for (r, (from, _)) in enumerate(m.transitions)
            src = state_index(ss, from)
            n_src = avail[src]
            p = clamp(m.rates[r](pars, counts[:, t], t), 0.0, 1.0)
            k = n_src > 0 ? rand(rng, Distributions.Binomial(n_src, p)) : 0
            ev[r] = k
            avail[src] -= k
        end
        events[:, t] .= ev
        counts[:, t + 1] .= counts[:, t] .+ A * ev
    end
    return counts, events
end

"""
    chain_binomial_loglik(pars, m::SimpleEpiTransitionMatrix, counts, events) -> Real

Log-likelihood of an observed compartmental trajectory under the chain-binomial
model: for each step and transition, `logpdf(Binomial(n_source, rate),
events[r,t])`. Pure and autodiff-friendly (differentiates w.r.t. `pars`;
`counts`/`events` are constants) — drop into a PracticalBayes `@model` via
`@addlogprob! chain_binomial_loglik(pars, m, counts, events)`.

This is the transition-matrix style's analogue of [`trajectory_loglik`](@ref);
for the same underlying process both target the same parameters (aggregate counts
vs per-individual trajectories) and recover the same values.
"""
function chain_binomial_loglik(pars, m::SimpleEpiTransitionMatrix, counts, events)
    ss = m.state_space
    T = eltype(pars)
    n_t = size(counts, 2)
    lp = zero(T)
    for t in 1:(n_t - 1)
        for (r, (from, _)) in enumerate(m.transitions)
            src = state_index(ss, from)
            n_src = counts[src, t]
            n_src == 0 && continue
            p = clamp(m.rates[r](pars, view(counts, :, t), t), T(1e-12), one(T) - T(1e-12))
            k = events[r, t]
            lp += _logbinom(n_src, k) + k * log(p) + (n_src - k) * log(one(T) - p)
        end
    end
    return lp
end

# log binomial coefficient log C(n, k) — constant w.r.t. parameters, kept in
# Float64 (it never needs a derivative). `loggamma(x+1) == log(x!)`.
@inline function _logbinom(n::Integer, k::Integer)
    return StatsFuns.loggamma(n + 1) - StatsFuns.loggamma(k + 1) - StatsFuns.loggamma(n - k + 1)
end
