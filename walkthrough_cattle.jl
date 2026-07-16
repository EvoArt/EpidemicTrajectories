using Random
using Distributions
using PracticalBayes
using AdvancedHMC: NUTS
import AbstractMCMC
using StableRNGs: StableRNG
using Statistics: mean, std
using DataFrames
using CSV
using JLD2: @save
using CairoMakie
using Dates

#### package code

struct TransitionSpec
    states::Vector{Symbol}
    transitions::Vector{Tuple{Symbol,Symbol}}
    rate_fns::Vector{Function}
    auto_self::Bool
end

_replace_state_sym(x) = x
_replace_state_sym(x::Symbol) = x === :state ? :s : x
function _replace_state_sym(ex::Expr)
    Expr(ex.head, map(_replace_state_sym, ex.args)...)
end

_invert_op(op) = op === :(+=) ? :(-=) :
                 op === :(-=) ? :(+=) :
                 op === :(*=) ? :(/=) :
                 op === :(/=) ? :(*=) :
                 error("Unsupported operator in derived summary: $op")

function _aggregate_line_to_lambda(line)
    cond = :(true)
    upd = line
    if line isa Expr && line.head == :if
        upd = line.args[1]
        cond = line.args[2]
    elseif line isa Expr && line.head == :call && line.args[1] == :count
        if length(line.args) == 3
            cond = line.args[2]
            target = line.args[3]
            upd = Expr(:(+=), target, 1)
        else
            error("Unsupported count(...) form in @aggregate")
        end
    end

    upd = _replace_state_sym(upd)
    cond = _replace_state_sym(cond)
    op = upd.head
    inv_op = _invert_op(op)
    rev = Expr(inv_op, upd.args[1], upd.args[2])

    lam = :((model, data, X, s, i, t; reverse=false) -> begin
        if $cond
            if reverse
                $rev
            else
                $upd
            end
        end
        nothing
    end)
    return esc(lam)
end

macro derived_summary(name, expr)
    expr2 = _replace_state_sym(expr)
    op = expr2.head
    inv_op = _invert_op(op)
    reverse_expr = Expr(inv_op, expr2.args[1], expr2.args[2])
    lam = :((model, data, X, s, i, t; reverse=false) -> begin
        if reverse
            $reverse_expr
        else
            $expr2
        end
        nothing
    end)
    return esc(:($name = $lam))
end

function save_results(draws; truths=NamedTuple(), outdir)
    mkpath(outdir)

    timestamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    draws_jld2_path = joinpath(outdir, "draws-$(timestamp).jld2")
    draws_csv_path = joinpath(outdir, "draws-$(timestamp).csv")
    summary_csv_path = joinpath(outdir, "summary-$(timestamp).csv")
    posterior_png_path = joinpath(outdir, "posterior-$(timestamp).png")

    param_names = collect(keys(draws))

    @save draws_jld2_path draws truths

    draws_df = DataFrame()
    for name in param_names
        draws_df[!, string(name)] = collect(getfield(draws, name))
    end
    CSV.write(draws_csv_path, draws_df)

    summary_df = DataFrame(
        parameter=String[],
        truth=Float64[],
        mean=Float64[],
        sd=Float64[],
    )
    for name in param_names
        values = getfield(draws, name)
        truth = hasproperty(truths, name) ? Float64(getfield(truths, name)) : NaN
        push!(summary_df, (string(name), truth, mean(values), std(values)))
    end
    CSV.write(summary_csv_path, summary_df)

    ncol = 3
    nrow = cld(length(param_names), ncol)
    fig = Figure(size=(400 * ncol, 350 * nrow))
    for (idx, name) in enumerate(param_names)
        row = cld(idx, ncol)
        col = mod1(idx, ncol)
        values = getfield(draws, name)
        ax = Axis(fig[row, col], title=string(name))
        hist!(ax, values; bins=50, normalization=:pdf)
        if hasproperty(truths, name)
            vlines!(ax, [Float64(getfield(truths, name))], color=:red, linewidth=2)
        end
    end
    save(posterior_png_path, fig)

    (
        outdir=outdir,
        draws_jld2_path=draws_jld2_path,
        draws_csv_path=draws_csv_path,
        summary_csv_path=summary_csv_path,
        posterior_png_path=posterior_png_path,
    )
end

macro aggregate(block)
    lines = block isa Expr && block.head == :block ? [x for x in block.args if !(x isa LineNumberNode)] : [block]
    lambdas = [_aggregate_line_to_lambda(line) for line in lines]
    return :([$(lambdas...)])
end

macro survival(args...)
    return nothing
end

_wrap_rate_expr(rate) = (rate isa Expr && rate.head == :->) ? rate : :((model, data, i, t) -> ($rate))

function _parse_transition_block(block)
    lines = [x for x in block.args if !(x isa LineNumberNode)]
    transitions = Tuple{Symbol,Symbol,Any}[]
    survival_expr = nothing
    death_state = nothing

    for line in lines
        if line isa Expr && line.head == :macrocall && line.args[1] == Symbol("@survival")
            length(line.args) >= 3 || error("@survival needs at least survival expression")
            survival_expr = line.args[3]
            if length(line.args) >= 4
                kw = line.args[4]
                if kw isa Expr && kw.head == :(=) && kw.args[1] == :death
                    death_state = kw.args[2]
                end
            end
            death_state === nothing && error("@survival requires death=:State")
            continue
        end

        (line isa Expr && line.head == :(->)) || error("Transition line must look like `A -> B = rate`")
        from = line.args[1]
        rhs = line.args[2]
        assign = rhs isa Expr && rhs.head == :block ? first(filter(x -> !(x isa LineNumberNode), rhs.args)) : rhs
        (assign isa Expr && assign.head == :(=)) || error("Transition line must include `= rate`")
        to = assign.args[1]
        rate = assign.args[2]
        push!(transitions, (from, to, rate))
    end

    if survival_expr !== nothing
        transitions = map(transitions) do (from, to, rate)
            if to == death_state
                (from, to, rate)
            else
                (from, to, :(($survival_expr) * ($rate)))
            end
        end
        src_states = unique(first.(transitions))
        for s in src_states
            s == death_state && continue
            if !any(t -> t[1] == s && t[2] == death_state, transitions)
                push!(transitions, (s, death_state, :(1 - ($survival_expr))))
            end
        end
    end

    return transitions
end

macro transitions(args...)
    rest = collect(args)
    style = :simple
    auto_self = false

    while !isempty(rest) && rest[1] isa QuoteNode
        tag = rest[1].value
        if tag == :individual
            style = :individual
        elseif tag == :auto_self
            auto_self = true
        else
            error("Unsupported @transitions tag: $tag")
        end
        rest = rest[2:end]
    end

    block = if length(rest) == 1
        rest[1]
    elseif length(rest) == 2
        rest[2]
    else
        error("@transitions expects [:individual] [:auto_self] [state_space] begin ... end")
    end

    style == :individual || error("This walkthrough supports only @transitions :individual")
    block isa Expr && block.head == :block || error("@transitions body must be begin...end")

    trs = _parse_transition_block(block)
    states = unique(vcat(Symbol[t[1] for t in trs], Symbol[t[2] for t in trs]))
    trans_pairs = [:(($(QuoteNode(t[1])), $(QuoteNode(t[2])))) for t in trs]
    rates = [_wrap_rate_expr(t[3]) for t in trs]

    quote
        TransitionSpec(
            Symbol[$(map(QuoteNode, states)...)],
            Tuple{Symbol,Symbol}[$(trans_pairs...)],
            Function[$(map(esc, rates)...)],
            $(auto_self),
        )
    end
end

mutable struct WalkData
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

members(data::WalkData, g) = data.members_by_group[g]

function build_affected_individuals_from_groups(group::Vector{Int}, n_timepoints::Int; include_self=false)
    n_individuals = length(group)
    out = Matrix{Vector{Int}}(undef, n_timepoints, n_individuals)
    for t in 1:n_timepoints
        for i in 1:n_individuals
            ids = [j for j in 1:n_individuals if group[j] == group[i] && (include_self || j != i)]
            out[t, i] = ids
        end
    end
    out
end

"""
    reset_aggregates!(data)

Zero every user-supplied aggregate array. The package does not know what the
aggregates represent; it just clears their storage before a fresh sum-up.
"""
function reset_aggregates!(data::WalkData)
    for (_, v) in data.aggregates
        v isa AbstractArray && fill!(v, zero(eltype(v)))
    end
    nothing
end

"""
    apply_derived_summaries!(model, data, X)

Forward-apply every user derived summary over the whole population and window.
The package has no knowledge of what is being aggregated (infected counts,
alive counts, ...) — it only calls the user's generated `@derived_summary`
functions. Used once to "sum up" from a fresh (zeroed) aggregate after X is
initialised; thereafter iFFBS keeps the aggregate current incrementally.
"""
function apply_derived_summaries!(model, data::WalkData, X)
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

@inline function _state_index(data::WalkData, s::Symbol)
    idx = findfirst(==(s), data.state_space)
    idx === nothing && error("State $s not found in state_space")
    idx
end

function transition_matrix_at(trans_mat::TransitionSpec, model, data::WalkData, X, i, t)
    N = data.n_states
    T = typeof(model.α)
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

no_rest_contribution(model, data, X, i, t, n_states) = ones(n_states)

function make_rest_contribution(; normalize=true, min_logprob=-1e12, affected_ids, neighbor_logprob)
    function rest_contribution(model, data::WalkData, X, i, t, n_states, affected_override=nothing)
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

function make_neighbor_logprob_from_transitions(trans_mat::TransitionSpec; eps_prob=1e-12)
    return function (model, data::WalkData, X, j, t, updated_id)
        Pj = transition_matrix_at(trans_mat, model, data, X, j, t)
        from_state = X[t, j]
        to_state = X[t + 1, j]
        log(max(Pj[from_state, to_state], eps_prob))
    end
end

function observation_process(model, data::WalkData, X, i, t)
    w = ones(Float64, data.n_states)
    y_r = data.test_mats[1][t, i]
    y_f = data.test_mats[2][t, i]

    if y_r >= 0
        θ = model.θʳ
        w[1] *= y_r == 1 ? 0.0 : 1.0
        w[2] *= y_r == 1 ? θ : (1 - θ)
    end

    if y_f >= 0
        θ = model.θᶠ
        w[1] *= y_f == 1 ? 0.0 : 1.0
        w[2] *= y_f == 1 ? θ : (1 - θ)
    end

    w
end

initialise_forward_filter(model, data::WalkData, X, i, t) = data.starting_state(model, data, X, i, t)

function forward_filter(xᵢ, start_sampling, end_sampling, model, data::WalkData, X, i)
    probs = zeros(Float64, length(xᵢ), data.n_states)
    trans_cache = Vector{Matrix{Float64}}(undef, length(xᵢ))

    t0 = start_sampling
    base = initialise_forward_filter(model, data, X, i, t0)
    obs = observation_process(model, data, X, i, t0)
    affected = data.affected_individuals === nothing ? nothing : data.affected_individuals[t0, i]
    rest = data.rest_contribution(model, data, X, i, t0, data.n_states, affected)
    init = base .* obs .* rest
    probs[1, :] .= init ./ sum(init)
    trans_cache[1] = zeros(Float64, data.n_states, data.n_states)

    for j in 2:length(xᵢ)
        t = start_sampling + j - 1
        tp = t - 1
        trans = transition_matrix_at(data.trans_mat, model, data, X, i, tp)
        trans_cache[j] = trans
        pred = trans' * view(probs, j - 1, :)
        obs_w = observation_process(model, data, X, i, t)
        affected = data.affected_individuals === nothing ? nothing : data.affected_individuals[t, i]
        rest_w = data.rest_contribution(model, data, X, i, t, data.n_states, affected)
        unnorm = pred .* obs_w .* rest_w
        z = sum(unnorm)
        probs[j, :] .= z > 0 ? unnorm ./ z : fill(1.0 / data.n_states, data.n_states)
    end

    probs, trans_cache
end

function backward_sample!(probs, trans_cache, xᵢ, start_sampling, end_sampling, model, data::WalkData, X, i, rng)
    n_t = length(xᵢ)
    xᵢ[n_t] = _sample_categorical(rng, view(probs, n_t, :))

    for j in (n_t - 1):-1:1
        trans = trans_cache[j + 1]
        bnext = xᵢ[j + 1]
        cond = view(probs, j, :) .* view(trans, :, bnext)
        z = sum(cond)
        w = z > 0 ? cond ./ z : fill(1.0 / data.n_states, data.n_states)
        xᵢ[j] = _sample_categorical(rng, w)
    end
    nothing
end

function iFFBS_individual!(model, data::WalkData, X, i, rng)
    start_sampling, end_sampling = data.sampling_period[i]
    xᵢ = @view X[start_sampling:end_sampling, i]

    for t in start_sampling:end_sampling
        for ds in data.derived_summaries
            ds(model, data, X, X[t, i], i, t; reverse=true)
        end
    end

    probs, trans_cache = forward_filter(xᵢ, start_sampling, end_sampling, model, data, X, i)
    backward_sample!(probs, trans_cache, xᵢ, start_sampling, end_sampling, model, data, X, i, rng)

    for t in start_sampling:end_sampling
        for ds in data.derived_summaries
            ds(model, data, X, X[t, i], i, t)
        end
    end

    nothing
end

function iFFBS!(model, data::WalkData, X, rng)
    # Aggregates are already consistent with `X` on entry (initial sum-up +
    # incremental maintenance), so iFFBS only reverses/forwards the focal
    # individual's contribution inside `iFFBS_individual!`.
    for i in 1:data.n_individuals
        iFFBS_individual!(model, data, X, i, rng)
    end
    X
end

function build_simulate(data::WalkData)
    function simulate(rng, model)
        X = zeros(Int, data.n_timepoints, data.n_individuals)

        for i in 1:data.n_individuals
            p0 = data.starting_state(model, data, X, i, 1)
            X[1, i] = _sample_categorical(rng, p0)
        end

        for t in 1:(data.n_timepoints - 1)
            # Populate this time slice's aggregates via the user's derived
            # summaries before they are read by the transition rates.
            for i in 1:data.n_individuals
                for ds in data.derived_summaries
                    ds(model, data, X, X[t, i], i, t)
                end
            end
            for i in 1:data.n_individuals
                P = transition_matrix_at(data.trans_mat, model, data, X, i, t)
                X[t + 1, i] = _sample_categorical(rng, view(P, X[t, i], :))
            end
        end

        X
    end

    simulate
end

function build_loglik(data::WalkData)
    # Relies on incrementally-maintained aggregates: whatever `infection_func`
    # (and any other rate) reads off `data` must already be consistent with
    # `X`. That invariant is established by the initial `apply_derived_summaries!`
    # sum-up and preserved by iFFBS, so loglik never rebuilds it here.
    function loglik(model, data::WalkData, X)
        ll = zero(model.α)

        for i in 1:data.n_individuals
            p0 = data.starting_state(model, data, X, i, 1)
            ll += log(p0[X[1, i]] + 1e-12)
        end

        for t in 1:(data.n_timepoints - 1)
            for i in 1:data.n_individuals
                P = transition_matrix_at(data.trans_mat, model, data, X, i, t)
                ll += log(P[X[t, i], X[t + 1, i]] + 1e-12)
            end
        end

        ll
    end

    loglik
end

function build_latent_sampler(data::WalkData)
    (rng, model, X) -> iFFBS!(model, data, X, rng)
end

function build_data(; n_individuals, n_timepoints, group, state_space, trans_mat, starting_state,
                      test_mats, derived_summaries, aggregates=Dict{Symbol,Any}())
    n_groups = maximum(group)
    members_by_group = Dict(g => findall(==(g), group) for g in 1:n_groups)
    sampling_period = [(1, n_timepoints) for _ in 1:n_individuals]

    affected_individuals = build_affected_individuals_from_groups(group, n_timepoints)
    affected_ids = (data, t, i) -> affected_individuals[t, i]
    neighbor_logprob = make_neighbor_logprob_from_transitions(trans_mat)
    rest_contribution = make_rest_contribution(affected_ids=affected_ids, neighbor_logprob=neighbor_logprob)

    WalkData(
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

function simulate_observations(rng, X; θʳ, θᶠ, observed_days)
    n_timepoints, n_individuals = size(X)
    R = fill(-1, n_timepoints, n_individuals)
    F = fill(-1, n_timepoints, n_individuals)

    for t in observed_days, i in 1:n_individuals
        infected = X[t, i] == 2
        R[t, i] = rand(rng) < (infected ? θʳ : 0.0) ? 1 : 0
        F[t, i] = rand(rng) < (infected ? θᶠ : 0.0) ? 1 : 0
    end

    R, F
end

#### user code

state_space = [:S, :I]

# User-defined accumulator: number infected per group at each time. The package
# knows nothing about it — the @aggregate macro builds a reversible update
# against the generic `data.aggregates` container, and the rate below reads it.
infected_summaries = @aggregate begin
    data.aggregates[:n_infected_per_group][data.group[i], t] += (s == 2)
end

function infection_func(model, data, i, t)
    g = data.group[i]
    I_minus = data.aggregates[:n_infected_per_group][g, t]
    clamp(-expm1(-(model.α + model.β * I_minus)), 1e-12, 1 - 1e-12)
end

recovery_func(model, data, i, t) = clamp(1 / model.m, 1e-12, 1 - 1e-12)

trans_mat = @transitions :individual :auto_self begin
    S -> I = infection_func(model, data, i, t)
    I -> S = recovery_func(model, data, i, t)
end

n_pens = 10
n_per_pen = 8
n_timepoints = 80
n_individuals = n_pens * n_per_pen
group = repeat(1:n_pens; inner=n_per_pen)
observed_days = collect(1:6:n_timepoints)

true_pars = (; α=0.01, β=0.02, m=6.0, ν=0.10, θʳ=0.8, θᶠ=0.5)

starting_state = (model, data, X, i, t) -> [1 - model.ν, model.ν]

dummy_tests = [fill(-1, n_timepoints, n_individuals), fill(-1, n_timepoints, n_individuals)]
sim_data = build_data(
    n_individuals=n_individuals,
    n_timepoints=n_timepoints,
    group=group,
    state_space=state_space,
    trans_mat=trans_mat,
    starting_state=starting_state,
    test_mats=dummy_tests,
    derived_summaries=infected_summaries,
    aggregates=Dict{Symbol,Any}(:n_infected_per_group => zeros(Int, n_pens, n_timepoints)),
)

simulate = build_simulate(sim_data)
rng = StableRNG(2024)
X_true = simulate(rng, true_pars)
Rmask, Fmask = simulate_observations(rng, X_true; θʳ=true_pars.θʳ, θᶠ=true_pars.θᶠ, observed_days=observed_days)

fit_data = build_data(
    n_individuals=n_individuals,
    n_timepoints=n_timepoints,
    group=group,
    state_space=state_space,
    trans_mat=trans_mat,
    starting_state=starting_state,
    test_mats=[Rmask, Fmask],
    derived_summaries=infected_summaries,
    aggregates=Dict{Symbol,Any}(:n_infected_per_group => zeros(Int, n_pens, n_timepoints)),
)

loglik = build_loglik(fit_data)
latent! = build_latent_sampler(fit_data)

struct TrajectoryLatent <: Distributions.DiscreteMatrixDistribution
    n_time::Int
    n_ind::Int
end
Base.size(d::TrajectoryLatent) = (d.n_time, d.n_ind)
Distributions.logpdf(::TrajectoryLatent, X::AbstractMatrix) = 0.0
Distributions.rand(rng::AbstractRNG, d::TrajectoryLatent) = fill(1, d.n_time, d.n_ind)

struct iFFBSKernel <: PracticalBayes.AbstractLatentKernel
    data::WalkData
    latent!::Function
end

PracticalBayes.latent_step(rng, k::iFFBSKernel, block_names, c::PracticalBayes.ModelConditional) = begin
    block_names == (:X,) || error("iFFBS kernel only handles :X")
    pars = (; α=c.values.α, β=c.values.β, m=c.values.m̃ + 1.0, ν=c.values.ν, θʳ=c.values.θʳ, θᶠ=c.values.θᶠ)
    X = copy(c.values.X)
    k.latent!(rng, pars, X)
    (; X=X)
end

struct NuKernel <: PracticalBayes.AbstractLatentKernel
    a::Float64
    b::Float64
end

PracticalBayes.latent_step(rng, k::NuKernel, block_names, c::PracticalBayes.ModelConditional) = begin
    X = c.values.X
    n_inf = count(==(2), @view X[1, :])
    n_sus = size(X, 2) - n_inf
    (; ν=rand(rng, Beta(k.a + n_inf, k.b + n_sus)))
end

struct RamsKernel <: PracticalBayes.AbstractLatentKernel
    a::Float64
    b::Float64
    R::Matrix{Int}
end

PracticalBayes.latent_step(rng, k::RamsKernel, block_names, c::PracticalBayes.ModelConditional) = begin
    X = c.values.X
    n_pos = 0
    n_tested_inf = 0
    for t in axes(k.R, 1), i in axes(k.R, 2)
        r = k.R[t, i]
        r < 0 && continue
        if X[t, i] == 2
            n_tested_inf += 1
            r == 1 && (n_pos += 1)
        end
    end
    (; θʳ=rand(rng, Beta(k.a + n_pos, k.b + (n_tested_inf - n_pos))))
end

struct FaecalKernel <: PracticalBayes.AbstractLatentKernel
    a::Float64
    b::Float64
    F::Matrix{Int}
end

PracticalBayes.latent_step(rng, k::FaecalKernel, block_names, c::PracticalBayes.ModelConditional) = begin
    X = c.values.X
    n_pos = 0
    n_tested_inf = 0
    for t in axes(k.F, 1), i in axes(k.F, 2)
        r = k.F[t, i]
        r < 0 && continue
        if X[t, i] == 2
            n_tested_inf += 1
            r == 1 && (n_pos += 1)
        end
    end
    (; θᶠ=rand(rng, Beta(k.a + n_pos, k.b + (n_tested_inf - n_pos))))
end

@model function cattle_model(data, n_time, n_ind, loglik_fn)
    α ~ Gamma(1, 1)
    β ~ Gamma(1, 1)
    m̃ ~ Gamma(2, 4)
    m := m̃ + 1.0

    ν ~ Beta(1, 1)
    θʳ ~ Beta(1, 1)
    θᶠ ~ Beta(1, 1)

    X ~ TrajectoryLatent(n_time, n_ind)

    pars = (; α=α, β=β, m=m, ν=ν, θʳ=θʳ, θᶠ=θᶠ)
    @addlogprob! loglik_fn(pars, data, X)
end

function run_fit(; n_sweeps=parse(Int, get(ENV, "WALKTHROUGH_SWEEPS", "2000")),
                    n_burn=parse(Int, get(ENV, "WALKTHROUGH_BURN", "800")),
                    n_adapts=parse(Int, get(ENV, "WALKTHROUGH_ADAPTS", "400")),
                    seed=7)
    m = cattle_model(fit_data, n_timepoints, n_individuals, loglik)

    spl = Gibbs(
        (:α, :β, :m̃) => NUTS(0.8),
        :ν => NuKernel(1.0, 1.0),
        :θʳ => RamsKernel(1.0, 1.0, Rmask),
        :θᶠ => FaecalKernel(1.0, 1.0, Fmask),
        :X => iFFBSKernel(fit_data, latent!),
    )

    X0 = copy(X_true)
    init = (; X=X0, α=0.05, β=0.05, m̃=4.0, ν=0.1, θʳ=0.7, θᶠ=0.4)

    # One-time generic sum-up: make the user's derived-summary aggregates
    # consistent with the initial X before the first (NUTS) loglik evaluation.
    # iFFBS keeps them consistent thereafter.
    init_pars = (; α=init.α, β=init.β, m=init.m̃ + 1.0, ν=init.ν, θʳ=init.θʳ, θᶠ=init.θᶠ)
    reset_aggregates!(fit_data)
    apply_derived_summaries!(init_pars, fit_data, X0)

    rng_fit = StableRNG(seed)
    chn = AbstractMCMC.sample(rng_fit, m, spl, n_sweeps;
        init=init,
        n_adapts=n_adapts,
        discard_initial=n_burn,
    )

    (
        α=collect(vec(chn[:α])),
        β=collect(vec(chn[:β])),
        m=collect(vec(chn[:m̃])) .+ 1.0,
        ν=collect(vec(chn[:ν])),
        θʳ=collect(vec(chn[:θʳ])),
        θᶠ=collect(vec(chn[:θᶠ])),
    )
end

function report(draws)
    println("\n=== Posterior recovery ===")
    truths = (α=true_pars.α, β=true_pars.β, m=true_pars.m, ν=true_pars.ν, θʳ=true_pars.θʳ, θᶠ=true_pars.θᶠ)
    for name in (:α, :β, :m, :ν, :θʳ, :θᶠ)
        post = getfield(draws, name)
        truth = getfield(truths, name)
        mn, sd = mean(post), std(post)
        within = abs(mn - truth) < 3 * (sd + sd / sqrt(length(post)))
        println(rpad(string(name), 4), ": posterior mean = ", rpad(round(mn; digits=4), 8),
                " (sd ", round(sd; digits=4), ")   truth = ", truth, within ? "   ✓" : "   ✗")
    end
end

println("Simulated herd: $n_pens pens x $n_per_pen animals x $n_timepoints days")
println("True parameters: α=$(true_pars.α), β=$(true_pars.β), m=$(true_pars.m), ν=$(true_pars.ν), θʳ=$(true_pars.θʳ), θᶠ=$(true_pars.θᶠ)")
println("Observed days: ", observed_days)
println("True infection prevalence: ", round(mean(X_true .== 2); digits=3))

if get(ENV, "WALKTHROUGH_RUN", "1") == "1"
    draws = run_fit()
    report(draws)
    outdir = get(ENV, "WALKTHROUGH_OUTDIR", joinpath(@__DIR__, "examples", "outputs"))
    paths = save_results(draws; truths=true_pars, outdir=outdir)
    println("\nSaved results:")
    println("  JLD2:   ", paths.draws_jld2_path)
    println("  CSV:    ", paths.draws_csv_path)
    println("  Summary:", paths.summary_csv_path)
    println("  PNG:    ", paths.posterior_png_path)
end
