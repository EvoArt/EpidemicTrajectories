# Leave-future-out cross-validation: the generic loop, the granularity axis,
# and the result object.
#
# Nothing below mentions epidemics; it needs a fit callback, a per-cell score
# callback and a truncation callback. It sits here because `truncate_data`
# does, and because this is where it is exercised.
#
# LFO scores ELPD = sum over cutoffs of log p(y_{t+1:t+M} | y_{1:t}). LOO and
# WAIC estimate something else: leaving out y_t while conditioning on both
# neighbours lets the model interpolate, using the future to predict the past.

"""
    Granularity

How much data goes inside the logarithm when a window's score is assembled.

Each Monte Carlo term is a product of `N * M` densities. If a single cell gets
probability zero under draw `s`, the whole term is zero and that draw
contributes nothing -- discarding the correct predictions it made for every
other cell. With mortality this is not rare: measured on a simulated model with
an absorbing dead state, `Joint()` lost 46-70% of windows to `-Inf` where
`Pointwise()` lost 4%.

The choice is a *modelling* one, not an implementation detail:

- `Joint()`, one cell for the whole window. This is Bürkner, Gabry & Vehtari's
  (2020) block predictive density, and what they prescribe for dependent series.
- `ByGroup(g)`, one cell per (group, step), with `g` a per-individual group
  vector. Keeps within-group dependence, discards between-group.
- `ByIndividual()`, one cell per individual's full forecast window. This keeps
  temporal dependence within an individual while scoring individuals separately.
- `Pointwise()`, one cell per (individual, step). This is the standard ELPD of
  Vehtari, Gelman & Gabry (2017) — "pointwise" is in the acronym.

These are different predictive targets. Even when individual trajectories are
independent conditional on parameters, integrating shared parameter uncertainty
can make their joint and separately scored predictive densities differ. The
realized log-score difference has no fixed sign.

**Never compare totals across granularities**, a pointwise total sums `N*M`
densities per window, a joint total sums one. [`compare`](@ref) refuses.
"""
abstract type Granularity end

struct Joint <: Granularity end
struct ByIndividual <: Granularity end
struct Pointwise <: Granularity end
struct ByGroup{V<:AbstractVector{<:Integer}} <: Granularity
    group::V
end

Base.show(io::IO, ::Joint) = print(io, "Joint()")
Base.show(io::IO, ::ByIndividual) = print(io, "ByIndividual()")
Base.show(io::IO, ::Pointwise) = print(io, "Pointwise()")
Base.show(io::IO, g::ByGroup) = print(io, "ByGroup(", length(unique(g.group)), " groups)")

_granularity_name(::Joint) = :joint
_granularity_name(::ByIndividual) = :by_individual
_granularity_name(::Pointwise) = :pointwise
_granularity_name(::ByGroup) = :by_group

"""
    cell_of(g::Granularity, i, m) -> Any

Which cell individual `i` at forecast step `m` is charged to. The only thing a
granularity does: same trajectories, same densities, same weights: only the
bucket changes. That is what makes `Joint()` reproduce an unbucketed scorer
exactly, which the test suite asserts.
"""
cell_of(::Joint, i, m) = 1
cell_of(::ByIndividual, i, m) = i
cell_of(::Pointwise, i, m) = (i, m)
cell_of(g::ByGroup, i, m) = (g.group[i], m)

"""
    aggregate_cells(logw, cell_lp) -> Float64

Combine per-draw, per-cell log densities into one window score:

    sum over cells c of  [ logsumexp_s( logw[s] + cell_lp[s][c] ) - logsumexp_s(logw) ]

`cell_lp[s]` maps cell key to that draw's log density for the cell. `logw` are
log importance weights; pass zeros under exact refitting.

**The normaliser matters, and getting it wrong is silent.** Each cell's score is
a self-normalised weighted average over draws, so `logsumexp_s(logw)` must be
subtracted per cell: with unnormalised weights that is `log S`. Omit it and every
cell is inflated by `log S`, so a pointwise total (many cells) gains
`n_cells * log S` over a joint total (one cell). Both totals stay finite and
plausible; only the comparison between them is destroyed, and nothing about the
numbers looks wrong.

A cell that is `-Inf` for every draw makes the whole window `-Inf`, which is
honest: no draw could explain that observation. A cell that is `-Inf` for SOME
draws costs only those draws, which is the entire point of the axis.
"""
function aggregate_cells(logw::AbstractVector{<:Real},
                         cell_lp::AbstractVector{<:AbstractDict})
    total = 0.0
    for v in values(cell_scores(logw, cell_lp))
        total += v
        isfinite(total) || return -Inf
    end
    total
end

"""
    cell_scores(logw, cell_lp) -> Dict

Each cell's own log predictive density, the terms [`aggregate_cells`](@ref)
sums: `logsumexp_s(logw[s] + cell_lp[s][c]) - logsumexp_s(logw)` per cell `c`.

These are what stacking over individual histories needs. A window total alone
cannot be split back into its cells, because the log is taken per cell.
"""
function cell_scores(logw::AbstractVector{<:Real},
                     cell_lp::AbstractVector{<:AbstractDict})
    S = length(cell_lp)
    S == length(logw) ||
        throw(ArgumentError("$(length(logw)) weights for $S draws"))
    keys_all = Set{Any}()
    for d in cell_lp, k in keys(d); push!(keys_all, k); end
    lognorm = logsumexp(logw)
    out = Dict{Any,Float64}()
    buf = Vector{Float64}(undef, S)
    for k in keys_all
        @inbounds for s in 1:S
            buf[s] = logw[s] + get(cell_lp[s], k, -Inf)
        end
        # StatsFuns.logsumexp returns -Inf for an all--Inf input, which is what a
        # cell no draw can explain should score.
        out[k] = logsumexp(buf) - lognorm
    end
    out
end

# How many draws gave each cell a finite density: the LOCAL support count. The
# window-level `n_finite` counts draws finite in EVERY cell, which says nothing
# about a cellwise score -- it can be 0 while every cell is well supported.
function _cell_support(cell_lp::AbstractVector{<:AbstractDict})
    out = Dict{Any,Int}()
    for d in cell_lp, (k, v) in d
        out[k] = get(out, k, 0) + isfinite(v)
    end
    out
end

"""
    WindowResult

One cutoff's outcome, with the diagnostics needed to judge it.

`n_informative` is the count of cells whose density actually depends on the
latent state, or `nothing` when the spec supplied no `is_informative`. It matters
more than it sounds: on the badger data only ~2% of cells carried an observation
at all, the rest contributed a state-independent constant identical under every
granularity — so the granularity axis had almost nothing to redistribute and all
three arms agreed to within a few nats. Reading that number first tells you
whether a comparison can resolve anything.

`nothing` is deliberately distinct from `0`: "nobody asked" and "not one cell is
informative" are opposite situations, and collapsing them would let the second —
which means the comparison is meaningless, hide as the first.
"""
struct WindowResult
    cutoff::Int
    elpd::Dict{Symbol,Float64}          # granularity name => window score
    n_draws::Int
    n_finite::Dict{Symbol,Int}          # draws finite in EVERY cell
    n_cells::Dict{Symbol,Int}
    n_informative::Union{Int,Nothing}   # `nothing` = not reported (see above)
    fit_seconds::Float64
    score_seconds::Float64
    # Per-cell scores and per-cell finite-draw counts, by granularity. The cell
    # scores sum to `elpd`; they are kept because stacking over individual
    # histories, and finding which individual failed, both need them.
    cells::Dict{Symbol,Dict{Any,Float64}}
    cell_support::Dict{Symbol,Dict{Any,Int}}
end

WindowResult(cutoff, elpd, n_draws, n_finite, n_cells, n_informative,
             fit_seconds, score_seconds) =
    WindowResult(cutoff, elpd, n_draws, n_finite, n_cells, n_informative,
                 fit_seconds, score_seconds,
                 Dict{Symbol,Dict{Any,Float64}}(), Dict{Symbol,Dict{Any,Int}}())

"""
    LFOResult

The whole sweep. Index granularities by their name (`:joint`, `:pointwise`,
`:by_group`).
"""
struct LFOResult
    windows::Vector{WindowResult}
    granularities::Vector{Symbol}
    L::Int
    M::Int
    meta::Dict{Symbol,Any}
end

cutoffs(r::LFOResult) = [w.cutoff for w in r.windows]

"""
    elpd(r::LFOResult, granularity=first(r.granularities)) -> Float64

Total ELPD across every window, under one granularity.

Only meaningful against another total computed under the same granularity and
the same cutoffs; see [`compare`](@ref), which enforces both.
"""
function elpd(r::LFOResult, g::Symbol = first(r.granularities))
    g in r.granularities ||
        throw(ArgumentError("no granularity $g; have $(r.granularities)"))
    sum(w.elpd[g] for w in r.windows)
end
elpd(r::LFOResult, g::Granularity) = elpd(r, _granularity_name(g))

"""
    cell_elpd(r::LFOResult, granularity=first(r.granularities)) -> NamedTuple

Every window's per-cell scores as flat columns `(; cutoff, cell, elpd,
n_finite)`, one row per (window, cell). Under `ByIndividual()` a cell is an
individual's whole forecast block, so these are the individual-history scores
stacking needs; under `Joint()` there is one row per window.

`n_finite` is how many posterior draws gave that cell a finite density. A cell
with `elpd = -Inf` is one no draw could explain: read it here rather than
inferring it from a window total.
"""
function cell_elpd(r::LFOResult, g::Symbol = first(r.granularities))
    g in r.granularities ||
        throw(ArgumentError("no granularity $g; have $(r.granularities)"))
    cutoff = Int[]; cell = Any[]; lp = Float64[]; nf = Int[]
    for w in r.windows
        haskey(w.cells, g) || throw(ArgumentError(
            "window $(w.cutoff) kept no per-cell scores (a result from before they were recorded)"))
        for k in sort!(collect(keys(w.cells[g])); by = string)
            push!(cutoff, w.cutoff); push!(cell, k)
            push!(lp, w.cells[g][k]); push!(nf, get(w.cell_support[g], k, 0))
        end
    end
    (; cutoff, cell, elpd = lp, n_finite = nf)
end
cell_elpd(r::LFOResult, g::Granularity) = cell_elpd(r, _granularity_name(g))

"""
    n_informative(r::LFOResult) -> Union{Int,Nothing}

Total cells across the sweep whose density depends on the latent state, or
`nothing` if the spec supplied no `is_informative`.

If this is a small fraction of the nominal cell count, no granularity can
discriminate much and the comparison is close to a null experiment: worth
knowing before interpreting a margin. Supplying `is_informative` is optional but
strongly recommended for exactly that reason.
"""
function n_informative(r::LFOResult)
    any(w -> w.n_informative === nothing, r.windows) && return nothing
    sum(w.n_informative for w in r.windows)
end

"""
    compare(a::LFOResult, b::LFOResult; granularity) -> NamedTuple

`a` minus `b` on the cutoffs they share, with a standard error.

Refuses to compare across granularities: a pointwise total sums `N*M` densities
per window and a joint total sums one, so they differ by thousands of nats for
reasons unrelated to fit. Compares only shared cutoffs, because a total over 30
windows against one over 24 is not a comparison.

The SE treats windows as independent. With `M > 1` they overlap, so it is
optimistic; `se_indep` uses every `M`-th window instead and is the conservative
reading.
"""
function compare(a::LFOResult, b::LFOResult; granularity::Symbol)
    granularity in a.granularities && granularity in b.granularities ||
        throw(ArgumentError("both results need granularity $granularity"))
    a.M == b.M ||
        throw(ArgumentError("different M ($(a.M) vs $(b.M)): not comparable"))

    ca, cb = Dict(w.cutoff => w for w in a.windows), Dict(w.cutoff => w for w in b.windows)
    shared = sort(collect(intersect(keys(ca), keys(cb))))
    isempty(shared) && throw(ArgumentError("no shared cutoffs"))

    d = [ca[t].elpd[granularity] - cb[t].elpd[granularity] for t in shared]
    all(isfinite, d) || return (; diff = NaN, se = NaN, se_indep = NaN,
                                n_windows = length(shared), granularity,
                                note = "non-finite windows present")
    n = length(d)
    total = sum(d)
    se = n > 1 ? std(d) * sqrt(n) : NaN
    indep = d[1:a.M:end]
    se_indep = length(indep) > 1 ? sqrt(n^2 / length(indep) * var(indep)) : NaN
    (; diff = total, se, se_indep, n_windows = n, granularity)
end

function Base.show(io::IO, r::LFOResult)
    println(io, "LFOResult: ", length(r.windows), " windows, L=", r.L, ", M=", r.M)
    for g in r.granularities
        nf = sum(w.n_finite[g] for w in r.windows)
        nd = sum(w.n_draws for w in r.windows)
        @printf(io, "  %-12s elpd = %14.3f   finite draws %d/%d\n",
                String(g), elpd(r, g), nf, nd)
    end
    ni = n_informative(r)
    println(io, "  informative cells: ", ni === nothing ? "not reported" : ni)
end

"""
    model_weights(results; granularity, method=:stacking) -> NamedTuple

Weights over a set of candidate models from their leave-future-out scores.

A margin says which model won and by how much; a weight says how much of the
predictive job each model should be given, which is the quantity a reader of a
model comparison usually wants. `results` is anything iterable of `LFOResult`,
optionally a `Dict`/`NamedTuple` whose keys name the models.

Two methods, and they answer different questions:

  * `:stacking` (the default) chooses `w` on the simplex to maximise
    `sum_t log sum_k w_k * exp(elpd_kt)` -- the predictive density of the
    WEIGHTED MIXTURE, window by window. It is the right default because the
    candidates are being combined, not ranked: a model that predicts badly
    where the others predict well still earns weight.

  * `:pseudo_bma` sets `w_k` proportional to `exp(elpd_k)` on the totals. It is
    a softmax of the totals, so it ignores the between-window correlation
    entirely and collapses onto the single best model as the series lengthens
    -- with many windows the exponent is a sum of many terms and the largest
    total wins by an arbitrarily wide margin. Reported because it is what most
    people mean by "model weight", and cheap; not because it is better.

Both refuse to mix granularities and both use only the cutoffs every result
shares, for the reasons [`compare`](@ref) documents: a pointwise total sums
`N*M` densities per window and a joint total sums one, and a total over 30
windows against one over 24 is not a comparison.

A window in which any candidate scored non-finite is DROPPED, and the number
dropped is returned as `n_dropped`. Silently keeping it would let one model's
degeneracy set the weights -- which is exactly the mortality failure this
package exists to correct, so it must not be hidden here.

Returns `(; weights, names, method, granularity, n_windows, n_dropped)`.
"""
function model_weights(results; granularity::Symbol, method::Symbol = :stacking)
    names, rs = _named_results(results)
    length(rs) >= 2 || throw(ArgumentError("need at least two models to weight"))
    for (nm, r) in zip(names, rs)
        granularity in r.granularities ||
            throw(ArgumentError("model $nm has no granularity $granularity; " *
                                "have $(r.granularities)"))
    end
    allequal(r.M for r in rs) ||
        throw(ArgumentError("different M across models: not comparable"))

    # Only the cutoffs every candidate scored, for compare()'s reason.
    shared = sort(collect(reduce(intersect,
                                 (Set(cutoffs(r)) for r in rs))))
    isempty(shared) && throw(ArgumentError("no shared cutoffs"))

    # lp[t, k] = model k's score in window t.
    idx = [Dict(w.cutoff => w for w in r.windows) for r in rs]
    lp = [idx[k][t].elpd[granularity] for t in shared, k in eachindex(rs)]

    keep = [all(isfinite, @view lp[t, :]) for t in axes(lp, 1)]
    n_dropped = count(!, keep)
    any(keep) || throw(ArgumentError(
        "every window has a non-finite score for at least one model, so no " *
        "weights can be formed; inspect n_finite per window"))
    lp = lp[keep, :]

    w = method === :stacking    ? _stacking_weights(lp) :
        method === :pseudo_bma  ? _pseudo_bma_weights(lp) :
        throw(ArgumentError("unknown method $method; use :stacking or :pseudo_bma"))

    (; weights = w, names, method, granularity,
       n_windows = size(lp, 1), n_dropped)
end

"""
    model_weights(lp::AbstractMatrix; names, method=:stacking) -> NamedTuple

Weights from a `windows x models` matrix of per-window scores.

The `LFOResult` method above is the one to reach for while the results are in
hand. This one exists because per-window scores SURVIVE a session and an
`LFOResult` does not: a fit holds a handle into a generated module, so a
pipeline that scores on a cluster and gathers later has the numbers but not the
objects. Rows with a non-finite entry are dropped, as above.
"""
function model_weights(lp::AbstractMatrix; names = nothing,
                       method::Symbol = :stacking)
    size(lp, 2) >= 2 || throw(ArgumentError("need at least two models to weight"))
    nms = names === nothing ? ["model$k" for k in axes(lp, 2)] :
                              String.(collect(names))
    length(nms) == size(lp, 2) ||
        throw(ArgumentError("got $(length(nms)) names for $(size(lp,2)) models"))

    keep = [all(isfinite, @view lp[t, :]) for t in axes(lp, 1)]
    n_dropped = count(!, keep)
    any(keep) || throw(ArgumentError(
        "every window has a non-finite score for at least one model, so no " *
        "weights can be formed"))
    kept = lp[keep, :]

    w = method === :stacking   ? _stacking_weights(kept) :
        method === :pseudo_bma ? _pseudo_bma_weights(kept) :
        throw(ArgumentError("unknown method $method; use :stacking or :pseudo_bma"))

    (; weights = w, names = nms, method, granularity = nothing,
       n_windows = size(kept, 1), n_dropped)
end

# Accept a Dict/NamedTuple (named models) or a plain collection (positional).
function _named_results(results)
    if results isa AbstractDict
        ks = collect(keys(results))
        (String.(ks), [results[k] for k in ks])
    elseif results isa NamedTuple
        (String.(collect(keys(results))), collect(values(results)))
    else
        rs = collect(results)
        (["model$k" for k in eachindex(rs)], rs)
    end
end

_pseudo_bma_weights(lp) = begin
    tot = vec(sum(lp, dims = 1))
    w = exp.(tot .- maximum(tot))
    w ./ sum(w)
end

# Maximise sum_t log sum_k w_k exp(lp[t,k]) over the simplex.
#
# Parameterised by a softmax of unconstrained `z` so the simplex constraint is
# automatic, and optimised by plain gradient ascent with backtracking: the
# objective is smooth and low-dimensional (one parameter per candidate), so a
# dependency on an optimisation package would not earn its place.
function _stacking_weights(lp; iters::Int = 500, tol::Float64 = 1e-10)
    T, K = size(lp)
    z = zeros(K)
    obj = function (z)
        w = exp.(z .- maximum(z)); w ./= sum(w)
        s = 0.0
        @inbounds for t in 1:T
            m = -Inf
            for k in 1:K
                w[k] > 0 && (m = max(m, lp[t, k] + log(w[k])))
            end
            acc = 0.0
            for k in 1:K
                w[k] > 0 && (acc += exp(lp[t, k] + log(w[k]) - m))
            end
            s += m + log(acc)
        end
        s
    end
    grad = function (z)
        w = exp.(z .- maximum(z)); w ./= sum(w)
        g = zeros(K)
        @inbounds for t in 1:T
            m = maximum(lp[t, k] + log(max(w[k], eps())) for k in 1:K)
            r = [exp(lp[t, k] + log(max(w[k], eps())) - m) for k in 1:K]
            den = sum(r)
            # d/dw_k of log(sum w_j e^{lp_tj}) = e^{lp_tk} / sum_j w_j e^{lp_tj}
            for k in 1:K
                g[k] += r[k] / (den * max(w[k], eps()))
            end
        end
        # chain rule through the softmax
        gw = g .* w
        gw .- w .* sum(gw)
    end

    f = obj(z)
    for _ in 1:iters
        g = grad(z)
        step = 1.0
        improved = false
        for _ in 1:40
            z2 = z .+ step .* g
            f2 = obj(z2)
            if f2 > f + tol
                z, f, improved = z2, f2, true
                break
            end
            step /= 2
        end
        improved || break
    end
    w = exp.(z .- maximum(z))
    w ./ sum(w)
end
