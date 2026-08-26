# Leave-future-out cross-validation: the generic loop, the granularity axis, and
# the result object.
#
# WHERE THIS SHOULD EVENTUALLY LIVE. Nothing below mentions epidemics: it needs a
# fit callback, a per-cell score callback, and a truncation callback. It sits
# here for now because `truncate_data` does, and because this is where it is
# exercised. If a second, non-epidemic user appears it should move to a
# workflow-level package -- see LFO_PACKAGING_GUIDE.md.
#
# THE ESTIMAND, AND WHY IT IS NOT LOO OR WAIC. LFO scores
#
#     ELPD = sum over cutoffs t of  log p(y_{t+1:t+M} | y_{1:t})
#
# LOO and WAIC are pointwise leave-ONE-out criteria and estimate the same
# quantity as each other. Leaving out y_t while conditioning on y_{t-1} AND
# y_{t+1} lets the model interpolate -- it uses the future to predict the past,
# which is not the predictive task. WAIC is arguably worse than PSIS-LOO here
# because it carries no k-hat diagnostic, so it fails silently.

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

- `Joint()` — one cell for the whole window. This is Bürkner, Gabry & Vehtari's
  (2020) block predictive density, and what they prescribe for dependent series.
- `ByGroup(g)` — one cell per (group, step), with `g` a per-individual group
  vector. Keeps within-group dependence, discards between-group.
- `Pointwise()` — one cell per (individual, step). This is the standard ELPD of
  Vehtari, Gelman & Gabry (2017) — "pointwise" is in the acronym.

`Pointwise` and `Joint` are DIFFERENT ESTIMANDS, equal only if cells are
independent given the data. For a transmission model they are not, and that
dependence is the signal. The gap is a multi-information term,
`sum_c log p(y_c) - log p(y_all) = -MI <= 0`.

**Never compare totals across granularities** — a pointwise total sums `N*M`
densities per window, a joint total sums one. [`compare`](@ref) refuses.
"""
abstract type Granularity end

struct Joint <: Granularity end
struct Pointwise <: Granularity end
struct ByGroup{V<:AbstractVector{<:Integer}} <: Granularity
    group::V
end

Base.show(io::IO, ::Joint) = print(io, "Joint()")
Base.show(io::IO, ::Pointwise) = print(io, "Pointwise()")
Base.show(io::IO, g::ByGroup) = print(io, "ByGroup(", length(unique(g.group)), " groups)")

_granularity_name(::Joint) = :joint
_granularity_name(::Pointwise) = :pointwise
_granularity_name(::ByGroup) = :by_group

"""
    cell_of(g::Granularity, i, m) -> Any

Which cell individual `i` at forecast step `m` is charged to. The ONLY thing a
granularity does: same trajectories, same densities, same weights — only the
bucket changes. That is what makes `Joint()` reproduce an unbucketed scorer
exactly, which the test suite asserts.
"""
cell_of(::Joint, i, m) = 1
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
subtracted per cell — with unnormalised weights that is `log S`. Omit it and every
cell is inflated by `log S`, so a pointwise total (many cells) gains
`n_cells * log S` over a joint total (one cell). Both totals stay finite and
plausible; only the comparison between them is destroyed, and nothing about the
numbers looks wrong.

A cell that is `-Inf` for EVERY draw makes the whole window `-Inf`, which is
honest: no draw could explain that observation. A cell that is `-Inf` for SOME
draws costs only those draws, which is the entire point of the axis.
"""
function aggregate_cells(logw::AbstractVector{<:Real},
                         cell_lp::AbstractVector{<:AbstractDict})
    S = length(cell_lp)
    S == length(logw) ||
        throw(ArgumentError("$(length(logw)) weights for $S draws"))
    keys_all = Set{Any}()
    for d in cell_lp, k in keys(d); push!(keys_all, k); end
    lognorm = logsumexp(logw)
    total = 0.0
    buf = Vector{Float64}(undef, S)
    for k in keys_all
        @inbounds for s in 1:S
            buf[s] = logw[s] + get(cell_lp[s], k, -Inf)
        end
        # StatsFuns.logsumexp returns -Inf for an all--Inf input, which is what a
        # cell no draw can explain should score.
        total += logsumexp(buf) - lognorm
        isfinite(total) || return -Inf
    end
    total
end

"""
    WindowResult

One cutoff's outcome, with the diagnostics needed to judge it.

`n_informative` is the count of cells whose density actually depends on the
latent state. It matters more than it sounds: on the badger data only ~2% of
cells carried an observation at all — the rest contributed a state-independent
constant identical under every granularity — so the granularity axis had almost
nothing to redistribute and all three arms agreed to within a few nats. Reading
that number first tells you whether a comparison can resolve anything.
"""
struct WindowResult
    cutoff::Int
    elpd::Dict{Symbol,Float64}          # granularity name => window score
    n_draws::Int
    n_finite::Dict{Symbol,Int}          # draws with a finite score
    n_cells::Dict{Symbol,Int}
    n_informative::Int
    fit_seconds::Float64
    score_seconds::Float64
end

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

Only meaningful against another total computed under the SAME granularity and
the SAME cutoffs; see [`compare`](@ref), which enforces both.
"""
function elpd(r::LFOResult, g::Symbol = first(r.granularities))
    g in r.granularities ||
        throw(ArgumentError("no granularity $g; have $(r.granularities)"))
    sum(w.elpd[g] for w in r.windows)
end
elpd(r::LFOResult, g::Granularity) = elpd(r, _granularity_name(g))

"""
    n_informative(r::LFOResult) -> Int

Total cells across the sweep whose density depends on the latent state. If this
is a small fraction of the nominal cell count, no granularity can discriminate
much and the comparison is close to a null experiment — worth knowing BEFORE
interpreting a margin.
"""
n_informative(r::LFOResult) = sum(w.n_informative for w in r.windows)

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
    @printf(io, "  informative cells: %d\n", n_informative(r))
end
