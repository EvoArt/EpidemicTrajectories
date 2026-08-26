# The leave-future-out driver: walk the cutoffs, fit, score, collect.
#
# Deliberately small. Everything hard lives in the pieces it calls -- truncation
# (truncate.jl), aggregation (lfo.jl), scoring (lfo_score.jl) -- and this file is
# just the loop plus the two things a loop must get right: caching fits, and
# never scoring a window whose fit saw the future.

"""
    LFOSpec(; fit, cell_logdensity, plan, kwargs...)

Everything `lfo_cv` needs from a model.

# Required
- `fit(train_data, cutoff) -> (draws, X_draws)` — fit to the truncated data.
  `draws[s]` is whatever your `cell_logdensity` wants as its `model` argument;
  `X_draws[s]` is draw `s`'s latent trajectory over the WHOLE series.
- `cell_logdensity(model, data, X, i, t) -> Float64` — log density of individual
  `i`'s observation at `t` under the (simulated) states in `X`. Return `-Inf` for
  an inadmissible trajectory; it is charged to that individual's cell alone.
- `plan::TruncationPlan` — from [`truncation`](@ref).

# Optional
- `is_informative(data, i, t) -> Bool` — whether that cell's density depends on
  the latent state. Strongly recommended: see [`n_informative`](@ref).
- `constrain` / `survival_weight` — from [`survival_constrained`](@ref). Pass
  both or neither; passing one alone is a bias, so it is rejected.
- `n_sim` — forward trajectories per draw (default 1).
- `seed` — base RNG seed; window `t`, draw `s` uses `seed + 1000s + t` so that
  every granularity sees IDENTICAL trajectories and the arms differ only in
  where the log is taken.
"""
Base.@kwdef struct LFOSpec{F,C,I,K,W}
    fit::F
    cell_logdensity::C
    plan::TruncationPlan
    is_informative::I = nothing
    constrain::K = nothing
    survival_weight::W = nothing
    n_sim::Int = 1
    seed::Int = 13
end

"""
    lfo_cv(spec, data; L, M, granularity, stride=1, cache=nothing, cutoffs=nothing,
           verbose=true) -> LFOResult

Run leave-future-out cross-validation.

For each cutoff `t` in `L:stride:(T-M)`: truncate the data to `t`, fit, then
score `t+1 : t+M` under every requested granularity **from the same fit and the
same forward trajectories**. Scoring all granularities together is not an
optimisation — it is what makes them comparable, since separate runs would see
different trajectories and the difference would no longer be attributable to the
granularity.

# Caching
`cache` is a directory for fitted chains. The key is what changes the POSTERIOR
— the cutoff and the spec's own fit function — and NOT the granularity or the
scorer. Rescoring then costs seconds instead of a refit. Always cache anything
expensive: adding one scoring arm to an uncached 120-fit sweep once meant
repeating every fit to redo four minutes of arithmetic.

# Example
```julia
spec = LFOSpec(
    fit  = (d, t) -> my_fit(d, t),
    cell_logdensity = my_density,
    plan = truncation(clamp=(:last_seen,), keep=(:sex,)),
)
res = lfo_cv(spec, data; L=20, M=2,
             granularity=(Pointwise(), ByGroup(data.group), Joint()))
elpd(res, :pointwise)
```
"""
function lfo_cv(spec::LFOSpec, data::EpidemicData;
                L::Int, M::Int,
                granularity = (Pointwise(),),
                stride::Int = 1,
                cache = nothing,
                cutoffs = nothing,
                verbose::Bool = true)

    (spec.constrain === nothing) == (spec.survival_weight === nothing) ||
        throw(ArgumentError("""
            pass `constrain` and `survival_weight` together or not at all.
            A constraint without its weight deletes a branch of the proposal with
            nothing to correct it -- a bias that does NOT shrink with n_sim.
            Build both from `survival_constrained(...)`."""))

    grans = granularity isa Granularity ? (granularity,) : Tuple(granularity)
    gnames = collect(_granularity_name.(grans))
    length(unique(gnames)) == length(gnames) ||
        throw(ArgumentError("repeated granularity: $gnames"))

    ts = cutoffs === nothing ? lfo_cutoffs(data; L, M, stride) : collect(cutoffs)
    cache === nothing || mkpath(cache)

    windows = WindowResult[]
    for t in ts
        train = truncate_data(data, spec.plan, t)

        t_fit = time()
        draws, X_draws = _fit_cached(spec, train, t, cache, verbose)
        fit_secs = time() - t_fit

        t_score = time()
        S = length(X_draws)
        cell_lp = Dict(g => Vector{Dict{Any,Float64}}(undef, S) for g in gnames)
        n_inf = 0
        for s in 1:S
            for (g, gname) in zip(grans, gnames)
                # SAME seed for every granularity => identical trajectories.
                rng = StableRNG_or_default(spec.seed + 1000 * s + t)
                cells, ni = score_window(draws[s], data, X_draws[s], t, M, g;
                                         cell_logdensity = spec.cell_logdensity,
                                         is_informative = spec.is_informative,
                                         n_sim = spec.n_sim, rng = rng,
                                         constrain = spec.constrain,
                                         survival_weight = spec.survival_weight)
                cell_lp[gname][s] = cells
                s == 1 && gname === first(gnames) && (n_inf = ni)
            end
        end

        logw = zeros(S)                      # exact refitting: no importance weights
        elpds = Dict(g => aggregate_cells(logw, cell_lp[g]) for g in gnames)
        nfin = Dict(g => count(s -> all(isfinite, values(cell_lp[g][s])), 1:S)
                    for g in gnames)
        ncell = Dict(g => length(cell_lp[g][1]) for g in gnames)
        score_secs = time() - t_score

        push!(windows, WindowResult(t, elpds, S, nfin, ncell, n_inf,
                                    fit_secs, score_secs))
        verbose && _report_window(windows[end], gnames)
    end

    LFOResult(windows, gnames, L, M,
              Dict{Symbol,Any}(:n_sim => spec.n_sim, :stride => stride,
                               :cache => cache))
end

# StableRNGs is not a dependency; fall back to the stdlib if it is absent. A
# seeded MersenneTwister is reproducible within a Julia version, which is what
# the granularity-comparability argument needs.
StableRNG_or_default(seed::Int) = Random.MersenneTwister(seed)

function _fit_cached(spec::LFOSpec, train, t::Int, cache, verbose::Bool)
    cache === nothing && return spec.fit(train, t)
    file = joinpath(cache, "fit_t$(lpad(t, 4, '0')).jls")
    if isfile(file)
        verbose && @info "reusing cached fit" cutoff = t file
        try
            return open(deserialize, file)
        catch err
            @warn "unreadable cache, refitting" file err
        end
    end
    out = spec.fit(train, t)
    # Saved the moment it exists, not at the end of the sweep: a run that dies at
    # window 40 must not lose the first 39 fits.
    open(f -> serialize(f, out), file, "w")
    out
end

function _report_window(w::WindowResult, gnames)
    parts = join((@sprintf("%s=%.2f", String(g), w.elpd[g]) for g in gnames), "  ")
    @info @sprintf("cutoff %d: %s  (fit %.1fs, score %.1fs, informative %d)",
                   w.cutoff, parts, w.fit_seconds, w.score_seconds, w.n_informative)
end
