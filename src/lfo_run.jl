# The leave-future-out driver: walk the cutoffs, fit, score, collect.
#
# Deliberately small. Everything hard lives in the pieces it calls, and this is
# the loop plus the two things a loop must get right: caching fits, and never
# scoring a window whose fit saw the future.

"""
    LFOScorer

How a window's predictive density is computed from a posterior draw. A
separate axis from [`Granularity`](@ref), which says where the log is taken.

- [`ForwardSimulation`](@ref)`()` (the default): simulate the population
  forward from the draw's sampled state at the cutoff and score the
  observations against the simulated states. Works for any model, coupled or
  not; needs latent draws.
- [`ExactHMM`](@ref)`()`: for individuals independent given the parameters,
  sum each one's cutoff state and future path out exactly by the forward
  algorithm (see [`hmm_forecast_logliks`](@ref)). No simulation, no Monte Carlo
  error beyond the posterior draws, and no latent draws needed.
"""
abstract type LFOScorer end

"Score by forward simulation from each draw's latent state. See [`LFOScorer`](@ref)."
struct ForwardSimulation <: LFOScorer end

"Score by exact per-individual forward filtering. See [`LFOScorer`](@ref)."
struct ExactHMM <: LFOScorer end

_scorer_name(::ForwardSimulation) = :forward_simulation
_scorer_name(::ExactHMM) = :exact_hmm

"""
    LFOSpec(; fit, plan, cell_logdensity=nothing, scorer=ForwardSimulation(), kwargs...)

Everything `lfo_cv` needs from a model.

# Required
- `fit(train_data, cutoff)`: fit to the truncated data. Returns either
  `(draws, X_draws)` or just `draws`. `draws[s]` is what the model's functions
  take as their `model` argument; `X_draws[s]`, when given, is draw `s`'s latent
  trajectory over the whole series. `ExactHMM()` never reads them, so a
  collapsed fit returns `draws` alone. `ForwardSimulation()` needs them, and
  for a collapsed fit of independent individuals draws them itself, by backward
  sampling from each draw's exact posterior given the training data.
  A fit may also return a NamedTuple `(draws, X, sampler)`. `sampler` is
  whatever the fit needs to start a later fit where this one ended (final
  values, adapted metric and step size); `lfo_cv` keeps it per cutoff in
  `result.meta[:sampler]` and never reads it.
- `plan::TruncationPlan`, from [`truncation`](@ref).

# Scoring
- `scorer`: [`ForwardSimulation`](@ref)`()` (default) or [`ExactHMM`](@ref)`()`.
- `cell_logdensity(model, data, X, i, t) -> Float64`: required by
  `ForwardSimulation()`. Log density of individual `i`'s observation at `t`
  under the (simulated) states in `X`. Return `-Inf` for an inadmissible
  trajectory; it is charged to that individual's cell alone. `ExactHMM()`
  scores with the model's own observation process instead, and refuses one,
  since two observation models for one score is a disagreement waiting to
  happen.

# Optional
- `is_informative(data, i, t) -> Bool` — OPTIONAL. Whether that cell's density depends on
  the latent state. Strongly recommended: see [`n_informative`](@ref).
- `constrain` / `survival_weight`: from [`survival_constrained`](@ref). Pass
  both or neither; passing one alone is a bias, so it is rejected. Simulation
  only: there is no proposal to correct under `ExactHMM()`.
- `n_sim`: forward trajectories per draw (default 1). Simulation only.
- `seed`, base RNG seed; window `t`, draw `s` uses `seed + 1000s + t` so that
  every granularity sees IDENTICAL trajectories and the arms differ only in
  where the log is taken.
- `entrants`: whether individuals entering after the cutoff are simulated into
  the forecast (see [`forward_simulate`](@ref)). `false` fixes the forecast
  population at the cutoff. Simulation only.
- `guide`: simulate from the observation-guided proposal, with its weights
  (see [`forward_simulate`](@ref)). Not with `constrain`. For a coupled model,
  `Joint()` only.
"""
Base.@kwdef struct LFOSpec{F,C,I,K,W,SC<:LFOScorer}
    fit::F
    cell_logdensity::C = nothing
    plan::TruncationPlan
    is_informative::I = nothing
    constrain::K = nothing
    survival_weight::W = nothing
    n_sim::Int = 1
    seed::Int = 13
    scorer::SC = ForwardSimulation()
    entrants::Bool = true
    guide::Bool = false
end

# A fit returns `(draws, X_draws)` or `draws` alone.
_split_fit(f::Tuple{Any,Any}) = f
_split_fit(f::AbstractVector) = (f, nothing)
_split_fit(f::NamedTuple) = (f.draws, get(f, :X, nothing))

# Refuse settings that mean nothing to the chosen scorer, rather than ignoring
# them: a run that looks survival-corrected and is not is worse than an error.
function _check_scorer(spec::LFOSpec, data::EpidemicData, grans, M::Int)
    if spec.scorer isa ExactHMM
        (spec.constrain === nothing && spec.survival_weight === nothing) ||
            throw(ArgumentError("ExactHMM() has no forward proposal to constrain; " *
                                "drop `constrain`/`survival_weight`."))
        spec.n_sim == 1 || throw(ArgumentError(
            "ExactHMM() integrates the future exactly; `n_sim` does not apply."))
        spec.guide && throw(ArgumentError(
            "ExactHMM() has no forward proposal to guide; drop `guide`."))
        spec.cell_logdensity === nothing || throw(ArgumentError(
            "ExactHMM() scores with the model's own observation process; drop " *
            "`cell_logdensity` so there is only one observation model."))
        require_independent(data; caller = "ExactHMM()")
        # The exact block density is the chain rule over the WHOLE block, so it
        # can only be charged to cells that hold a whole block: those whose key
        # does not change with the forecast step. Joint and ByIndividual do;
        # Pointwise and ByGroup key on the step, and would need step-marginal
        # forecasts that skip the intervening observations.
        if M > 1
            for g in grans
                cell_of(g, 1, 1) == cell_of(g, 1, 2) || throw(ArgumentError(
                    "ExactHMM() scores whole forecast blocks, so for M > 1 it " *
                    "supports Joint() and ByIndividual(), not $g."))
            end
        end
    else
        spec.cell_logdensity === nothing && throw(ArgumentError(
            "ForwardSimulation() needs a `cell_logdensity`."))
        spec.guide && spec.constrain !== nothing && throw(ArgumentError(
            "use the observation-guided proposal or the survival constraint, not both"))
        # A guided move of one individual changes another's dynamics when they
        # are coupled, so only the product of every weight corrects a density.
        if spec.guide && !_independent(data) && any(g -> !(g isa Joint), grans)
            throw(ArgumentError("the observation-guided proposal on a coupled " *
                                "model gives a valid estimate only for Joint()"))
        end
    end
    nothing
end

_independent(data) = try
    require_independent(data); true
catch err
    err isa ArgumentError || rethrow()
    false
end

"""
    lfo_cv(spec, data; L, M, granularity, stride=1, cache=nothing, cutoffs=nothing,
           verbose=true) -> LFOResult

Run leave-future-out cross-validation.

For each cutoff `t` in `L:stride:(T-M)`: truncate the data to `t`, fit, then
score `t+1 : t+M` under every requested granularity **from the same fit and the
same forward trajectories**. Scoring all granularities together is not an
optimisation: it is what makes them comparable, since separate runs would see
different trajectories and the difference would no longer be attributable to the
granularity.

# Scoring
`spec.scorer` chooses how each draw's predictive density is computed (see
[`LFOScorer`](@ref)); the granularity chooses where the log is taken. The two
are independent, except that `ExactHMM()` scores whole forecast blocks and so,
for `M > 1`, takes only `Joint()` and `ByIndividual()`. The result's
`meta[:scorer]` records which was used.

# Caching
`cache` is a directory for fitted chains. The key is what changes the POSTERIOR
— the cutoff and the spec's own fit function — and not the granularity or the
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
                backend = LocalBackend(),
                verbose::Bool = true)

    (spec.constrain === nothing) == (spec.survival_weight === nothing) ||
        throw(ArgumentError("""
            pass `constrain` and `survival_weight` together or not at all.
            A constraint without its weight deletes a branch of the proposal with
            nothing to correct it -- a bias that does not shrink with n_sim.
            Build both from `survival_constrained(...)`."""))

    grans = granularity isa Granularity ? (granularity,) : Tuple(granularity)
    gnames = collect(_granularity_name.(grans))
    length(unique(gnames)) == length(gnames) ||
        throw(ArgumentError("repeated granularity: $gnames"))

    _check_scorer(spec, data, grans, M)

    ts = cutoffs === nothing ? lfo_cutoffs(data; L, M, stride) : collect(cutoffs)
    cache === nothing || mkpath(cache)

    # RE-ENTRANCY. Under a scheduler backend this same script runs twice: once as
    # the launcher (no work item in the environment) and once per task as a
    # worker (one item set). The spec is rebuilt simply because everything above
    # this call re-executes -- which is why a closure never has to be serialised.
    if !(backend isa LocalBackend)
        item = work_item()
        if item === nothing
            return _launch(backend, spec, data, ts, M, gnames, cache, verbose)
        end
        1 <= item + 1 <= length(ts) ||
            throw(ArgumentError("work item $item outside 0:$(length(ts) - 1)"))
        ts = [ts[item + 1]]              # this task does exactly one cutoff
    end

    known = known_times(data, spec.plan)
    windows = WindowResult[]
    samplers = Dict{Int,Any}()
    for t in ts
        train = truncate_data(data, spec.plan, t)

        t_fit = time()
        fitted = _fit_cached(spec, train, t, cache, verbose)
        draws, X_draws = _split_fit(fitted)
        fitted isa NamedTuple && haskey(fitted, :sampler) && (samplers[t] = fitted.sampler)
        fit_secs = time() - t_fit

        t_score = time()
        S = length(draws)
        cell_lp = Dict(g => Vector{Dict{Any,Float64}}(undef, S) for g in gnames)
        if spec.scorer isa ExactHMM
            n_inf = _score_exact!(cell_lp, spec, data, draws, t, M, grans, gnames,
                                  known)
        else
            # A collapsed fit has no trajectories. For independent individuals
            # they can be drawn afterwards from their exact posterior given the
            # TRAINING data -- the joint posterior of (theta, X) the simulation
            # scorer expects -- so the simulated and exact scores can be run on
            # the very same parameter draws.
            X_draws === nothing &&
                (X_draws = _posterior_paths(draws, train, spec.seed + t))
            length(X_draws) == S ||
                throw(ArgumentError("$S parameter draws but $(length(X_draws)) trajectories"))
            n_inf = spec.is_informative === nothing ? nothing : 0
            for s in 1:S
                for (g, gname) in zip(grans, gnames)
                    # same seed for every granularity => identical trajectories.
                    rng = StableRNG_or_default(spec.seed + 1000 * s + t)
                    cells, ni = score_window(draws[s], data, X_draws[s], t, M, g;
                                             cell_logdensity = spec.cell_logdensity,
                                             is_informative = spec.is_informative,
                                             n_sim = spec.n_sim, rng = rng,
                                             constrain = spec.constrain,
                                             survival_weight = spec.survival_weight,
                                             known = known, entrants = spec.entrants,
                                             guide = spec.guide)
                    cell_lp[gname][s] = cells
                    s == 1 && gname === first(gnames) && n_inf !== nothing && (n_inf = ni)
                end
            end
        end

        logw = zeros(S)                      # exact refitting: no importance weights
        cells = Dict(g => cell_scores(logw, cell_lp[g]) for g in gnames)
        elpds = Dict(g => sum(values(cells[g]); init = 0.0) for g in gnames)
        nfin = Dict(g => count(s -> all(isfinite, values(cell_lp[g][s])), 1:S)
                    for g in gnames)
        ncell = Dict(g => length(cells[g]) for g in gnames)
        support = Dict(g => _cell_support(cell_lp[g]) for g in gnames)
        score_secs = time() - t_score

        push!(windows, WindowResult(t, elpds, S, nfin, ncell, n_inf,
                                    fit_secs, score_secs, cells, support))
        verbose && _report_window(windows[end], gnames)
    end

    res = LFOResult(windows, gnames, L, M,
                    Dict{Symbol,Any}(:n_sim => spec.scorer isa ExactHMM ? nothing : spec.n_sim,
                                     :scorer => _scorer_name(spec.scorer),
                                     :stride => stride, :cache => cache,
                                     :sampler => samplers))

    # A worker writes its one window where `sweep_status` looks for it. Existence
    # of this file is what marks the item done -- never a progress log, which is
    # only written by tasks that reach the end and so cannot record a task that
    # died at startup.
    if !(backend isa LocalBackend)
        item = work_item()
        if item !== nothing
            outdir = get(ENV, "LFO_OUTDIR", ".")
            mkpath(outdir)
            open(f -> serialize(f, res), _item_file(outdir, item), "w")
        end
    end
    res
end

"""
    _launch(backend, spec, data, cutoffs, ...) -> SweepHandle

Submit the sweep and return immediately. Called when a scheduler backend is given
and no work item is set, i.e. this process is the launcher.
"""
function _launch(be::SlurmArray, spec::LFOSpec, data::EpidemicData,
                 ts::Vector{Int}, M::Int, gnames, cache, verbose::Bool)
    script = be.script === nothing ? _calling_script() : be.script
    script === nothing && throw(ArgumentError("""
        could not determine the script to re-run; pass `script=` to SlurmArray.
        A scheduler task starts cold and cannot receive a closure, so the sweep
        works by re-running your script with LFO_WORK_ITEM set."""))

    outdir = abspath(get(ENV, "LFO_OUTDIR", "lfo_sweep"))
    mkpath(joinpath(outdir, "logs"))
    manifest = joinpath(outdir, "manifest.txt")
    open(manifest, "w") do io
        for t in ts; println(io, t); end
    end

    # MaxArraySize caps the array index, not the count, so slices always start at
    # 0 and an offset shifts them onto the right part of the manifest.
    n = length(ts)
    job_ids = String[]
    offset = 0
    while offset < n
        ntasks = min(n - offset, be.max_array_index + 1)
        sh = joinpath(outdir, "submit_$(offset).sh")
        open(io -> write_sbatch(io, be, script, outdir, offset, ntasks), sh, "w")
        id = try
            strip(read(`sbatch --parsable $sh`, String))
        catch err
            throw(ErrorException("sbatch failed for offset $offset: $err"))
        end
        push!(job_ids, String(id))
        verbose && @info "submitted" offset ntasks job = id
        offset += ntasks
    end
    SweepHandle(job_ids, manifest, outdir, n, be)
end

"The path of the script that called into here, or `nothing` from a REPL."
function _calling_script()
    for f in stacktrace()
        p = String(f.file)
        (isempty(p) || startswith(p, "REPL") || occursin("EpidemicTrajectories", p)) && continue
        endswith(p, ".jl") && isfile(p) && return abspath(p)
    end
    isinteractive() ? nothing : (isempty(PROGRAM_FILE) ? nothing : abspath(PROGRAM_FILE))
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

# One trajectory per parameter draw, from p(X | y_train, theta_s), by backward
# sampling. `train` is the truncated copy, so nothing after the cutoff conditions
# the path; cells outside each training window keep the placeholder 1.
function _posterior_paths(draws, train::EpidemicData, seed::Int)
    require_independent(train; caller = "ForwardSimulation() of a fit with no latent draws")
    check_independent(first(draws), train)
    rng = StableRNG_or_default(seed)
    [hmm_sample!(rng, th, train, _scaffold(train)) for th in draws]
end

# Exact scoring of one window: per draw, each cohort individual's whole-block
# log predictive density from `hmm_forecast_logliks`, charged to its cell. The
# log is then taken per cell over draws by `cell_scores`, exactly as for the
# simulated densities, so Joint() puts the product over individuals INSIDE the
# average over draws and ByIndividual() averages each individual separately.
function _score_exact!(cell_lp, spec::LFOSpec, data::EpidemicData, draws, t::Int,
                       M::Int, grans, gnames, known)
    S = length(draws)
    X0 = _scaffold(data)
    # The numeric half of the independence check, once per window at the first
    # draw: cheap beside S forward passes, and it catches a rate that reads a
    # latent aggregate before any score is formed from it.
    check_independent(draws[1], data)
    for s in 1:S
        fc = hmm_forecast_logliks(draws[s], data, t, M; known, X = X0)
        for (g, gname) in zip(grans, gnames)
            d = Dict{Any,Float64}()
            for (i, lp) in fc.logp
                k = cell_of(g, i, 1)
                d[k] = get(d, k, 0.0) + lp
            end
            cell_lp[gname][s] = d
        end
    end
    spec.is_informative === nothing && return nothing
    # The same count `score_window` reports: cohort cells in the window whose
    # density depends on the latent state.
    last_t = min(t + M, data.n_timepoints)
    n = 0
    for u in (t + 1):last_t, i in 1:data.n_individuals
        f_i, l_i = data.sampling_period[i]
        (known[i] <= t && f_i <= u <= min(l_i, data.n_timepoints)) || continue
        spec.is_informative(data, i, u) && (n += 1)
    end
    n
end

function _report_window(w::WindowResult, gnames)
    parts = join((@sprintf("%s=%.2f", String(g), w.elpd[g]) for g in gnames), "  ")
    inf = w.n_informative === nothing ? "n/a" : string(w.n_informative)
    @info @sprintf("cutoff %d: %s  (fit %.1fs, score %.1fs, informative %s)",
                   w.cutoff, parts, w.fit_seconds, w.score_seconds, inf)
end
