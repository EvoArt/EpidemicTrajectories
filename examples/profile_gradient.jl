# Profile the GRADIENT — now the dominant cost of a sweep (~62-74%) and the only
# block with real headroom left.
#
# Profiles the full log-density gradient (transitions + observation, scalar obs
# path), which is exactly what AdvancedHMC calls L=30 times per sweep.
#
# READING THE OUTPUT (per CLAUDE.md, learned the hard way on 2026-07-17):
#   * Sort by TOTAL when deciding WHAT TO CHANGE. A self%-sorted profile of the
#     coupling term once looked flat at 6.4% while its true total was 69.5%, and
#     nearly sent an investigation in the wrong direction.
#   * Sort by SELF only when reading what ONE line does.
# Both are printed below, in that order, for that reason.
#
# Also flags runtime dispatch (red in a flamegraph) and GC per line — which is
# how the earlier concrete-types and allocation wins were found.
#
# Run: JULIA_NUM_THREADS=8 julia --project=examples examples/profile_gradient.jl
# Env: PROF_N (gradient calls to profile, default 20)

ENV["BADGER_RUN"] = "0"
include(joinpath(@__DIR__, "badger_fit_reststotal_hmc.jl"))
include(joinpath(@__DIR__, "badger_model_obssplit.jl"))
using Profile, ProfileToLLM
using LogDensityProblems
using Printf
using StableRNGs: StableRNG

bs = badger_data_obssplit(DATA_DIR)
DATA, RAW = bs.data, bs.raw

trans_ll = epidemic_loglik(DATA)
obs_ll   = epidemic_obs_loglik(DATA; observation_process=badger_obs_tests,
                                     observation_weight=badger_obs_tests_weight)

@model function badger_prof(data, n_time, n_ind, n_groups, n_tests, n_seasons, n_nu, ll_fn, obs_fn)
    tau ~ Exponential(10.0); alpha ~ PracticalBayes.filldist(Exponential(1.0), n_groups)
    lambda ~ Exponential(1.0); beta ~ Exponential(1.0); q ~ Beta(1, 1)
    c1 ~ Exponential(1.0); a1 ~ Exponential(1.0); b1 ~ Exponential(1.0)
    a2 ~ Exponential(1.0); b2 ~ Exponential(1.0)
    thetas ~ PracticalBayes.filldist(Beta(1, 1), n_tests)
    rhos ~ PracticalBayes.filldist(Beta(1, 1), n_tests)
    phis ~ PracticalBayes.filldist(Beta(1, 1), n_tests)
    etas ~ PracticalBayes.filldist(Beta(1, 1), n_seasons)
    nu ~ NuSimplex(n_nu, [8.0, 1.0, 1.0])
    X ~ TrajectoryLatent(n_time, n_ind)
    pars = (; tau, alpha, lambda, beta, q, a1, b1, a2, b2, c1, thetas, rhos, phis, etas, nu)
    @addlogprob! ll_fn(pars, data, X) + obs_fn(pars, data, X)
end

init0 = badger_initial_params(RAW; rng=StableRNG(13))
X0 = copy(RAW.X_init)
init = (; X=X0, tau=init0.tau, alpha=init0.alpha, lambda=init0.lambda,
        beta=init0.beta, q=init0.q, c1=init0.c1, a1=init0.a1, b1=init0.b1,
        a2=init0.a2, b2=init0.b2, thetas=init0.thetas, rhos=init0.rhos,
        phis=init0.phis, etas=init0.etas, nu=hcat(init0.nuE, init0.nuI))
reset_aggregates!(DATA)
apply_derived_summaries!(init0, DATA, X0)

m = badger_prof(DATA, DATA.n_timepoints, DATA.n_individuals, G, NT, NS, NNU, trans_ll, obs_ll)
names = (:tau, :alpha, :lambda, :beta, :q, :c1, :a1, :b1, :a2, :b2, :thetas, :rhos, :phis)
layout, theta0, store0 = build_layout(m; flat=names, values=(:X, :etas, :nu), init=init)
ldf = PracticalBayes.LogDensityFunction(m, layout, store0, ADTYPE; θ0=theta0)

# Warm up OUTSIDE the profile, so JIT never lands in the samples.
LogDensityProblems.logdensity_and_gradient(ldf, theta0)

n = parse(Int, get(ENV, "PROF_N", "20"))
t = minimum(@elapsed(LogDensityProblems.logdensity_and_gradient(ldf, theta0)) for _ in 1:3)
@printf("gradient: %.4f s (%d params), profiling %d calls\n\n", t, length(theta0), n)

Profile.clear()
Profile.@profile for _ in 1:n
    LogDensityProblems.logdensity_and_gradient(ldf, theta0)
end

rows = profile_table(Profile.fetch())

nrows = parse(Int, get(ENV, "PROF_ROWS", "45"))

# Drop the pure-scaffolding frames (Base/client/loading/Profile) from the
# total%-sorted view: they are all ~99.9% total by construction and crowd out the
# frames that actually distinguish where the work goes.
is_scaffold(r) = occursin(r"client\.jl|loading\.jl|boot\.jl|Base\.jl|Profile\.jl|essentials\.jl", String(r.file))

println("="^100)
println("SORTED BY TOTAL%  — use THIS to decide what to change (scaffolding frames dropped)")
println("="^100)
print_profile(sort(filter(!is_scaffold, rows); by = r -> -r.total_pct); max_rows=nrows)

println()
println("="^100)
println("SORTED BY SELF%   — use this only to read what ONE line does")
println("="^100)
print_profile(rows; max_rows=nrows)

# Where does `log` actually get called from? It is the single largest
# identifiable cost (25% total in the first run), and both epidemic_loglik and
# epidemic_obs_loglik call log(p + 1e-12) once per (i,t) — under AD, on Duals.
println()
println("="^100)
println("PACKAGE / MODEL FRAMES ONLY (ours, not Base/ForwardDiff internals)")
println("="^100)
ours(r) = occursin(r"EpidemicTrajectories|badger_model|PracticalBayes", String(r.file))
print_profile(sort(filter(ours, rows); by = r -> -r.total_pct); max_rows=nrows)
