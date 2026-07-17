# Benchmark the badger model's gradient AND iFFBS sweep through PracticalBayes'
# own machinery — the gradient via LogDensityFunction (exactly what NUTS calls),
# the sweep via epidemic_latent_sampler (exactly what iFFBSKernel calls). Reports
# min/mean/median over several evaluations, comparable to badger_ref/bench_ref.jl's
# grad_pars/iFFBS timing.
#
# Both are warmed up with one untimed call first, so the first-ever-call JIT
# compilation cost (which can dominate a single sample) never lands in the timed
# statistics — see bench_ref.jl's equivalent treatment on the reference side.
#
# Run:  julia --project=examples examples/bench_gradient.jl [backend]
#   backend ∈ {forwarddiff, polyesterforwarddiff}   (default: forwarddiff)
# Env:  BENCH_NGRAD (gradient evaluations, default 15)
#       BENCH_NSWEEP (iFFBS sweeps, default 3)
ENV["BADGER_RUN"] = "0"   # badger_fit.jl auto-runs run_badger_fit() unless told not to
include(joinpath(@__DIR__, "badger_fit.jl"))
using ADTypes, LogDensityProblems, Statistics
using PolyesterForwardDiff: PolyesterForwardDiff
using StableRNGs: StableRNG

which_backend = length(ARGS) >= 1 ? ARGS[1] : "forwarddiff"
adtype = which_backend == "polyesterforwarddiff" ? ADTypes.AutoPolyesterForwardDiff(; chunksize=nothing) :
         which_backend == "forwarddiff" ? ADTypes.AutoForwardDiff() :
         error("unknown backend '$which_backend', use forwarddiff|polyesterforwarddiff")

init0 = badger_initial_params(raw; rng=StableRNG(13))
X0 = copy(raw.X_init)
init = (; X=X0, tau=init0.tau, alpha=init0.alpha, lambda=init0.lambda,
        beta=init0.beta, q=init0.q, c1=init0.c1, a1=init0.a1, b1=init0.b1,
        a2=init0.a2, b2=init0.b2, thetas=init0.thetas, rhos=init0.rhos,
        phis=init0.phis, etas=init0.etas, nu=hcat(init0.nuE, init0.nuI))
reset_aggregates!(data)
apply_derived_summaries!(merge(init0, (; nu=hcat(init0.nuE, init0.nuI))), data, X0)

m = badger_base(data, data.n_timepoints, data.n_individuals, G, NT, NS, NNU, loglik)

# `X`, `etas` and `nu` are kernel-owned (value store); the rest is what NUTS moves.
nuts_names = (:tau, :alpha, :lambda, :beta, :q, :c1, :a1, :b1, :a2, :b2, :thetas, :rhos, :phis)
layout, theta0, store0 = build_layout(m; flat=nuts_names,
                                      values=(:X, :etas, :nu), init=init)
println("=== Package benchmark: backend=$which_backend ===")
println("NUTS parameter vector length: ", length(theta0))

ldf = PracticalBayes.LogDensityFunction(m, layout, store0, adtype; θ0=theta0)
LogDensityProblems.logdensity_and_gradient(ldf, theta0)          # warm up

n_grad = parse(Int, get(ENV, "BENCH_NGRAD", "15"))
grad_times = Float64[]
for _ in 1:n_grad
    t = @elapsed LogDensityProblems.logdensity_and_gradient(ldf, theta0)
    push!(grad_times, t)
end
println("full gradient (", length(theta0), " params), ", n_grad, " evals, warm-up call excluded:")
println("  min=", round(minimum(grad_times), digits=3), "s  mean=", round(mean(grad_times), digits=3),
        "s  median=", round(median(grad_times), digits=3), "s")

# iFFBS sweep — same pattern as the gradient above: one untimed warm-up sweep,
# then n_sweep timed sweeps. X0/aggregates carry state forward between sweeps
# exactly as a real Gibbs run would (each sweep conditions on the previous one's
# result), which is also why the warm-up sweep's STATE change is kept even though
# its TIME isn't recorded.
latent! = epidemic_latent_sampler(data)
rng = StableRNG(1)
# The rate functions read `model.nu` (the combined S/E/I mixing matrix), not the
# separate nuE/nuI fields badger_initial_params returns — same merge `init`/`ldf`
# above already does, and the same one badger_fit.jl's own `_pars(c)` does.
sweep_pars = merge(Base.structdiff(init0, (; nuE=0, nuI=0)), (; nu=hcat(init0.nuE, init0.nuI)))
latent!(rng, sweep_pars, X0)   # warm up

n_sweep = parse(Int, get(ENV, "BENCH_NSWEEP", "3"))
sweep_times = Float64[]
for _ in 1:n_sweep
    t = @elapsed latent!(rng, sweep_pars, X0)
    push!(sweep_times, t)
end
println("iFFBS sweep (", data.n_individuals, " individuals), ", n_sweep, " sweeps, warm-up sweep excluded:")
println("  min=", round(minimum(sweep_times), digits=3), "s  mean=", round(mean(sweep_times), digits=3),
        "s  median=", round(median(sweep_times), digits=3), "s")
