# Acceptance test for the type-stability / loop-restructure fixes to
# EtaKernel / NuKernel / iFFBSKernel (see perf_gap_log.md).
#
# Speed is worthless if the posterior moved. The eta rewrite changed the LOOP
# ORDER and the accumulation shape, and although bench_eta.jl already showed the
# COUNTS are bit-identical on the initial X, the counts feed `rand(rng, Beta(...))`
# — so the real question is whether the whole sampler still produces the exact
# same chain from the same seed.
#
# This runs the full Gibbs sampler twice from an identical seed — once with the
# fixed kernels, once with reference reimplementations of the ORIGINAL kernel
# bodies (globals, NS-pass eta loop) — and compares the draws exactly.
#
# The RNG draw order must match for this to be a fair test: the fixed EtaKernel
# does all its counting first and then NS Beta draws, while the original
# interleaved count-then-draw per season. Both make the same NS draws in the same
# order from the same generator, so the streams should coincide exactly.
#
# Run: JULIA_NUM_THREADS=8 julia --project=examples examples/check_kernel_fix_exact.jl

ENV["BADGER_RUN"] = "0"
include(joinpath(@__DIR__, "badger_fit_reststotal_hmc.jl"))
using Printf
using StableRNGs: StableRNG

# --- Reference reimplementations of the ORIGINAL (pre-fix) kernel bodies ------
struct EtaKernelOrig <: PracticalBayes.AbstractLatentKernel
    a::Float64
    b::Float64
end
function PracticalBayes.latent_step(rng, k::EtaKernelOrig, block_names, c::PracticalBayes.ModelConditional)
    X = c.values.X
    etas = zeros(Float64, NS)
    for s in 1:NS
        caught = 0
        available = 0
        for t in 1:data.n_timepoints
            data.season[t] == s || continue
            for i in 1:data.n_individuals
                g = data.social_group[i, t]
                (g > 0 && data.capt_effort[g, t] == 1) || continue
                X[t, i] == 4 && continue
                available += 1
                caught += data.capture[t, i] == 1
            end
        end
        etas[s] = rand(rng, Beta(k.a + caught, k.b + max(available - caught, 0)))
    end
    (; etas=etas)
end

struct NuKernelOrig <: PracticalBayes.AbstractLatentKernel
    hp::Vector{Float64}
end
function PracticalBayes.latent_step(rng, k::NuKernelOrig, block_names, c::PracticalBayes.ModelConditional)
    X = c.values.X
    nu = Matrix{Float64}(undef, NNU, 2)
    for (idx, nt) in enumerate(data.nu_times)
        counts = zeros(Int, 3)
        for i in 1:data.n_individuals
            start_time = data.sampling_period[i][1]
            (start_time == nt && data.birth_time[i] < start_time) || continue
            s = X[nt, i]
            s <= 3 && (counts[s] += 1)
        end
        p = rand(rng, Dirichlet(counts .+ k.hp))
        nu[idx, 1], nu[idx, 2] = p[2], p[3]
    end
    (; nu=nu)
end

# --- Run the sampler with a given pair of conjugate kernels -------------------
function run_variant(eta_k, nu_k; n_sweeps, seed=13)
    m = badger_base(data, data.n_timepoints, data.n_individuals, G, NT, NS, NNU, loglik, obs_loglik)
    spl = Gibbs(
        (:tau, :alpha, :lambda, :beta, :q, :c1, :a1, :b1, :a2, :b2,
         :thetas, :rhos, :phis) => make_hmc_block(HMC_EPS, HMC_L),
        :etas => eta_k,
        :nu   => nu_k,
        :X    => iFFBSKernel(latent!),
    )
    init0 = badger_initial_params(raw; rng=StableRNG(seed))
    X0 = copy(raw.X_init)
    init = (; X=X0, tau=init0.tau, alpha=init0.alpha, lambda=init0.lambda,
            beta=init0.beta, q=init0.q, c1=init0.c1, a1=init0.a1, b1=init0.b1,
            a2=init0.a2, b2=init0.b2, thetas=init0.thetas, rhos=init0.rhos,
            phis=init0.phis, etas=init0.etas, nu=hcat(init0.nuE, init0.nuI))
    reset_aggregates!(data)
    apply_derived_summaries!(init0, data, X0)
    AbstractMCMC.sample(StableRNG(seed), m, spl, n_sweeps;
                        init=init, adtype=ADTYPE, n_adapts=0, discard_initial=0,
                        save_states=(X=:buffer,))
end

n_sweeps = 5
println("Running $n_sweeps sweeps with FIXED kernels...")
chn_fix = run_variant(EtaKernel(1.0, 1.0), NuKernel([1.0, 1.0, 1.0]); n_sweeps=n_sweeps)
println("Running $n_sweeps sweeps with ORIGINAL kernels...")
chn_orig = run_variant(EtaKernelOrig(1.0, 1.0), NuKernelOrig([1.0, 1.0, 1.0]); n_sweeps=n_sweeps)

# Comparison lives in a FUNCTION, not top-level script scope. At top level, `for`
# introduces a soft scope, so assigning to `allsame` inside the loop creates a
# new local (and `b` collides with the module-level `b` from the fit script).
# A function body has none of those problems.
function compare_chains(chn_fix, chn_orig)
    allsame = true
    for name in (:tau, :lambda, :beta, :q, :c1, :a1, :b1, :a2, :b2)
        x = vec(chn_fix[name]); y = vec(chn_orig[name])
        same = x == y
        allsame &= same
        @printf("%-8s %s   maxdiff=%.3e\n", name, same ? "IDENTICAL" : "DIFFERS  ",
                maximum(abs.(x .- y)))
    end
    for name in (:etas, :thetas, :rhos, :phis)
        x = reduce(hcat, chn_fix[name]); y = reduce(hcat, chn_orig[name])
        same = x == y
        allsame &= same
        @printf("%-8s %s   maxdiff=%.3e\n", name, same ? "IDENTICAL" : "DIFFERS  ",
                maximum(abs.(x .- y)))
    end
    allsame
end

println("\n=== comparing draws ===")
allsame = compare_chains(chn_fix, chn_orig)

println()
if allsame
    println(">>> PASS: every draw is bit-identical. The fixes are pure speed.")
else
    println(">>> FAIL: the fixes changed the chain. Investigate before using them.")
end

