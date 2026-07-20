# Measure, rather than infer, how a real Gibbs sweep's time splits between the
# iFFBS (X) block and the HMC (continuous parameters) block.
#
# The arithmetic "gradient x L" is only a MODEL of the HMC block's cost; it
# ignores AdvancedHMC's own overhead, the loglik calls at the trajectory
# endpoints, PracticalBayes' layout/unflatten work, and the conjugate blocks. So
# we wrap the actual kernels and accumulate wall-clock per block across a real
# `sample` call — the same code path the 5000-sweep script runs.
#
# Method: each kernel is wrapped in a timing shim that adds its elapsed time to a
# global accumulator. The shims are the ONLY change; the model, sampler, data and
# init are identical to badger_fit_reststotal_hmc.jl.
#
# Run: JULIA_NUM_THREADS=8 julia --project=examples examples/bench_blocks.jl [nsweeps]

ENV["BADGER_RUN"] = "0"    # don't let the fit script auto-run its own long run
include(joinpath(@__DIR__, "badger_fit_reststotal_hmc.jl"))

using Printf

const T_IFFBS = Ref(0.0)
const T_ETA   = Ref(0.0)
const T_NU    = Ref(0.0)
const N_IFFBS = Ref(0)

# --- Timing shims -----------------------------------------------------------
# Each wraps an existing kernel, forwards to it, and accumulates elapsed time.
struct TimedLatent{K} <: PracticalBayes.AbstractLatentKernel
    inner::K
    acc::Base.RefValue{Float64}
    count::Base.RefValue{Int}
end
function PracticalBayes.latent_step(rng, k::TimedLatent, block_names, c::PracticalBayes.ModelConditional)
    t0 = time_ns()
    out = PracticalBayes.latent_step(rng, k.inner, block_names, c)
    k.acc[] += (time_ns() - t0) / 1e9
    k.count[] += 1
    out
end

n_sweeps = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
seed = 13

m = badger_base(data, data.n_timepoints, data.n_individuals, G, NT, NS, NNU, loglik, obs_loglik)
hmc_kernel = make_hmc_block(HMC_EPS, HMC_L)

spl = Gibbs(
    (:tau, :alpha, :lambda, :beta, :q, :c1, :a1, :b1, :a2, :b2,
     :thetas, :rhos, :phis) => hmc_kernel,
    :etas => TimedLatent(EtaKernel(1.0, 1.0), T_ETA, Ref(0)),
    :nu   => TimedLatent(NuKernel([1.0, 1.0, 1.0]), T_NU, Ref(0)),
    :X    => TimedLatent(iFFBSKernel(latent!), T_IFFBS, N_IFFBS),
)

init0 = badger_initial_params(raw; rng=StableRNG(seed))
X0 = copy(raw.X_init)
init = (; X=X0, tau=init0.tau, alpha=init0.alpha, lambda=init0.lambda,
        beta=init0.beta, q=init0.q, c1=init0.c1, a1=init0.a1, b1=init0.b1,
        a2=init0.a2, b2=init0.b2, thetas=init0.thetas, rhos=init0.rhos,
        phis=init0.phis, etas=init0.etas, nu=hcat(init0.nuE, init0.nuI))

function run_timed(nsw; label)
    reset_aggregates!(data)
    apply_derived_summaries!(init0, data, X0)
    T_IFFBS[] = 0.0; T_ETA[] = 0.0; T_NU[] = 0.0; N_IFFBS[] = 0
    t0 = time_ns()
    AbstractMCMC.sample(StableRNG(seed), m, spl, nsw;
                        init=init, adtype=ADTYPE, n_adapts=0, discard_initial=0,
                        save_states=(X=:buffer,))
    total = (time_ns() - t0) / 1e9
    (; label, nsw, total, iffbs=T_IFFBS[], eta=T_ETA[], nu=T_NU[])
end

# Warm-up run (JIT) — discarded, so compilation never lands in the reported split.
println("### compile pass (1 sweep, untimed) ###")
run_timed(1; label="warmup")

println("\n### timed pass ($n_sweeps sweeps) ###")
r = run_timed(n_sweeps; label="timed")

other = r.total - r.iffbs - r.eta - r.nu
@printf("\n%-28s %10s %10s %8s\n", "block", "total (s)", "per sweep", "share")
@printf("%-28s %10.2f %10.3f %7.1f%%\n", "iFFBS (X)", r.iffbs, r.iffbs/r.nsw, 100*r.iffbs/r.total)
@printf("%-28s %10.2f %10.3f %7.1f%%\n", "conjugate etas", r.eta, r.eta/r.nsw, 100*r.eta/r.total)
@printf("%-28s %10.2f %10.3f %7.1f%%\n", "conjugate nu", r.nu, r.nu/r.nsw, 100*r.nu/r.total)
@printf("%-28s %10.2f %10.3f %7.1f%%\n", "HMC + overhead (residual)", other, other/r.nsw, 100*other/r.total)
@printf("%-28s %10.2f %10.3f %7.1f%%\n", "TOTAL", r.total, r.total/r.nsw, 100.0)
println("\n(HMC row is a RESIDUAL: everything not inside a timed kernel — the HMC")
println(" block itself plus PracticalBayes' per-sweep layout/unflatten overhead.)")
