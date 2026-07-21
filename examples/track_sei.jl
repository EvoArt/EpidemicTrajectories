# Track S/E/I/D population totals over N iFFBS iterations, for the Siler or
# Gompertz-Makeham (fixed) model. Diagnoses whether/how the infection collapses.
#
# Instruments the iFFBS kernel to count S/E/I/D on the X it produces each sweep,
# so we get the per-sweep trajectory of the epidemic without saving the full X
# chain to disk. Runs the SAME Gibbs sampler each model's fit script builds.
#
# Run: JULIA_NUM_THREADS=8 julia --project=examples examples/track_sei.jl [siler|gompertz] [n]

const WHICH = length(ARGS) >= 1 ? ARGS[1] : "gompertz"
const NSW   = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 100

ENV["BADGER_RUN"] = "0"
const FIT = WHICH == "siler"    ? "badger_fit_reststotal_hmc.jl" :
            WHICH == "gompertz" ? "badger_fit_gompertz_fixed_hmc.jl" :
            error("arg must be siler or gompertz")
include(joinpath(@__DIR__, FIT))

using Printf, StableRNGs
import AbstractMCMC

# Per-sweep S/E/I/D log, filled by the instrumented kernel below.
const SEI_LOG = NamedTuple{(:s,:e,:i,:d),NTuple{4,Int}}[]

function sei_counts(X)
    s=e=i=d=0
    for ii in 1:data.n_individuals
        lo,hi = data.sampling_period[ii]
        for t in lo:min(hi,data.n_timepoints)
            v = X[t,ii]
            v==1 && (s+=1); v==2 && (e+=1); v==3 && (i+=1); v==4 && (d+=1)
        end
    end
    (; s,e,i,d)
end

# Wrap the real iFFBS kernel: after it produces the new X, record the counts.
struct TrackingIFFBS{K} <: PracticalBayes.AbstractLatentKernel
    inner::K
end
function PracticalBayes.latent_step(rng, k::TrackingIFFBS, block_names, c::PracticalBayes.ModelConditional)
    out = PracticalBayes.latent_step(rng, k.inner, block_names, c)
    push!(SEI_LOG, sei_counts(out.X))
    out
end

# Rebuild the sampler exactly as the fit script's run_badger_fit does, but swap in
# the tracking kernel and use whatever HMC block names the loaded model declared.
# The fit script defines `run_badger_fit`; we replicate its spl construction here
# by reading the model's own parameter set. Simplest robust approach: reuse the
# fit script's own run by monkeypatching iFFBSKernel is messy — instead we detect
# the continuous names from the model.

# Both fit scripts expose G,NT,NS,NNU,data,raw,loglik,obs_loglik,latent!,
# badger_base, make_hmc_block, HMC_EPS, HMC_L, EtaKernel, NuKernel, iFFBSKernel,
# badger_initial_params. The ONLY difference is a1/b1 presence. Detect it:
const HAS_A1B1 = WHICH == "siler"
const HMC_NAMES = HAS_A1B1 ?
    (:tau,:alpha,:lambda,:beta,:q,:c1,:a1,:b1,:a2,:b2,:thetas,:rhos,:phis) :
    (:tau,:alpha,:lambda,:beta,:q,:c1,:a2,:b2,:thetas,:rhos,:phis)

function run_tracked(nsweeps; seed=13)
    empty!(SEI_LOG)
    m = badger_base(data, data.n_timepoints, data.n_individuals, G, NT, NS, NNU, loglik, obs_loglik)
    spl = Gibbs(
        HMC_NAMES => make_hmc_block(HMC_EPS, HMC_L),
        :etas => EtaKernel(1.0, 1.0),
        :nu   => NuKernel([1.0, 1.0, 1.0]),
        :X    => TrackingIFFBS(iFFBSKernel(latent!)),
    )
    init0 = badger_initial_params(raw; rng=StableRNG(seed))
    X0 = copy(raw.X_init)
    base_init = (; X=X0, tau=init0.tau, alpha=init0.alpha, lambda=init0.lambda,
                 beta=init0.beta, q=init0.q, c1=init0.c1, a2=init0.a2, b2=init0.b2,
                 thetas=init0.thetas, rhos=init0.rhos, phis=init0.phis,
                 etas=init0.etas, nu=hcat(init0.nuE, init0.nuI))
    init = HAS_A1B1 ? merge(base_init, (; a1=init0.a1, b1=init0.b1)) : base_init

    reset_aggregates!(data)
    apply_derived_summaries!(init0, data, X0)

    AbstractMCMC.sample(StableRNG(seed), m, spl, nsweeps;
                        init=init, adtype=ADTYPE, n_adapts=0, discard_initial=0,
                        save_states=(X=:buffer,))
    nothing
end

# Xinit baseline
xinit = sei_counts(copy(raw.X_init))
@printf("=== %s model, %d sweeps ===\n", WHICH, NSW)
@printf("Xinit:     S=%d E=%d I=%d D=%d\n\n", xinit.s, xinit.e, xinit.i, xinit.d)

run_tracked(NSW; seed=13)

@printf("%-6s %8s %8s %8s %8s\n", "sweep", "S", "E", "I", "D")
for (k, c) in enumerate(SEI_LOG)
    (k <= 10 || k % 10 == 0 || k == length(SEI_LOG)) &&
        @printf("%-6d %8d %8d %8d %8d\n", k, c.s, c.e, c.i, c.d)
end

# Save the full per-sweep counts
outdir = joinpath(@__DIR__, "outputs"); mkpath(outdir)
open(joinpath(outdir, "sei_track_$(WHICH).csv"), "w") do io
    println(io, "sweep,S,E,I,D")
    println(io, "0,$(xinit.s),$(xinit.e),$(xinit.i),$(xinit.d)")
    for (k,c) in enumerate(SEI_LOG); println(io, "$k,$(c.s),$(c.e),$(c.i),$(c.d)"); end
end
last = SEI_LOG[end]
@printf("\nfinal:     S=%d E=%d I=%d D=%d  (I: %d -> %d, %.0f%% of Xinit)\n",
        last.s, last.e, last.i, last.d, xinit.i, last.i, 100*last.i/max(xinit.i,1))
