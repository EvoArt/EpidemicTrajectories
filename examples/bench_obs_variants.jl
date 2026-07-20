# Benchmark the observation-likelihood blocking variants.
#
# All variants now include the observation likelihood (the model WITHOUT it is
# simply wrong — thetas/rhos/phis get no likelihood information; see
# check_grad_zeros.jl). The question is how to BLOCK it.
#
#   naive      : one HMC block, everything in it (tau..b2, thetas, rhos, phis,
#                etas), full observation likelihood in @addlogprob!. No conjugate
#                blocks at all except nu. 65 differentiated params.
#   split      : TWO HMC blocks — epidemic/survival (43 params) and tests
#                (thetas/rhos/phis, 18 params) — plus conjugate etas. The test
#                factor only goes in @addlogprob!; etas stays conjugate. This is
#                closest to badger_ref's C++ blocking.
#   onehmc     : ONE HMC block over the 61 epidemic+test params (as the current
#                fit script), but WITH the test likelihood added, plus conjugate
#                etas. Isolates "did splitting help?" from "did adding the obs
#                likelihood cost?".
#   noobs      : the CURRENT (incorrect) model — no observation likelihood at
#                all. Included only as the speed baseline everything else is
#                measured against; its posterior is wrong.
#
# Timing protocol: each variant runs BENCH_WARM sweeps untimed (JIT), then
# BENCH_N sweeps timed, reporting per-block splits via the same shim harness as
# bench_blocks.jl.
#
# Run: JULIA_NUM_THREADS=8 julia --project=examples examples/bench_obs_variants.jl
# Env: BENCH_N (timed sweeps, default 50), BENCH_WARM (untimed, default 5)

ENV["BADGER_RUN"] = "0"
include(joinpath(@__DIR__, "badger_fit_reststotal_hmc.jl"))
include(joinpath(@__DIR__, "badger_model_obssplit.jl"))
using Printf
using Dates: now, format
using StableRNGs: StableRNG

const N_TIMED = parse(Int, get(ENV, "BENCH_N", "50"))
const N_WARM  = parse(Int, get(ENV, "BENCH_WARM", "5"))

# Rebuild data with the SPLIT observation process (same model, factored obs).
bs = badger_data_obssplit(DATA_DIR)
const DATA = bs.data
const RAW  = bs.raw

trans_loglik = epidemic_loglik(DATA)
latent_s!    = epidemic_latent_sampler(DATA)

# Both observation terms use the SCALAR (`observation_weight`) path: the
# likelihood needs one entry of the weight vector, and the vector-returning path
# allocates an array per (i,t) — ~187k allocations per call, of Duals under AD.
# Verified entry-for-entry against the vector path in check_obs_split.jl.
# Set BENCH_OBS_VECTOR=1 to measure the slow path instead (A/B).
const USE_VECTOR_OBS = get(ENV, "BENCH_OBS_VECTOR", "0") == "1"
obs_full = USE_VECTOR_OBS ?
    epidemic_obs_loglik(DATA; observation_process=badger_observations_split) :
    epidemic_obs_loglik(DATA; observation_process=badger_observations_split,
                              observation_weight=badger_obs_split_weight)
obs_tests = USE_VECTOR_OBS ?
    epidemic_obs_loglik(DATA; observation_process=badger_obs_tests) :
    epidemic_obs_loglik(DATA; observation_process=badger_obs_tests,
                              observation_weight=badger_obs_tests_weight)

# --- Timing shims -----------------------------------------------------------
struct TimedLatent{K} <: PracticalBayes.AbstractLatentKernel
    inner::K
    acc::Base.RefValue{Float64}
end
function PracticalBayes.latent_step(rng, k::TimedLatent, block_names, c::PracticalBayes.ModelConditional)
    t0 = time_ns()
    out = PracticalBayes.latent_step(rng, k.inner, block_names, c)
    k.acc[] += (time_ns() - t0) / 1e9
    out
end

# --- Models, one per likelihood shape ---------------------------------------
@model function badger_noobs(data, n_time, n_ind, n_groups, n_tests, n_seasons, n_nu, ll_fn)
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
    @addlogprob! ll_fn(pars, data, X)
end

# `ll_fn` + `obs_fn`: the observation term is a SEPARATE argument so the same
# model serves both "tests only" (etas conjugate) and "full obs" (etas in HMC).
@model function badger_withobs(data, n_time, n_ind, n_groups, n_tests, n_seasons, n_nu, ll_fn, obs_fn)
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

# --- Per-parameter HMC step sizes, sliced per block --------------------------
const EPI_NAMES  = (:tau, :alpha, :lambda, :beta, :q, :c1, :a1, :b1, :a2, :b2)
const TEST_NAMES = (:thetas, :rhos, :phis)
const EPS_EPI  = vcat(0.002, fill(0.2, G), 0.01, 0.05, 0.05, 0.005, 0.001, 0.001, 0.001, 0.001)
const EPS_TEST = fill(0.005, 3 * NT)
const EPS_ETA  = fill(0.005, NS)

function build_variant(name)
    init0 = badger_initial_params(RAW; rng=StableRNG(13))
    X0 = copy(RAW.X_init)
    init = (; X=X0, tau=init0.tau, alpha=init0.alpha, lambda=init0.lambda,
            beta=init0.beta, q=init0.q, c1=init0.c1, a1=init0.a1, b1=init0.b1,
            a2=init0.a2, b2=init0.b2, thetas=init0.thetas, rhos=init0.rhos,
            phis=init0.phis, etas=init0.etas, nu=hcat(init0.nuE, init0.nuI))
    t_iffbs = Ref(0.0); t_eta = Ref(0.0); t_nu = Ref(0.0)
    iffbs_k = TimedLatent(iFFBSKernel(latent_s!), t_iffbs)
    nu_k    = TimedLatent(NuKernel([1.0, 1.0, 1.0], DATA), t_nu)
    eta_k   = TimedLatent(EtaKernel(1.0, 1.0, DATA), t_eta)

    if name == "noobs"
        m = badger_noobs(DATA, DATA.n_timepoints, DATA.n_individuals, G, NT, NS, NNU, trans_loglik)
        spl = Gibbs((EPI_NAMES..., TEST_NAMES...) => make_hmc_block(vcat(EPS_EPI, EPS_TEST), HMC_L),
                    :etas => eta_k, :nu => nu_k, :X => iffbs_k)
        npar = "61 (1 block)"
    elseif name == "onehmc"
        m = badger_withobs(DATA, DATA.n_timepoints, DATA.n_individuals, G, NT, NS, NNU,
                           trans_loglik, obs_tests)
        spl = Gibbs((EPI_NAMES..., TEST_NAMES...) => make_hmc_block(vcat(EPS_EPI, EPS_TEST), HMC_L),
                    :etas => eta_k, :nu => nu_k, :X => iffbs_k)
        npar = "61 (1 block)"
    elseif name == "split"
        m = badger_withobs(DATA, DATA.n_timepoints, DATA.n_individuals, G, NT, NS, NNU,
                           trans_loglik, obs_tests)
        spl = Gibbs(EPI_NAMES  => make_hmc_block(EPS_EPI, HMC_L),
                    TEST_NAMES => make_hmc_block(EPS_TEST, HMC_L),
                    :etas => eta_k, :nu => nu_k, :X => iffbs_k)
        npar = "43 + 18 (2 blocks)"
    elseif name == "naive"
        # Everything in ONE HMC block INCLUDING etas -> full observation
        # likelihood (capture * tests), no conjugate eta block.
        m = badger_withobs(DATA, DATA.n_timepoints, DATA.n_individuals, G, NT, NS, NNU,
                           trans_loglik, obs_full)
        spl = Gibbs((EPI_NAMES..., TEST_NAMES..., :etas) =>
                        make_hmc_block(vcat(EPS_EPI, EPS_TEST, EPS_ETA), HMC_L),
                    :nu => nu_k, :X => iffbs_k)
        npar = "65 (1 block, etas in HMC)"
    else
        error("unknown variant $name")
    end
    (; m, spl, init, init0, X0, t_iffbs, t_eta, t_nu, npar)
end

function run_variant(name; nsweeps)
    v = build_variant(name)
    reset_aggregates!(DATA)
    apply_derived_summaries!(v.init0, DATA, v.X0)
    v.t_iffbs[] = 0.0; v.t_eta[] = 0.0; v.t_nu[] = 0.0
    t0 = time_ns()
    chn = AbstractMCMC.sample(StableRNG(13), v.m, v.spl, nsweeps;
                              init=v.init, adtype=ADTYPE, n_adapts=0,
                              discard_initial=0, save_states=(X=:buffer,))
    total = (time_ns() - t0) / 1e9
    (; name, total, iffbs=v.t_iffbs[], eta=v.t_eta[], nu=v.t_nu[], npar=v.npar, chn)
end

variants = split(get(ENV, "BENCH_VARIANTS", "noobs,onehmc,split,naive"), ",")

# Results are printed AND appended to disk as each variant finishes, so a run
# that is killed or errors partway still leaves everything completed so far.
# (An earlier run buffered all output to a redirect and was killed at 35 min
# having written nothing — no way to see progress or which variant was slow.)
const RESULT_PATH = get(ENV, "BENCH_OUT", joinpath(@__DIR__, "outputs", "bench_obs_variants.txt"))
mkpath(dirname(RESULT_PATH))

function emit(io_line)
    println(io_line)
    flush(stdout)
    open(RESULT_PATH, "a") do io
        println(io, io_line)
        flush(io)
    end
end

emit("# bench_obs_variants  $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
emit("# threads=$(Threads.nthreads())  warm=$N_WARM  timed=$N_TIMED")
emit(@sprintf("%-9s %-26s %9s %9s %9s %9s %9s", "variant", "HMC params", "total", "s/sweep", "iFFBS", "etas", "HMC*"))
emit("-"^88)

results = []
for v in variants
    # Warm up and time each variant back to back, so a variant's compile cost is
    # paid immediately before its own timed run rather than all four up front.
    t_warm = @elapsed run_variant(v; nsweeps=N_WARM)
    @printf("  [%s] warm-up (%d sweeps, incl. compile): %.1f s — timing %d sweeps...\n",
            v, N_WARM, t_warm, N_TIMED)
    flush(stdout)

    r = run_variant(v; nsweeps=N_TIMED)
    push!(results, r)
    hmc = r.total - r.iffbs - r.eta - r.nu
    emit(@sprintf("%-9s %-26s %9.1f %9.3f %9.3f %9.3f %9.3f",
                  r.name, r.npar, r.total, r.total/N_TIMED,
                  r.iffbs/N_TIMED, r.eta/N_TIMED, hmc/N_TIMED))
end

emit("")
emit("* HMC is a residual (everything outside a timed kernel).")
emit("  `noobs` has NO observation likelihood and is statistically WRONG;")
emit("  it is here only as the speed baseline.")
emit("")
if !isempty(results)
    base = results[1].total
    for r in results
        emit(@sprintf("%-9s %.3f s/sweep   (%.2fx vs %s)",
                      r.name, r.total/N_TIMED, base/r.total, results[1].name))
    end
end
println("\nresults also written to: ", RESULT_PATH)
