#!/usr/bin/env julia
# =============================================================================
# Self-contained server run: badger base model, reststotal coupling, fixed-eps
# HMC (per-parameter step sizes matching badger_ref/run_base_exp.jl), 5000
# sweeps, TIMED. Latent X is dropped from output (`buffer` mode) so memory stays
# flat — the earlier chain-retaining run exhausted RAM at ~15 GB.
#
# ---------------------------------------------------------------------------
# 2026-07-20: THE MODEL CHANGED. Runs from before this date are INVALID.
# ---------------------------------------------------------------------------
# The observation likelihood was missing from the log density. `epidemic_loglik`
# covers only the starting state and the transitions; `observation_process` is
# used solely by the iFFBS forward filter and never entered `@addlogprob!`. So
# thetas/rhos/phis received NO likelihood information and were effectively drawn
# from their Beta(1,1) priors — verified by halving all of them and seeing the
# log density change by exactly 0.000e+00.
#
# The fit script now adds `epidemic_obs_loglik(...)` alongside the transition
# term. `etas` stays a conjugate Gibbs block (the test factor is provably
# independent of it, so nothing is double-counted).
#
# Cost of the fix: ~14% per sweep (3.222 -> 3.663 s/sweep measured on an 8-thread
# laptop), NOT ~3x, because the observation term uses the allocation-free
# `observation_weight` path (gradient 0.226 -> 0.075 s; 18 MB -> 16 bytes/call).
#
# Blocking: ONE HMC block for all 61 continuous parameters, with etas/nu
# conjugate. Splitting the test parameters into their own block — mirroring the
# C++ reference's grad_/gradThetasRhos split — was measured 23% SLOWER (4.527 vs
# 3.663 s/sweep), because PracticalBayes evaluates the whole model body per
# block, so two blocks means two full primal evaluations AND two leapfrog
# trajectories. See perf_gap_log.md for the full benchmark table.
#
# ---------------------------------------------------------------------------
# 2026-07-20: PERFORMANCE — iFFBS 3.0x faster, and L halved to match the reference
# ---------------------------------------------------------------------------
# Two package-internal iFFBS fixes (bit-identical output, verified against the
# pre-change code on both X and every aggregate array):
#
#   * `apply_summaries!` — the derived-summaries loops were iterating a Tuple of
#     distinct closure types with a plain `for`, which infers their union and
#     dispatches at RUNTIME on every call. Two lines, 42.7% of a sweep's self
#     time. Now tuple-recursed, so each step sees one concrete function.
#   * in-place forward/backward — sweep-level typed scratch instead of a fresh
#     (and zeroed) `N x N x n_t` cache per individual, and fused loops that write
#     into reused buffers rather than allocating `pred`/`unnorm`/`cond` per (i,t).
#
#     iFFBS sweep: 1.129 s / 717 MB  ->  0.375 s / 85.6 MB   (3.01x, 8.4x less)
#
# HMC_L is now 15, not 30. The reference draws `intL = ceil(runif(0,1)*30)` —
# uniform on {1..30}, mean 15.5, so ~16.5 gradients per HMC step. A fixed L=30
# was doing 1.82x the reference's gradient work for no added fidelity. We take
# the expected value; a randomised-L kernel is buildable from AdvancedHMC's
# public API but needs mutable state in the sampler struct, which is not worth
# the fragility here (see badger_fit_reststotal_hmc.jl for the full reasoning).
# Override with BADGER_HMC_L.
#
# MEASURED, all of today's changes together (8-thread laptop, 5 sweeps):
#
#   before today (L=30, slow iFFBS, no obs likelihood)   3.222 s/sweep
#   corrected model + fast iFFBS + L=15                  2.033 s/sweep
#     iFFBS 0.401 (19.7%) | etas 0.002 | HMC 1.631 (80.2%)
#
# ~1.59x faster per sweep than the pre-existing run AND statistically correct
# (the old one sampled thetas/rhos/phis from their priors). 5000 sweeps:
# ~4.5 h -> ~2.8 h on that laptop.
#
# Installs everything into a throwaway project (a temp depot-local env), pulling
# the two unregistered packages straight from GitHub at the exact commits this
# script was written against, so the run is reproducible and uses the latest code.
#
# USAGE (Linux server):
#   BADGER_DATA_DIR=/path/to/RData2 julia server_run_badger_5000.jl
#
# Only RData2 is needed (WPbadgerData is NOT used by this model's data loader).
# Set JULIA_NUM_THREADS to the core count you want PolyesterForwardDiff to use,
# e.g.  JULIA_NUM_THREADS=16 BADGER_DATA_DIR=/data/RData2 julia server_run_badger_5000.jl
#
# It runs an UNTIMED compile sweep first (to keep JIT cost out of the timings),
# then a TIMED 100-sweep pass, then the TIMED main (5000-sweep) run — each timed
# pass writes its own timing file. ALL output (install logs included) is tee'd to
# a live, line-flushed log under BADGER_OUT, so the run is followable even when
# detached — no shell redirect needed for the log (though your own redirect
# still works and also streams).
#
# USAGE (detach it yourself):
#   JULIA_NUM_THREADS=16 BADGER_DATA_DIR=/data/RData2 nohup julia server_run_badger_5000.jl &
#   tail -f badger_out/run_*.log        # the self-written, live log
#
# Env knobs (all optional except BADGER_DATA_DIR):
#   BADGER_DATA_DIR      path to the RData2 directory              (REQUIRED here)
#   BADGER_SWEEPS        sweeps for the MAIN run                   (default 5000)
#   BADGER_WARMUP_SWEEPS sweeps for the timed warmup pass          (default 100)
#   BADGER_SEED          RNG seed                                  (default 13)
#   BADGER_OUT           output dir for chain + timing + log       (default ./badger_out)
#   BADGER_X_SAVE        "buffer" | "disk" | "chain"              (default "buffer")
#   BADGER_X_FLUSH       if X_SAVE=disk, sweeps per flushed file   (default 500)
#   BADGER_ET_REV        EpidemicTrajectories git rev             (default "master")
#   BADGER_PB_REV        PracticalBayes git rev                   (default: pinned SHA)
#   BADGER_HMC_L         leapfrog steps per HMC move              (default 15)
#   BADGER_MODEL         "siler" | "gompertz"                     (default "siler")
#                        gompertz matches the C++ (classic rcpp): Gompertz-Makeham
#                        survival, no a1/b1, tau~Exp(100), nu~Dir(1,1,1).
# =============================================================================

using Pkg
using Dates: Dates

# --- Live tee-logging: mirror all stdout/stderr to a flushed log file ---------
# Determined up front (before install) so the whole run — package install
# included — is captured. `redirect_stdout` accepts only real OS streams, not a
# custom IO, so we redirect to a `Pipe` and run an async reader that copies each
# chunk to BOTH the original console and the log file, flushing the log after
# every chunk so a `tail -f` streams live (Julia otherwise fully buffers stdout
# to a file, so a plain `> log` would sit empty until the buffer fills).
const OUT_DIR = get(ENV, "BADGER_OUT", joinpath(pwd(), "badger_out"))
mkpath(OUT_DIR)
const LOG_PATH = joinpath(OUT_DIR, "run_$(Dates.format(Dates.now(), "yyyymmdd-HHMMSS")).log")

const _LOGFILE = open(LOG_PATH, "w")
const _ORIG_STDOUT = stdout
const _ORIG_STDERR = stderr
const _TEE_PIPE = Pipe()
Base.link_pipe!(_TEE_PIPE; reader_supports_async = true, writer_supports_async = true)
# One reader task drains the pipe (fed by both redirected stdout and stderr) and
# fans each chunk out to the console + the log, flushing the log every time.
const _TEE_TASK = @async begin
    try
        while !eof(_TEE_PIPE)
            chunk = readavailable(_TEE_PIPE)
            write(_ORIG_STDOUT, chunk); flush(_ORIG_STDOUT)
            write(_LOGFILE, chunk); flush(_LOGFILE)
        end
    catch
    end
end
redirect_stdout(_TEE_PIPE)
redirect_stderr(_TEE_PIPE)
# Restore real streams and finish draining/closing the log at exit, so nothing
# is lost if the run ends (normally or via error).
atexit() do
    try
        redirect_stdout(_ORIG_STDOUT)
        redirect_stderr(_ORIG_STDERR)
        close(_TEE_PIPE.in)
        wait(_TEE_TASK)
        flush(_LOGFILE); close(_LOGFILE)
    catch
    end
end
@info "Logging to" LOG_PATH

# --- Pinned versions: the exact commits pushed for this run ------------------
# A pinned commit hash cannot be written into the same commit that creates it, so
# EPITRAJ_REV defaults to the branch name and is overridable. Set BADGER_ET_REV
# to a specific SHA to reproduce an exact past run; leave it unset to take the
# current tip of master (what you want when running the latest work).
const PRACTICALBAYES_URL = "https://github.com/EvoArt/PracticalBayes.git"
const PRACTICALBAYES_REV = get(ENV, "BADGER_PB_REV",
                               "21b0576377893ad50608e0c3da1372b118a54dce")
const EPITRAJ_URL        = "https://github.com/EvoArt/EpidemicTrajectories.git"
const EPITRAJ_REV        = get(ENV, "BADGER_ET_REV", "master")

# --- Install into a fresh temp project so nothing on the server is disturbed --
const PROJECT_DIR = mktempdir(; prefix = "badger_run_")
Pkg.activate(PROJECT_DIR)
@info "Installing packages into throwaway project" PROJECT_DIR

# Registered packages the run needs directly. (Random/Statistics/Dates are
# stdlibs; AbstractMCMC comes transitively via PracticalBayes but is added
# explicitly so the script's `import AbstractMCMC` resolves.)
Pkg.add([
    Pkg.PackageSpec(name = "CSV"),
    Pkg.PackageSpec(name = "DataFrames"),
    Pkg.PackageSpec(name = "Distributions"),
    Pkg.PackageSpec(name = "AdvancedHMC"),
    Pkg.PackageSpec(name = "ADTypes"),
    Pkg.PackageSpec(name = "PolyesterForwardDiff"),
    Pkg.PackageSpec(name = "JLD2"),
    Pkg.PackageSpec(name = "StableRNGs"),
    Pkg.PackageSpec(name = "AbstractMCMC"),
    Pkg.PackageSpec(name = "Statistics"),
])

# The two unregistered packages, pinned to the exact commits above.
Pkg.add(Pkg.PackageSpec(url = PRACTICALBAYES_URL, rev = PRACTICALBAYES_REV))
Pkg.add(Pkg.PackageSpec(url = EPITRAJ_URL, rev = EPITRAJ_REV))

Pkg.instantiate()
Pkg.precompile()

# --- Locate the example model/data code inside the installed EpidemicTrajectories
# We install the package from GitHub, so its `examples/` (the model + data
# loaders + the fit script) ships alongside `src/`. Find the installed package
# root via the loaded module, then run the reststotal HMC script from there.
using EpidemicTrajectories
const ET_ROOT = normpath(joinpath(dirname(pathof(EpidemicTrajectories)), ".."))
const EXAMPLES = joinpath(ET_ROOT, "examples")
@info "EpidemicTrajectories examples dir" EXAMPLES
isdir(EXAMPLES) || error("could not find the installed examples/ dir at $EXAMPLES")

# --- Sanity-check the data dir before the (long) run ------------------------
haskey(ENV, "BADGER_DATA_DIR") ||
    error("set BADGER_DATA_DIR to the path of your RData2 directory")
const RDATA2 = ENV["BADGER_DATA_DIR"]
isdir(RDATA2) || error("BADGER_DATA_DIR=$RDATA2 is not a directory")
for f in ("dimensions.csv", "Xinit.csv", "TestMat.csv", "Kay.csv", "k.csv")
    isfile(joinpath(RDATA2, f)) ||
        error("expected $f in BADGER_DATA_DIR=$RDATA2 — is this really the RData2 dir?")
end

# --- Config for the two-pass run --------------------------------------------
# The fit script runs a TIMED warmup pass then a TIMED main pass — each writes
# its own `*-<n>iter-*-timing.txt`. We use the warmup as a real 100-sweep timing
# run (a quick read on per-sweep cost + a smoke test that the whole pipeline
# works) BEFORE committing to the long 5000-sweep main run. Set what we want to
# fix here; leave anything the caller already set untouched.
get!(ENV, "BADGER_SEED", "13")
get!(ENV, "BADGER_X_SAVE", "buffer")            # drop X from output — memory-light
get!(ENV, "BADGER_WARMUP_SWEEPS", "100")        # timed 100-sweep pass first...
get!(ENV, "BADGER_MAIN_SWEEPS", get(ENV, "BADGER_SWEEPS", "5000"))  # ...then 5000
ENV["BADGER_OUT"] = OUT_DIR                      # already created; keep log + outputs together
ENV["BADGER_RUN"] = "1"

@info "Run configuration" data=RDATA2 warmup_sweeps=ENV["BADGER_WARMUP_SWEEPS"] main_sweeps=ENV["BADGER_MAIN_SWEEPS"] seed=ENV["BADGER_SEED"] x_save=ENV["BADGER_X_SAVE"] out=ENV["BADGER_OUT"] threads=Threads.nthreads()

# --- Run the fit script (it prints timing and writes the chain + timing file) -
# BADGER_MODEL selects which survival model to fit:
#   "siler"    (default) — Siler survival (a1,b1,a2,b2,c1), matches run_base_exp.jl
#   "gompertz"           — Gompertz-Makeham (a2,b2,c1), matches the C++ classic rcpp,
#                          with C++-matched priors (tau~Exp(100), nu~Dir(1,1,1)).
# The two write DIFFERENTLY-NAMED outputs (badger-reststotal-* vs badger-gompertz-*),
# so both can live in the same BADGER_OUT without clobbering each other.
const BADGER_MODEL = lowercase(get(ENV, "BADGER_MODEL", "siler"))
const FIT_SCRIPT =
    BADGER_MODEL == "siler"    ? "badger_fit_reststotal_hmc.jl" :
    BADGER_MODEL == "gompertz" ? "badger_fit_gompertz_hmc.jl"   :
    error("BADGER_MODEL must be \"siler\" or \"gompertz\", got \"$BADGER_MODEL\"")
@info "Fitting model" BADGER_MODEL FIT_SCRIPT
include(joinpath(EXAMPLES, FIT_SCRIPT))

@info "Done. Chain and timing written under" ENV["BADGER_OUT"]
