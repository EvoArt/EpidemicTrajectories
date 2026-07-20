#!/usr/bin/env julia
# =============================================================================
# Self-contained server run: badger base model, reststotal coupling, fixed-eps
# HMC (per-parameter step sizes matching badger_ref/run_base_exp.jl), 5000
# sweeps, TIMED. Latent X is dropped from output (`buffer` mode) so memory stays
# flat — the earlier chain-retaining run exhausted RAM at ~15 GB.
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
# Env knobs (all optional except BADGER_DATA_DIR):
#   BADGER_DATA_DIR   path to the RData2 directory              (REQUIRED here)
#   BADGER_SWEEPS     number of sweeps                          (default 5000)
#   BADGER_SEED       RNG seed                                  (default 13)
#   BADGER_OUT        output directory for chain + timing       (default ./badger_out)
#   BADGER_X_SAVE     "buffer" | "disk" | "chain"              (default "buffer")
#   BADGER_X_FLUSH    if X_SAVE=disk, sweeps per flushed file   (default 500)
# =============================================================================

using Pkg

# --- Pinned versions: the exact commits pushed for this run ------------------
const PRACTICALBAYES_URL = "https://github.com/EvoArt/PracticalBayes.git"
const PRACTICALBAYES_REV = "21b0576377893ad50608e0c3da1372b118a54dce"
const EPITRAJ_URL        = "https://github.com/EvoArt/EpidemicTrajectories.git"
const EPITRAJ_REV        = "105a23679260f15259050cbd5da84cd001d835cf"

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

# --- Defaults tuned for a one-shot 5000-sweep timed run ----------------------
# The fit script reads all of these from ENV; set the ones we want to fix here,
# leaving anything the caller already set untouched.
get!(ENV, "BADGER_SWEEPS", "5000")
get!(ENV, "BADGER_SEED", "13")
get!(ENV, "BADGER_X_SAVE", "buffer")      # drop X from output — memory-light
get!(ENV, "BADGER_OUT", joinpath(pwd(), "badger_out"))
# The fit script runs a warmup pass then a main pass; for a single 5000-sweep
# timed run we make the warmup trivial and the main pass the real thing.
get!(ENV, "BADGER_WARMUP_SWEEPS", "1")
ENV["BADGER_MAIN_SWEEPS"] = ENV["BADGER_SWEEPS"]   # main pass = the sweeps we want
ENV["BADGER_RUN"] = "1"

@info "Run configuration" data=RDATA2 sweeps=ENV["BADGER_MAIN_SWEEPS"] seed=ENV["BADGER_SEED"] x_save=ENV["BADGER_X_SAVE"] out=ENV["BADGER_OUT"] threads=Threads.nthreads()

mkpath(ENV["BADGER_OUT"])

# --- Run the fit script (it prints timing and writes the chain + timing file) -
include(joinpath(EXAMPLES, "badger_fit_reststotal_hmc.jl"))

@info "Done. Chain and timing written under" ENV["BADGER_OUT"]
