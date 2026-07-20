# Why is the `etas` block 0.541 s/sweep when its counting loop is 0.0018 s?
#
# bench_eta.jl showed the count accumulation — the only real work the EtaKernel
# does — takes 1.8 ms. bench_blocks.jl measured the whole block at 541 ms. That
# is a 300x gap, so ~99.7% of the block is NOT the algorithm. This isolates
# where it goes.
#
# Prime suspect: the kernel's own body does `X = c.values.X`, and `latent_step`
# is called from gibbs.jl:243-248 as
#     c = ModelConditional(model, values)
#     newvals = latent_step(...)
#     values = merge(values, newvals)
# `values` is a NamedTuple containing X — a 161x2384 Int matrix, ~3 MB. If any
# of that path copies X (or if `data` / globals are accessed untyped inside the
# kernel, forcing boxing), the cost lands here rather than in the arithmetic.
#
# Second suspect, and the more likely one: the EtaKernel closure reads the
# GLOBAL `data` and `NS` (module-level, non-const bindings in the fit script).
# Every access to a non-const global is a dynamic lookup returning `Any`, so the
# whole inner loop runs untyped and allocates. That would not show up in
# bench_eta.jl, which passed `data` in as an ARGUMENT (hence typed).
#
# This script tests exactly that difference: same loop, global vs argument.
#
# Run: JULIA_NUM_THREADS=8 julia --project=examples examples/bench_eta_overhead.jl

ENV["BADGER_RUN"] = "0"
include(joinpath(@__DIR__, "badger_fit_reststotal_hmc.jl"))
using Printf
using StableRNGs: StableRNG

X0 = copy(raw.X_init)

# --- (A) reads `data`/`NS` as GLOBALS, exactly like the real EtaKernel body ---
function eta_counts_globals(X)
    caught = zeros(Int, NS); available = zeros(Int, NS)
    for s in 1:NS
        c = 0; a = 0
        for t in 1:data.n_timepoints
            data.season[t] == s || continue
            for i in 1:data.n_individuals
                g = data.social_group[i, t]
                (g > 0 && data.capt_effort[g, t] == 1) || continue
                X[t, i] == 4 && continue
                a += 1
                c += data.capture[t, i] == 1
            end
        end
        caught[s] = c; available[s] = a
    end
    caught, available
end

# --- (B) identical loop, but `data`/`NS` passed as ARGUMENTS (typed) ---------
function eta_counts_args(data, X, NS)
    caught = zeros(Int, NS); available = zeros(Int, NS)
    for s in 1:NS
        c = 0; a = 0
        for t in 1:data.n_timepoints
            data.season[t] == s || continue
            for i in 1:data.n_individuals
                g = data.social_group[i, t]
                (g > 0 && data.capt_effort[g, t] == 1) || continue
                X[t, i] == 4 && continue
                a += 1
                c += data.capture[t, i] == 1
            end
        end
        caught[s] = c; available[s] = a
    end
    caught, available
end

eta_counts_globals(X0); eta_counts_args(data, X0, NS)   # warm up

n = 10
t_glob = minimum(@elapsed(eta_counts_globals(X0)) for _ in 1:n)
t_args = minimum(@elapsed(eta_counts_args(data, X0, NS)) for _ in 1:n)

a_glob = @allocated eta_counts_globals(X0)
a_args = @allocated eta_counts_args(data, X0, NS)

@printf("globals (as EtaKernel does) : %.4f s   %10d bytes allocated\n", t_glob, a_glob)
@printf("arguments (typed)           : %.4f s   %10d bytes allocated\n", t_args, a_args)
@printf("ratio                       : %.1fx\n\n", t_glob / t_args)

if t_glob / t_args > 5
    println(">>> CONFIRMED: reading `data`/`NS` as untyped globals is the cost.")
    println(">>> Fix is user-side and trivial: make the kernel carry its data in")
    println(">>> typed struct fields (or mark the globals const), so the loop is typed.")
else
    println(">>> Globals are NOT the dominant cost; the overhead is elsewhere")
    println(">>> (ModelConditional construction / merge / X copying in gibbs.jl).")
end
