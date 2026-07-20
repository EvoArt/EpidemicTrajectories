# Isolate and fix the `etas` conjugate block, measured at 11.7% of a sweep
# (0.541 s) by bench_blocks.jl — nearly half the cost of the whole iFFBS sweep,
# for what is ultimately four Beta draws.
#
# THE PROBLEM. The original loops `for s in 1:NS` on the OUTSIDE and re-scans
# every (t, i) inside, discarding ~75% of the work each pass via
# `data.season[t] == s || continue`. That is NS x T x N = 4 x 161 x 2384 ~= 1.5M
# iterations to accumulate 8 integers.
#
# THE FIX. Season is a function of `t` alone, so one pass over (t, i) can bucket
# straight into `caught[season[t]]` / `available[season[t]]` — T x N ~= 384k
# iterations, a 4x reduction in work, with identical results.
#
# Also fixed: loop order. `X` is `X[t, i]` (t fastest-varying, column-major), so
# the inner loop must run over `t`, not `i`, to walk memory contiguously. The
# original's inner `for i` strides by `n_timepoints` on every access to
# `X[t, i]` and `data.capture[t, i]`. But `data.social_group[i, t]` is the
# OPPOSITE layout ([i, t]), so whichever order we choose, one of the two arrays
# strides. We hoist the season lookup and keep `t` inner (2 of 3 arrays favour
# it), then verify by measurement rather than assuming.
#
# Run: JULIA_NUM_THREADS=8 julia --project=examples examples/bench_eta.jl

ENV["BADGER_RUN"] = "0"
include(joinpath(@__DIR__, "badger_fit_reststotal_hmc.jl"))
using Printf, Random
using StableRNGs: StableRNG

X0 = copy(raw.X_init)

# --- Original: NS passes over the whole (t, i) grid -------------------------
function eta_counts_original(data, X, NS)
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

# --- Fixed: ONE pass, bucketed by season, t innermost -----------------------
function eta_counts_onepass(data, X, NS)
    caught = zeros(Int, NS); available = zeros(Int, NS)
    @inbounds for i in 1:data.n_individuals
        for t in 1:data.n_timepoints
            g = data.social_group[i, t]
            (g > 0 && data.capt_effort[g, t] == 1) || continue
            X[t, i] == 4 && continue
            s = data.season[t]
            available[s] += 1
            caught[s] += data.capture[t, i] == 1
        end
    end
    caught, available
end

# --- Correctness first ------------------------------------------------------
c1, a1 = eta_counts_original(data, X0, NS)
c2, a2 = eta_counts_onepass(data, X0, NS)
println("original  caught=", c1, "  available=", a1)
println("one-pass  caught=", c2, "  available=", a2)
if c1 == c2 && a1 == a2
    println(">>> IDENTICAL counts — the rewrite is exact.\n")
else
    error("MISMATCH — the rewrite changes the answer, do not use it")
end

# --- Then speed -------------------------------------------------------------
eta_counts_original(data, X0, NS); eta_counts_onepass(data, X0, NS)   # warm up

n = 20
t_orig = minimum(@elapsed(eta_counts_original(data, X0, NS)) for _ in 1:n)
t_new  = minimum(@elapsed(eta_counts_onepass(data, X0, NS))  for _ in 1:n)

@printf("original : %.4f s  (min of %d)\n", t_orig, n)
@printf("one-pass : %.4f s  (min of %d)\n", t_new, n)
@printf("speedup  : %.2fx\n", t_orig / t_new)
@printf("\nblock was 0.541 s/sweep (11.7%% of a 4.64 s sweep);\n")
@printf("projected -> %.3f s/sweep, saving %.3f s/sweep\n",
        0.541 * t_new / t_orig, 0.541 * (1 - t_new / t_orig))
