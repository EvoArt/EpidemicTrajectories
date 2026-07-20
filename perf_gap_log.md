# Closing the gap to the C++ reference — working log

Goal: the badger power-user model (`examples/server_run_badger_5000.jl`) is ~2x
slower than `badger_ref/classic rcpp`. Close that WITHOUT the package learning
anything about user code ahead of time — every speedup must be an opt-in
user-supplied function or a generic core improvement, never a badger special case.

Prior state (`badger_repro_log.md`, 2026-07-19): three additive power-user levers
(`coupled_transitions`, `coupling_trans_mat`, `rest_contribution`) took the iFFBS
sweep 2.737s -> 0.845s, ~1.5x faster than the Julia reference's sweep. Gradient
untouched at ~0.10s.

## What has been changed so far

**Package code (`src/`) — two additions, both opt-in seams:**
- `src/build.jl`: **`epidemic_obs_loglik(data; observation_process=...,
  observation_weight=...)`**, exported from `src/EpidemicTrajectories.jl`. The
  observation likelihood was MISSING from the model entirely (a bug — see the
  entry below); this supplies it. `observation_process` lets a user pass a
  REDUCED factor so conjugate parameters stay out of the differentiated density;
  `observation_weight` is the allocation-free scalar path.
- Nothing else. `iffbs.jl`, `transitions.jl`, `data.jl`, `aggregates.jl`,
  `spec.jl` are untouched.

Both keywords follow the established pattern (`rest_contribution`,
`coupling_trans_mat`): the package takes a user function on trust and never
learns what it means.

**User code:**
- `examples/badger_fit_reststotal_hmc.jl` — typed `data` fields on
  `EtaKernel`/`NuKernel`, single-pass eta count loop, parameterised `iFFBSKernel{F}`.
- `examples/badger_model_obssplit.jl` — the factored observation process
  (capture x tests) plus the three scalar weight functions.

**Benchmarks/checks** (not imported by the package):
`bench_blocks.jl`, `bench_eta.jl`, `bench_eta_overhead.jl`, `bench_obs_cost.jl`,
`bench_obs_variants.jl`, `profile_gradient.jl`, `check_grad_zeros.jl`,
`check_kernel_fix_exact.jl`, `check_obs_split.jl`, and this log.

## Measurement caveat (2026-07-20)

All timings below are TENTATIVE: taken on a laptop whose power state changed
mid-session, with other Julia jobs running concurrently. Treat them as
order-of-magnitude only. The structural findings (next section) are read off
source code and do NOT depend on the machine.

| date | what | iFFBS sweep | gradient | notes |
|---|---|---|---|---|
| 2026-07-20 | reststotal + PolyesterFD, 8 threads | min 1.46 / med 1.50 / mean 1.70 s | min 0.078 / med 0.092 s | contended machine; earlier quiet-ish run gave iFFBS min 1.09 / med 1.20, grad min 0.070 / med 0.075 |

## The cost model — MEASURED, not inferred

`examples/bench_blocks.jl` wraps each Gibbs kernel in a timing shim and runs the
real `sample` call (same model/sampler/data as the 5000-sweep script), after an
untimed compile pass. 10 sweeps, 8 threads, contended machine:

| block | total (s) | per sweep | share |
|---|---|---|---|
| iFFBS (X) | 12.09 | 1.209 | **26.1%** |
| conjugate etas | 5.41 | 0.541 | **11.7%** |
| conjugate nu | 0.22 | 0.022 | 0.5% |
| HMC + overhead (residual) | 28.67 | 2.867 | **61.8%** |
| TOTAL | 46.40 | 4.640 | 100% |

This confirms the reframing and adds one surprise:

1. **HMC/AD is ~62% of the sweep; iFFBS is ~26%.** Every optimisation to date has
   been spent on the 26%. Even making iFFBS free would only give ~1.35x.
2. **The `etas` conjugate update is 11.7% — nearly half the cost of iFFBS**, for
   what should be a trivial Beta draw. Looking at `EtaKernel` in the fit script,
   it loops `n_seasons x n_timepoints x n_individuals` = 4 x 161 x 2384 ~= 1.5M
   iterations per sweep to accumulate 4 counts. It is O(S x T x N) when it could
   be O(T x N) (one pass, bucket by season) — an easy ~4x on that block, pure
   user-side change, no package involvement. Cheapest real win available.

At 4.64 s/sweep, 5000 sweeps is ~6.4 hours.

## C++ source analysis — the three structural differences

Read from `badger_ref/classic rcpp/BIID/src/`. These are design differences, not
implementation-quality differences, and they compound.

### 1. The C++ gradient is HAND-DERIVED, not autodiffed (the big one)

`grad_.cpp` computes the gradient analytically in **one pass over `(i,t)`**, with
a `Float64` accumulator per parameter (`likeas`, `likelam`, `likeb`, `likeq`,
`liketau`, `likea2`, `likeb2`, `likec1`). Helpers `Dlogpt_a2/b2/c1`,
`DlogS_a2/b2/c1` are closed-form derivatives of the Siler survival term.
`gradThetasRhos.cpp` does the same for the test parameters.

Cost: **~1x the loglik**, with scalar doubles throughout.

Ours: `epidemic_loglik` -> ForwardDiff over 61 parameters. With
`AutoPolyesterForwardDiff` and chunk size C, that is `ceil(61/C)` passes over the
whole `(i,t)` loop, each carrying `Dual` numbers with a C-wide partials tuple.
Even at the best case this is several times the work of one scalar pass, and it
touches several times the memory per element.

**This is the primary source of the 2x.** It is also exactly the kind of thing
the "hardcoded Julia version that was close to C++" would have had.

### 2. The C++ randomizes trajectory length; we do a fixed L

`HMC_.cpp:71` and `HMC_thetas_rhos.cpp:51`:
```cpp
int intL = ceil(runif(1,0,1)[0]*L);
```
With `L=30`, the C++ averages **~15.5 gradient calls per sweep**, not 30. Our
`HMC(L=30)` (AdvancedHMC) does a fixed 30 every sweep.

So on gradient COUNT alone we do ~1.94x the work of the reference — before any
per-gradient cost difference. This is a genuine algorithmic difference in the
samplers being compared, not a Julia-vs-C++ effect.

### 3. The C++ splits parameters into two blocks — and we differentiate 18
###    parameters the likelihood does not even depend on

`grad_`/`logPost_HMC` handle ONLY the epidemic/survival parameters (alphas,
lambda, beta, q, tau, a2, b2, c1) — read `logPost_HMC.cpp`: the loglik loop
contains transition and survival terms and **nothing about tests**.
`gradThetasRhos` handles thetas/rhos separately, looping over test observations
only. Two small independent problems, never one big one.

We put all 61 parameters in ONE HMC block over ONE `epidemic_loglik`. But
**`epidemic_loglik` (src/build.jl:81-107) sums only the starting-state term and
`transition_prob` — it never calls `observation_process` at all.** Grepped: the
only uses of `observation_process` in the package are `src/iffbs.jl:46,60` (the
forward filter). In the badger model `thetas`/`rhos`/`phis`/`etas` appear ONLY in
`badger_observations`.

**Therefore 18 of our 61 differentiated parameters (thetas, rhos, phis — 3 x 6
tests) contribute nothing to the log-density being differentiated.** Their
gradient entries are prior/Jacobian terms only. Forward-mode AD cost scales with
`ceil(n_params / chunksize)` passes, so we are paying roughly
`61/43 ~= 1.4x` more AD passes than the problem needs — differentiating a
constant with respect to 18 directions.

**VERIFIED EMPIRICALLY** (`examples/check_grad_zeros.jl`):

```
epidemic_loglik with original thetas/rhos/phis/etas : -81609.8745139124
epidemic_loglik with all four HALVED               : -81609.8745139124
difference                                          : 0.000e+00
```

Bit-identical. And the gradient magnitudes confirm it — epidemic/survival
entries are O(1e2-1e4) (`tau` -1.6e4, `lambda` +3.0e2, `beta` -2.0e2), while
every thetas/rhos/phis entry is O(0.1-1), i.e. the Beta(1,1) logit-Jacobian term
alone.

**This is a modelling question before it is a performance question.** In the
reference, thetas/rhos/phis are informed by the TEST likelihood
(`gradThetasRhos.cpp` differentiates exactly that). In our model those
parameters enter only `badger_observations`, which reaches the sampler through
the iFFBS forward filter but never enters `@addlogprob!`. So the HMC block is
proposing moves in 18 directions on prior information alone.

Whether that is intended needs the user's call — the iFFBS filter does use the
observation weights, so the test parameters are not unused, but they are not
being informed by an explicit likelihood term in the HMC block the way the
reference informs them. **Do not "optimise" this away until that is settled**;
the fix might be to ADD the observation likelihood (making the model match the
reference and the gradient more expensive), not to drop the parameters.

### 4. Data access patterns (Julia-side implications)

- **`X` layout.** C++ `arma::imat X(m, maxt)` is `X(i, j)` = individual-major,
  column-major storage -> **`i` fastest-varying**. Its hot loops are
  `for i { for j }`, i.e. striding over `j`... but `grad_` reads `X(i,j-1)`,
  `X(i,j)` adjacently in `j`, so it strides. Ours is `X[t, i]` with `t`
  fastest-varying and loops `for i { for t }` — **ours is actually the
  cache-friendly one for the loglik/gradient loop.** No change indicated here.
- **`logProbRest(ttt, s, i)`** in `iFFBScalcLogProbRest.cpp` is `(maxt-1, 4, m)`
  — time fastest, then state. The 4 states written per `(i,t)` are NOT contiguous
  (stride `maxt-1`). So the reference's own coupling cache is not perfectly laid
  out either; its win is algorithmic (running total), which we already match via
  `rest_contribution`.
- **Per-sweep amortization.** `MCMCiFFBS_.cpp:661-711` precomputes
  `logProbStoSgivenSorE/I/D` and `logProbStoEgivenSorE/I/D` as `G x (maxt-1)`
  matrices ONCE per sweep, then every `(i,t)` lookup is a single array read. We
  recompute the three scenario FOIs on demand inside `rest_contribution`
  (`_grp_foi`, involving a `pow`). `G x maxt = 34 x 161 = 5474` entries — trivial
  to precompute. **This is a cheap, in-philosophy win**: it is user code in
  `badger_model_reststotal.jl`, needing at most a package hook to run a
  user-supplied callback once per sweep.

### 5. Per-sweep amortization of parameter-derived quantities

`MCMCiFFBS_.cpp:624-711` precomputes, ONCE per sweep, right after the new
parameters are set:
- `LogProbSurvMat(i,t)` / `LogProbDyingMat(i,t)` — the Siler survival term, an
  `exp`/`expm1` chain, for every `(i, t)`.
- `logProbEtoE[i]` / `logProbEtoI[i]` — the progression terms.
- `logProbStoSgivenSorE/I/D`, `logProbStoEgivenSorE/I/D` — `G x (maxt-1)` FOI
  matrices for all three focal scenarios.

Every subsequent iFFBS lookup is then a single array read. Ours recomputes
`siler_survival` (with `exp`, two `expm1`) per `(i,t)` per candidate state
inside the forward filter, and `_grp_foi` (including a `pow`) per candidate.

`G x maxt = 34 x 161 = 5474` entries; `m x maxt = 2384 x 161 = 384k` for the
survival matrix. Both trivial to precompute once per sweep.

This is the SAME seam `coupling_trans_mat` uses and is safe for the same
reason: it feeds the iFFBS path only, which is never differentiated. It needs a
package hook — "run this user callback once per sweep, before the sweep" — which
is in-philosophy (the package runs a user function; it does not know what the
cache means).

### 6. The etas block: 300x more expensive than its own arithmetic

`bench_blocks.jl` put the etas conjugate block at 0.541 s/sweep (11.7%).
`bench_eta.jl` timed the counting loop it performs — the only real work — at
**0.0018 s**. A 300x gap.

Two fixes, independent:
- *Algorithmic*: the loop is `for s in 1:NS` OUTSIDE, re-scanning every `(t,i)`
  and discarding ~75% via `season[t] == s || continue`. One pass bucketing by
  `season[t]` gives **1.92x, verified bit-identical counts**
  (`caught=[1579,3466,4463,2599]` both ways).
- *Type stability*: `examples/badger_fit_reststotal_hmc.jl:57` binds
  `data, raw = b.data, b.raw` as **non-const globals**. `EtaKernel`'s body reads
  `data.n_timepoints`, `data.season[t]`, `data.social_group[i,t]` etc. directly
  from that global, so every access is a dynamic lookup returning `Any` and the
  whole 1.5M-iteration loop runs untyped. `bench_eta.jl` did not see this
  because it passed `data` as an argument. This is exactly the failure mode
  CLAUDE.md's performance section warns about ("Concrete types... invisible to
  inference checks. Benchmark, don't infer") — reappearing in USER code rather
  than package code.

## Ranked plan

**DONE — 0. Type-stability fix in the conjugate kernels (user-side, free).**
`EtaKernel`/`NuKernel`/`iFFBSKernel` now carry their data/closure in
type-parameterised fields instead of reading non-const module globals; the etas
count loop is single-pass (bucketed by season, verified bit-identical counts).

**MEASURED RESULT (same harness, 10 sweeps):**

| block | before | after (run 1) | after (run 2) |
|---|---|---|---|
| iFFBS (X) | 1.209 s | 0.859 s | 1.407 s |
| conjugate etas | 0.541 s | **0.002 s** | **0.002 s** |
| conjugate nu | 0.022 s | 0.000 s | 0.000 s |
| HMC + overhead | 2.867 s | 2.502 s | 3.129 s |
| **TOTAL** | **4.640 s** | **3.363 s** | **4.538 s** |

**Read this table carefully — only the etas row is trustworthy.**

- **etas: 0.541 -> 0.002 s (270x) is REAL.** An effect that size cannot be
  machine noise, and it is reproduced identically in both post-fix runs.
- **Everything else in this table is contaminated.** Run 2 executed CONCURRENTLY
  with another benchmark on the same 8 cores, so its iFFBS (1.407 s) and HMC
  (3.129 s) are inflated. Run 1's iFFBS (0.859 s) vs the pre-fix 1.209 s is NOT
  evidence of a speedup — run 2 shows 1.407 s for the same or better code.
- **An earlier version of this log claimed "iFFBS also got 1.41x faster, almost
  certainly GC pressure." That claim was WRONG** and is retracted. It was one
  measurement of a noisy quantity on a contended machine, reasoned into a
  mechanism after the fact. The 164 MB/sweep the old kernel allocated is real
  and *may* have taxed other blocks, but this data cannot establish it.
- **Whole-sweep totals are therefore also unreliable**; the honest statement is
  "the etas block went from 11.7% of the sweep to ~0.1%", which implies roughly
  1.13x overall, NOT the 1.38x run 1 suggests.

**To measure the rest properly, re-run on a quiet machine with no concurrent
jobs.** The one number worth carrying forward from today is the etas fix.

Recovering ~11.6% of the sweep (the etas block) with no algorithmic change to
the sampler, no package change, and no loss of generality.

**Where the remaining gap goes, as RATIOS** (ratios survive a noisy machine;
absolute seconds do not, so none are quoted here):

| step | effect on the HMC block | gated on | outcome |
|---|---|---|---|
| randomized `intL` (match C++, avg L/2 vs fixed L) | ~2x fewer gradient calls | nothing — pure sampler config | **untested** |
| drop the 18 no-information params from AD (61->43) | ~1.4x fewer AD chunk-passes | the modelling question in §3 | **VOID** — see below |

**The second row is now void.** It assumed thetas/rhos/phis could be dropped from
the differentiated density. The user confirmed the missing observation likelihood
was a model BUG, so those parameters must stay and must be informed — the
question was never "remove them" but "where do they belong". And splitting them
into their own block, measured, is 23% SLOWER (see the variants entry below).

The first row stands and is now the largest single remaining lever:
- the `intL` one is us doing ~1.94x the leapfrog work the reference does, BY
  CHOICE (fixed L=30 vs its randomized average of ~15.5);
- the second is us differentiating a constant in 18 directions (§3), which is
  gated on the modelling question, not on performance.

So the ~2x gap looks reachable WITHOUT an analytic gradient — the analytic
gradient (plan item 3) is what would take us past the C++ rather than merely
level with it.

1. **Resolve the thetas/rhos/phis modelling question FIRST** (see §3). Either
   the observation likelihood belongs in `@addlogprob!` (model currently differs
   from the reference; gradient gets MORE expensive but correct), or those 18
   parameters should not be in the HMC block at all (gradient gets ~1.4x
   cheaper). **This is a correctness decision, and it changes the direction of
   the optimisation — do not proceed past it on perf grounds alone.**
2. ~~**Split the HMC block**~~ — **TRIED AND REJECTED, measured 23% SLOWER**
   (4.527 vs 3.663 s/sweep). PracticalBayes evaluates the whole model body per
   block, so two blocks = two full primals + two L=30 trajectories. The partials
   saving does not pay for that. Keep ONE HMC block. See the entry below.
3. **Analytic / partially-analytic gradient as a power-user lever.** Now the
   LARGEST remaining lever, and the profile supports it: the gradient has zero
   dispatch and zero GC, so it is pure arithmetic — the only way to make it
   substantially cheaper is to do LESS arithmetic, which is exactly what the
   C++'s hand-derived `grad_` does (one scalar pass vs 61-partial forward-mode).
   Same philosophy as `rest_contribution`: an optional seam for a user-supplied
   gradient, defaulting to AD. Needs design + sign-off.

3b. **Keep rates in log-space** (smaller, user-side). `badger_infection` returns
   `-expm1(-foi)` and `loglik` then takes `log` of it; the C++ never leaves log
   space (`logProbStoSgivenSorE = -alpha - beta*inf`). Worth ~8-10% on the
   profile's `exp`/`log` share — NOT the 25% an earlier noisy run suggested.
   Would need the package to accept log-rates (a `log_rates=true` style flag on
   `TransitionSpec`), so it is a package change, not pure user code.
4. **Per-sweep precompute hook** for survival/FOI matrices (§5) — in-philosophy,
   needs a small package hook ("run this user callback once per sweep").
5. **Match the reference's randomized `intL`** (§2) if we want a like-for-like
   comparison; currently we do ~1.94x the leapfrog work by construction. This is
   arguably the single largest "gap" contributor that is not a real efficiency
   difference at all.
6. `@batch` on the iFFBS individual loop — **not safe as-is**: individuals within
   a sweep share mutable aggregate state (the reverse->refilter->reapply
   invariant), so they are not independent. The repro log also found the
   reference's own fine-grained `@batch` to be a wash. Low priority, high risk.

### 2026-07-20 — gradient profile (post obs-scalar fix)

`examples/profile_gradient.jl`, 20 gradient calls, quiet machine, gradient
~0.076-0.084 s. Printed total%-sorted FIRST per the CLAUDE.md warning.

**The gradient is structurally clean: ZERO runtime dispatch, essentially zero GC**
(one 1.4% `Array` line). It is raw floating-point work. The earlier
type-stability and allocation classes of win are exhausted here.

Self%: `*` 12.6, `+` 8.7, `fma_llvm` 8.1, `<` 7.6, `getindex` 6.6+4.7,
`muladd` 6.2, `-` 4.4, `transition_prob` 2.9, `_log` 1.6, `exp` 1.5.

**Package/model frames only, by TOTAL% — this is the actionable view:**

| total% | frame |
|---|---|
| 58.6 | `loglik` — `src/build.jl:97` (the TRANSITION likelihood) |
| 54.1 | `transition_prob` — `src/transitions.jl:83` |
| 37.8 / 28.2 | `_accum_row` — `transitions.jl:97` / `:101` (the rate-tuple walk) |
| 22.1 | `#22` — `src/spec.jl:97` (a generated rate closure) |
| 13.1 | `loglik` — `build.jl:99` (the `log(p + 1e-12)` line) |
| **10.5** | `obs_loglik` — `src/build.jl:216` |
| 9.1 | `siler_survival` — `examples/badger_model.jl:82` |

**Two conclusions:**

1. **The observation likelihood is now only 10.5%**, against 58.6% for the
   transition likelihood. The `observation_weight` scalar path did its job; the
   obs term is no longer a target.
2. **The remaining cost is `transition_prob` -> `_accum_row` -> rate closures**
   (54% / 38% / 22%). That is the package's central hot path, already optimised
   once (tuple recursion, no allocation). Further gains there are arithmetic-level,
   not structural.

**CORRECTION to an earlier reading in this session.** A first profiling run showed
`log` at 25.2% total and I reported it as "the one standout", with a proposed
log-space rewrite. The cleaner re-run puts `_log` at **1.6% self** and `exp` at
8.5% total. The 25% figure came from a noisier run (fewer distinguishing frames,
scaffolding included) and **should not be acted on**. The log-space observation
about the C++ (`logProbStoSgivenSorE = -alpha - beta*inf` never leaves log space,
whereas `badger_infection` does `-expm1(-foi)` and then `loglik` takes `log`) is
still STRUCTURALLY true and worth ~8-10% at most on this evidence — not 25%.

### 2026-07-20 — iFFBS: the derived-summaries loop was 42.7% of the sweep

The user questioned whether ~1 s per iFFBS sweep was plausible on the optimised
reststotal path. It was not. Measured first, then profiled:

    iFFBS sweep: 1.129 s   751,503,904 bytes (716.7 MB)
    186,850 (i,t) cells  =>  4,022 bytes PER CELL

4 KB per cell, to produce a 4-element probability vector.

**Inspection pointed at the wrong thing.** The obvious suspects were the ~8
array allocations per `(i,t)` in `forward_filter`/`backward_sample!` (`pred`,
`unnorm`, `cond`, the `./ z` temporaries). But 8 small arrays is ~114 MB of the
717 MB — the arithmetic said ~48 small arrays per cell, so most of the cost was
somewhere else. **Profiling, not reading, found it:**

| self% | line | flags |
|---|---|---|
| **23.1** | `iffbs.jl:109` — `ds(...; reverse=true)` | **gc? Y  dispatch? Y** |
| **19.6** | `iffbs.jl:118` — `ds(...)` | **gc? Y  dispatch? Y** |
| 18.8 | `Array` (boot.jl:477) | gc? Y |

The whole forward filter was only 15.5% total; `rest_contribution` 6.7%.

**The cause.** `data.derived_summaries` is a `Tuple` of four DIFFERENT concrete
closure types. `aggregates.jl` already documented that it is a Tuple
"so `for ds in data.derived_summaries` in the hot loops specialises on one
concrete function at a time" — **but that is not what a `for` loop does.** It
infers the loop variable as the UNION of the element types and dispatches at
runtime on every call, boxing arguments. Storing them as a Tuple is necessary but
not sufficient; `rate_fns` got explicit tuple recursion (`_fill_rates!`) for
exactly this reason, and `derived_summaries` never did.

**The fix.** `apply_summaries!(summaries, model, data, X, s, i, t, reverse)` in
`aggregates.jl` — recurses over the tuple so each step sees one concrete function
type. `reverse` is passed POSITIONALLY, not as a keyword: a kwarg on a call the
compiler cannot resolve forces the slow kwarg path (a NamedTuple allocation per
call). Applied at all six call sites (`iffbs.jl` x2, `transitions.jl` x2 inside
`rest_contribution`, `build.jl` x2).

**Result — this change ALONE:**

| | before | after | gain |
|---|---|---|---|
| iFFBS sweep | 1.129 s | **0.435 s** | **2.60x** |
| allocated | 716.7 MB | **199.5 MB** | 3.59x |
| per (i,t) cell | 4,022 B | 1,120 B | |

The lesson is the one CLAUDE.md already records and this session keeps
re-learning: **profile, don't infer.** Reading the code pointed confidently at
the forward filter; the profiler pointed at two lines that looked inert.

### 2026-07-20 — iFFBS change 2: in-place forward/backward

The allocations inspection originally pointed at, now actually removed:

* **Sweep-level scratch** (`_filter_scratch`). `probs` and `trans_cache` are
  created by the forward pass, consumed by the backward pass, and dead by the end
  of `iffbs_individual!` — nothing holds a reference across individuals (checked
  before relying on it). So one buffer sized to the LONGEST sampling window
  serves the whole sweep, each individual using the leading `1:n_t` slice.
  Previously every individual allocated AND ZEROED a fresh `N x N x n_t`
  `trans_cache` (~20 KB) plus an `n_t x N` `probs` — ~48 MB of pointless zeroing
  per badger sweep, in a fresh cold region each time.
* **Fused forward loop** (`forward_filter!`). `pred = trans' * probs[j-1, :]`
  (which also materialised a transpose), `unnorm = pred .* obs_w .* rest_w`, and
  the `./ z` normalisation were three fresh N-vectors per `(i,t)`. Now one pass
  over the N candidate states writing into a reused buffer, with an explicit
  matvec — N is 4 in the reference model, well below where BLAS pays for its call
  overhead.
* **Backward pass**. `cond = probs[j,:] .* trans[:,bnext]` and `cond ./ z` were
  two more N-vectors per timepoint (~374k allocations per sweep); both now write
  into a reused buffer.

The scratch lives in a module-level `Ref` rather than being threaded through the
signatures, which is safe ONLY because `iffbs!` is single-threaded by
construction: individuals within a sweep share mutable aggregate state through
the reverse/re-apply invariant, so they cannot be run in parallel anyway. If that
ever changes, this becomes per-task state and the `Ref` must go.

**VERIFIED EXACT** (`/tmp/iffbs_exact.jl`, reference generated from the
pre-optimisation commit, 5 sweeps from a fixed seed):

```
X        : IDENTICAL
aggregates: IDENTICAL
>>> PASS: optimised iFFBS reproduces the reference sweep exactly.
```

Both the full 161x2384 `X` and all four aggregate arrays. This mattered: reusing
one buffer across individuals is exactly the shape of change that can silently
leak state between them, and a wrong-but-fast sampler is worthless.

**FIRST ATTEMPT MADE IT SLOWER — worth recording.** The scratch was initially
held in a `const Ref{Any}(nothing)` containing a NamedTuple. Result:

| | sweep | allocated |
|---|---|---|
| after change 1 only | 0.435 s | 199.5 MB |
| + in-place, `Ref{Any}` scratch | **0.546 s** | 139.5 MB |

**Allocation fell 30% and the sweep got 26% SLOWER.** `Ref{Any}` hands back
`Any`, so every `s.probs` / `s.cur` / `s.w` was a dynamic field lookup and the
buffers arrived in `forward_filter!` / `backward_sample!` untyped — reintroducing
type instability across the entire inner loop, ~4,768 scratch fetches per sweep
plus untyped array accesses inside them. The instability cost more than the
allocations it removed.

Fixed by making the scratch a concretely-typed `struct FilterScratch` with a
`Ref{Union{Nothing,FilterScratch}}` and a `::FilterScratch` return annotation.

Two lessons, both already in CLAUDE.md and both re-learned the hard way here:
1. **Fewer allocations is not automatically faster.** Allocation and type
   stability are separate axes, and trading one for the other can lose.
2. **Measure each change on its own.** Had these two changes been made together,
   the summaries fix (2.6x) would have masked the in-place regression entirely,
   and the `Ref{Any}` instability would have shipped invisibly.

**With the typed struct** (10 sweeps each, same harness, quiet machine):

| version | min | median | allocated |
|---|---|---|---|
| after change 1 only | — | 0.435 s | 199.5 MB |
| + in-place, `Ref{Any}` | 0.447 | 0.510 s | 139.5 MB |
| **+ in-place, typed** | **0.338** | **0.375 s** | **85.6 MB** |

Re-verified bit-identical after the typing fix (`X` and all four aggregates).

### iFFBS: cumulative result

| stage | sweep | allocated |
|---|---|---|
| original | 1.129 s | 717.0 MB |
| + `apply_summaries!` tuple recursion | 0.435 s | 199.5 MB |
| + in-place forward/backward (typed scratch) | **0.375 s** | **85.6 MB** |

**3.01x faster, 8.4x less allocation, bit-identical output.**

Effect on the whole Gibbs sweep: 3.663 -> ~2.909 s (**1.26x**), i.e. 5000 sweeps
goes from ~5.09 h to ~4.04 h on this laptop. Both changes are package internals
(`src/aggregates.jl`, `src/iffbs.jl`) — no API change, no user involvement.

Note the split shifts again: iFFBS is now ~13% of a sweep (0.375 of 2.909) and
the HMC/AD block ~85%. The gradient is once more the only thing that matters,
and per the gradient profile it has no dispatch and no GC left to remove — so
the remaining levers are the two structural ones (randomized `intL`, and a
hand-derived gradient seam), not micro-optimisation.

### 2026-07-20 — L=30 -> 15: we were doing 1.82x the reference's gradient work

**Verified the reference's behaviour by direct simulation**, not by reading:
`ceil(runif(0,1)*30)` is uniform on {1,...,30}, mean 15.498 over 2M draws
(exactly the analytic (1+L)/2 = 15.5). The C++ then does `intL - 1` gradient
calls in the loop plus a half-step either side = `intL + 1` ~= **16.5 gradients
per HMC step**. Our fixed `HMC(L=30)` does 30. So we were doing **1.82x** the
reference's gradient work — pure cost, no added fidelity.

**Checked that L is not doubled/rescaled anywhere before reaching the HMC code**
(user's request, and worth it): `runmodel.R:39` sets `L <- 30` and passes it
unchanged (`L=L`, line 359); `MCMCiFFBS_.cpp` mentions `L` exactly three times —
the parameter and two pass-throughs (lines 805, 851); `HMC_.cpp:71` and
`HMC_thetas_rhos.cpp:51` both use it directly as `ceil(runif(0,1)*L)`. No
doubling. Counting the loop body: 1 initial gradient, then `intL - 1` iterations
of (position update + gradient), then a final position update + gradient — i.e.
**`intL` position updates and `intL + 1` gradients**. Our `HMC(L=15)` does 15
position updates and ~16 gradients: correctly matched.

*Separate fidelity note, NOT a count issue.* The reference divides the momentum
update by 2 at EVERY step including interior ones (`HMC_.cpp:77-80`), where
textbook leapfrog uses full interior steps and halves only at the ends. So its
effective interior step size is half nominal. That is an epsilon difference, not
a trajectory-length one, and our epsilons are matched separately via the mass
matrix. Flagged here because it means our sampler is not a step-for-step clone of
the reference's dynamics even with L matched — worth knowing if posteriors are
ever compared draw-for-draw rather than in distribution.

**Can we randomise L without touching internals? YES — and it was still
rejected.** The initial answer here was "no public option", which was WRONG:
`Trajectory`, `HMCKernel`, `FixedNSteps`, `EndPointTS` are all exported, and
`HMCSampler(κ, metric, adaptor)` passes any kernel straight through
(`make_kernel(spl::HMCSampler, _) = spl.κ`). `HMC`/`NUTS` are genuinely just
conveniences over that, so a custom termination criterion is supported API.

It was implemented, then reverted on the user's "only if it's clean and unlikely
to cause a slowdown" constraint, because of one real wrinkle:
`AdvancedHMC.nsteps(τ)` takes no RNG and is called TWICE per transition — for the
trajectory (trajectory.jl:337) and for the reported `n_steps` stat (:288).
Drawing independently in each simulates a different L than it reports. The
workaround needs a `Ref{Int}` in the termination criterion, redrawn from the
momentum-refreshment hook. That works but puts MUTABLE STATE in the sampler and
depends on `refresh` firing exactly once per transition — an ordering assumption
about a package we do not control, which if broken fails SILENTLY. Not worth it
for a 5000-sweep production run when a fixed L=15 has the same expected cost.

Tradeoff recorded: a fixed L can hit resonance in geometries where a randomised
one would not. If the chain shows that, the randomised kernel is the fix — not a
larger fixed L.

**Result (5 sweeps, 8-thread laptop):**

| | s/sweep | iFFBS | etas | HMC |
|---|---|---|---|---|
| L=30 (with today's iFFBS fixes) | 3.625 | 0.400 | 0.003 | 3.222 |
| **L=15** | **2.033** | 0.401 | 0.002 | **1.631** |

HMC halved exactly as predicted; **1.78x on the full sweep.**

### Cumulative: where today ended up

| | s/sweep | correct? |
|---|---|---|
| start of day (L=30, slow iFFBS, NO obs likelihood) | 3.222 | **NO** |
| end of day (L=15, fast iFFBS, obs likelihood) | **2.033** | yes |

**~1.59x faster AND statistically correct** — the original sampled
thetas/rhos/phis from their priors. 5000 sweeps: ~4.5 h -> ~2.8 h on this laptop.

Block split is now iFFBS 19.7% / HMC 80.2%, so the gradient remains the only
thing that matters. Per the gradient profile it has no dispatch and no GC left,
so the remaining lever is the structural one: a hand-derived gradient seam
(what the C++ actually does). Needs sign-off.

## Entries

### 2026-07-20 — the observation likelihood was MISSING (a model bug, now fixed)

Confirmed by the user as a genuine error, not a design choice: the model had no
observation likelihood at all, so thetas/rhos/phis were sampled from their priors.

**Package change (the first one in this effort):** `epidemic_obs_loglik(data;
observation_process=data.observation_process)` in `src/build.jl`, exported. It is
the counterpart to `epidemic_loglik` (which covers the starting state and
transitions only). Neither includes the other, so a model wanting both writes
`@addlogprob! loglik(...) + obs_loglik(...)`.

**The `observation_process` keyword is the seam that makes C++-style blocking
possible.** The package CANNOT decide which observation factors belong in the
differentiated density and which belong in a conjugate block — `observation_process`
is one opaque function returning a weight vector, and nothing in it says which
factor goes with which parameter. So the user splits their own process and hands
this function whichever factor they want differentiated. The package stays
ignorant; it just builds a likelihood from the function it is given.

**User-side split** (`examples/badger_model_obssplit.jl`), mirroring
`ObsProcess_.cpp`, which builds the corrector as a PRODUCT
(`{eta,eta,eta,0}` then `*= productIfSuscep`):

    w(state) = badger_obs_capture(state) * badger_obs_tests(state)

`badger_observations_split` (the product) goes to `epidemic_data` so the iFFBS
filter still sees the full weights; only `badger_obs_tests` goes to
`epidemic_obs_loglik`, so `etas` stays conjugate without double-counting.

**Verified** (`examples/check_obs_split.jl`):

| claim | result |
|---|---|
| `capture .* tests == badger_observations` (186,850 cells) | max diff **1.11e-16** |
| `obs_loglik(full) == obs_loglik(capture) + obs_loglik(tests)` | diff **1.5e-07** on ~4e4 |
| test factor gives thetas/rhos/phis real information | moves by **1.2e+04** (was 0.000e+00) |
| test factor independent of `etas` | diff **0.000e+00** — conjugate block safe |

**The four benchmark variants** (`examples/bench_obs_variants.jl`, 50 timed
sweeps after 5 untimed warm-up sweeps):

| variant | obs term in `@addlogprob!` | HMC blocks | `etas` | correct? |
|---|---|---|---|---|
| `noobs` | none | 1 x 61 params | conjugate | **NO — current model** |
| `onehmc` | tests only | 1 x 61 params | conjugate | yes |
| `split` | tests only | **2: 43 + 18** | conjugate | yes |
| `naive` | full (capture x tests) | 1 x **65** (etas in HMC) | in HMC | yes |

The two decisive comparisons:
- `onehmc` vs `split` — does splitting thetas/rhos into their own block help?
  Same likelihood, same parameters, ONLY the block structure differs.
- `naive` vs `split` — is there anything wrong with putting everything in one
  HMC block?

**RESULT (50 timed sweeps each, 5 untimed warm-up, quiet machine, scalar obs path):**

| variant | s/sweep | iFFBS | etas | HMC* | vs noobs |
|---|---|---|---|---|---|
| `noobs` (WRONG, baseline) | 3.222 | 0.895 | 0.002 | 2.325 | — |
| `onehmc` (1 block, 61) | 3.682 | 0.925 | 0.002 | 2.755 | 0.87x |
| **`split` (2 blocks, 43+18)** | **4.527** | 0.837 | 0.002 | **3.688** | **0.71x** |
| `naive` (1 block, 65, etas in HMC) | 3.663 | 0.856 | 0.000 | 2.807 | 0.88x |

**SPLITTING IS SLOWER — 23% slower, the worst of the three correct variants.
This refutes the prediction made earlier in this log.**

*Why the prediction was wrong.* The earlier analysis noted that
`_logdensity_call` evaluates the WHOLE model body for every block, and treated
the doubled primal as a minor caveat against the partials saving (61 -> 43 and
61 -> 18 AD chunk-passes). That was backwards. Two HMC blocks means:
- 2 full primal evaluations per gradient instead of 1, AND
- a SECOND AdvancedHMC block running its own L=30 leapfrog trajectory.

So the split roughly doubles the leapfrog work to save partials on each half.
The partials saving does not come close to paying for a second full trajectory.

*Why the C++ split works and ours does not.* `grad_` and `gradThetasRhos` are
separate HAND-WRITTEN functions over disjoint data — the epidemic gradient never
touches test data and vice versa. Our two blocks each re-evaluate the entire
model body. Same block structure, opposite economics. **The C++'s blocking is
not portable to a PPL that evaluates the whole model per block.**

*Answers to the three questions asked:*
1. **Does splitting thetas/rhos out speed things up?** No — it is the slowest
   correct variant (4.527 vs 3.663 s/sweep).
2. **What about conjugate etas?** On WALL-CLOCK it buys nothing: `naive` (etas in
   HMC) at 3.663 s/sweep vs `onehmc` (etas conjugate) at 3.682 s/sweep is a wash.
   But see the ESS caveat below — s/sweep is the wrong metric for this one.
3. **Anything wrong with the naive one-block model?** Not on speed. It is the
   joint-fastest correct variant.

*Cost of correctness:* 3.222 -> 3.663 s/sweep = **0.44 s/sweep (~14%)** to add the
observation likelihood the model was missing. Cheap — and it would have been
~4.5 s/sweep without the `observation_weight` scalar path.

**CAVEAT — this benchmark measures WALL-CLOCK ONLY.** `naive` samples `etas` by
HMC rather than conjugately. A conjugate draw is an exact independent sample from
the correct conditional; an HMC step is a correlated move with a hand-set step
size (eta epsilon = 0.005 here, not tuned). So `naive` could win on s/sweep while
producing FEWER effective samples per second. **If `naive` looks competitive,
measure ESS before concluding anything** — s/sweep is the wrong metric for
comparing a conjugate block against an HMC block.

**Bug found by the `naive` variant (mine, in the split obs code).**
`badger_obs_capture` allocated `ones(Float64, n_states)` and then wrote
`1 - eta` into it. That is fine while `etas` is CONJUGATE (always Float64), but
the `naive` blocking puts `etas` in the HMC block, where it arrives as a
`ForwardDiff.Dual` — and writing a Dual into a Float64 array throws. Fixed to
`ones(eltype(model.etas), ...)`.

Worth recording because of what it says about the seam: **an observation factor's
element type depends on the BLOCKING, not just on the model.** The same function
is called with Float64 when its parameters are conjugate and with Duals when they
are in an HMC block. Any user-written factor must therefore take its eltype from
the parameters (as the package's own `_param_eltype` does) rather than hard-code
Float64. The earlier `check_obs_split.jl` pass did NOT catch this — it only ever
exercised the Float64 path, so it remains valid but was not sufficient.

### 2026-07-20 — the obs likelihood repeated a mistake the transition term had already fixed

The first four-variant benchmark ran 35 min against a ~16 min estimate and was
killed. Diagnosis, from reading the code rather than the (buffered, empty) log:

`epidemic_obs_loglik` as first written called
`observation_process(model, data, X, i, t)` at every `(i,t)` and read ONE entry
of the returned vector. Each call allocates a fresh `n_states` array — ~187k
allocations per likelihood call on the badger model, each of `Dual`s under AD.

**That is precisely the pattern `epidemic_loglik` was already optimised to
avoid.** Its docstring records that building the full transition matrix per
`(i,t)` was "~380k matrix allocations per call" and dominated the gradient, which
is why it uses the scalar `transition_prob` instead. I reintroduced the same
shape in the observation term.

**Fix — a second package seam, `observation_weight`:**

    observation_weight(model, data, X, i, t, s) -> P(observation at (i,t) | state s)

`epidemic_obs_loglik(data; observation_process=..., observation_weight=...)`
calls it with `s = X[t,i]` and never materialises a vector. Exactly the
relationship `transition_prob` has to `transition_matrix_at`. The vector path
stays the default and stays correct; this is opt-in, like `rest_contribution`.
The sampler still needs the vector form (the forward filter reads every state),
so BOTH exist and must agree — which is checked, not assumed.

Badger side: `badger_obs_tests_weight` / `badger_obs_capture_weight` /
`badger_obs_split_weight`, branching on `s` instead of filling all four states.

**Verified exact before benchmarking** (`examples/check_obs_split.jl`):

| check | entries | max diff |
|---|---|---|
| `badger_obs_tests` scalar vs vector | 747,400 | **0.000e+00** |
| `badger_obs_capture` scalar vs vector | 747,400 | **0.000e+00** |
| `badger_observations_split` scalar vs vector | 747,400 | **0.000e+00** |
| assembled `obs_loglik` both ways | — | **0.000e+00** |

**Measured (quiet machine, `examples/bench_obs_cost.jl`):**

| | primal, one call | allocations |
|---|---|---|
| `epidemic_loglik` (transitions) | 0.0097 s | 228,880 B |
| `obs_loglik` — vector path | 0.0148 s | **17,937,616 B** |
| `obs_loglik` — scalar path | **0.0017 s** | **16 B** |

**8.45x on the primal; 18 MB -> 16 bytes.** ~96 bytes per `(i,t)` cell across
186,850 cells. Note the vector-path observation term was MORE EXPENSIVE than the
entire transition likelihood it was being added to.

**Gradient (what HMC calls L=30 times per sweep):**

| | gradient | x30 (per sweep) |
|---|---|---|
| obs vector path | 0.2263 s | 6.79 s |
| obs scalar path | **0.0749 s** | **2.25 s** |
| speedup | **3.02x** | **saves 4.54 s/sweep** |

That 4.54 s/sweep is the explanation for the killed 35-minute run: four variants
x 55 sweeps x ~4.5 s of pure avoidable overhead is ~16 min on its own.

For scale, the gradient WITHOUT any observation likelihood measured ~0.075-0.095 s
earlier — so the scalar path adds essentially nothing to the gradient, while the
vector path tripled it. The model is now correct AND costs no more than the
(incorrect) original.

**What reading the C++ settled about blocking.** `MCMCiFFBS_.cpp:849-868` shows
the reference does NOT put all observation parameters in HMC:

- `thetas`, `rhos` -> HMC (`HMC_thetas_rhos`)
- `phis` -> **conjugate Gibbs** (`CheckSensSpec_` + `rbeta`), NOT HMC
- `etas` -> conjugate Gibbs
- `nu` -> conjugate Gibbs

Our model has `phis` in HMC. That is a third discrepancy with the reference (after
the missing obs likelihood and the fixed-vs-randomized `intL`). It is not wrong —
`phis` is informed either way once the test factor is in the likelihood — but a
`CheckSensSpec_`-equivalent conjugate kernel would be cheaper. Not built yet.

### 2026-07-20 — reframing: we have been optimising the wrong third

Established the cost model above and read the C++ gradient. Conclusion: iFFBS is
~1/4-1/3 of the sweep and is already faster than the Julia reference's; the
HMC/AD block is ~2/3-3/4 and is where the entire remaining gap lives. The C++
wins there by (a) a hand-derived one-pass gradient instead of 61-partial
forward-mode AD, (b) ~half the leapfrog steps via randomized trajectory length,
and (c) splitting the parameters into two independent, much cheaper gradient
problems.

### 2026-07-20 — the etas fix is exact (acceptance test PASSED)

`examples/check_kernel_fix_exact.jl` runs the FULL Gibbs sampler twice from the
same seed — once with the fixed kernels, once with reference reimplementations of
the original kernel bodies (untyped globals, NS-pass eta loop) — and compares
every draw.

```
tau lambda beta q c1 a1 b1 a2 b2 etas thetas rhos phis
    ALL IDENTICAL,  maxdiff = 0.000e+00
>>> PASS: every draw is bit-identical. The fixes are pure speed.
```

Five full sweeps, all 13 parameter groups, exact. This matters because the eta
rewrite changed both the loop order and the accumulation shape; bit-identical
COUNTS (bench_eta.jl) did not by itself prove the CHAIN was unchanged, since the
counts feed `rand(rng, Beta(...))`. It now is proven end-to-end.

### 2026-07-20 — retraction: the "iFFBS also got faster" claim

An earlier version of this log recorded iFFBS improving 1.41x and a 1.38x overall
speedup from the kernel fixes, attributed to reduced GC pressure. **Retracted.**
A second run of the identical harness gave iFFBS 1.407 s where the first gave
0.859 s — the spread is contention from concurrently-running benchmarks on the
same 8 cores, not a real effect. The mechanism (164 MB/sweep of garbage from the
old kernel taxing other blocks) is plausible but unproven by this data.

Lesson, and it is the same one CLAUDE.md already records: **one measurement of a
noisy quantity is not a result.** The etas number (270x) survives only because
the effect is far larger than the noise. Everything else needs a quiet machine.
