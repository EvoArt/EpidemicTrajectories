# Handover — badger fit state, fixes, and open crash

## TL;DR
On branch `fix-badger-focal-package`. A large body of work is **uncommitted in the
working tree**. There is a **`ReadOnlyMemoryError` crash** under investigation. A
clean isolation test (running the known-good script on the current tree) was
**still running when this conversation ended — its result is unknown.** Get that
result first; it decides everything below.

## Where the code lives (verified via git)
- Branches `fix-badger-focal-package` and `fix-badger-focal-m-user` are at the SAME
  commit `0c889fc`. Neither is ahead. The branch name is currently irrelevant.
- **Everything below is UNCOMMITTED** (working tree only). No stash, no snapshot.
  A stray `git checkout`/`reset` destroys it all.
- The other LLM's committed history (through `0c889fc`) is safe: survival off-by-one,
  entry conditioning, death-ban separation, starting-state scoring, siler/gompertz.
- **The focal-contribution package fix (`_call_rate_with_focal`, `_focal` field, X
  threaded through `_fill_rates!`/`_accum_row`) was NEVER committed** — it was the
  other LLM's uncommitted work, present in the tree when this session started. It is
  committed on NO branch (master/m-user/package all show 0 hits).

## What the OTHER LLM did (working baseline — get correct params)
- Ran `examples/badger_fit_siler_max.jl` for 1000 iters → CORRECT params (tau≈15,
  beta<1). This is the known-good reference. Run it with `julia --project=.`
  (it lacks a self-activating Pkg bootstrap).
- Its uncommitted package contribution: the focal fix (above).

## What THIS (Claude) session changed — all uncommitted
Package `src/` (6 files: aggregates, build, data, iffbs, spec, transitions):
1. **`focal_self_contribution` flag** (data.jl field + kwarg + one-time @warn;
   transitions.jl guard `data.focal_self_contribution && data._focal[] == i`).
   Default true. Opt-out for users whose rates do their own focal accounting.
2. **Coupling survival-annihilation FIX** — the big one. `@survival` now stashes a
   survival-free view of the transitions on a NEW `TransitionSpec.coupling` field
   (struct changed `{RF}` → `{RF,C}`, 5-arg outer ctor keeps 4-arg calls working;
   spec.jl `_parse_transition_block` returns `(transitions, bare_transitions)`;
   macro builds the coupling spec). `epidemic_data` resolves the coupling spec:
   explicit `coupling_trans_mat` wins, else `trans_mat.coupling`, else `trans_mat`.
   Reason: default coupling scored neighbours through the survival-scaled
   `transition_prob`; when survival underflows to 0.0 the infection signal is
   annihilated (proven: 100% of 804 known-alive S→E events), degenerating the chain.
   Verified: fixed default coupling == reststotal coupling to 4e-14. 100/100 tests
   pass. See `memory/coupling-survival-annihilation-bug.md`.
3. **`observation_weight` scalar API** — either/both forms; scalar-only auto-derives
   the filter vector; `epidemic_obs_loglik` defaults both from `data`.
4. Positional `reverse` in derived summaries (was already partly done); docstrings.

New example scripts (untracked): `badger_run.jl`, `badger_run_reststotal.jl`,
`badger_run_allhmc.jl`, `badger_run_allhmc_nutshmc.jl` (older exploratory variant),
`badger_common.jl` + staged set (`badger_naive/coupled/obsweight/reststotal.jl`),
`badger_siler.jl`.

## THE OPEN PROBLEM: ReadOnlyMemoryError crash
- Crashed twice on the new scripts (e.g. `badger_run_reststotal.jl`) with
  `Internal error: ReadOnlyMemoryError()` early in the fit, under
  `AutoPolyesterForwardDiff` (multithreaded AD).
- **PRIME SUSPECT: the `TransitionSpec.coupling` change (#2).** The struct now holds
  a nested `TransitionSpec` (`TransitionSpec{Tuple{...}, TransitionSpec{Tuple{...},
  Nothing}}`), a recursively-shaped type that flows into every threaded-AD closure.
  Recursive/odd struct types + Polyester threaded AD is a known bad-codegen →
  ReadOnlyMemoryError failure mode.
- **Isolation test IN PROGRESS at handover:** running the known-good
  `badger_fit_siler_max.jl` for 1000 iters on the current tree (loaded fine, was
  sweeping). Output file:
  `.../scratchpad/max_1000b.txt` (task brqhzesub).
  - If it CRASHES → my src change broke the baseline → bisect (start by reverting
    src/spec.jl to HEAD; that also needs the data.jl `.coupling` resolution hunk
    guarded/reverted so it loads).
  - If it COMPLETES with tau≈15 → my src changes are safe on the known-good path;
    the crash is script-specific (AdaptiveHMC sampler, or the reststotal-run wiring).

## Proposed next steps (priority order)
1. **Read the isolation-test result** (max_1000b.txt) before anything else.
2. If baseline crashed: **de-recursivize the coupling field.** Instead of storing a
   nested `TransitionSpec`, store the bare transitions + bare rate_fns as plain
   fields (or a small non-recursive struct), so `TransitionSpec`'s type parameter
   doesn't nest. Re-test with `badger_fit_siler_max.jl`.
3. If baseline is clean: the crash is in the new scripts. Test
   `badger_run_reststotal.jl` (now on fixed HMC, not AdaptiveHMC) 1000 iters; if it
   also crashes, suspect AD/threading interaction with the new obs_weight
   auto-derived-vector closure; try `AutoForwardDiff()` (single-thread) to confirm
   threading, and `chunksize`/`tag` settings.
4. **Second, separate regression already fixed:** prior-draw init started chains far
   out (tau~Exp(100) mean 100 vs true 15). All 3 badger_run scripts now pass
   explicit `init` (tau=5, alpha=0.001, ...). `badger_run_allhmc_nutshmc.jl` is
   STALE (no init, NUTSthenHMC, N_ADAPT=200) — bring in line or delete.
5. **Git safety (do once crash is understood):** commit a checkpoint so nothing is
   lost. The other LLM's focal fix + my fixes are interleaved in the same files, so
   a perfectly clean "other-LLM-only" commit is not reconstructable from git — the
   pre-me working tree exists nowhere. Simplest safe move: commit the whole current
   tree as one checkpoint, then split/revert from a committed base.

## Verified-correct facts to rely on
- Death-ban (D weight 0 while `t <= last_capture_time`) AND survival gate
  (`epidemic_loglik(; entry_time=first_capture, survival)`) are present + consistent
  in all 3 new scripts. All-HMC uses ONE obs function; conjugate-etas scripts split
  into capture_weight×tests_weight only because etas is conjugate.
- Coupling fix is mathematically verified (4e-14 vs reststotal); the QUESTION is
  only whether its type change causes the memory crash.
- Memory notes written this session: `coupling-survival-annihilation-bug.md`,
  `observation-weight-api.md`, `focal-self-contribution-flag.md`,
  `badger-staged-optimization-example.md`.
