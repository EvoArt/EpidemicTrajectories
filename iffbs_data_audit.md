# Audit: reference `iFFBS_Data` fields vs. what we have

Every field in the semi-Markov Julia ref's `iFFBS_Data` struct
(`semi-markov_src.jl/MCMCiFFBS_.jl:9`), classified. Purpose: surface data
structures we lack that could indicate a MISSING MODEL COMPONENT (not just a
scratch buffer or a different-variant field).

Legend: **HAVE** = we have an equivalent · **SCRATCH** = per-iteration working
buffer, no model content · **VARIANT** = for a model variant we don't fit (sex,
GP frailty, semi-Markov Weibull) · **MISSING?** = we lack it AND it may carry
model meaning — investigate.

## Core data (all HAVE)

| ref field | ours | notes |
|---|---|---|
| `X` | the `X` matrix | latent trajectory |
| `TestMat`/`TestField`/`TestTimes` | `data.tests[t,i,j]` | test results |
| `CaptHist` | `data.capture` | capture history |
| `birthTimes` | `data.birth_time` | |
| `startSamplingPeriod`/`endSamplingPeriod` | `data.sampling_period` | |
| `nuTimes` | `data.nu_times` | |
| `CaptEffort` | `data.capt_effort` | |
| `numSeasons`/`seasonStart`/`seasonVec` | `data.n_seasons`/`data.season` | |
| `maxt`/`k`/`K`/`numTests`/`m`/`G`/`numStates`/`numNuTimes` | scalars in `data` | |
| `SocGroup` | `data.social_group` | |
| `ageMat` | `data.age` | |
| `numInfecMat`/`mPerGroup` | `data.aggregates.n_infectious`/`n_alive` | leave-one-out counts |
| `totalNumInfec`/`totalmPerGroup` | recomputed on demand | with-focal counts |
| `hp_theta`/`hp_rho`/`hp_phi`/`hp_eta` | priors in the `@model` | |
| `xi` | `BROCK_CHANGEPOINT` (fixed) | we fix it, ref infers it |

## Capture / lifetime bookkeeping

| ref field | ours | status |
|---|---|---|
| `lastCaptureTimes` | `data.last_capture_time` | HAVE (+ CAM already baked into our CaptHist) |
| `firstCaptureTimes` | `raw.first_capture_time` | HAVE (added; used for entry conditioning) |
| `capturesAfterMonit` | `raw.captures_after_monit` | HAVE (no-op for us — see progression_bug.md) |
| `lastObsAliveTimes` | recomputed in-loop from X | HAVE (implicit: D absorbing, loop bound equivalent) |

## Filter scratch (SCRATCH — no model content)

`corrector`, `predProb`, `filtProb`, `corrector_theta_rho`, `test_*_buf`,
`idVecAll`, `mh_*_indiv` counters — all per-iteration working state. We allocate
the equivalent (`FilterScratch`, the corrector via `observation_process`). No gap.

## Precomputed group-level transition logs (HAVE, computed differently)

| ref field | ours |
|---|---|
| `logProbStoSgivenSorE/I/D`, `logProbStoEgivenSorE/I/D` | `_grp_foi` recomputes on demand in `rest_contribution` |
| `logProbRest` (maxt×4×m), `logProbRestTotal` (maxt×4) | our `nSE`/`nSS` reversible aggregates (verified equal to 5e-13) |
| `probDyingMat`, `LogProbDyingMat`, `LogProbSurvMat` | survival fn + obs death-ban |

## Parameter transform machinery (HAVE via PracticalBayes)

`par_indices`, `named_indices`, `log_jac`, `transformations`,
`back_transformations` — the ref's hand-rolled unconstrained-space transforms +
Jacobians. PracticalBayes does this (link/invlink, logjac) automatically. No gap.

## VARIANT fields (for models we don't fit — NOT gaps)

- `surv_group` — discrete survival frailty (2-group mixture). A model extension.
- `sex`, `has_sex_data`, `numInfecMat_F/M`, `totalNumInfec_F/M`,
  `cached_logProbEtoI/E` (F/M) — SEX-STRATIFIED transmission. run_base_exp
  explicitly filters to known-sex but fits the BASE (no sex effect) model, so
  these are unused in our target.
- `gp_chol_L`, `gp_logdet` — Gaussian-process prior on alpha (whitened GP). A
  prior variant; our alpha is iid Exponential(1), matching the base model.
- `likelihood_progression_fn`, Weibull/half-Cauchy progression — semi-Markov
  progression variants; we use the exponential (memoryless) one.
- `TestFieldProposal` — for the xi (Brock changepoint) RWMH proposal; we FIX xi.

## VERDICT

No missing MODEL component found in `iFFBS_Data` beyond what this session already
addressed (obs likelihood, progression convention, survival timing, entry
conditioning, death-ban). Every remaining field is either something we have (by a
different mechanism), a scratch buffer, or a variant we deliberately don't fit.

The ONE thing worth double-checking is `xi` (Brock changepoint): the reference
INFERS it (RWMH), we FIX it at BROCK_CHANGEPOINT=101. That is a deliberate,
documented simplification (see CLAUDE.md / badger_repro_log.md), not a bug — but
if the reference's posterior xi differs materially from 101, our fixed value would
shift the test-parameter fits (thetas/rhos/phis via the Brock1/Brock2 split).
Flagged, not acted on.
