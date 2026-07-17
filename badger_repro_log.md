# Badger model reproduction log

A running log of reproducing the badger bovine-TB model (`badger_ref/`) as **user
code** against EpidemicTrajectories. Records the mission, the plan, every
challenge hit, and anything that had to change in the package to make it work.

---

## Mission statement

Reproduce the base badger model from `badger_ref/run_base_exp.jl` using
EpidemicTrajectories, writing only *user code* — no badger-specific machinery in
the package. The package must keep making **no assumptions about which arrays the
user tracks or how they update**; the badger model's group statistics
(infectious-per-group, alive-per-group) must be declared by the user as ordinary
reversible aggregates, exactly as the cattle model declares its own.

This is the real test of the design. The cattle model is two states, one
aggregate, no deaths, one group structure, two tests. The badger model is four
states with mortality, two coupled aggregates, a density-scaled force of
infection, time-varying group membership, an age-dependent survival curve, six
diagnostic tests with three-way state-dependent accuracy, and a capture process.
If that fits in user code, the abstraction is right; wherever it doesn't, the
package is wrong and this log records why.

### Deviation from the reference (agreed)

The reference infers a **changepoint `xi`** for when the Brock diagnostic test
switched from version 1 to version 2 (prior `hp_xi = [81, 60]`, init `xi = 80`,
updated by RWMH; `TestMatAsFieldProposal` swaps the Brock1/Brock2 columns for
tests between the old and proposed changepoint). We do **not** infer it: Brock1 is
assumed for `t <= 101`, Brock2 for `t > 101`. This makes the Brock column
assignment a static, one-time relabelling of the data, and removes the only part
of the model that genuinely needs a discrete RWMH step.

Consequently: **all remaining parameters are to be inferred with NUTS** where
possible, rather than reproducing the reference's split of RWMH/conjugate-Gibbs/
HMC blocks. The latent trajectory still goes through iFFBS (that is the point of
the package). Conjugate Gibbs updates are kept only where NUTS is not applicable.

---

## What the reference model actually is

Read from `badger_ref/run_base_exp.jl` (the entry point, 226 lines) plus
`semi-markov_src.jl/{observation,transitions,model,MCMCiFFBS_,Helper_funcs}.jl`.

**Dimensions** (`RData2/dimensions.csv`): `m = 2391` individuals (filtered to
known-sex, so fewer), `maxt = 161` timepoints, `G = 34` groups, `numTests = 6`,
`numSeasons = 4`, `numNuTimes = 6`.

**States**: 4 — susceptible, exposed, infectious, dead. (The reference encodes
these sparsely as S=0, E=3, I=1, D=9; ours will be `[:S, :E, :I, :D]` = 1..4.)

**Transitions** — S → E → I → D, with **no recovery** (unlike the cattle model,
`I` never returns to `S`) and **D absorbing**:
- `S -> E`: force of infection `alpha_g + beta * I_g / ((M_g / K)^q)`, where
  `alpha_g = lambda * alpha[g]` (per-group), `I_g` is the infectious count in the
  group, `M_g` the alive count, `K = 85` a fixed scaling constant, and `q` a
  density-dependence exponent. Probability of infection over a step is
  `1 - exp(-foi)`.
- `E -> I`: progression, `erlang_cdf_at_1(k, tau/k)` with `k = 1` (so exponential)
  and `tau = progression_scale`.
- `(S,E,I) -> D`: death, `1 - survival`, where survival is the **Siler** curve in
  the individual's age: `exp(-c1 + (a2/b2)*Δexp_late + (a1/b1)*Δexp_early)`.
- Every non-death transition is conditional on surviving the step (the reference
  multiplies each by the survival probability) — exactly what our
  `@survival ... death=:D` sugar expresses.
- Survival is forced to 1 before an individual's last capture time
  (`MCMCiFFBS_.jl:851-857`): a badger seen later is known to have been alive, so
  death is impossible until then. This is data-dependent, not a rate.

**Observation process** (`observation.jl`, `ObsProcess!`) — per individual/time:
- Capture: if captured, the alive states get weight `eta[season(t)]` and dead
  gets `0`; if not captured, alive get `1 - eta` and dead gets `1`.
- Tests, only at capture times, three-way by true state:
  - susceptible: specificity, `(1-phi)^x * phi^(1-x)`
  - exposed: reduced sensitivity, `(theta*rho)^x * (1 - theta*rho)^(1-x)`
  - infectious: full sensitivity, `theta^x * (1 - theta)^(1-x)`
  where `x` is the 0/1 result and only results in `{0,1}` count (others missing).

**Parameters** (base model, no sex effects), from the `model = (...)` NamedTuple:
`progression_scale`, `alpha` (length G), `lambda`, `beta`, `q`, Siler
`a1,b1,a2,b2,c1`, and per-test `thetas`, `rhos`, `phis` (length 6 each). Plus
`nuE`/`nuI` (initial-state mixing at `nuTimes`) and `etas` (per-season capture
probability), which the reference updates by conjugate Gibbs.

**Group statistics** the FOI reads: `numInfecMat` (infectious per group per time,
excluding the individual being updated) and `mPerGroup` (alive per group per
time, likewise). These are the badger analogue of the cattle model's
`n_infected` — and in our design they are just two user-declared reversible
aggregates. Note group membership is **time-varying** (`SocGroup[i, t]`), unlike
the cattle model's fixed pens.

---

## The code path (traced, not guessed)

`run_base_exp.jl` calls `MCMCiFFBS_(...)` (`MCMCiFFBS_.jl:191`). One MCMC
iteration, in order:

1. **Refresh survival** (`MCMCiFFBS_.jl:836-862`). For every `(i, t)` with
   `ageMat[i,t] > 0` and `t > lastCaptureTimes[i]`, evaluate
   `groupLevelSurvivalProb` (= `siler_surv`) → `individualLevelSurvivalProb`, and
   store `LogProbDyingMat` / `LogProbSurvMat`, both `(m, maxt, 3)` with the third
   dim indexing the *previous* state `(S, E, I)`. Before the last capture, an
   individual is known alive: death log-prob `log(0)`, survival `log(1)`.
2. **Refresh group-level FOI logs** (`:867`):
   `update_group_level_logs_from_counts!(pars, model, data, 1:G, maxt, 1)`
   (`transitions.jl:142`) — recomputes `logProbStoSgivenSorE`, `...givenI`,
   `...givenD` (`G × maxt-1` each) from `numInfecMat`/`mPerGroup`, with the
   "+1" variants accounting for whether the individual about to be updated is in
   that group. Written to exclude `current_id = 1`, the first individual of the
   sweep.
3. **Rebuild the coupling cache** (`:873-884`): `data.logProbRest =
   getLogProbRest(...)`, a `(4, m, maxt-1)` array — for each candidate state of
   the individual being updated, each other individual's contribution — then
   `logProbRestTotal[s, tt] = sum(logProbRest[s, :, tt])` (`4 × maxt-1`).
4. **Per-individual sweep** (`:892-923`): `for jj in 1:m`, call `iFFBS(jj, ...)`.
5. **Refresh totals** (`:940-961`): `lastObsAliveTimes`, then
   `updateTotalNumInfecAndTotalmPerGroup` → `totalNumInfec`/`totalmPerGroup`
   (the *including-everyone* counts the FOI reads).

`iFFBS(id, ...)` (`iFFBS.jl:1-115`), per individual:

1. **Observation corrector** (`:26-32`): `effective_test_vectors!` then
   `ObsProcess!` (`observation.jl:1`) fills `corrector[t, 1:4]`.
2. **Forward filter**: `iFFBS_initializeForwardFiltering!` (start state from
   `nuEs`/`nuIs` at `nuTimes`), `iFFBS_forwardFilteringFirstStep!`,
   `iFFBS_forwardFilteringLoop!`, `iFFBS_forwardFilteringFinalStep!` — all in
   `iFFBS_modular.jl`, over the individual's own window
   `startSamplingPeriod[id] .. endSamplingPeriod[id]`.
3. **Backward sample**: `iFFBS_backwardSampling!` writes `X[id, :]`.
4. **Re-apply / roll forward** (`:107-114`):
   `iFFBS_updateGroupStatistics_dispatch!(data, X, id, idNext, maxt)`
   (`iFFBS_modular.jl:346`) swaps the excluding-current counts from `id` to
   `idNext`; then `update_group_level_logs_from_counts!` for just the affected
   groups, and `updateLogProbRestTotalIndiv!` / `updateLogProbRestTotal!`
   (`updaters.jl:290`, `:253`) patch the coupling cache incrementally.

**The transition structure** (`iFFBS_forwardFilteringLoop!`, `iFFBS_modular.jl`):
`compute_individual_transition_probs` (`transitions.jl:2`) returns
`p00, p01, p11, p12, p22` and a death probability per live state, and the filter
pushes them forward as

```
predProb[t,1] = p00*filt[t-1,1]                                    # S
predProb[t,2] = p01*filt[t-1,1] + p11*filt[t-1,2]                  # E
predProb[t,3] = p12*filt[t-1,2] + p22*filt[t-1,3]                  # I
predProb[t,4] = ΣprDeath_from_s*filt[t-1,s] + filt[t-1,4]          # D (absorbing)
```

So: no recovery (`I` never returns to `S`, unlike the cattle model), and **dead is
absorbing** — its filtered mass carries forward. Then the filter combines the three
terms:

```
unnormFiltProb[s] = corrector[t,s] * predProb[t,s] * transProbRest[s]
```

**How this maps onto our design.** That last line is structurally identical to our
`forward_filter`'s `pred .* obs_w .* rest_w` — observation × prediction × coupling.
The maths is the same; the difference is caching. Steps 1-3 of the outer loop are
caches the reference maintains by hand (`compute_trans_prob_rest!` reads the
prebuilt `logProbRest`); our package has no equivalent and recomputes live.

`iFFBS`'s shape is also the same as ours (filter → sample → re-apply); the
difference is that the reference's `iFFBS_updateGroupStatistics_dispatch!` rolls
the counts from `id` to `idNext`, whereas we reverse `id`'s contribution at the
*start* of `iffbs_individual!` and re-apply at the end. Both give leave-one-out
statistics; ours needs no `idNext` bookkeeping and is generic over user-declared
aggregates.

### Future performance work: the `logProbRest` cache

The reference's biggest performance device is the `logProbRest` / `logProbRestTotal`
coupling cache (`(4, m, maxt-1)` + `(4, maxt-1)`), rebuilt once per MCMC iteration
and patched per individual. Our `make_rest_contribution` instead recomputes the
affected individuals' transition probabilities for each candidate state, every
time. **We stick with our way for now** — it is generic (it works for any
user-declared aggregates and any coupling structure, with no assumption about what
is cached) and correct, and correctness-before-speed is the right order.

Worth revisiting later: memoise the coupling per `(i, t)` within an MCMC iteration
and clear the cache at the end of each — e.g. with Memoization.jl or a hand-rolled
buffer. That would recover most of the reference's saving without hard-coding what
the cached quantity means. Log any measurement here when it is attempted.

---

## Plan of action

1. **Load and reshape the data.** `RData2/*.csv` → the package's conventions.
   Note the reference is individual-major (`X[i, t]`) while our package is
   time-major (`X[t, i]`); transpose on load. Filter to known-sex badgers as the
   reference does (keeps the dataset comparable), even though the base model has
   no sex effects.
2. **Static Brock assignment.** Apply `xi = 101` once when building the test
   data: results at `t <= 101` are Brock1, `t > 101` are Brock2. No swapping.
3. **Declare the aggregates** as user code: `n_infectious[g, t]` and
   `n_alive[g, t]`, both keyed by the *time-varying* group `data.social_group[i, t]`.
   This is the first real test of whether `@aggregate` handles a time-varying
   index and two coupled arrays.
4. **Declare the transitions** with `@survival siler_surv death=:D`, and rate
   functions reading the aggregates.
5. **Observation process.** The package's default `observation_process` is
   cattle-specific (two tests, sensitivity only). The badger model needs capture
   + six tests + three-way state-dependent accuracy. Expect this to force a
   package change: the observation process must be user-suppliable, not baked in.
   Log it when it happens.
6. **Starting state** from `nuE`/`nuI` at `nuTimes`, and per-individual sampling
   windows from `startSamplingPeriod`/`endSamplingPeriod` (the cattle model
   assumes every individual is observed over the whole window; the badger model
   does not — another likely package change).
7. **Fit** with `Gibbs(NUTS for the continuous parameters, iFFBS for X)`, adding
   conjugate kernels only where NUTS cannot apply.
8. **Verify** on simulated data first (simulate from known parameters with the
   badger structure, recover them), then run against the real data.

### Anticipated package changes (to confirm or refute as I go)

- ~~**The observation process must be user-supplied.**~~ **CONFIRMED and FIXED** —
  see the log entry below.
- ~~**Per-individual sampling windows.**~~ **CONFIRMED and FIXED** — below.
- ~~**Time-varying group membership.**~~ **PARTLY WRONG, and FIXED** — below.
- ~~**`@survival` is a stub.**~~ **CONFIRMED and FIXED** — below.

---

## Log

### 2026-07-17 — reading the reference, writing this plan

Found that "brock_1 / brock_2" are **not** groups or a spatial split, as I first
assumed from the brief: they are two of the six diagnostic tests (`TestMat`
columns are `time, idNumber, group, Brock1, Brock2, Culture, DPP, Gamma,
StatPak`). The reference infers the changepoint between them. Fixing it at
`t = 101` is therefore a static column relabelling — see "Deviation" above.

No package changes yet.

### 2026-07-17 — three package changes, made before touching the badger model

Arthur reviewed the four anticipated changes and pushed back on them. He was right
on every count, and the outcome is three fixes to the package plus one correction
to my own claim. All three were the *same* flaw wearing different hats: **the
package deciding something the user should decide, with no way to override.**

**1. The observation process WAS hard-coded (my error).** `src/iffbs.jl` carried a
cattle-specific `observation_process` — two tests, sensitivity only, reading
`model.θʳ`/`model.θᶠ` by name — and `forward_filter` called it unqualified, so a
user's own definition could never be reached. This was not in the walkthrough's
design: there it lived in the *package-code* section of a single-file script, where
a user could simply redefine it. Porting it verbatim into a real package silently
promoted a placeholder into a hard assumption — the same class of mistake as the
old `epidemic_model` API, and a direct violation of the rule in CLAUDE.md.

Fixed: `observation_process` is now a user-supplied field on `EpidemicData`,
defaulting to `no_observations` (every state equally likely — the honest default,
since the package cannot know what you observe). The cattle model now supplies its
own, as user code, which is where it always belonged.

**2. Per-individual sampling windows WERE hard-coded.** `epidemic_data` set
`sampling_period = [(1, n_timepoints) ...]` unconditionally. `walkthrough.jl` had
them as user-supplied, so this was a regression I introduced in the port. Fixed:
`sampling_period` is a keyword, defaulting to the whole window for everyone, and
validated against `n_individuals`.

**3. Time-varying group membership — I was wrong to call this a package change.**
Arthur said he'd be unhappy if the design blocked it. It didn't: `affected_individuals`
is already a `Matrix{Vector{Int}}` indexed `[t, i]`, so time-varying coupling was
designed for from the start. What was wrong was narrower — `epidemic_data`
*unconditionally overwrote* it by calling `build_affected_individuals_from_groups`
on a fixed `group::Vector{Int}`, giving no way to pass your own. The core was fine;
the constructor was the problem. Fixed: `affected_individuals` is a keyword
(validated for shape), with the fixed-group build as a convenience default, and
`group` itself now defaults to "everyone in one group" for models that have no
groups at all.

**Also added: `extras`.** The badger model needs to carry capture histories, test
matrices, ages, and time-varying group membership on `data`. Rather than the
package naming any of them, `epidemic_data(...; anything=...)` stores them and
`data.anything` reaches them via `getproperty`. The package never looks inside.
This is what `test_mats` should have been all along — it was another package-level
assumption about what a model observes, and it is now gone.

Verified: cattle user code still recovers all six parameters (numerically identical
to before, since the model is unchanged — only who supplies the observation process
moved). Test suite 76 → 84, the new tests covering the three fixes.

### 2026-07-17 — `@survival` implemented

The fourth anticipated change, confirmed. Two real bugs, both of which the badger
model would have hit immediately:

1. **`death=:D` never worked.** The parser read `kw.args[2]` straight into a
   `Tuple{Symbol,Symbol,Any}`, but `death=:D` gives a `QuoteNode(:D)`, not a
   `Symbol` — so any use of `@survival` died with `Cannot convert QuoteNode to
   Symbol`. It had never been exercised. Now unwrapped, and `death=D` works too.
2. **Absorbing states got no death transition.** The leftover-mass loop collected
   live states from the transitions' *sources* only, so `I` — which in an S→E→I→D
   model only ever appears as a destination — never got its `I -> D`. Badgers in
   the infectious state would have been immortal. Now collects live states from
   both sides of every transition.

So `@transitions [:S,:E,:I,:D] begin; @survival siler death=:D; S -> E = foi;
E -> I = prog; end` yields (S,E), (E,I), (S,D), (E,D), (I,D) — the badger
structure, with the two death transitions the user never wrote. 84 → 92 tests.

All four anticipated package changes are now done. Next: the badger model itself.

### 2026-07-17 — data loader, and a misunderstanding of the changepoint

Wrote `examples/badger_data.jl` (user code: reads the CSVs, filters, converts the
reference's conventions to ours). Two conventions differ and are converted on
load: the reference is individual-major (`X[i,t]`) where we are time-major
(`X[t,i]`), and it encodes states sparsely (S=0, E=3, I=1, D=9) where we use the
position in `state_space` (S=1, E=2, I=3, D=4).

What the data turns out to be, after the known-sex filter (2391 → **2384**
badgers, matching the reference), over 161 timepoints and 34 groups:

- All four states occur, and **40% of cells are dead** — mortality is a dominant
  feature of this model, unlike the cattle model which has none.
- **505 badgers change social group.** Time-varying membership is real, not
  hypothetical. Good thing the package supports it (see the previous entry).
- 12,430 captures, 25,521 test results across the six tests.

**I got the changepoint wrong, and the data caught it.** My first
`apply_brock_changepoint!` blanked whichever Brock column "didn't apply" at each
time — I had assumed `xi` decides which column is valid when. Checking against the
raw data showed Brock1 spans t=6–80 and Brock2 t=81–125 **already**, and my code
silently destroyed 1675 Brock2 results (3493 → 1684).

Re-reading the reference: `TestField = TestMatAsField_CORRECTED(TestMat_infer, m)`
builds the test field **straight from the raw columns, with no `xi` applied**. So
the raw data is already correct *at the initial* `xi = 80`, and `xi` only ever
acts incrementally — `TestMatAsFieldProposal` **swaps** the Brock1/Brock2 columns
for tests lying between the current and proposed changepoint. Confirmed in the
window t=80..100: Brock1 has 92 results, Brock2 has 1675, and **never both** —
each occasion fills exactly one column, so a swap is lossless.

So fixing the changepoint at 101 means: take the raw data and swap the two Brock
columns for the tests in `[80, 101)`. Now implemented that way, and verified —
Brock1 6706→8143 (now spanning t=6–100), Brock2 3201→1764, **total unchanged at
9907**, and the other four tests untouched.

Lesson for the rest of this: the reference's parameters are not always what their
name suggests. `xi` is not "which test applies when", it is an offset against a
raw encoding that already has a changepoint baked in at 80.

### 2026-07-17 — the model builds and evaluates; one package bug; performance is the wall

`examples/badger_model.jl` — the whole badger base model as user code. It builds
and every piece evaluates against the real data:

- `@survival siler_survival death=:D` + `S -> E` + `E -> I` produced exactly
  `(S,E), (E,I), (S,D), (E,D), (I,D)` — including the `I -> D` never written.
- Transition matrix rows sum to 1; observation weights sensible; **loglik finite**
  (-95,604) on the reference's own `Xinit`.
- Both aggregates (`n_infectious`, `n_alive`) fill correctly, indexed by the
  **time-varying** `social_group[i, t]`, with the `g == 0` ("not present") guard.

**The design holds.** Four states, mortality, two coupled aggregates, a
density-scaled FOI, an age-dependent survival curve, time-varying groups, six
tests with three-way accuracy, and a capture process — all expressed as user code,
with the package knowing none of it. That was the thing this exercise was meant to
test.

**One package bug, which only the badger model could find.** `_param_eltype` did
`promote_type(map(typeof, values(model))...)`, which lands on `Any` when the
parameters mix scalars with vectors (`(; beta=0.1, alpha=[...])`) — and
`zeros(Any, n, n)` throws. The cattle model has only scalar parameters, so it
never hit this. Fixed to reach through containers to the number type inside, and
to ignore non-numeric entries.

**Resolved, not a bug: progression = 0.993 at the initial values.** Open question
1 in this log. My implementation is faithful — `erlang_cdf_at_1(k, tau/k)` with
`k=1, tau=5` really is `1 - exp(-5) = 0.993`, and the reference computes the same.
It is just a poor starting point; the prior is `Exponential(10)` and the sampler
moves it. Not a discrepancy to chase.

**Performance is the wall.** Measured on the real data (2384 badgers × 161 times):

| | cost |
|---|---|
| one `loglik` | **3.0 s** (1.07 GiB allocated) |
| one iFFBS individual | 0.118 s |
| one iFFBS sweep (2384) | **~4.7 min** |
| 1000 sweeps | **~78 hours** — and that is *before* NUTS |

NUTS needs many gradient evaluations of that 3-second likelihood per sweep, so a
full run is out of reach as things stand. This is the cost of our generic
on-the-fly coupling: `make_rest_contribution` recomputes every affected
individual's transition probabilities for each candidate state, at every
(i, t) — the exact work the reference avoids with its `logProbRest` cache
(rebuilt once per iteration, patched per individual).

Next: verify correctness on a subsample (a few groups), which is the honest thing
to do before optimising, and report timings rather than sink hours into a run
that cannot finish. The memoisation idea in "Future performance work" above is
now the obvious next step, not a someday-maybe.
