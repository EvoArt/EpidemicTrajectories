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

### 2026-07-17 — the badger iFFBS is correct (after two self-inflicted false alarms)

`examples/badger_check.jl`: simulate a 4-state badger-structured epidemic from
known parameters, then start the sampler from an all-susceptible `X` and see if it
rebuilds the truth.

| | truth | recovered |
|---|---|---|
| S | 0.053 | 0.041 |
| E | 0.046 | 0.046 |
| I | 0.901 | 0.914 |
| D | 0.0 | 0.0 |

**97.5% state agreement**, `P(E or I) at test-positive cells = 1.0`, and the
incremental aggregate still equals a from-scratch recompute — now on a four-state
model with mortality and two coupled aggregates, not just the cattle model's one.

It failed twice first, and **both were my test, not the package**:

1. *"97% dead when the truth is 72%"*. My "survival ≈ 1" parameters were nothing
   of the sort: with `b2 = 1.0` the Siler senescence term `exp(b2 * age)` explodes
   and survival is 0.047 by age 20. My badgers aged 10→40, so they nearly all
   died. Fixed with `b2 = 0.01`, `a2 = 1e-12` and younger animals.
2. *"P(E or I) at test-positive cells = 0.024"* — the sampler putting badgers in a
   state a near-perfect positive test rules out. The cause: I set `etas = 1.0`
   ("capture is certain") while capturing only every 3rd step. If capture is
   certain then **not being seen logically implies dead**, so the observation
   process — correctly — forced every unobserved step to D. Incoherent test
   parameters, not a bug. Fixed with `etas = 0.6`.

Isolating it took a single-badger, no-coupling case, which showed obs weights
`[1,1,1,0]` when captured, survival 1.0, a correct S→E→I transition matrix with D
absorbing, and an all-infectious sampled trajectory with no deaths — i.e. the
machinery was right all along and the harness was lying.

Worth keeping in mind for the real fit: `eta` and capture frequency are coupled
assumptions. A high `eta` with sparse capture is not a neutral choice — it is a
strong statement that unobserved badgers are dead.

**Coupling scope** (Arthur asked): our coupling is group-scoped —
`affected_individuals[t, i]` holds only groupmates at time `t`, so an individual's
state only ever affects same-group, same-time badgers. What we do *not* have is
the reference's `whichRequireUpdate`, which further restricts recomputation to
individuals whose contribution actually changed. We recompute for every groupmate,
for every candidate state, at every `(i, t)` — which is a large part of the
4.7 min/sweep.

### 2026-07-17 — profiling says the problem is types, not the coupling

Used ProfileToLLM (`~/.julia/dev/ProfileToLLM`) on a real badger sweep rather than
guessing. The result overturned what I was about to do.

```
 self%  total%  gc?  dispatch?   function              file:line
  13.7    84.2    Y      Y       transition_matrix_at  src/transitions.jl:28
  10.2    11.5    Y      Y       siler_survival        badger_model.jl:82
   7.1     8.7    Y      Y       siler_survival        badger_model.jl:84
   6.4    12.0    Y      Y       siler_survival        badger_model.jl:85
   6.4     6.6    Y      Y       siler_survival        badger_model.jl:81
   6.1     6.4    Y      Y       siler_survival        badger_model.jl:83
   ...     ...    Y      Y       (every row)
```

`transition_matrix_at` is 84% of total, as expected — it is the hub. But
`siler_survival` alone is **~45% of self time, spread evenly across every line of
the formula**, and — the real finding — **every row is flagged for both runtime
dispatch and GC**. That is not "the coupling is inherently expensive". It is
type instability, everywhere.

Confirmed the cause directly:

| field | type | consequence |
|---|---|---|
| `extras` | `Dict{Symbol,Any}` | **`data.age` infers to `Any`** |
| `aggregates` | `Dict{Symbol,Any}` | every FOI read is `Any` |
| `rate_fns` | `Vector{Function}` | every rate call dispatches |

So each `data.age[i,t]`, `data.social_group[i,t]`, `data.capture[t,i]`,
`data.tests[t,i,j]` and `data.aggregates[:n_infectious][g,t]` returns `Any`, and
the arithmetic on top of it dispatches at runtime and boxes. That is the 1.07 GiB
per likelihood call, and most of the 4.7 min/sweep.

**The irony is mine.** `extras` was added *precisely so the package would not
assume what a model carries* — the central design rule. But `Dict{Symbol,Any}` was
a lazy way to spell "anything", and it made the hot path pay for that generality on
every single read. The rule doesn't require this: a `NamedTuple` is just as open —
the user still puts whatever they like in it — while being concretely typed, so
`data.age` infers to `Matrix{Int64}` and the arithmetic compiles. The cattle model
never showed this because its rates read almost nothing off `data`.

**Next (in priority order), and note this comes BEFORE the coupling cache:**
1. `extras::Dict{Symbol,Any}` → `NamedTuple`. Should be the single biggest win,
   and it costs the user nothing (`epidemic_data(...; age=..., capture=...)` is
   unchanged).
2. `aggregates::Dict{Symbol,Any}` → `NamedTuple` too. Same argument. `@aggregate`
   already knows the names and element types at macro time, so it can build a
   concretely-typed container — which is what the old `GroupStat` plan was trying
   to achieve, obtainable here for free.
3. `rate_fns::Vector{Function}` → a `Tuple`, so each rate's concrete type is known
   and the calls devirtualise.
4. Only then reconsider the coupling cache / memoisation. It may not be needed.

This is why the profiler was worth using: I would have built the cache first and
left a 10x type-instability tax underneath it.

### 2026-07-17 — the fix: NamedTuple + `Val`-dispatched `getproperty` (measured)

Arthur's steer: after construction the arrays must have known types, as the
reference achieves with its hand-written `iFFBS_Data` — but **without the user
writing a struct**, and not necessarily by the same mechanism.

A `NamedTuple` *is* "a struct with known types, generated for you", so the user's
call site never changes. But getting it right took two false starts, both worth
recording:

**False start 1 — "NamedTuples are slower".** My first benchmark said NT was 2x
*slower* than Dict, while allocating 13x less. Those two facts cannot both be a
real property of NamedTuples, so I dug instead of reporting it.

**The culprit was my own `getproperty`**, in both versions:

```julia
Base.getproperty(d, s::Symbol) = s in fieldnames(EpidemicData) ? getfield(d, s) : ...
```

`s in fieldnames(T)` is a **runtime search over a tuple of symbols, on every single
property access**, and it allocates. Dispatching on `Val(s)` instead moves the
decision to compile time:

```julia
@inline Base.getproperty(d::EpidemicData, s::Symbol) = _get(d, Val(s))
@inline _get(d::EpidemicData, ::Val{:extras}) = getfield(d, :extras)
@inline _get(d::EpidemicData, ::Val{s}) where {s} = getfield(getfield(d, :extras), s)
```

Isolated: 11.7 ms / 20k allocations → **18.9 µs / 0 allocations. A 620x
difference**, from the accessor alone.

**False start 2 — inference was a false all-clear.** Both the broken and the fixed
version report `Base.return_types(d -> d.age) == Matrix{Int64}`. Inference being
clean says nothing about whether the *accessor* is cheap. Only benchmarking found
it.

**The real comparison**, on the badger model's two hottest functions over 20k
(i, t) cells:

| | today (`Dict{Symbol,Any}`) | proposed (NamedTuple + Val) | gain |
|---|---|---|---|
| `siler_survival` | 15.1 ms, 520k allocs, 7.9 MiB | **0.82 ms, 0 allocs, 0 B** | **18x** |
| force of infection | 7.6 ms, 240k allocs, 3.7 MiB | **1.05 ms, 0 allocs, 0 B** | **7x** |

Zero allocations in both. This is the whole 1.07 GiB per likelihood call.

Implementing now: `extras` and `aggregates` become NamedTuples, `EpidemicData`
gains type parameters for them, `getproperty` dispatches on `Val`, and `rate_fns`
becomes a `Tuple`. The user-facing API does not change.

### 2026-07-17 — types fixed: 5.3x, and the coupling was never touched

Implemented the three type fixes and measured after each.

| | before | after | gain |
|---|---|---|---|
| one `loglik` | 3.02 s | **0.64 s** | 4.7x |
| one iFFBS individual | 0.118 s | **0.022 s** | 5.4x |
| one full sweep (2384) | 4.7 min | **0.9 min** | 5.2x |
| 1000 sweeps | 78 h | **14.7 h** | 5.3x |

What changed:

1. **`extras`/`aggregates`: `Dict{Symbol,Any}` → `NamedTuple`**, and `EpidemicData`
   parameterised on their types. Gave ~2x on its own, and `siler_survival`
   **disappeared from the profile entirely** — it had been ~45% of self time purely
   from `Any`-typed arithmetic.
2. **`getproperty` dispatches on `Val(s)`** rather than `s in fieldnames(T)` (a
   runtime search that allocates on every access).
3. **`rate_fns`: `Vector{Function}` → `Tuple`**, with `TransitionSpec` parameterised
   on it, and `transition_matrix_at` recursing over the tuple via `_fill_rates!`
   instead of looping. This was the remaining 75% of self time: a plain loop over a
   heterogeneous tuple still infers `Any` per element, so each rate call
   dispatched. Recursion specialises on one concrete function type at a time. Gave
   the second ~2.6x.

The profile is now clean: no `siler_survival`, no dispatch-flagged rows at the top,
and the remaining leaves are ordinary arithmetic (`muladd`, `getindex`, `float.jl`)
— what healthy code looks like.

**The point worth keeping**: none of this touched the coupling. My plan before
profiling was to build the `logProbRest` cache first; that would have left a 5x
type tax underneath it and I would have concluded the coupling was the problem.
The profiler said otherwise, and it was right.

14.7 h for 1000 sweeps is still not a casual run, but it is now in reach for an
overnight fit, and the coupling cache/memoisation remains available if more is
needed. Correctness first: the cattle model still recovers 6/6 and all 92 tests
pass after the change.

### 2026-07-17 — `coupled_transitions`: 78 h -> 1.4 h, and a bug caught by insisting on the maths

Arthur's idea: the package needn't recompute every neighbour's full transition
matrix for every candidate focal state, because the focal only affects *some* of a
neighbour's transitions. Let the user declare which. (Inferring it from the
transition spec is possible later; user-declared is the easy first pass.)

Implemented as `coupled_transitions=[(:S, :E)]` on `epidemic_data` — "the only way
one badger affects another is through its force of infection, which only the S->E
rate reads".

**My first implementation was wrong, and Arthur's "make sure the math still
balances" caught it.** I masked exactly the named transition, reasoning that a
neighbour whose move isn't coupled contributes an identical constant to every
candidate and cancels on normalisation. The reasoning was fine; the mask was not.
Checked against the unmasked computation over 10,008 (i,t) cells:

```
worst absolute difference in NORMALISED weights: 0.2486
*** DIFFERS - the skip is NOT valid ***
```

A quarter, in normalised weights — it would have silently corrupted the sampler.
Diagnosing it took printing one neighbour's log-prob against each focal state:

```
neighbour j=31, move S->S, "not coupled":
  logp per focal state:  -7.0e-6  -7.0e-6  -0.017593  -7.0e-6
                            S        E         I         D
```

**`S -> S` moves with the focal's state.** Of course it does: probabilities out of
a state sum to one, so if the focal being infectious raises a neighbour's `S -> E`,
it necessarily lowers that neighbour's `S -> S` by the same amount. I had conflated
"the transition whose RATE FUNCTION reads the aggregates" with "the transitions
whose PROBABILITY depends on the focal". With `auto_self` the self-transition takes
the leftover mass, so it is coupled too.

The fix: the mask is the **closure** — every transition out of any state that has a
coupled transition out of it. Re-checked:

```
worst absolute difference in NORMALISED weights: 1.3e-13
IDENTICAL - the skip is exact
```

So declaring `[(:S, :E)]` skips only neighbours currently in `E`, `I` or `D` — but
in this model 40% of cells are dead, and both gates are now permanent tests
(`coupled_transitions: the skip is exact`, and one asserting a coupled source
couples all its outgoing transitions).

**The result:**

| | original | after the type fixes | **after `coupled_transitions`** | total |
|---|---|---|---|---|
| one `loglik` | 3.02 s | 0.64 s | **0.52 s** | 5.8x |
| one iFFBS individual | 0.118 s | 0.022 s | **0.002 s** | **59x** |
| one full sweep (2384) | 4.7 min | 0.9 min | **~5 s** | **~56x** |
| 1000 sweeps | 78 h | 14.7 h | **1.4 h** | **56x** |

**78 hours to 1.4 hours**, exactly, with the sampler's answers unchanged to 1e-13.
The `logProbRest` cache is no longer needed for this model.

Lesson, and it is the same one twice now: I reasoned correctly about the *shape* of
the optimisation and wrongly about its *precondition*, and only checking against
the unoptimised path found it. "The maths must balance" is not a formality.

Still to do: inferring the coupled set automatically from the transition spec —
the package can see which rates touch the aggregates, in principle. User-declared
works and is explicit, so this is a convenience, not a gap.

### 2026-07-17 — wiring into PracticalBayes; the nu simplex; the likelihood's matrix waste

`examples/badger_fit.jl`: the model wired into PracticalBayes as
`Gibbs(NUTS(13 continuous blocks) + conjugate(etas, nu) + iFFBS(X))`, with the
reference's own priors.

**A real modelling bug of mine, caught by a `DomainError`.** I declared
`nuE ~ filldist(Beta(1,1), n_nu)` and `nuI` likewise — independently. But the
starting state is `(1 - nuE - nuI, nuE, nuI)`: they are two components of ONE
simplex, and independent Betas let them sum past 1, making the susceptible
probability negative. The likelihood takes its log. The reference uses
`Dirichlet([8,1,1])` for exactly this reason. Fixed: `nu` is now a single
`n_nu × 2` latent with a `NuSimplex` prior that draws a proper Dirichlet, updated
only by the conjugate kernel.

Note `NuSimplex <: DiscreteMatrixDistribution` despite holding continuous values.
That is deliberate and worth remembering: **in PracticalBayes it is the DISCRETE
declaration that routes a site to the value store**, where a latent kernel owns it
and NUTS never touches it. Declared continuous, `nu` went to NUTS and died on
`linked_vec_length` — and NUTS could not have handled a simplex it knows nothing
about anyway. Same mechanism carries `X`.

**Then: is the slowness PracticalBayes or us?** (Arthur's question.) Profiled a
real gradient through `LogDensityFunction` — what NUTS actually calls:

```
 self%  total%  gc?  dispatch?  function   file:line
  33.2    95.8    Y      Y      loglik     src/build.jl:84
  14.7    14.7    Y      -      Array      boot.jl:479
  13.1    13.1    Y      -      Array      boot.jl:477
  12.0    12.0    -      -      setindex!  array.jl:1021
```

**Ours, unambiguously.** PracticalBayes' machinery does not appear at all; `loglik`
is 95.8% of total, and ~40% of self time is `Array`/`setindex!` — pure allocation.
Corroborating: one gradient over 9 scalars cost only 2.3x a bare loglik, which is
healthy for ForwardDiff. The gradient machinery was never the problem; the
likelihood is just expensive and NUTS calls it ~32 times a sweep.

**The waste**: `loglik` built a whole `n_states × n_states` transition matrix for
every `(i, t)` — 2384 × 160 = **381,000 matrix allocations per likelihood call**,
of Dual numbers under AD — in order to read **one entry**, the move the individual
actually made. It also evaluated all five transition rates when one was wanted.

Added `transition_prob(trans_mat, model, data, X, i, t, from, to)`: walks the rate
tuple once, accumulating only the requested entry and the row total (for the
`auto_self` leftover). The sampler still uses the full matrix — it genuinely needs
every entry. Verified entry-for-entry against `transition_matrix_at` to 1e-12,
including the self-transition, and that is now a permanent test.

**Gradient: 7.4 s → 3.63 s**, and the `Array`/`setindex!` rows vanished from the
profile entirely. 100 tests pass.

### 2026-07-17 — gradient benchmarking against reference implementation

Ran comparative gradient benchmarks between the EpidemicTrajectories package and the reference implementation on the badger dataset to measure AD performance.

**Setup:**
- Dataset: 2384 badgers × 161 timepoints × 34 groups, 6 tests
- AD Backend: ForwardDiff (AutoForwardDiff) for both
- Reference: `badger_ref/bench_ref.jl` (61 continuous parameters)
- Package: `examples/bench_badger_fit.jl` (PracticalBayes LogDensityFunction)

**Results:**

| Implementation | Gradient Eval (median) | Speedup |
|----------------|------------------------|---------|
| Reference      | 0.194s                 | 11.3x   |
| Package        | 2.188s                 | 1x      |

The reference implementation shows significantly faster gradient computation (~11x speedup). Both use ForwardDiff on the same dataset.

**Profiling findings:**
User profiled the package gradient computation and found that almost all time was spent on dynamic dispatching at:
- `src/build.jl:97` — `transition_prob` call (dominant)
- `src/build.jl:99` — `log(p + 1e-12)` (secondary)

This suggests the package's gradient computation is still type-unstable or has excessive runtime dispatch compared to the reference's hand-tuned implementation. The reference likely has more specialized, type-stable code paths for gradient evaluation.

**Potential optimization question:**
Can we move `_param_eltype(model)` out of the hot loop (currently computed at `src/transitions.jl:56-57`) and pass it to `transition_prob`? This might help with type stability and reduce runtime dispatch. Diagnosing the runtime dispatch will likely close most of the speed gap.

**Analysis of runtime dispatch source (Cascade's assessment):**
Note: This analysis comes from a weaker model and should be verified.

The likely sources of runtime dispatch in `transition_prob` are:

1. **Rate function calls**: `first(rates)(model, data, i, t)` in `_accum_row` — if the tuple elements are generic `Function` rather than concrete callable types, each call dispatches at runtime. The earlier fix changed `rate_fns` from `Vector{Function}` to `Tuple`, but the tuple elements might still be inferred as generic `Function`.

2. **Type instability propagation**: If `model` or `data` have any `Any`-typed fields (even after the NamedTuple fixes), reads from them return `Any`, causing instability to propagate through the rate functions.

3. **`_state_index` dispatch**: This function likely does dictionary lookups or similar on the state space, which may not be type-stable.

The dominant dispatch at `transition_prob` suggests the issue is in the rate function evaluation chain, not the `log(p + 1e-12)` itself — that's just where the instability manifests after the type-unstable `p` is computed.

**Recommended next step:**
Use `@code_warntype` on `transition_prob` with the actual badger model to see where inference breaks, then trace back to the source. Tools like `JET.jl` or `Cthulhu.jl` could also help diagnose the type instability.

### 2026-07-17 — investigating remaining iFFBS timing gap and PolyesterForwardDiff performance

Now profiling the reference iFFBS to understand the remaining performance gap. Current iFFBS timings:

| Implementation | iFFBS sweep (median) |
|----------------|---------------------|
| Reference      | 1.708s              |
| Package        | ~5s (estimated)     |

After closing the iFFBS gap, we should investigate why PolyesterForwardDiff is twice as slow on our code compared to the reference. The reference uses DifferentiationInterface's storage for in-place evaluation, which may explain the difference. This performance gap might only be noticeable for PolyesterForwardDiff because it is so fast (making overhead more apparent).

**Note:** The reference code uses multithreading in iFFBS to update transition probabilities, which likely contributes to its performance advantage.

### 2026-07-17 — the actual root cause: `EpidemicData.trans_mat` was unparametrized

Picked back up with a priority from Arthur: find and fix the runtime dispatch
flagged above, comparing properly against the reference throughout.

**Two real bugs found and fixed on the way, unrelated to type instability:**

1. **`epidemic_loglik` ignored `sampling_period` entirely** — summed every
   individual's transitions over the full `1:n_timepoints`, not that individual's
   own sampling window, even though `iffbs.jl` already correctly restricts to it.
   On the badger dataset the average window is 78 of 161 timepoints (48.4% of the
   full range), so this was roughly 2x unnecessary work on every gradient call.
   Fixed in `src/build.jl`'s `loglik`: now loops `first_t:min(last_t,
   n_timepoints)-1` per individual, matching what the reference's own
   `logPost_pars` does (`for j in mint_i:(lastObsAliveTimes[i]-1)`,
   `posterior.jl:145`). `sampling_period` already defaults to `(1, n_timepoints)`
   for every individual when the user doesn't supply one, so this costs nothing
   when it isn't needed and is exact when it is.

2. **`badger_data()` never passed `capt_effort` into `epidemic_data`'s extras** —
   `examples/badger_data.jl` loads it into the raw struct (`d.capt_effort`,
   from `CaptEffort.csv`) but `examples/badger_model.jl`'s `epidemic_data(...)`
   call never forwarded it, even though `badger_fit.jl`'s `EtaKernel` reads
   `data.capt_effort[g, t]`. This meant **the badger fit could not run past its
   first `EtaKernel` conjugate update** — it always threw
   `ArgumentError: EpidemicData has no field or extra 'capt_effort'`. This is
   almost certainly why the earlier "2-sweep smoke test exceeded 600 s" never
   produced output: it wasn't slow, it was crashing immediately and the
   `timeout`/background-capture machinery swallowed the stack trace. Fixed with
   one line in `badger_model.jl`'s `epidemic_data(...)` call.

**Comparative benchmarking, done properly.** Built `badger_ref/bench_ref.jl` +
`badger_ref/bench_mcmc_body.jl`: a verbatim copy of `MCMCiFFBS_`'s setup code
(struct construction, survival refresh, group-level FOI log precompute — literally
copy-pasted, not reimplemented) up to the point the real function starts its
`for iter in 1:N` loop, then a small timing harness bolted on in place of the real
loop body that times (a) one full iFFBS sweep over all 2384 individuals and (b)
several `grad_pars` calls through the SAME cached `prepare_gradient` `prep` and
`backend` the real HMC/NUTS step would use — reporting min/mean/median rather than
a single sample. The reference's `Project.toml`/`Manifest.toml` were stale/empty
(referenced packages like `Polyester`/`DifferentiationInterface`/`ForwardDiff`
weren't even in the manifest) — deleted both and re-resolved a fresh environment
from the actual `using`/`import` statements across every reference source file.

Reference results (`bench_ref.jl forwarddiff`, `BENCH_REPEATS=2 BENCH_NGRAD=2`):

```
iFFBS sweep (2 sweeps):      min=5.326s  mean=7.582s  median=7.582s
grad_pars (4 evals, 61 continuous params):
                              min=0.099s  mean=0.846s  median=0.167s
```

Our gradient at that point (`grad_profile.jl`, `sampling_period` fix in place, 10
evals): **min=1.493s mean=1.809s median=1.658s** — a genuine ~10-15x gap, too big
to be explained by anything other than a real inefficiency, matching the
`dispatch?=Y, gc?=Y` flags ProfileToLLM had already pinned to `loglik` at
`build.jl:97` (`transition_prob` call) and `:99` (`log(p + 1e-12)`), accounting for
87.3% of self time between them.

**Root cause, found by reading the struct, not guessing at the call site.** Arthur
asked directly: *what does the reference do differently in how it feeds data and
functions to the likelihood?* The reference's rate functions
(`groupLevelInfectionForce`, `siler_surv`, `progression_fn`, ...) are **free
top-level functions** — `iFFBS_Data` never stores a function or a rate-function
bundle as a field, only plain arrays/scalars. Every call site sees the function's
own literal, concrete, singleton type at compile time; there is no struct-field
indirection to erase it.

Our design can't do that — `EpidemicData` genuinely has to hold the transition
spec as *data*, because the package has no idea ahead of time what rate functions
a user will supply (that's the whole point of `TransitionSpec{RF<:Tuple}`, whose
own docstring already explains why `RF` is a type parameter: so
`transition_matrix_at`'s tuple recursion sees one concrete function type per rate
instead of dispatching through `Any`). The bug: `EpidemicData`'s struct
declaration — `trans_mat::TransitionSpec` (src/data.jl:49, pre-fix) — dropped the
`{RF}` parameter. A struct field's *declared* type is what the compiler uses to
infer everything downstream of a field read, regardless of how carefully the
*value* stored there is typed. So every `data.trans_mat.rate_fns` access (i.e.
every single call inside `transition_prob`/`_accum_row`, the hottest site in the
whole gradient) saw the abstract `TransitionSpec` — RF erased, tuple length and
element types unknown — and fell through to runtime dispatch, exactly undoing the
whole point of parametrizing `TransitionSpec` on `RF` one layer further out. The
`_accum_row`/`_fill_rates!` tuple-recursion trick (see `transitions.jl`'s own
docstrings) was never wrong; the container holding it was silently erasing the
type information the recursion depends on before the recursion ever got a chance
to specialize.

Fix, `src/data.jl`:

```julia
struct EpidemicData{SS,OP,RC,EX<:NamedTuple,AG<:NamedTuple,RF<:Tuple}
    ...
    trans_mat::TransitionSpec{RF}
    ...
end
```

`RF` is inferred automatically from the `trans_mat` argument at construction —
no change needed to `epidemic_data(...)`'s body, since it already passes a real
`TransitionSpec{RF}` instance positionally into the default constructor.

**Result — 100 tests still pass, and the gradient now matches the reference:**

| | before | after `sampling_period` | after `TransitionSpec{RF}` | reference |
|---|---|---|---|---|
| gradient (61 params) | 3.63 s | 1.5–1.8 s | **0.125–0.136 s** | 0.099–0.167 s |
| profile: dispatch?/gc? on `loglik` | Y/Y (87.3% self) | Y/Y | **all `-`** | — |

The post-fix profile is dominated entirely by genuine floating-point arithmetic
(`*`, `+`, `fma_llvm`, `getindex`) — no dispatch, no GC in the top rows — which is
what a numerically-bound gradient is supposed to look like. Our median (0.134 s)
now sits almost exactly on the reference's median (0.167 s); the reference's own
`mean` (0.846s) is pulled up by one compilation-affected outlier in a 4-sample
run, so the real comparison is the medians, and they match.

**Revised 10,000-sweep feasibility.** At the new gradient cost, ~32 leapfrogs/sweep
(tree depth 5) is ~4.3 s of NUTS gradient work; with iFFBS around 5–8 s/sweep
(same ballpark as the reference's own 5.3–9.8 s), a full sweep is roughly
**10–13 s**, putting 10,000 sweeps at **~30–35 hours (about a day and a half)** —
down from the earlier ~330-hour estimate. Not yet attempted end-to-end (the
`capt_effort` bug meant no full sweep had ever actually completed before this
session), but the two blocking problems — a crash on the first conjugate update,
and a ~12x-too-slow gradient — are both fixed now.

**Not fixed, lower priority, noted for later (at the time):** `derived_summaries::Vector{Function}`
(`src/data.jl`) has the identical unparametrized-field shape, but it's read only
by the iFFBS/simulator/aggregate-update paths (`build.jl`'s `simulate`,
`iffbs.jl`, `transitions.jl`'s `make_rest_contribution`) — never by
`epidemic_loglik`, the NUTS gradient hot path this session was about. The
reference's iFFBS sweep and ours are already in a similar ballpark (5-8s each),
so this isn't blocking; fixing it would need the same kind of struct
reparametrization (`EpidemicData{...,DS<:Tuple}`) and is a candidate for a
follow-up pass if iFFBS timing becomes the bottleneck again.

**This turned out to be wrong** — see the next entry. The "similar ballpark"
comparison was contaminated by JIT-compilation time on both sides (a single,
uncontrolled sample per side), and once measured cleanly the iFFBS gap was ~6x,
not "similar."

### 2026-07-17 — proper apples-to-apples benchmarking: methodology, then the real iFFBS fix

Arthur's next ask, precisely: *"cant we copy the mcmciffbs function/script and make
a new one that follows all the same steps, except that when it gets to the hmc
bit, it benchmarks gradient evaluation"* — i.e. stop eyeballing single numbers
from different runs and build a real, controlled comparison. Then, mid-benchmark:
*"never run benchmarks concurrently with other code"* (two jobs had been launched
in parallel to save wall-clock time, which contaminates both timings on a shared
8-core machine — added to memory as a standing rule) and *"the 1 sweep contains
compilation time? we need compilation-free benchmarks."*

**What got built**, all still in the repo:
- `badger_ref/bench_ref.jl` + `badger_ref/bench_mcmc_body.jl` — a byte-for-byte
  copy of `MCMCiFFBS_`'s setup code (struct construction, survival refresh,
  group-level FOI precompute) up to its `for iter in 1:N` loop, with the real
  loop body replaced by a small harness timing (a) one full iFFBS sweep and (b)
  several `grad_pars` calls through the same cached `prep`/`backend` the real
  HMC/NUTS step uses. The reference's own `Project.toml`/`Manifest.toml` were
  stale (referenced packages weren't even in the manifest) — deleted and
  re-resolved from the actual `using`/`import` statements across the source.
- `examples/bench_gradient.jl` — the package-side equivalent: gradient timing
  through `PracticalBayes.LogDensityFunction` (exactly what NUTS calls) AND an
  iFFBS sweep through `epidemic_latent_sampler` (exactly what `iFFBSKernel`
  calls), same reporting shape.
- Both warm up once, untimed, before their timed loop — the first call to any
  freshly-JIT'd function compiles it, and that cost had been silently dominating
  single-sample readings on both sides (reference: repeat-1 iFFBS = 4.5-4.6s,
  repeat-2 = 1.2-1.3s, SAME sweep, only compilation differs).
- `badger_ref/run_all_benchmarks.sh` runs all variants **strictly sequentially** —
  no `&`, no background jobs racing each other, each `julia` call blocks until it
  exits before the next starts.

**Clean, compilation-free, non-concurrent numbers** (both sides, `AutoForwardDiff`):

| | Reference | Package (before this session's iFFBS fix) |
|---|---|---|
| iFFBS sweep | min=1.216s mean=1.216s median=1.216s (1 clean sample) | min=7.689s mean=7.714s median=7.714s (2 sweeps) |
| Gradient (61 params) | min=0.073s mean=0.102s median=0.099s (30 evals) | min=0.135s mean=0.16s median=0.148s (15 evals) |

Also benchmarked `AutoPolyesterForwardDiff` (Arthur: *"try both with
polyesterforwarddiff aswell as forwarddiff"*): gives the reference a genuine ~3x
gradient speedup (median 0.099s → 0.032s). On our side it does not work at all —
crashes inside `Bijectors.VectorBijectors`'s `with_logabsdet_jacobian` when
PolyesterForwardDiff's threaded chunk-splitting (`StrideArraysCore` arrays) meets
our link/invlink transform (`ArgumentError: tuple must be non-empty`,
`StrideArraysCore/ptr_array.jl:998`). Real cross-package incompatibility, not
attempted to fix this session — flagged for whoever picks up PolyesterForwardDiff
support on our side later.

**So: gradient parity confirmed (median ~0.13-0.15s vs reference's ~0.10s, both
close), but iFFBS was genuinely ~6x slower (7.7s vs 1.2s), not "similar ballpark"
as the previous entry concluded from contaminated data.**

**Chasing the real cause — two real fixes that didn't move the number, a third
that did.** Arthur: *"please go ahead and investigate the iffbs slowness, based on
your suspicion first"* — the suspicion being the same unparametrized-abstract-
field pattern already found for `trans_mat`.

1. **`derived_summaries::Vector{Function}` → `Tuple`.** `AggregateDeclaration`
   (`src/aggregates.jl`) and `EpidemicData` (`src/data.jl`) both parametrized
   (`AggregateDeclaration{DS<:Tuple}`, `EpidemicData{...,DS<:Tuple}`); the
   `@aggregate` macro now emits `($(lams...),)` instead of `Function[lams...]`;
   `epidemic_data`'s verbose-fallback constructor converts with `Tuple(...)`
   instead of `Vector{Function}(...)`. **Confirmed real** by profiling before/
   after: the `dispatch?=Y` flags on `rest_contribution`
   (`transitions.jl:234/251`) and `iffbs_individual!` (`iffbs.jl:91/100`)
   completely disappeared — every row in the post-fix profile shows `dispatch?=-`.
   **But the sweep timing did not move** (still 8.1-8.5s). Dispatch was real but
   was never the dominant cost — profiling percentages can mislead about which
   fix will actually matter if something else with a larger absolute cost has no
   dispatch flag to notice it by.
2. **`transition_matrix_at!` buffer reuse in `forward_filter`.** The forward
   filter built a fresh `zeros(N,N)` + `zeros(N)` on every one of ~185,000
   `(individual, timepoint)` calls per sweep (~30MB of short-lived allocation,
   `gc?=Y` on `Array` at `boot.jl:477/479`, ~24% of self time). Added
   `transition_matrix_at!(P, rowsum, ...)` (in-place core; `transition_matrix_at`
   is now a thin allocating wrapper around it, so every OTHER caller —
   `simulate`, tests — is unaffected). `forward_filter` now preallocates the
   whole `trans_cache` as one `N × N × n_t` array up front and reuses one
   `rowsum` scratch vector across the loop; `backward_sample!` updated to index
   the 3D array instead of a `Vector{Matrix}`. **Confirmed via 100/100 tests**
   passing (the buffer-reuse path produces bit-identical results to fresh
   allocation — same arithmetic, just not re-allocated). **Sweep timing again did
   not move** (8.1-8.9s). Two real, profiler-motivated, test-verified fixes in a
   row, zero measurable effect — a strong signal to stop trusting the profiler's
   percentage table and instrument directly instead.
3. **Direct instrumentation, not profiling.** Timed `iffbs_individual!`'s four
   phases separately (reverse aggregates / `forward_filter` / `backward_sample!`
   / reapply aggregates) for one individual, and extrapolated per-cell cost across
   every individual's own window — this matched the observed sweep total
   (6.8s extrapolated vs ~8s observed), confirming nothing was hiding outside
   what was measured. `forward_filter` alone was 87% of one individual's time
   (3.575ms of 4.091ms). Both prior fixes touched code INSIDE `forward_filter`'s
   own loop — so what else was in there?
   **`make_neighbor_logprob_from_transitions`** (`transitions.jl`), called from
   `rest_contribution` inside `forward_filter`'s per-timepoint coupling term
   (`data.rest_contribution(...)`, called twice per timepoint — once for the
   initial state, once per subsequent step). Its old body called the
   **allocating** `transition_matrix_at` (not `!`) to read ONE entry
   (`Pj[from_state, to_state]`) — exactly the "build the whole matrix for one
   entry" waste `transition_prob` was already built to avoid, in the *likelihood*
   (`epidemic_loglik`, fixed two sessions ago). Nobody had made the same swap
   here, one level deeper: `rest_contribution` calls this once per affected
   individual per candidate state (`n_states × |affected|` times per `(i,t)`) —
   the real multiplier neither of the previous two fixes touched, because they
   were both inside `forward_filter`'s OWN transition-matrix build, not this
   nested call through the coupling term. Even the function's own docstring
   comment ("saves building that neighbour's whole transition matrix") was
   sitting right next to the code that still built it whenever a neighbour WAS
   coupled.

   Fix: `make_neighbor_logprob_from_transitions` now calls `transition_prob`
   instead of `transition_matrix_at`. One-line change to the hot line; no other
   file touched.

**Result — the actual iFFBS fix, verified clean (100/100 tests, then a fresh
non-concurrent benchmark):**

| | Reference | Package: before any iFFBS fix | after `derived_summaries` (no change) | after buffer reuse (no change) | after `neighbor_logprob` fix |
|---|---|---|---|---|---|
| iFFBS sweep | 1.2-1.3s | 7.7-8.9s | 8.1-8.5s | 8.1-8.9s | **2.9-3.2s** |

**~2.7x speedup**, closing the gap from ~6-7x to ~2.3-2.5x versus the reference.
Gradient timing was never touched by any of this and stayed at 0.10-0.15s
throughout, matching the reference's own 0.10s median.

**Lesson for next time, stated plainly since it cost real wall-clock this
session:** a profiler's self-time percentage table is not the same claim as "this
is the binding constraint." Two fixes here were independently real (confirmed by
before/after profiling and by tests), yet neither changed wall-clock time at all,
because both were inside the wrong function relative to where the actual
multiplier lived. What broke the deadlock was per-phase direct timing
(`@elapsed`/`@allocated` on hand-picked sub-calls) that pointed at
`forward_filter` as 87% of one individual's cost, followed by reading the
`rest_contribution`/`neighbor_logprob` call chain by eye rather than trusting the
next profiler run to point at the right line automatically.

**Remaining gap (2.9-3.2s vs 1.2-1.3s, ~2.3-2.5x) not yet investigated.** Plausible
next places to look, not yet checked: the reference precomputes per-`(g,t)`
group-level FOI log-probabilities once per sweep (`update_group_level_logs_from_counts!`,
`logProbStoSgivenSorE` etc.) and its discrete forward-filter reads those cached
scalars directly, whereas our `badger_infection` rate function recomputes the FOI
independently for every individual sharing a group (previously noted as a
possible optimisation, never implemented — see the earlier "gradient benchmarking
against reference implementation" entry, point about `groupLevelInfectionForce`
being called per-individual). That asymmetry applies to `forward_filter`'s
per-timepoint `transition_matrix_at!` call too, not just the likelihood. Not
confirmed as the cause of the remaining gap — would need the same
profile-then-instrument discipline as above before touching it.

### 2026-07-17 — closing the iFFBS gap: `coupling_trans_mat`, a power-user seam

What we are chasing, and how, written down before the code so the reasoning can be
checked against the result rather than reconstructed after it.

**The target.** Our iFFBS sweep is ~2.6-2.9s; the reference's is ~1.47s. The
gradient is NOT the target — it is already at parity (0.113s vs the reference's
0.108s median, ForwardDiff; 0.063s vs 0.033s under PolyesterForwardDiff, which
now works on our side after PracticalBayes commit f708858).

**Where the time is, measured not guessed.** Profiling the sweep sorted by TOTAL%
(not self%):

```
iffbs_individual!    98.2%
  forward_filter     84.3%
    rest_contribution   69.5%     <- the coupling term
      transition_prob   65.8%
```

The coupling term owns ~70% of the sweep. Worth recording that the SELF%-sorted
table of the same profile shows `rest_contribution` at 6.4% and looks completely
flat — reading that table, this entry nearly got written as "the coupling loop is
NOT the bottleneck, the logProbRest idea is wrong-targeted". Self% attributes
cycles to leaf arithmetic (`getindex`, `muladd`, `fma_llvm`, `exp`); it does not
tell you which caller drives them. **Sort by total when the question is "what
should I change", by self when the question is "what is this line doing".** Third
time this session that self% nearly sent the work in the wrong direction.

**Why it costs that much.** `rest_contribution` (transitions.jl) answers a
counterfactual: for each of the `n_states` candidate states of the focal `i` at
time `t`, apply the user's summaries, loop over EVERY affected neighbour `j`
calling `neighbor_logprob`, then reverse the summaries. That is
`n_states × |affected|` rate evaluations per `(i, t)` — on badgers, 4 × ~70
groupmates × ~78 timepoints × 2384 individuals. The reference does not have this
loop at all: it keeps `logProbRest[s, j, t]` / `logProbRestTotal[s, t]` and
patches them incrementally per individual (`updaters.jl:290`), so its coupling
cost is an O(1) array read.

**Three designs considered, two rejected.**

1. *Memoise the transition funcs generically, dropping `i` and `t` from the key.*
   Rejected: the package cannot know what `(i,t)` REDUCES to. `badger_infection`
   reduces it to `(g,t)`; `siler_survival` to `(age[i,t], t<=last_capture[i])`;
   `badger_progression` to `()`. Those mappings live in user code behind arbitrary
   data lookups. A generic "drop `i`" rule would silently return one badger's rate
   for another. A user-declared cache key (`@cached (data.social_group[i,t], t)
   badger_infection`) would be sound, but then hits the invalidation problem below.

2. *Package-side `logProbRest[s, j, t]` cache, built from the counterfactuals
   `rest_contribution` already runs.* Genuinely generic — the package knows
   `n_states`, the user already declares `affected_individuals[t,i]` (which `j`
   are affected) and `coupled_transitions` (which of `j`'s moves are sensitive),
   so invalidation IS derivable from what we already have. Rejected on expected
   value, not correctness: on badgers `affected_individuals[t,i]` IS the focal's
   groupmates, so resampling `i` dirties ~70 neighbour-entries — roughly
   everything we would have cached. The reference only gets away with the
   per-individual cache because a GROUP-level cache (`logProbStoSgivenSorE[g,t]`,
   34×161) sits underneath it: an individual's move dirties one group-cell, not 70
   neighbour-cells. So the group-level array is the load-bearing part, and the
   per-individual array is bookkeeping on top of it.

3. **Chosen: let the power user supply the group-level array, via a separate
   transition spec for the coupling term.** The user already CAN maintain a
   `foi[g,t]` array — a derived summary that recomputes one cell from the counts
   the other summaries just updated, O(1) per individual. What they could not do
   is use it, because the one rate spec (`trans_mat`) serves both the AD
   likelihood and the sampler, and a Float64 FOI cache read under AD freezes the
   parameter dependence (measured: gradient w.r.t. lambda/alpha/beta/q collapses
   to the PRIOR gradient, ~±1 instead of +300/-195/-20/+26 — while the
   log-density stays BIT-IDENTICAL, so nothing warns you; see the
   `badger_model_foicache.jl` check earlier today).

**The seam already exists.** `rest_contribution` reaches the rates only through
`neighbor_logprob`, a closure `make_rest_contribution` already accepts as a
keyword. `epidemic_data` simply hard-wires it to `trans_mat` (data.jl:248). So:

```julia
epidemic_data(...; coupling_trans_mat = trans_mat)   # default: unchanged behaviour
```

- `trans_mat` → the honest recomputing rate. Used by `epidemic_loglik` (AD) and
  `forward_filter`. Unchanged.
- `coupling_trans_mat` → the cached rate. Used ONLY by `neighbor_logprob`, i.e.
  only inside `rest_contribution`, which the AD path never reaches.

This is why it is safe where the earlier all-in-one FOI cache was not: the cached
spec is structurally unreachable from the gradient, not merely kept away from it
by discipline. The package stays ignorant — it does not know one spec is "cached",
only that it was handed two, exactly like `coupled_transitions` is a pure
optimisation it takes on trust.

**Known sharp edge:** nothing stops a user passing a cached spec as the MAIN
`trans_mat` and silently getting the frozen gradient measured above. Documented
loudly; a build-time check that both specs agree at one `(i,t)` would catch the
common case cheaply.

**Prediction to check against.** If the coupling term is 70% of the sweep and the
FOI redundancy is ~70 individuals per group, caching it should take a large bite
out of that 70% — but `transition_prob` also evaluates the survival and
progression rates, which this does NOT cache, so the sweep will not fall to the
reference's 1.47s. Anything from "no change" (in which case the FOI arithmetic was
never the cost inside that loop, and the remaining gap is elsewhere) to ~1.5-2x is
consistent with what is known. Recorded here so the benchmark can contradict it.

**The result: the seam works, the cache pays ~1.19x, and the prediction landed at
its pessimistic end.** Six variants, one clean sequential run, warm-up excluded:

| variant | iFFBS (median) | gradient (median) |
|---|---|---|
| reference — ForwardDiff | **0.98s** | 0.091s |
| reference — PolyesterForwardDiff | 1.201s | **0.033s** |
| package — ForwardDiff | 2.527s | 0.112s |
| package — PolyesterForwardDiff | 2.467s | 0.063s |
| package — ForwardDiff + cached-FOI coupling | **2.12s** | 0.110s |
| package — Polyester + cached-FOI coupling | **2.117s** | 0.061s |

Two things confirmed, one refuted.

*Confirmed — the seam does what it is for.* The gradient is unmoved (0.112 → 0.110s;
0.063 → 0.061s, i.e. noise) while the sweep improves, which is exactly the
signature of a cached rate confined to the coupling term. The pre-flight gate said
the same thing more strictly: gradient bit-identical to base on all 61 parameters
(max|diff| = 0.000e+00), including the four (lambda/alpha/beta/q) that collapsed
to the prior gradient when the same file wrongly fed the cached spec to
`trans_mat`. Same model file, same cache, one line different — broken vs exact.

*Confirmed — the cache pays.* 2.527 → 2.12s and 2.467 → 2.117s: ~1.19x, consistent
across both AD backends, for one keyword and one changed line.

*Refuted — FOI redundancy is NOT the substance of the coupling cost.* The
prediction allowed anything from "no change" to ~1.5-2x; the outcome is ~1.19x,
the pessimistic end. If the ~70-individuals-per-group FOI recomputation were what
made `rest_contribution` 70% of the sweep, caching it should have taken a far
bigger bite. It did not. So the coupling term is expensive because of the LOOP
ITSELF — `n_states` candidate states × ~70 neighbours × (apply summaries, call
`transition_prob`, reverse summaries) per `(i, t)` — not because of the arithmetic
inside one rate call. Cheapening each iteration is not where the money is.

**Which is evidence FOR the design rejected earlier in this entry.** The reference
does not win by making iterations cheaper; it wins by not having the loop at all
(`logProbRestTotal[s, t]` is an O(1) read, patched incrementally by
`updateLogProbRestTotalIndiv!`). Option 2 above (package-side `logProbRest[s,j,t]`
cache) was rejected on expected value — the guess being that on badgers a focal
dirties ~70 neighbour-entries, i.e. most of what was cached. That guess is now the
only thing standing between us and the remaining gap, and it is still a guess. The
cheap measurement that would settle it: instrument one sweep, count `(s,j,t)`
entries reused vs recomputed. If reuse is low the idea is dead regardless of how
cleanly it is expressed; if high, we know the ceiling before building.

**Gap now.** ~2.12s vs the reference's 0.98-1.47s (note the reference's own iFFBS
varies run to run — 1.468s in the previous session's run, 0.98s in this one, both
single post-warm-up samples; ours is stable at 2.1-2.6 across runs). So call it
1.5-2x, down from ~6x at the start of the day. The gradient is at parity on
ForwardDiff (0.110s vs 0.091s) and ~2x adrift on PolyesterForwardDiff (0.061s vs
0.033s) — the latter is a new observation, not yet investigated, and worth noting
that PolyesterForwardDiff buys the reference 2.8x but us only 1.8x.

### 2026-07-17 (cont.) — profiling the reference, and a real bug found in it

Following the ~1.19x result above, user's direction: profile the reference itself
and get an algorithmic breakdown, rather than keep inferring from source reading
alone. Two things came out of this — one closes the algorithmic question, one is
a genuine finding about the reference's own code.

**First profile attempt: nearly all noise, not signal.** `BENCH_PROFILE=1` (added
to `bench_mcmc_body.jl` earlier) had a scoping bug (`sumLogCorrector` was local to
a loop that had already ended) that crashed print_profile after `Profile.@profile`
had already captured data — fixed with a dedicated `profile_sumLogCorrector`
local. Once fixed, the profile showed something unexpected: **`Polyester.batch` /
`_batch_no_reserve` owned 92.4% of TOTAL time, `wait` 85.2%, and self-time was
dominated by a bare integer `<` comparison (39.7%) and `process_events` (27.8%)**
— i.e. thread-scheduler dispatch and idle-spin, not the algorithm. The real
per-individual work (FOI arithmetic, `iFFBScalcLogProbRest!`) was invisible even
at `max_rows=60`, buried under 44 more omitted rows of task-queue bookkeeping.

**Why: `@batch per=thread`/`@batch per=core` is called PER INDIVIDUAL, twice.**
Traced the call graph precisely this time (redefinition matters — Julia keeps the
LAST definition when a function is defined twice in the same file, and
`updaters.jl` has two definitions each of `updateLogProbRestTotal!` and
`updateLogProbRestTotalIndiv!`; the dead first pair uses plain `@batch`, the live
second pair uses `@batch per=thread`). Both live functions, plus
`update_group_level_logs_from_counts!` (`transitions.jl:151`, `@batch per=core`),
run once per `tt in 1:maxt-1` **inside `iFFBS()`, called once per individual** —
so a sweep launches a threaded batch region 3 × 2384 = 7152 times, each over only
160 iterations doing ~13 cheap array ops. That is a textbook too-fine-grained
threading anti-pattern: the launch/dispatch/join cost is being paid far more
often than the parallel work it buys.

**Confirmed by disabling all four `@batch` sites (replaced with plain `for`,
reverted immediately after — these are third-party reference files, not ours to
keep edited).** Single-threaded, same profiling protocol: **iFFBS sweep = 1.064s**
— at or below every threaded run measured today (0.98s, 1.201s, 1.468s, 1.705s,
1.862s, noisy but none clearly faster than 1.064s). More importantly, the profile
is now real: `updateLogProbRestTotal!` 31.5% total (15.0% self — the running-total
patch), `iFFBScalcLogProbRest!` 4.4%+3.2% self across its two sites,
`update_group_level_logs_from_counts!` 2.1% self. **The reference's own threading
is a wash at best on this workload, and its published wall-clock numbers all day
were measuring thread-scheduler overhead as much as algorithm.**

**What this settles.** The O(maxt)-per-individual running-total design (`logProbRest[s,jj,tt]`
+ `logProbRestTotal[s,tt]`, patched by subtract-old/recompute-one-row/add-new
rather than rebuilt) is genuinely cheap — ~13 ops/timepoint/individual, single-
threaded, no neighbour loop at all. That is categorically different from our
`rest_contribution`, which is O(n_states × |affected|) per `(i,t)` — for badgers,
`4 × ~70 ≈ 280` neighbour-visits per timepoint against their ~13 array ops. The
~20x-per-cell estimate from reading source alone holds up under profiling; it was
never really in doubt. What was in doubt — whether the reference's ~1-1.5s
wall-clock number was a fair thing to chase — is resolved: it is, and arguably
beatable, since the reference is not even collecting on its own algorithmic
advantage cleanly (threading tax eating into it).

**Two structural reasons ours is slower than a same-shape port would be, beyond
the O(n_states × |affected|) vs O(maxt) difference:**

1. *Array layout.* Reference: `logProbRest[s, jj, tt]`, `s` fastest-varying
   (column-major Julia) — the 4-state read/write clump at the top of
   `updateLogProbRestTotalIndiv!` is one cache line. Ours: `X[t, i]`, `t`
   fastest-varying — `rest_contribution`'s inner `for j in ids` loop reads
   `X[t,j]`/`X[t+1,j]` for varying `j` at fixed `t`, striding across
   `n_timepoints` per neighbour: a cache miss on every one of the ~280
   neighbour-visits per timepoint. Not measured directly (would need a
   cache-miss counter, not just wall-clock), but the layouts are unambiguous from
   the code and consistent with the direction of the gap.
2. *No running total at all.* Even ignoring cache effects, we literally
   recompute the sum over affected individuals from scratch on every `(i,t)`
   call, where the reference maintains it as state and only ever touches the ONE
   row (`logProbRest[:, id, :]`) that actually changed.

**Open design question — NOT implemented, for the user to decide.** Port the same
shape: a package-side `logProbRest`-equivalent (`n_states × n_individuals ×
n_timepoints`, or scoped to whatever each individual's own window needs) plus a
running per-`(s,t)` total, patched incrementally per individual rather than
rebuilt by `rest_contribution`'s counterfactual loop. This is Option 2 from the
earlier entry today, rejected then on a guess that invalidation would touch most
of what was cached (~70 neighbour-entries per resampled individual). That guess
was about which CELLS go stale, not about whether maintaining a running TOTAL
instead of recomputing a SUM is cheaper regardless — and today's profiling says
yes, decisively, for the reference's own version of exactly this problem.
Building it single-threaded (no `@batch`, per today's finding that threading this
shape of loop is not obviously worth it) is the natural first attempt. This is a
package-side structural change, materially bigger than `coupling_trans_mat`
(one keyword) — needs sign-off before starting, and needs its own invalidation
correctness worked out generically from `affected_individuals`/`coupled_transitions`
(both already user-declared) rather than assumed.

### 2026-07-17 (cont.) — the gap closed, as power-user code: `rest_contribution` keyword

The previous entry ended proposing a package-side running-total cache and flagging
it as "materially bigger than coupling_trans_mat, needs sign-off." It turned out
NOT to need a package-structural change at all — the same seam that worked for the
FOI cache works here: expose the coupling term itself as a keyword.

**The seam (`src/data.jl`).** `rest_contribution` was already a per-instance field
(`RC` type parameter) and already the ONLY route `forward_filter` uses to reach the
coupling; `epidemic_data` simply always built it via `make_rest_contribution(...)`
and never let the caller override. Added `rest_contribution=nothing` keyword,
defaulting to the brute-force builder. One keyword, one branch. A power user can
now supply their own coupling term of the same signature
`(model, data, X, i, t, n_states, affected_override=nothing) -> weight vector`,
exploiting their model's structure to skip the default's
`O(n_states × |affected|)` neighbour loop. Documented with the contract (must be
side-effect-free, return per-candidate weights, ones at the final timepoint) and a
worked sketch. Like `coupled_transitions`/`coupling_trans_mat`, the package stays
ignorant — it takes whatever function it's handed.

**The badger implementation (`examples/badger_model_reststotal.jl`).** The
derivation that made this both O(n_states) AND exact — arrived at only after two
wrong attempts, each caught by a direct entry-for-entry check against the
brute-force `rest_contribution` (`scratchpad/rest_match.jl`, the discipline that
mattered here):

A susceptible neighbour's realised one-step move contributes, per focal candidate
scenario c, only:
- `S -> E`: `survival_j · foi_c`   → c-dependent part `log(foi_c)`
- `S -> S`: `survival_j · (1-foi_c)` → c-dependent part `log(1-foi_c)`
- `S -> D`: `1 - survival_j`         → **does NOT depend on c**

(`S -> I` is impossible in one step.) The individual survival factor is not coupled
to the focal, so it is constant across c; the `S -> D` term is entirely c-independent.
BOTH cancel when `rest_contribution` normalises (`logw .-= maximum(logw)`). So the
only thing that survives is

    restTotal_c = nSE · log(foi_c) + nSS · log(1-foi_c)

where nSE / nSS are just COUNTS of susceptible neighbours moving S->E / S->S — two
integer reversible aggregates (a susceptible individual doing S->E increments nSE,
S->S increments nSS), order-independent, no staleness. The three scenario FOIs
(`SorE`/`I`/`D`, from the three distinct (I,M) the focal's four candidate states
induce) are cheap enough to recompute on demand — 3 numbers — so nothing else is
cached. Because iFFBS reverses the focal out of the aggregates before
`forward_filter`, nSE/nSS already exclude the focal; no self-subtraction needed.

**Two dead ends first, both worth recording:**
1. Caching a per-neighbour float `restRow[j,t,c]` + group float `restTotal[g,t,c]`,
   rebuilt by a per-individual summary. Fragile: the per-individual zero-and-re-sum
   left `restTotal` reflecting whichever individual fired last during the
   incremental `apply_derived_summaries!` build, with stale counts. Differed from
   brute-force by 0.346.
2. Even with the total made consistent, the row used a binary `to_E ? foi : 1-foi`,
   which mis-scored `S -> D` (death) moves as `S -> S` — injecting spurious
   candidate-dependence where the true `S -> D` term is candidate-constant. Same
   0.346 error. The single-cell dissection (`scratchpad/rest_five.jl`) that showed
   `S->D` neighbours with a flat `transition_prob = 1e-12` across all four
   candidates is what revealed the S->D-cancels insight and led to the
   integer-count decomposition.

The integer-count version matches brute-force to **5.1e-13** across 4846 cells
(4381 with informative coupling).

**Result (ForwardDiff, warm-up excluded, min/mean/median):**

| variant | iFFBS sweep | gradient (61 params) |
|---|---|---|
| base (brute-force coupling) | 2.604 / 2.784 / 2.737s | 0.104 / 0.149 / 0.148s |
| **reststotal (this)** | **0.784 / 0.832 / 0.845s** | 0.101 / 0.115 / 0.115s |
| reference (context) | ~1.0-1.5s | ~0.1s |

**iFFBS 3.24x faster** (2.737 → 0.845s), now ~1.5x FASTER than the reference's own
sweep — from ~6x SLOWER at the start of the day. Gradient untouched, as it must be:
`rest_contribution` is never on the AD path. Exact posterior (5e-13 vs the
brute-force coupling).

The whole "close the iFFBS gap" arc landed as three additive, opt-in power-user
levers over an unchanged generic core — `coupled_transitions` (skip uncoupled
neighbours), `coupling_trans_mat` (cache the rate the coupling reads), and now
`rest_contribution` (replace the coupling loop with a running total) — none of
which the package understands beyond their signatures. The central design rule
held: every speedup is user code the package takes on trust, not a special case
baked into it.
