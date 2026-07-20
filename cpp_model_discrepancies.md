# Model discrepancies: our package vs the C++ reference (classic rcpp)

Audit to build an *exactly* C++-matching variant (`badger_fit_gompertz_hmc.jl`
already matches survival family + priors; this covers the remaining SUBTLE
differences in how entry, death, and post-monitoring captures are handled).

**The C++ variable names are misleading — decoded here from source, not names.**

Sources read: `logPost_HMC.cpp` (the differentiated survival/transition loglik),
`iFFBS_fixedPars.cpp` (the forward filter + backward sample), `MCMCiFFBS_.cpp`
(the driver that builds `lastObsAliveTimes`, `probDyingMat`, `ageMat`), and our
`src/build.jl`, `src/transitions.jl`, `examples/badger_model*.jl`,
`examples/badger_data.jl`.

Legend: **[MATCH]** already agrees · **[DIFF]** genuine discrepancy · **[CHECK]**
needs a runtime check to be sure.

---

## 1. [DIFF] The C++ does NOT condition survival on entry to the study

`logPost_HMC.cpp:59-61`:
```cpp
if(birthTimes[i] < startSamplingPeriod[i]){
    loglik += logS(ageMat(i, startSamplingPeriod[i] - birthTimes[i]), a2, b2, c1);
}
```
For any badger born BEFORE the study started, the C++ adds `logS(age_at_entry)` —
the log survival probability from BIRTH to STUDY ENTRY — as a likelihood term.
`logS` (DlogSt.cpp) is the cumulative Gompertz-Makeham log-survival, NOT a
one-step transition.

This means the C++ likelihood is **P(survive birth→entry) × P(trajectory during
study)**, i.e. it does NOT condition on the badger being alive at entry — it pays
the full survival probability from birth. A correctly-conditioned model would
DIVIDE this out (condition on survival to entry), which the C++ does not.

**Ours:** `epidemic_loglik` has no such term at all. Our survival enters only
through the per-timestep transition matrix over `sampling_period` (see §2); we
never add a birth→entry cumulative-survival factor. So the two differ by exactly
this `logS(age_at_entry)` term for every pre-study-born individual — which is
MOST of the population.

To match the C++ *exactly* (even though it is arguably wrong), we would need to
add `logS(age_at_entry)` per pre-study-born individual to the differentiated
likelihood. This is a package-level thing: it is a per-individual term keyed on
`birth_time < start`, not a per-transition term, so it does not fit the
`@transitions` mechanism. It would go in a user `@addlogprob!` alongside
`epidemic_loglik` — a third likelihood piece, `entry_survival_loglik`.

**`ageMat(i, startSamplingPeriod - birthTimes)` is the age at entry.** Note the
index arithmetic: `startSamplingPeriod - birthTimes` is the number of steps from
birth to entry, and `ageMat(i, that)` reads the age at that offset. Age indexing
otherwise matches (see §6).

---

## 2. [MATCH, mechanism differs] Where survival enters the per-step likelihood

**C++** (`logPost_HMC.cpp:69-96`): loops `j = mint_i .. lastObsAliveTimes[i]-1`
and for each realised transition adds `log_pti = TrProbSurvive_(age)` PLUS the
infection/progression log-prob. Death (`z_t==9`) adds `log_qti =
TrProbDeath_(age)` and no survival. So survival multiplies every alive→alive move.

**Ours:** `@survival` makes every non-death rate `survival * rate` and adds
`state→D = 1 - survival`, so `transition_prob` returns `survival * move_prob` for
alive→alive and `1 - survival` for →D. Same factorisation.

**These agree in structure.** The bounds differ, though — see §3.

---

## 3. [DIFF] Loop bounds: `lastObsAliveTimes` vs our `sampling_period` end

**`lastObsAliveTimes` is misleadingly named.** `MCMCiFFBS_.cpp:753-761`:
```cpp
which_deadTimes = find(X.row(jj) == 9L);
lastObsAliveTimes[jj] = which_deadTimes.n_elem>0 ? min(which_deadTimes)+1 : endSamplingPeriod[jj];
```
It is **the first DEAD time** (first `t` where `X==9`), or `endSamplingPeriod` if
the individual never dies. It is NOT "last time observed alive". The survival
loop runs `mint_i .. lastObsAliveTimes-1`, i.e. up to and INCLUDING the first
death transition (the `j` at which `X(i,j)==9`), then STOPS — no terms after death.

**Ours:** `epidemic_loglik` loops `first_t : min(last_t, n_timepoints)-1` where
`(first_t, last_t) = sampling_period[i]` = `(startSamplingPeriod, endSamplingPeriod)`.
We loop the WHOLE sampling window regardless of when death occurs.

Whether this differs depends on the state after death. **VERIFIED [MATCH]:**
`transition_prob(D→D) = 1.0` exactly (so `log = 0`) and `transition_prob(D→S) = 0`.
D is absorbing — no outgoing transition in `@transitions`, so it takes the
self-transition leftover = 1. Post-death steps contribute exactly 0 for us, which
is equivalent to the C++ stopping the loop at first death. **Not a real
difference.**

**Subtlety:** the C++ death state is `9`, and it distinguishes "dead" (9) from the
alive states {0=S, 3=E, 1=I}. Our state_space is {S,E,I,D} = {1,2,3,4}. The C++
`z_t==9` death branch adds `TrProbDeath_` at the FIRST 9; we add `1-survival` at
the S/E/I → D transition. Same event, need to confirm same age index is used.

---

## 4. [DIFF] Captures after the monitoring period — we do not handle these at all

**C++** (`logPost_HMC.cpp:100-117`):
```cpp
for(int ir=0; ir<capturesAfterMonit.n_rows; ir++){
    i = capturesAfterMonit(ir,0)-1; lastCaptTime = capturesAfterMonit(ir,1);
    for(int j=lastObsAliveTimes[i]; j<lastCaptTime; j++)
        loglik += TrProbSurvive_(ageMat(i,j), a2, b2, c1, true);  // log
}
```
For badgers CAPTURED AFTER their monitoring period ended, the C++ adds
`log(survival)` for every step from `lastObsAliveTimes` up to `lastCaptTime` — a
correction saying "we know it was alive then, because it was caught". This is a
survival-only term (no infection dynamics) over the post-monitoring tail.

**Ours:** we do NOT load `capturesAfterMonit` at all (`badger_data.jl` reads it
nowhere), and `epidemic_loglik` has no post-monitoring term. So for any badger
in `capturesAfterMonit`, we are missing `sum log(survival)` over its
post-monitoring alive steps.

To match: load `capturesAfterMonit.csv`, and add these survival terms to the
differentiated likelihood (again a user `@addlogprob!` piece, not a transition).

---

## 5. [DIFF] "Born after study start ⇒ forced susceptible at t0"

**C++** iFFBS (`iFFBS_fixedPars.cpp:250-252`):
```cpp
if((tt==0L)&&(birthTime>=startTime)){
    probs = {1.0, 0.0, 0.0, 0.0};   // forced S at its first step
}
```
A badger born AT OR AFTER the study start is forced to be Susceptible at its
first modelled time — its backward-sample draw at t0 is deterministic S.

**Ours:** `badger_starting_state` returns `p = (1-nuE-nuI, nuE, nuI, 0)` and only
sets nuE/nuI when `birth_time < start_time`. For `birth_time >= start_time`, nuE=nuI=0,
so `p = (1, 0, 0, 0)` — forced S. **[MATCH]** — same behaviour, arrived at
differently. Worth a runtime check that the nu_times lookup can't accidentally
give nonzero nu for a post-start birth.

---

## 6. [MATCH] Age indexing

C++ `ageMat(i, tt) = tt+1 - birthTimes[i]` (0-based `tt`). Ours
`age[i, t] = t - birth_time[i]` (1-based `t`). At 1-based `t == tt+1` both give
`t - birth`. Agree. `siler_survival`/`gompertz_makeham_survival` read `data.age[i, t]`
at the same `t` the C++ reads `ageMat(i, t-1)`. Consistent.

---

## 7. [MATCH] Death forbidden before last capture

C++ `probDyingMat` (`MCMCiFFBS_.cpp:632-643`): death prob = 0 (survival forced to
1) when `tt+1 <= lastCaptureTimes[i]`. Ours: `siler_survival`/`gompertz` return
`1.0` when `t <= last_capture_time[i]`. **Agree** — both forbid death up to and
including the last capture, so a badger seen alive at `t` cannot have died before `t`.

---

## 8. [MATCH, verified] Starting-state term in the likelihood

**Ours** (`build.jl:82-85`): adds `log(p0[X[1,i]])` — the starting-state
log-prob — at the GLOBAL `t=1` for every individual, using
`data.starting_state(model, data, X, i, 1)`.

**C++:** the starting distribution enters through the iFFBS FILTER at each
individual's own `t0 = startTime`, not as a separate global-t=1 likelihood term,
and not in `logPost_HMC` (nu is conjugate, absent from the differentiated density).

I worried our global-t=1 term scores a garbage cell for late-entry badgers
(2336 of 2384 have `startTime > 1`). **VERIFIED it does NOT:** for every
late-entry badger `X_init[1, i] = 1` (state S), and `starting_state` returns
`[1, 0, 0, 0]`, so `log(p0[X[1,i]]) = log(1) = 0`. The term contributes exactly
ZERO for late-entry individuals — the pre-window fill value is S and the forced-S
starting distribution scores it at probability 1. **Not a bug, not a difference.**

(This holds for the CURRENT `X_init`. It relies on the pre-window fill being S;
if a future `X_init` put a non-S fill before entry, this term would start
contributing spuriously. Worth a guard, but not a live problem.)

---

## Summary — to build an EXACTLY C++-matching variant

Already done (badger_fit_gompertz_hmc.jl): Gompertz-Makeham survival, no a1/b1,
tau~Exp(100), nu~Dir(1,1,1).

After verification, only TWO genuine differences remain, both additive survival
terms in the differentiated likelihood — pure user code via `@addlogprob!`, NO
package change:

1. **§1 entry survival**: add `logS(age_at_entry)` per pre-study-born individual
   (the C++'s failure to condition on entry — statistically wrong, but this is
   the "match exactly" variant).
2. **§4 captures-after-monitoring**: load `capturesAfterMonit.csv`, add
   `sum log(survival)` over the post-monitoring tail for those individuals.

Verified NOT differences (checked, don't touch): §2 survival factorisation, §3
post-death (D absorbing, contributes 0), §5 forced-S at entry, §6 age indexing,
§7 death-before-last-capture, §8 starting term (0 for late entry).

**Explicitly NOT matched (and shouldn't be, unless asked):** the C++'s failure to
condition on entry (§1) is a statistical error — it double-counts survival from
birth. A "match the C++ exactly" variant reproduces it; the *correct* model omits
it (as ours currently does). Keep both, clearly labelled.
