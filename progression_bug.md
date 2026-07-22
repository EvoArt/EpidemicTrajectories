# The progression bug: tau is inverted (rate vs mean-time)

**Symptom** (user report): our fit gives tau~0.3, beta~24, q~0.7; both references
(C++ classic rcpp AND the semi-Markov Julia ref) give tau~15, beta~0.019, q~0.6.
Low ESS and occasional poor rhat on the HMC params.

**Root cause: our E->I progression uses the RATE convention where the reference
uses the MEAN-TIME convention. Our `tau` is the reference's `1/tau`.**

## The two `erlang_cdf_at_1` are different

Ours (`examples/badger_model.jl:68`), copied from `run_base_exp.jl`'s top-level:
```julia
erlang_cdf_at_1(k, tau) = 1.0 - exp(-tau) * sum((tau^j)/factorial(j) for j in 0:(k-1))
```
At k=1: `1 - exp(-tau)`. `tau` is a RATE.

The reference ENGINE (`badger_ref/semi-markov_src.jl/transitions.jl:250`), which is
what `run_base_exp.jl` actually RUNS:
```julia
function erlang_cdf_at_1(k, theta)
    x_over_theta = 1.0 / theta          # <-- 1/theta
    ...
    return 1.0 - sum_val * exp(-1/theta)
end
prate(k,tau) = erlang_cdf_at_1(k, tau/k)
```
At k=1: `1 - exp(-1/tau)`. `tau` is a MEAN TIME.

The C++ agrees with the engine: `logProbEtoI = log(1 - exp(-1/tau))`
(`logPost_HMC.cpp:66`), `p12 = (1-prDeath)*(1-exp(-1/tau))`
(`iFFBS_fixedPars.cpp:115`).

**We copied the WRONG `erlang_cdf_at_1`.** `run_base_exp.jl` has an inline
top-level definition using `exp(-tau)`, but that function is a decoy — the code
path that actually runs uses `semi-markov_src.jl/transitions.jl`'s version with
`exp(-1/theta)`. We matched the decoy.

## Numerical confirmation

```
k=1, P(E->I):
  tau=0.0667  ours(rate)=0.0645   ref(mean-time)=1.0000
  tau=15.0    ours(rate)=1.0000   ref(mean-time)=0.0645
```
`ours(tau=0.0667) == ref(tau=15) == 0.0645`. Exact reciprocal.

## Why this drags beta and q too

Progression sets how fast E->I happens, which sets how many I animals the latent X
contains. Wrong progression -> wrong X -> the FOI counts (n_infectious) feeding
beta/q are wrong -> beta and q move to compensate. The FOI FORMULA is identical
across all three implementations (verified term-by-term: `a + b*I/(M/K)^q`,
`alpha_g = lambda*alpha[g]`, K=85, I counts I-only, M counts alive); so beta being
off ~1000x is NOT a FOI bug — it is beta absorbing the damage from a corrupted X.

## Also wrong: the prior

`tau ~ Exponential(10)` means "mean RATE 10" under our convention but should mean
"mean TIME 10" under the reference's. So even the prior pulled tau the wrong way.
The Gompertz variant already set tau ~ Exponential(100) to match the C++ Gamma(1,
scale=100) — that is the MEAN-TIME prior, correct ONLY IF the formula is also
mean-time. With our current rate-convention formula, that prior is doubly wrong.

## The fix

Change `erlang_cdf_at_1` to the mean-time convention (`exp(-1/theta)`), matching
the engine and the C++. One-line change to the formula; the prior then also needs
to be the mean-time one (Exp(10) or Exp(100) as intended, now meaning mean time).

Two things to decide:
1. Fix in the SILER model (badger_model.jl) — affects the existing default and
   the reststotal fit. This is a genuine bug fix, not a variant.
2. The Gompertz variant inherits `badger_progression` from badger_model.jl, so
   fixing it there fixes both.

NOT verified yet: whether flipping the convention brings beta/q back into line
too. It SHOULD (via a corrected X), but that needs a run to confirm. Do NOT assume
the fit matches until a converged run shows it.

## What was NOT the cause (ruled out)

- k/K swap: verified data.K=85.0, data.k=1 at runtime. Correct.
- FOI formula / K scaling / I,M counts: identical across all three.
- Randomised L: reverted; current code is fixed L=15. And L only affects mixing.
- thetas/rhos sharing an HMC block: affects mixing (ESS/rhat), never the posterior.
- Low ESS/rhat: SYMPTOMS of the mis-specified target, not causes.

---

# Second bug: survival evaluated at SOURCE time, not DESTINATION (off-by-one)

After the progression fix, a run still gave tau~43, beta~2.4 (vs reference ~15,
~0.019) and rhos collapsed to ~0.1. User suspected an off-by-one in the
transitions. Confirmed:

For the transition `t -> t+1`, BOTH references evaluate SURVIVAL at t+1:
- C++ (logPost_HMC.cpp): move (j-1)->j, `TrProbSurvive_(ageMat(i,j))` — age at
  DEST j. `lastCaptureTimes` compared vs j (the dest).
- semi-Markov Julia (transitions.jl:6-18): `t_next = t+1`,
  `groupLevelSurvivalProb(..., t_next, ...)`, `t_next <= lastCaptureTimes`.

But FOI counts and group are read at the SOURCE (j-1 / t): both references and
ours agree there.

OURS evaluated EVERYTHING at the source t: `siler_survival`/`gompertz_makeham_survival`
read `data.age[i, t]` and guard on `t <= last_capture_time`. So our survival age
was one step behind the reference, systematically, for every individual and
timepoint. Since `@survival` multiplies survival into every transition, c1/a2/b2
were pulled to compensate, distorting the per-step likelihood and dragging
beta/tau/q.

FIX (badger_progression_meantime.jl): `gompertz_makeham_survival_tp1` reads age at
t+1 and guards on t+1, matching the reference. Wired into
badger_fit_gompertz_fixed_hmc.jl via badger_transitions_meantime_gompertz.

VERIFIED MATCH (not off-by-one): FOI counts (source t), progression (source t),
group (source t), test emission (S/E/I formulas identical to ObsProcess_.cpp).

CAVEAT: 3-sweep smoke tests show scale, NOT convergence. Whether the fit now
MATCHES the reference needs a converged run. The residual gap after the
progression fix could be this off-by-one OR non-convergence OR a further
discrepancy — a short run cannot distinguish them.

---

# CORRECTION + the REAL cause: iFFBS collapses the epidemic

**The survival off-by-one was overclaimed.** A one-step age shift is ~0.01% per
factor — it cannot drive tau to 42. The user was right to be skeptical (mean
progression 42 seasons means most exposed badgers never become infectious, which
no survival-indexing detail explains). The off-by-one is a real bug and worth
fixing, but it is NOT the cause of the parameter divergence.

**The real cause, measured (count E/I states before and after iFFBS):**

    Xinit         E=4016  I=2728  | S->E=801  E->I=299  E->E=3212
    after 5 iFFBS E=2015  I= 776  | S->E=368  E->I=138  E->E=1645

Xinit's own E->I fraction is 299/(299+3212) ≈ 0.085 -> tau ≈ 11, right in the
reference's ballpark. But our iFFBS DESTROYS infections: after 5 sweeps E halves
and I drops to ~28% of its initial count. With few E and I states left, the
likelihood forces tau up (few E->I events) and beta up (few I -> need high
transmission). tau=42 and beta=2.4 are DOWNSTREAM of a latent trajectory that has
lost most of its infections.

So the bug is in the LATENT SAMPLER (forward filter / observation weights /
coupling), not in the survival age or the progression convention (both of which
are now correct). The iFFBS is systematically down-weighting E and I states and
resampling them to S.

STATUS: root cause localised to the iFFBS producing an infection-poor X.
Progression fix (mean-time tau) and survival off-by-one are correct and should
stay, but they are NOT sufficient — the trajectory collapse must be fixed for the
epidemic parameters to match. Investigation ongoing.

## The collapse is STRUCTURAL (parameter-independent), and hits I specifically

Ran 5 iFFBS sweeps at three parameter settings:

    tau=15 beta=0.019: E=2015 I=776   (Xinit E=4016 I=2728)
    tau= 3 beta=0.500: E=2025 I=348
    tau= 3 beta=2.000: E=2727 I=257

The I collapse happens at EVERY setting — cranking beta 100x makes it WORSE, not
better. So it is not a parameter basin; it is structural. And it is SPECIFIC to
I: at tau=3/beta=2, E recovers toward Xinit (2727 vs 4016) but I craters to 257.

=> The forward filter / observation weights are systematically preventing badgers
from STAYING in I. Next: inspect the I-row of the transition matrix and the
observation weight vector at an I cell.

## I-collapse: NOT a test-alignment or state-code bug

Counts of Xinit I/E cells by test result:
    I-cells: POS=374  NEG=677  no-test=1677   (64% of tested I are NEGATIVE)
    E-cells: POS=1487 NEG=916  no-test=1613   (62% of tested E are POSITIVE)

Positive tests cluster on E, not I. Checked: this is CORRECT, not a bug —
runmodel.R builds Xinit by setting the positive-test time to state 3L (=E in the
C++'s codes {S=0,E=3,I=1,D=9}), then "E becomes I some quarters later". Our loader
maps codes correctly (REF_STATE_CODE = 0->1,3->2,1->3,9->4), so E lands on E.
Test times: C++ applies a test at TestTimes=T to corrector time T; ours reads
tests[t] at t. Same alignment. Observation weights are correct arithmetic (a
negative-testing badger genuinely looks S).

So the reference reconstructs a healthy epidemic FROM this same Xinit, but our
iFFBS collapses I regardless of parameters. Transition matrix is correct
(P[I,:]=[0,0,1,0], I absorbing). Remaining suspect: the forward/backward
recursion or the coupling term — how E->I probability propagates through the
filter. Investigation continues there.

## Audit vs Julia reference (semi-markov_src.jl) — findings

CONFIRMED MATCHING (read line-by-line):
- Forward-filter indexing: ref `compute_individual_transition_probs(tt-1+t0)` ->
  predProb[tt+t0] (move src->dest); ours transition_matrix_at!(t-1)->probs[j].
  Same.
- First step: ref filtProb[t0] = corrector·predProb·transProbRest normalized;
  ours base·obs·rest normalized. Same.
- Backward: ref P(state|next) = p[state->next]·filtProb[t]/predProb[t+1,next];
  ours cond[a]=probs[j,a]·trans[a,bnext] normalized. Same (predProb is the norm const).
- Space: both work in probability space in the filter (corrector·predProb·rest,
  sums of products), log-space only for the coupling normalization. Ours matches
  (logw then exp). No prob/logprob confusion found.
- Transition matrix correct: E-row P[E,:]=[0, 0.9355, 0.0645, 0] (E->I = 1-exp(-1/15)
  exactly), I-row [0,0,1,0] (absorbing). No leak.
- Freshness: aggregates seeded at run start (reset+apply), maintained by
  reverse/reapply; X flows unbroken through the iFFBS kernel copy. Consistent.

STILL DIVERGENT: I collapses despite all per-cell quantities being correct. This
is emergent over the trajectory: E->I is only 0.0645/step, and 64% of I-cells have
NEGATIVE tests (pull toward S), so the forward filter puts little mass on I. May be
correct given the model — but the reference does NOT collapse, so a difference
remains in how I mass is propagated/retained. Prime remaining suspects: the
coupling total (logProbRestTotal running-total vs our nSE/nSS decomposition) under
the FULL global X, and the numerical normalization (ref uses logsumexp; ours
subtract-max-then-exp — check for a regime where ours underflows where ref doesn't).

## SEI trajectories over 100 sweeps — it is an EQUILIBRIUM problem, not an iFFBS sink

Tracked S/E/I/D per sweep (examples/track_sei.jl). BOTH models PLATEAU by ~sweep
10 (not sink to zero) — so the iFFBS is NOT buggy; it faithfully samples a
stationary trajectory. The problem is WHERE the stationary point sits:

               E        I     E:I ratio
  Xinit(data)  4016   2728    1.47
  Siler(100)    455   2853    0.16   <- E collapses, I fine (E->I too FAST)
  Gompertz(100)3690    958    3.85   <- I decays, E fine (E->I too SLOW)

The two models sit on OPPOSITE sides of the data's E:I ratio, by ~10x and ~2.6x.
This is governed by the E->I progression rate vs the death rates — a model
EQUILIBRIUM, not a filter bug. Consistent with the tau-convention:
  Siler  1-exp(-tau), tau small -> high E->I -> E drains to I -> E collapses.
  Gompertz 1-exp(-1/tau), tau=15 -> E->I=0.065 -> E stuck -> I decays.

FEEDBACK LOOP: HMC co-fits tau from the sampled X. Gompertz stationary I=958 ->
likelihood sees few I -> infers slow progression (tau->42) -> even fewer I. The
sampler + likelihood reinforce a wrong basin. Neither model anchors E:I to the
Xinit/data ratio the way the reference presumably does.

Totals conserved (186850 both, all sweeps) — no counting/window bug. D shifts are
just states redistributing.

OPEN: why does the reference hold E:I near the data? Candidates: (a) the entry
survival + captures-after-monit terms (cpp_model_discrepancies.md §1,§4) that we
lack change the effective E/I evidence; (b) a difference in how the reference's
starting-state / nu mixing seeds E vs I; (c) the progression prior/init anchoring.
Next: compare the reference's actual equilibrium E:I (run it, or read its saved
output) — we may be chasing a match to numbers the reference ALSO doesn't hold.

## CAM + capture-timing audit — the real difference is survival CREDITING

User's key distinction: iFFBS forbids death before LAST capture; the likelihood
forbids death before FIRST capture. Verified:
- first_capture <= last_capture for ALL 2384 (0 violations). So the filter's
  last-capture death-ban is STRICTER than the likelihood's first-capture ban ->
  the likelihood gate never fires on a filter-produced X -> no separate
  first-capture code needed in the likelihood.
- CAM extension of last_capture_time: extended 0 of 2384. Because our
  CaptHist.csv ALREADY contains the post-monitoring captures (id=4: last_capt=31
  while endSampling=30). The reference needs the separate CAM file only because it
  builds last_capture from a TRIMMED CaptHist_infer; ours isn't trimmed, so we get
  it for free. Our filter already forbids death up to the post-monit capture.

THE REAL DIFFERENCE (survival crediting):
Our survival forces s=1 up to LAST capture, in BOTH the filter and (because they
share the survival fn) the likelihood. The reference forces s=1 in the FILTER
(probDyingMat, last-capture) but in the LIKELIHOOD gates only on FIRST capture --
so for a badger alive throughout [first,last], the reference CHARGES real survival
probability from first_capture onward, while we credit free s=1 up to last_capture.
=> our likelihood under-penalises survival over [first_capture .. last_capture],
which biases the survival params (c1/a2/b2) and, through them, the E:I death
balance.

This is the survival-crediting asymmetry, NOT CAM (which is a no-op for us). The
CAM likelihood term I added is therefore also ~a no-op (survival=1 over the tail
that our last_capture already covers -> log(1)=0). The fix that WOULD matter:
make the LIKELIHOOD survival gate on FIRST capture (charge real survival in
[first,last]) while the FILTER survival gates on LAST capture. These need to be
DIFFERENT survival functions for the two roles.

## Confirmation: c1 inflation = the survival-crediting bug (user spotted it)

The pasted Gompertz summary had c1=6.99. That is the Makeham CONSTANT hazard:
per-step survival exp(-6.99) ≈ 0.0009 — i.e. the fit says 99.9% of badgers die
EVERY quarter at EVERY age. Physically absurd (badgers live years).

This is the SMOKING GUN for the survival-crediting bug (fixed this session): with
survival forced to 1 across the observed window, the likelihood saw ONLY death
events (alive->D) and no survival events, so c1 inflated toward "everyone dies" —
there was no survival to explain, only deaths. Charging real survival over
[first_capture .. last_capture] (this session's fix) gives the likelihood ~186k
survival events to pin c1 to a realistic ~0.01-0.05 (per-step survival ~0.95-0.99).

Imputation window (user asked): we impute ONLY [start_sampling, end_sampling] per
individual (iffbs.jl:224), and the loglik loops [max(first,entry) .. end-1] — both
confined to the monitoring period, matching the reference. The post-monitoring
tail is neither imputed nor in the transition loglik. This was ALREADY correct
before this session; it was never the bug.

## NOTE (do not investigate): gompertz_cpp is pathological
The gompertz_cpp variant (full C++ parity, non-conditioning on entry + birth->entry
survival term) gave nonsense parameter estimates on a full run. Noted for the
record; the user asked not to investigate it. The entry-non-conditioning +
birth->entry term is the prime suspect but is NOT being pursued.

## siler_fixed vs run_base_exp after 25000 iters — should be identical, isn't

siler_fixed should reproduce run_base_exp.jl (same Siler model). It doesn't:

  param   siler_fixed  base_exp   diff
  tau     13.74        15.00      -8%
  lambda  0.00548      0.00468    +17%
  beta    0.01104      0.01871    -41%   <- big
  q       0.779        0.602      +29%   <- big
  c1      0.07627      0.07467    +2%    (survival baseline: MATCHES)
  a1      0.00015      0.44223    -99.97% <- HUGE (early-life mortality)
  b1      0.01669      2.36494    -99.3%  <- HUGE
  a2      0.00019      0.00011    +73%
  b2      0.13116      0.16272    -19%
  etas    [0.18,0.36,0.47,0.28]  [0.085,0.20,0.13,0.12]  2-4x higher

The a1/b1 divergence is the tell: base_exp has SUBSTANTIAL early-life mortality
(a1~0.44, b1~2.36), siler_fixed has ~none (a1~0, b1~0.02). c1 (the constant
Makeham hazard) MATCHES (0.076 vs 0.075). So the disagreement is entirely in the
AGE-DEPENDENT survival terms (a1/b1 early-life, a2/b2 late-life) — consistent with
a survival-likelihood/imputation difference, exactly the user's hunch. etas 2-4x
higher also points at survival: if our badgers "die" (leave the alive states)
differently, the alive-and-available denominator for the capture-probability
conjugate update changes.

## siler_fixed a1/b1 divergence localised to AGE-1 survival + a pre-entry loglik bug

Comparing the implied survival curves:
  age  siler_fixed  base_exp
   1     0.926       0.783    <- differ ~14% ONLY at age 1
   2     0.926       0.913
   3+    ~identical
c1 (constant hazard) matches (0.076 vs 0.075). So the ENTIRE a1/b1 disagreement
is first-year (age-1) mortality: base_exp sees young badgers dying (a1=0.44),
siler_fixed does not (a1~0).

CANDIDATE BUG (entry conditioning too aggressive): the reference (posterior.jl:145)
loops from START_SAMPLING and gates only the SURVIVAL factor on entry
(`log_pti = j >= entry_i ? log(isp) : 0`), STILL scoring the infection/progression
transition branches (S->E, E->I) in [start_sampling, first_capture) with survival=1.
Our `entry_time` fix skips the WHOLE window (loop from max(first_t, entry)), so we
DROP those pre-entry S->E / E->I terms. That plausibly drives beta (-41%), q, tau.

NOT YET EXPLAINED: whether this pre-entry gap accounts for the AGE-1/a1-b1
divergence specifically. a1/b1 are informed by DEATHS, which occur post-last-capture
(older ages), not by pre-entry infection terms. There may be a SECOND survival
difference in how/where death events are scored. Do NOT claim solved until the
age-1 death evidence confirms it. Next: confirm pre-entry cell count + ages, then
fix entry conditioning to gate only the survival factor (not the whole transition),
and re-run.

## RESOLVED: a1/b1 divergence is WEAK IDENTIFIABILITY, not a bug

Death-age distribution in Xinit: deaths start at AGE 4-5, peak at 5-6, NONE at
age 1-2. And 0 badgers have their last capture at age 1-2 -> death can NEVER be
imputed young (death-ban forbids death before last capture, in both models). So
neither model has any age-1/age-2 death DATA.

The Siler early-life term (a1/b1) governs age-1/2 mortality. With NO data there,
it is unidentified and drifts. Proof: over the OBSERVED death-age range (4-15),
siler_fixed's survival and base_exp's survival differ by AT MOST 0.19% — despite
a1/b1 differing by 99.97%. The observable survival is identical; only the
data-free young-age region is parameterised differently.

=> a1/b1 divergence is NOT a survival-likelihood bug. It is two MCMC runs landing
in different spots of a flat/ridged posterior region. (base_exp a1=0.44 is no more
"correct" than our a1~0 — neither is informed by data.) The user's survival-bug
hunch is CORRECT about there being a survival-side issue, but it is the PRE-ENTRY
likelihood gap (below), not a1/b1.

## The REAL bug: pre-entry infection/progression terms dropped

The reference (posterior.jl:145) loops from START_SAMPLING and, in
[start_sampling, first_capture), scores infection (S->E: log1mexp(-foi)) and
progression (E->I: log(progRate)) transitions with the SURVIVAL factor gated to 0
(log_pti=0) and death forbidden (log_qti=-Inf). Our siler_fixed `entry_time` fix
skips the WHOLE pre-entry window, so we DROP those infection/progression terms.

Measured: 2367 of 2384 badgers have a pre-entry gap, 9452 transition-steps total
(4672 at age 1-2). Dropping the S->E/E->I likelihood there plausibly drives the
REAL divergences: beta -41%, q +29%, tau -8%.

FIX: entry conditioning must gate only the SURVIVAL factor (as the reference
does), NOT skip the whole transition. i.e. loop from start_sampling, but zero the
survival contribution before first_capture while KEEPING infection/progression.
This is a package-level change to epidemic_loglik's entry_time semantics.

## Division of remaining work (2026-07-22)
- a1/b1 divergence: user investigating — likely an HMC EPSILON issue. a1/b1 were
  NOT in the original manuscript, so their per-parameter step sizes were never
  optimised (HMC_EPS uses 0.001 for both, copied as a guess). Weak identifiability
  (above) + un-tuned epsilon = the two runs explore the flat region differently.
  NOT a likelihood bug. FUTURE PracticalBayes: learn a mass matrix (e.g. via a
  NUTS warmup) and reuse it for the fixed-eps HMC — would fix the un-tuned-epsilon
  class of problem generally.
- entry-conditioning survival-only gate: THIS agent — package feature so the
  likelihood can condition on entry by zeroing only the SURVIVAL factor before
  first-capture, while KEEPING infection/progression (matching posterior.jl).

## Entry-conditioning FIX: survival-only gate (package feature)

Implemented `epidemic_loglik(data; entry_time=..., survival=...)`. The loop now
covers the WHOLE window; before entry it subtracts log(survival) from
log(transition_prob), removing the survival factor while keeping the disease move.

VERIFIED the subtraction is exact for every alive->alive move (our @survival makes
each `survival * conditional-move`, so dividing survival out leaves the move):
  S->S: trans_prob = survival*exp(-foi)      -> -log(surv) -> -foi        = ref
  S->E: trans_prob = survival*(1-exp(-foi))  -> -log(surv) -> log1mexp(-foi) = ref
  E->E: trans_prob = survival*(1-progRate)   -> -log(surv) -> log(1-progRate) = ref
  E->I: trans_prob = survival*progRate       -> -log(surv) -> log(progRate)   = ref
  I->I: trans_prob = survival                -> -log(surv) -> 0               = ref
alive->D pre-entry cannot occur (death-ban obs forbids it), so the one inexact
case (trans_prob=1-survival there) never fires. Documented.

Ergonomics: user passes entry_time (Vector{Int}) + the survival fn; error if
entry_time given without survival. entry_time=nothing => unchanged. Wired into
siler_fixed and gompertz_fixed (gompertz_cpp intentionally does NOT condition).
