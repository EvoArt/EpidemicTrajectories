# iFFBS-MH in EpidemicTrajectories — design proposal

**Status:** proposal, nothing implemented. **Date:** 2026-08-22.

Grounded in:

* **The paper.** Touloupou, Finkenstädt & Spencer (2020), *Scalable Bayesian
  Inference for Coupled Hidden Markov and Semi-Markov Models*, JCGS 29(2):238–249.
  §3.3.1 (iFFBS, Algorithm 1), §3.3.2 (MHiFFBS, Algorithm 2), §4.3 (SM-iFFBS).
* **The Julia reference**, `BadgeR/inst/julia/src/iFFBS_mh.jl` (384 lines) and
  `iFFBS_mh_modular.jl` (481 lines) — full read.
* **The C++ reference**, `BIID/src/iFFBS_MH.cpp` (507 lines) — full read of the
  ratio and the accept branch.
* **This package**, `src/iffbs.jl`, `src/transitions.jl`, `src/data.jl`,
  `src/build.jl`, `src/spec.jl`, `src/aggregates.jl`.

---

## 0. The one-paragraph version

iFFBS is an exact Gibbs step **only when the chain the forward filter runs is the
same chain the likelihood scores**. Three common situations break that: a
semi-Markov target (the E→I hazard depends on time-since-exposure, which no
first-order filter can represent), a deliberately cheapened proposal (drop the
coupling term — the paper's MHiFFBS), and a target that carries factors the filter
does not (this repo's `entry_time` gate is one, live in the badger examples today).
The fix in all three cases is the same: keep iFFBS, but treat its output as a
**proposal** in an independence Metropolis–Hastings step, and accept with

```
log α  =  [logπ(x_can | x_-i) − logπ(x_cur | x_-i)]  −  [log q(x_can) − log q(x_cur)]
```

This document proposes `iffbs_mh!` as an explicit sibling of `iffbs!`, an
`IFFBSProposal` object carrying the (optionally different) proposal chain, a
per-individual conditional target built from the same pieces `epidemic_loglik` and
`epidemic_obs_loglik` already use, and a test plan whose keystone is: **when the
proposal equals the target, every acceptance ratio must be exactly 1.** That single
property tests the proposal density, the conditional target, the aggregate
bookkeeping and the focal-self-contribution machinery simultaneously — and it turns
"is my spec iFFBS-compatible?" from an undecidable static question into a one-line
runtime check.

---

## 1. What the correction is for

### 1.1 The paper's two variants — one kernel

The paper names three algorithms that are the same kernel with different proposals:

| Paper's name | Proposal differs from target by | Why |
|---|---|---|
| **MHiFFBS** (§3.3.2, Alg. 2) | drops the between-chain term — assumes `P(X[c']_{t+1} \| X[1:C]_t) ≈ P(X[c']_{t+1} \| X[-c]_t)` | the coupling term is the expensive part of the filter |
| **SM-iFFBS** (§4.3, Alg. D.2) | uses a **Markov** kernel (κ=1, geometric sojourn) for a **semi-Markov** target | FFBS structurally requires a first-order chain |
| **SM-MHiFFBS** | both at once | "since it already includes a MH step, no further corrections are needed" (§4.3) |

The paper is explicit that the uncorrected version is *wrong*, not merely
approximate: dropping the between-chain term without the MH step is the
"uncorrected-iFFBS", used by Sherlock et al. (2013) and Fintzi et al. (2017), and
"failing to include the MH step may lead to poor behavior of the resulting MCMC
chains" (Supplementary §A).

Measured behaviour worth carrying into our expectations:

* acceptance rates **> 0.84** for every population size tried, up to C = 1000 per
  pen, declining slowly with C (§4.2, Supplementary Fig. C.2);
* in the **Markov** case, plain iFFBS beats MHiFFBS on relative speed — the
  correction is not free and the exact step is already cheap;
* in the **semi-Markov** case, MHiFFBS had the highest relative speed in 15 of 18
  simulated datasets, with SM-iFFBS a close second. The choice between them "depends
  on how important the missing arrows were" (§4.3).

### 1.2 What the Julia reference actually does

`iFFBS_mh.jl` is the SM-iFFBS shape, not MHiFFBS: the filter **keeps** the coupling
(`compute_trans_prob_rest!` / `normTransProbRest!` are called at every timepoint),
and the mismatch is entirely in the E→I progression. The filter uses
`compute_individual_transition_probs`, whose `progRate` is a plain per-step
hazard; the target uses `progression_fn`, which scans back through `X` to recover
`s`, the time since exposure, and evaluates a discrete Weibull / gamma /
half-Cauchy hazard (`progression_likelihood.jl`). Same states, different chain.

Its ratio is

```julia
ratio = (logq_cur - logq_can) + delta_loglik + (sumCorr_can - sumCorr_cur)
```

with `delta_loglik` covering the focal's own steps **and** the neighbours' S→S /
S→E steps whose FOI the focal enters, and `sumCorr` the observation ("corrector")
term. Note the split: `q` folds the observation weights in (they are in `filtProb`),
and the observation term appears *again* in the target as `sumCorr`. That is not
double counting — `q` is the proposal, `π` is the target, and each needs its own
copy of every factor it uses.

The **C++** `iFFBS_MH.cpp` is a different animal: it computes the same ratio but
then asserts `|ratio − 1| < 1e-5` and prints a diagnostic when it is not. That code
is a *verification harness* for an exact-Gibbs model, not a correction. We should
build both uses out of one implementation — see §7.

### 1.3 What is already latent in **this** repo

Three places where our filter and our likelihood already score different chains.
None is currently flagged; all three are exactly what MH corrects.

**(a) The `entry_time` gate.** `epidemic_loglik(data; entry_time, survival)`
subtracts `log(survival)` from every pre-entry step (`build.jl`), so the target has
no pre-entry survival factor. The filter has one — `transition_matrix_at!` reads
`data.trans_mat`, which `@survival` scaled by survival at every step. Ten of the
badger example scripts pass `entry_time=raw.first_capture_time`.

Whether this actually biases anything is decidable, and the answer is instructive.
`@survival p death=:D` (spec.jl:178-201) applies **one** survival expression
`(model, data, i, t)` to every live transition — it is *not* state-dependent. So
across all live source states the pre-entry factor is the same constant, it cancels
on normalisation, and the gate changes nothing… **for paths that stay alive.** For
a path that dies pre-entry the row entry is `1 - p_surv`, which the gate's
`-log p_surv` does not correspond to at all. So:

> plain iFFBS stays exact under the entry gate **iff** the observation process
> gives zero weight to death before entry.

The badger observation process is believed to do exactly that, which is why nothing
has gone visibly wrong. But "believed to" is the wrong basis, and the ratio-1
diagnostic of §7 answers it in one run.

**(b) `coupling_trans_mat`.** `epidemic_data` accepts a separate spec for the
coupling term, sold in CLAUDE.md as safe because "the cache equals the true
value". When it does, the ratio is 1 and nothing is needed. When it drifts — a
stale cache, a deliberately approximate FOI — the sampler is silently targeting the
wrong conditional, with no signal at all. MH converts that silent failure into a
visible acceptance rate.

(The *survival-free* coupling view `trans_mat.coupling` is **not** in this category:
stripping survival from the neighbour scoring removes a factor that does not depend
on the focal, so it changes every candidate's weight by the same constant and
cancels. Exact. Same for the death rows the coupling view omits — a neighbour's
alive→dead move gets the constant `log(1e-12)` under every candidate.)

**(c) The observation split.** `epidemic_obs_loglik(data; observation_process=...)`
lets a user pass only the non-conjugate *factor* of a multiplicative observation
model. **The MH target must NOT use that reduced factor.** The full conditional of
`X` contains every term that depends on `X`, and the capture factor depends on `X`
(you cannot capture a dead badger) even when its parameters are drawn conjugately.
The conditional target must use `data.observation_process` / `data.observation_weight`
— what the *filter* sees — not what was handed to `epidemic_obs_loglik`. This is
the mirror image of CLAUDE.md's existing warning and the ratio-1 test catches it.

---

## 2. The maths, in this package's vocabulary

Notation matching `src/`: individuals `i = 1..C`, timepoints `t`, states `1..N`,
window `(a_i, b_i) = data.sampling_period[i]`, latent path `x_i = X[a_i:b_i, i]`,
`A(t,i) = data.affected_individuals[t, i]`.

### 2.1 The target conditional

The joint the package scores is `epidemic_loglik + epidemic_obs_loglik`. Collecting
every factor of it that depends on `x_i`:

```
log π(x_i | x_-i)  =  const
  + log p0_i(x_{i,a_i})                                   # data.starting_state
  + Σ_{t=a_i}^{b_i-1} log P_i(x_{i,t} → x_{i,t+1})         # transition_prob(data.trans_mat, …)
  − Σ_{t < entry_i}  log surv_i(t)                         # the entry gate, when used
  + Σ_{t=a_i}^{b_i}   log g_i(x_{i,t})                     # FULL observation weight
  + Σ_{t=a_i}^{b_i}   Σ_{j ∈ A(t,i)} log P_j(x_{j,t} → x_{j,t+1} ; x_{i,t})
                                                           # neighbours the focal influences
```

The last line is the only term with no counterpart in `epidemic_loglik`'s
per-individual loop — it belongs to the neighbours' own likelihood contributions
and enters *i*'s conditional because their rates read aggregates *i* feeds.

Two exact restrictions on that last sum, both already relied on elsewhere in the
package:

* **Coupled-mask restriction.** A neighbour move outside
  `coupled_transition_mask` has the same probability under every value of `x_{i,t}`
  (`transitions.jl` documents and tests this). Constant ⇒ cancels in a difference.
* **Changed-timepoint restriction.** The aggregates contract is that a summary's
  update for `(s, i, t)` is *i*'s entire contribution at time `t`
  (`apply_derived_summaries!` builds them exactly that way, and
  `make_rest_contribution`'s counterfactual is already built on it). So the
  neighbour block at time `t` depends on `x_i` **only through `x_{i,t}`**, and
  cancels wherever `x_can[t] == x_cur[t]`.

### 2.2 The proposal

The proposal is the exact conditional of a possibly *different* chain
`(p̃0, P̃, g̃, r̃)` — the `IFFBSProposal` of §4.1. Its density is what the backward
pass already computes, one factor per timepoint:

```
log q(x_i) = log probs[n_t, x_{n_t}] + Σ_{j=1}^{n_t-1} log w_j(x_j),
      w_j(a) ∝ probs[j, a] · trans_cache[a, x_{j+1}, j+1]
```

which is exactly `backward_sample!`'s `w` after normalisation (`iffbs.jl:198-220`).
Two consequences that make the implementation small:

* **Sampling and scoring are the same walk.** Drawing a path accumulates `log q_can`
  for free; scoring the *current* path is the identical loop with the draw replaced
  by a lookup, and consumes no RNG.
* **Per-timepoint constants are irrelevant.** The filter renormalises `probs[j, :]`
  at every step, so multiplying the weight vector at time `t` by any constant
  changes nothing. In particular `make_rest_contribution`'s `logw .-= maximum(logw)`
  is invisible to `q`, and to the ratio.

### 2.3 The ratio

Independence proposal (the filter does not see `x_i` — it runs at the
leave-one-out aggregate state), so:

```
log α = [log π(x_can) − log π(x_cur)] − [log q(x_can) − log q(x_cur)]
```

### 2.4 The exactness lemma (why proposal == target ⇒ α ≡ 1)

Worth writing down because it is the specification of the keystone test.

When the proposal chain is the target chain, `q(·) = π(· | x_-i)` and the ratio is
identically 1. Term by term, our two evaluations must agree:

| Target term (evaluated at **full** aggregates, `_focal[] = -1`) | Filter term (evaluated at **leave-one-out** aggregates, `_focal[] = i`) |
|---|---|
| `p0_i(x_{i,a})` | `initialise_forward_filter` — same function |
| `g_i(x_{i,t})` | `data.observation_process(…)[s]` — same function, no aggregate dependence |
| `Σ_j log P_j(… ; x_{i,t})` | `log rest_t(x_{i,t})` up to the `max` constant — `rest_contribution` sets `X[t,i]=s` and applies the summaries forward from leave-one-out, reproducing precisely "aggregates including *i* in state *s*" |
| `log P_i(a → b)` | `transition_matrix_at!` under `_call_rate_with_focal`, which temporarily re-adds *i* in state `a` with `X[t+1,i] = b` and reverses — again reproducing "aggregates including *i*" |

The last row is the interesting one: it is the **focal-self-contribution machinery**
(`transitions.jl:88-110`), and the lemma holds *only* when
`focal_self_contribution = true`. With it `false`, the user's rates are written to
be evaluated against leave-one-out aggregates inside the filter but are called by
`epidemic_loglik` against full aggregates, and the package cannot know which of the
two the target intends.

> **Decision:** `iffbs_mh!` errors on `focal_self_contribution = false` in v1, with
> a message pointing here. Revisit only with a concrete model that needs it.

A pleasing corollary: the ratio-1 test is a **test of the focal re-insertion**,
which currently has no direct test at all.

---

## 3. The step, and what it costs

Ordering matters, because the aggregates are the expensive shared state. This
ordering costs the **same number of summary passes as plain iFFBS on an accept**,
and one extra round-trip on a reject.

```
iffbs_mh_individual!(model, data, X, i, rng, proposal, target, stats)

 0. invariant on entry: aggregates agree with X, which holds x_cur.
 1. logπ_cur  ← target(model, data, X, i)          # free: we are already at the
                                                   # full-aggregate state
 2. copy x_cur into scratch
 3. reverse i's summaries                          # 1 reverse  → leave-one-out
 4. data._focal[] = i
    forward_filter!(… proposal …)                  # the proposal chain
 5. log q_cur ← backward_score(probs, trans_cache, x_cur)     # no RNG
    x_can, log q_can ← backward_sample_logq!(scratch, …, rng) # writes scratch, NOT X
    data._focal[] = -1
 6. if x_can == x_cur:  re-apply summaries; record; return    # 1 apply
 7. write x_can into X;  apply i's summaries       # 1 apply   → full aggregates (can)
    logπ_can ← target(model, data, X, i)
 8. log α = (logπ_can − logπ_cur) − (log q_can − log q_cur)
    accept?  → done. X and aggregates already hold x_can.
    reject?  → reverse x_can's summaries; write x_cur into X; apply
                                                   # +1 reverse +1 apply
```

**Cost accounting**, per individual, against plain `iffbs!`:

| | plain iFFBS | iFFBS-MH, accept | iFFBS-MH, reject |
|---|---|---|---|
| summary passes | 1 rev + 1 app | 1 rev + 1 app | 2 rev + 2 app |
| forward filter | 1 | 1 | 1 |
| backward walks | 1 | 2 (one scores, no RNG) | 2 |
| target evaluations | 0 | 2 | 2 |

The target evaluation is `O(window × (1 + |A|))` — the neighbour sum touches each
affected individual **once per path**, versus the filter's `N` times per timepoint.
At `N = 4` on the badger model the two target evaluations cost roughly half of one
forward filter. With the changed-timepoint restriction of §2.1 they cost much less.

### 3.1 Performance: state the claim honestly

**MHiFFBS is not a speed play in this package, and the doc should say so.** The
temptation is: the coupling term was ~70% of an iFFBS sweep, so dropping it from
the filter is a huge win. It is not, for two reasons.

1. Dropping the coupling from the filter saves `N × |A|` neighbour visits per
   timepoint but the ratio still needs `2 × |A|` of them (cur and can) — the saving
   is `N/2 ≈ 2×` on the coupling term, not elimination.
2. Since the 2026-07-20 fixes, iFFBS is **~17% of a badger sweep** and the gradient
   is ~83% (CLAUDE.md). A 2× saving on part of 17% is at best ~8% of a sweep, before
   the MH step's own costs are subtracted.

So: build this for **correctness** — semi-Markov targets, mismatched proposals, and
the ability to *know* whether the sampler is exact. Any speed benefit is incidental
and must be measured, alone, per CLAUDE.md's rule.

---

## 4. API

### 4.1 `IFFBSProposal` — the chain the filter runs

```julia
struct IFFBSProposal{TM,SS,OP,RC}
    trans_mat::TM              # default: data.trans_mat
    starting_state::SS         # default: data.starting_state
    observation_process::OP    # default: data.observation_process
    rest_contribution::RC      # default: data.rest_contribution
end

iffbs_proposal(data; trans_mat = data.trans_mat, starting_state = data.starting_state,
                     observation_process = data.observation_process,
                     rest_contribution = data.rest_contribution)
```

Every field is parameterised, and — per CLAUDE.md's third-instance trap
(`trans_mat::TransitionSpec{RF}` with `C` unbound) — **`TM` must bind both of
`TransitionSpec`'s parameters** if it is ever declared as `TransitionSpec{...}`.
Simplest is to leave it a free `TM`, since a proposal need not be a `TransitionSpec`
at all.

`forward_filter!` changes from reading `data.trans_mat` / `data.observation_process`
/ `data.rest_contribution` directly to reading them off a `proposal` argument that
**defaults to `iffbs_proposal(data)`**. That is a pure refactor with no behaviour
change, and it is Phase 0 precisely so it can be verified as such (§8, test 0).

Two convenience constructors for the paper's named variants:

```julia
uncorrected_proposal(data)  = iffbs_proposal(data; rest_contribution = no_rest_contribution)
markov_proposal(data, spec) = iffbs_proposal(data; trans_mat = spec)
```

### 4.2 The conditional target

```julia
epidemic_conditional_loglik(data; entry_time = nothing, survival = nothing,
                                  observation_process = data.observation_process,
                                  observation_weight = data.observation_weight,
                                  step_logprob = nothing,
                                  coupled_transitions = nothing)
    -> (model, data, X, i) -> Real
```

Returns log π(x_i | x_-i) up to a constant, evaluated **at the full-aggregate
state** — i.e. under exactly the conditions `epidemic_loglik` assumes. Built from
the same calls: `data.starting_state`, `transition_prob(data.trans_mat, …)`, the
observation weight, the entry gate, plus the neighbour sum.

The neighbour sum needs `neighbor_logprob` and `coupled_mask`, which
`epidemic_data` builds today and then closes over inside `rest_contribution`
(`data.jl:418-428`). **Store them on `EpidemicData` as fields** so the conditional
can reuse the identical function — that is what makes exactness structural rather
than a coincidence two constructors happen to agree on.

> **The mismatch hazard, stated once.** `entry_time` / `survival` here MUST match
> what was given to `epidemic_loglik`, and `observation_process` here MUST be the
> FULL one (§1.3c), not `epidemic_obs_loglik`'s possibly-reduced factor. Getting
> either wrong makes the Gibbs step and the HMC block target different
> distributions — silently.
>
> **Structural fix (recommended, Phase 3):** introduce a small `TargetSpec` holding
> `(entry_time, survival, step_logprob)` once, and build **both**
> `epidemic_loglik(spec)` and `epidemic_conditional_loglik(spec)` from it. Keyword
> forms stay as thin wrappers, so nothing breaks. Until then, §8 test 1 is the
> guard.

### 4.3 The kernel

```julia
iffbs_mh_individual!(model, data, X, i, rng; proposal, target, stats = nothing)
iffbs_mh!(model, data, X, rng; proposal, target, stats = nothing)
```

and the builder:

```julia
epidemic_latent_sampler(data; mh::Bool = false, proposal = nothing,
                              target = nothing, stats = nothing)
```

**Recommendation: explicit `iffbs_mh!`, plus `mh::Bool` on the builder.** Not
`iffbs!(…; mh=true)`. Reasons:

* the MH variant needs two things the plain one does not (a proposal and a target),
  so a boolean would leave two arguments meaningless half the time;
* it has different failure modes and different diagnostics (an acceptance rate);
* the reference does exactly this — `MCMCiFFBS_.jl:892` branches on an `MH` flag
  between two whole functions — and that has proved workable;
* a user reading `iffbs_mh!` in a script knows a correction is in play. A user
  reading `iffbs!` and not noticing `mh=true` three lines up does not.

Argument validation on the builder:

* `mh = false` with a `proposal` given → **error**. Silently running an uncorrected
  mismatched proposal is the exact failure the paper warns about, and it should not
  be reachable by forgetting a keyword.
* `mh = true` with no `proposal` → fine, and useful: it is the ratio-1
  self-check running as the sampler.
* `focal_self_contribution = false` → **error** (§2.4).

### 4.4 Acceptance statistics

```julia
struct MHStats
    proposed::Vector{Int}       # per individual, excluding no-change proposals
    accepted::Vector{Int}
    identical::Vector{Int}      # proposal == current, auto-accepted
    last_logratio::Vector{Float64}
    max_abs_logratio::Base.RefValue{Float64}   # for the ratio-1 diagnostic
end

acceptance_rate(stats)             # population, excluding `identical`
acceptance_rate(stats, i)
```

`identical` is counted **separately, not as an acceptance**. The reference folds
them into the total (`iFFBS_mh.jl:113-119` returns early *before* incrementing, but
`data.mh_total_indiv[id] += 1` on the main path counts every non-identical proposal
including rejects), which makes the reported rate depend on how often the chain is
frozen rather than on how good the proposal is. Report both.

`max_abs_logratio` exists so the sampler *is* the diagnostic (§7).

The stats object is created by the user and passed in, rather than stashed on
`data` as the reference does — `data` is the model's fixed structure, and sampler
telemetry is not that.

### 4.5 Worked usage

**(a) The paper's MHiFFBS** — Markov target, cheap proposal:

```julia
prop   = uncorrected_proposal(data)                     # filter drops the coupling
target = epidemic_conditional_loglik(data; entry_time, survival)
stats  = MHStats(data.n_individuals)
latent! = epidemic_latent_sampler(data; mh = true, proposal = prop,
                                        target = target, stats = stats)
```

**(b) SM-iFFBS** — semi-Markov target, Markov proposal, coupling kept:

```julia
# proposal: the geometric/κ=1 chain, expressible with @transitions today
geom = @transitions state_space begin
    S -> E = infection
    E -> I = mean_matched_hazard          # (model, data, i, t)
    @survival surv death = :D
end

# target: the Weibull sojourn — needs the path, so it goes through `step_logprob` (§5)
target = epidemic_conditional_loglik(data; step_logprob = weibull_step_logprob,
                                           entry_time, survival)
latent! = epidemic_latent_sampler(data; mh = true,
                                        proposal = markov_proposal(data, geom),
                                        target  = target, stats = stats)
```

**(c) The verification harness** — is my current sampler exact?

```julia
report = check_iffbs_exact(model, data, X; rng, n_sweeps = 3)   # §7
```

---

## 5. The semi-Markov gap: rate functions cannot see `X`

This is a genuine prerequisite for use case (b) and should be called out rather
than discovered.

Rate functions in this package take `(model, data, i, t)`
(`transitions.jl:88`, `_fill_rates!`, `_accum_row`). A semi-Markov hazard needs
time-since-exposure, which is a function of the *path*. The reference gets it by
reading `data.X` and scanning backwards (`progression_likelihood.jl:12-19`) — an
alias to the very matrix the sampler mutates, which works but is implicit and
ordering-sensitive.

Three options, in increasing order of blast radius:

1. **`step_logprob` seam (recommended).** One optional function
   `step_logprob(model, data, X, i, t) -> Real`, defaulting to
   `log(transition_prob(data.trans_mat, model, data, X, i, t, X[t,i], X[t+1,i]) + 1e-12)`.
   It receives `X`, so a semi-Markov user overrides *only* this and keeps the
   starting-state, observation, entry-gate and neighbour machinery unchanged.
   Used by **both** `epidemic_loglik` and `epidemic_conditional_loglik`, from the
   one `TargetSpec` of §4.2 — which is what stops the population likelihood and the
   Gibbs step drifting apart.

   The reason this covers the canonical case with one hook: **semi-Markov structure
   bites only on the focal's own path.** The neighbour terms are restricted to
   *coupled* transitions (§2.1), which in every model here are the FOI-driven S→E
   moves — Markov in the neighbour's own state. A neighbour's own E→I sojourn does
   not depend on `x_i`, so it cancels.

2. **Path-aware rate arity.** Let `@transitions` accept
   `(model, data, X, i, t)` rates, with `TransitionSpec` carrying a compile-time
   arity tag. More general (the *filter* could then use a path-aware rate too, which
   it must not — that is the point of the approximation), and a larger change to
   the macro, the clamping and every existing model.

3. **`data.X` by convention.** What the reference does. Cheapest, and the worst:
   it makes correctness depend on when the alias is written relative to when a rate
   is read, with nothing checking it.

Take (1). Note (2) is not needed for MH and would be actively wrong to use for the
proposal chain.

---

## 6. What to copy from the reference, and what not to

**Copy: skip the `rand()` draw when `log α ≥ 0`.** `iFFBS_mh.jl:341` does this
deliberately — "when acc >= 1, avoid drawing RNG so forced-accept runs reproduce
classic iFFBS random stream". It makes test 6 of §8 possible: with
proposal == target, `α ≡ 1`, no RNG is consumed by the accept step, and
`iffbs_mh!` must reproduce `iffbs!` draw for draw under the same seed. That is a
far stronger test than "the answers look similar".

**Copy: the identical-proposal early exit.** `x_can == x_cur` skips the whole
target evaluation. Cheap, and common at high acceptance.

**Do NOT copy: the windowed delta.** The reference restricts the delta to
`[first_change − 1, …]` (`iFFBS_mh.jl:229-236`). Our version of that restriction
must be split in two, because the two halves have different preconditions:

* **Neighbour / observation / starting-state terms** cancel wherever
  `x_can[t] == x_cur[t]`. Exact under the aggregates contract (§2.1) — the same
  assumption `make_rest_contribution` already makes. **Default on.**
* **Own-step terms** cancel at `t` only if `x_can` and `x_cur` agree at both `t`
  and `t+1` **and** the step probability does not depend on the rest of the path.
  For a semi-Markov target that second condition fails: changing when an E-episode
  *started* shifts the sojourn clock for every later step in that episode, even
  where the state labels agree. The reference's `j_hi` happens to extend to the
  death time, which covers the forward case; a `±1` padding would not have.
  **Default: full window**, with an explicit
  `path_dependent_steps::Bool = false` on the target that narrows it. `false` is
  correct by construction today, because `@transitions` rates literally cannot see
  the path — so this knob only becomes load-bearing once §5's `step_logprob` lands,
  and it is the same flag that says a `step_logprob` was supplied.

**Do NOT copy: the `RowOverrideMatrix` / `DataXRowOverride` proxy**
(`iFFBS_mh.jl:1-14`). It exists to feed a candidate row to a function that reads
`data.X`; with `step_logprob(model, data, X, i, t)` taking `X` directly, and the
candidate written into `X` before evaluation (§3, step 7), there is nothing to
proxy.

**Do NOT copy: temporarily mutating `data` to feed counterfactual counts**
(`_foi_from_counts`, `iFFBS_mh.jl:140-166`, which writes `data.totalNumInfec[g,t]`
and restores). We have a principled version of exactly this — the reversible
aggregates — and it is what `rest_contribution` already uses.

**Do NOT copy: `acc = exp(ratio)` before the comparison.** `exp` of a large
positive ratio overflows to `Inf` and of a large negative underflows to `0`; the
reference then has to special-case `isnan`. Compare in log space:
`accept = logα ≥ 0 || log(rand(rng)) < logα`.

---

## 7. Compatibility detection: make it a runtime check, not a static one

The brief asks whether the package could detect that a user's spec is
iFFBS-incompatible. Statically: no, and it is worth saying why rather than
shelving it vaguely. Rates are opaque closures; "does this hazard depend on the
path?" is not decidable from a `TransitionSpec`, and the interesting mismatches
(§1.3) live in *keyword arguments to a different function* (`entry_time`), in a
*second spec* (`coupling_trans_mat`), or in a *split observation model*.

Dynamically it is trivial, and it falls out of this design for free:

```julia
check_iffbs_exact(model, data, X; rng, n_sweeps = 1, atol = 1e-8) -> report
```

Runs `iffbs_mh!` with `proposal = iffbs_proposal(data)` (i.e. proposal == target),
records `log α` for every individual, and reports `max |log α|` with the worst
offenders. By §2.4 an exact spec gives `max |log α| ≤ atol`; anything larger names
the individuals whose conditional the filter is getting wrong. This is precisely
what `iFFBS_MH.cpp` does with its `|ratio − 1| > 1e-5` warning, generalised.

Two ways to run it, both worth shipping:

* **as a check** — `check_iffbs_exact`, called once against a real fit;
* **as a guard** — `iffbs_mh!(…; warn_if_exact_violated = atol)`, so a long run
  reports drift rather than producing a quietly wrong posterior.

Numerical floor to expect: the filter and the likelihood do not use identical
guards — `transition_matrix_at!` clamps rates to `(1e-12, 1-1e-12)` then takes the
self-transition as `1 - rowsum`, `epidemic_loglik` does `log(p + 1e-12)`, and
`make_neighbor_logprob_from_transitions` does `log(max(p, 1e-12))`. On a model
whose probabilities sit well inside the band these agree to ~1e-12 relative;
`atol = 1e-8` is a reasonable default, and a model that fails it *only* because it
is riding the clamp has a different problem worth knowing about.

---

## 8. Test plan

The user's emphasis. Ordered by how much each one buys.

**0. The Phase-0 refactor changes nothing.** After `forward_filter!` takes a
`proposal`, `iffbs!` with the default proposal must produce a **bit-identical** `X`
from the same seed on the existing `test/iffbs.jl` models, and the aggregates must
match. Guards the one change that touches the existing hot path.

**1. Δ-consistency — the conditional really is the joint's restriction.**
For random `(model, X, i)` and a random candidate row:

```
target(model, data, X_can, i) − target(model, data, X_cur, i)
  ==  (loglik + obs_loglik)(X_can) − (loglik + obs_loglik)(X_cur)
```

with the right-hand side computed from scratch (`reset_aggregates!` +
`apply_derived_summaries!` on each of `X_cur`, `X_can`) so nothing incremental is
trusted. Catches every missing term: neighbours, observations, entry gate,
starting state. Run it with and without an entry gate, and with a split
observation model, since §1.3 says those are where it will fail.

**2. Ratio-1 — the keystone.** With `proposal = iffbs_proposal(data)`, every
`log α` must be `0` to `1e-8`, over full sweeps, on:
  * the S/I model of `test/iffbs.jl` with a perfect test observation process,
  * the same with `coupled_transitions` declared,
  * a model with `@survival` (so the coupling view is exercised),
  * a model with an `entry_time` gate — **expected to fail** if the observation
    process permits pre-entry death, and that is the point (§1.3a). Assert the
    documented condition, both ways.

This one test covers the proposal density, the target, the aggregate bookkeeping
and the focal-self-contribution equivalence at once.

**3. Aggregate consistency across both branches.** After sweeps with
`force_accept = true` and again with `force_reject = true`, the incrementally
maintained aggregates must equal a from-scratch recompute. The **reject** path —
reverse candidate, rewrite current, re-apply — is the single most likely place for
a bug, and it is the one branch the ratio-1 test never exercises.

**4. Exact-posterior enumeration — the correctness proof.** A deliberately tiny
model: `C = 2`, `T = 4`, `N = 2` ⇒ 2^8 = 256 possible `X`. Enumerate the exact
posterior by evaluating `loglik + obs_loglik` on all 256 and normalising. Then run
`iffbs_mh!` for ~10^6 sweeps with a **deliberately mismatched** proposal (drop the
coupling; perturb a rate) and compare the empirical distribution over `X` to the
exact one — total-variation distance, and a χ² over the states with non-negligible
mass. This is the only test that proves the kernel targets the right distribution
when the correction is actually doing work. Everything else proves internal
consistency.

Run the same enumeration against plain `iffbs!` with the mismatched *filter* and
show that it **fails** — otherwise the test cannot distinguish "MH is correct" from
"the mismatch did not matter".

**5. Detailed balance, spot check.** On the same tiny model, for a handful of
`(x, x')` pairs, check `π(x) q(x→x') α(x,x') == π(x') q(x'→x) α(x',x)` numerically.
Cheap, and it localises a sign error that test 4 would only show as a slow drift.

**6. RNG-stream equivalence.** With proposal == target and the `log α ≥ 0`
no-draw rule (§6), `iffbs_mh!` and `iffbs!` must produce identical `X` from the
same seed. Verifies that the candidate is drawn with exactly the same categorical
calls in the same order, and that scoring consumes no RNG.

**7. Semi-Markov end-to-end.** Once `step_logprob` lands: a small model with a
Weibull sojourn, a geometric proposal, and the enumeration of test 4. Plus the
check that `epidemic_loglik` and `epidemic_conditional_loglik` built from the same
`TargetSpec` agree (test 1 again, with the override active).

**8. Acceptance sanity.** On a badger-shaped synthetic, a mismatched proposal must
give an acceptance rate in a sane band, not ~0. The paper's > 0.84 is the
reference point; a rate near zero means the proposal is unusable rather than the
kernel being wrong, and the test should say which.

**9. Degenerate guards.** `log q_cur = -Inf` (current path unreachable under the
proposal — legitimate after initialisation elsewhere) must **accept**.
`log π_can = -Inf` must **reject**. `NaN` must reject and warn, and there should be
an `on_nonfinite = :error` mode for debugging.

---

## 9. Phasing

| Phase | Content | Gate |
|---|---|---|
| **0** | `IFFBSProposal`; `forward_filter!` takes it; default reproduces today | test 0 (bit-identical) |
| **1** | `backward_score` / `backward_sample_logq!`; store `neighbor_logprob` + `coupled_mask` on `EpidemicData`; `epidemic_conditional_loglik` | test 1 |
| **2** | `iffbs_mh!`, `MHStats`, `epidemic_latent_sampler(; mh)` | tests 2, 3, 4, 5, 6, 9 |
| **3** | `TargetSpec` + `step_logprob`, shared by `epidemic_loglik` and the conditional; semi-Markov example | test 7 |
| **4** | `check_iffbs_exact`; run it on the badger model and settle §1.3(a) | — |
| **5** | Changed-timepoint restriction as a measured optimisation | test 2 must still pass; measure alone |
| *later* | Static compatibility detection | shelved — see §7 |

Phases 0 and 1 are worth doing on their own even if MH is never used: Phase 0
makes the filter's inputs explicit, and Phase 1's `epidemic_conditional_loglik` is
the piece any *other* latent kernel (particle Gibbs, blocked Gibbs, MH move-events
— all named in CLAUDE.md's roadmap) will need too. That is the argument for the
shape: a per-individual conditional target is a reusable seam, not MH plumbing.

---

## 10. Decisions needed before implementation

1. **`focal_self_contribution = false` is an error in v1** (§2.4). Agree, or is
   there a live model that needs it?
2. **New fields on `EpidemicData`** for `neighbor_logprob` and `coupled_mask`
   (§4.2). Two more type parameters on a struct CLAUDE.md already flags as the
   package's hub — worth it for structural exactness, but it is a change to the
   hot type and must be checked with `isconcretetype(fieldtype(...))` on the
   *whole* field type, per the `trans_mat::TransitionSpec{RF}` precedent.
3. **`TargetSpec` (Phase 3) vs keyword duplication (Phases 1–2).** The keyword
   route ships sooner and test 1 guards it; the `TargetSpec` route makes the
   mismatch unrepresentable. Proposal above is: keywords now, `TargetSpec` in
   Phase 3, keyword forms retained as wrappers.
4. **Should `rest_contribution` gain a log-returning sibling?** The conditional
   does not need it under design (B) of §3 — it scores neighbours directly — but a
   `rest_log_contribution` would remove an `exp`/`log` round-trip and its underflow
   risk from the filter's own path. Independent change; measure alone.
5. **Where does §1.3(a) land?** If `check_iffbs_exact` shows the badger entry gate
   does bias the filter, the fix is either an MH correction or an entry-aware
   proposal `trans_mat`. Worth resolving before building on top of it.
