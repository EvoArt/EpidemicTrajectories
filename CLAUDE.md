# CLAUDE.md

Guidance for Claude Code (or any agent) working in EpidemicTrajectories.jl.

## Why this package exists (the founding motivation — don't lose this)

The reason this package was started: to **sample latent states within a
PracticalBayes model without that sampler running on every NUTS/HMC gradient
call.** In a plain PPL, a data-augmentation step written inside the model body
would re-run on every gradient evaluation — expensive, and for a stochastic
sampler it also makes the gradient itself stochastic, which can invalidate the
inference. PracticalBayes solves this structurally: the latent block is updated
once per Gibbs sweep by an `AbstractLatentKernel`, lexically outside every
gradient call, and is held constant (AD-constant `ValueSlot`) while the
continuous parameters are differentiated. This package provides the pieces that
plug into that seam.

**iFFBS is only ONE example of a latent-trajectory sampler.** The package is not
about iFFBS specifically — it's about generating, from a model spec, a
latent-state sampler (whatever kind is appropriate) plus a matching likelihood
and simulator that a PPL can use. iFFBS is the first one implemented because it
fits the two-state discrete-time cattle model; other latent samplers (particle
Gibbs / conditional SMC, block/blocked Gibbs, other FFBS variants, MH-within-
Gibbs move-events, etc.) are equally in scope and should slot into the same
`latent!`-shaped role. Keep the design general enough that the latent sampler is
pluggable, not hard-coded to FFBS.

## What this package is

EpidemicTrajectories.jl builds **discrete-time individual-level epidemic models**
as three reusable, **PPL-agnostic** pure functions, generated from a model spec:

1. a **simulator**,
2. an **autodiff-friendly likelihood** — usable as an HMC target via
   `@addlogprob!`, and
3. a **latent-state sampler** — currently iFFBS, but this is one instance of a
   pluggable role, not the whole story (see "Why this package exists" above).

**The likelihood comes in TWO halves and you almost always need both.**
`epidemic_loglik` covers the starting state and the transitions ONLY.
`epidemic_obs_loglik` covers the observation process. Neither includes the other:

```julia
@addlogprob! loglik(pars, data, X) + obs_loglik(pars, data, X)
```

**Omitting the second term is a silent modelling bug**, and it shipped in the
badger example for months: `observation_process` is used only by the iFFBS
forward filter, never by `epidemic_loglik`, so every observation parameter
(sensitivities, specificities, capture probabilities) received NO likelihood
information and was effectively sampled from its prior. Halving all four changed
the log density by exactly `0.000e+00`. If a model's observation parameters look
prior-like, check this first.

**It has no dependency on any probabilistic-programming framework.** That is
deliberate: the likelihood drops into a PracticalBayes (or Turing) `@addlogprob!`,
and the latent sampler is exactly what a PracticalBayes `AbstractLatentKernel`'s
`latent_step` calls once per Gibbs sweep — outside every gradient call.

## The central design rule (do not violate this)

**The package NEVER assumes ahead of time what arrays (if any) the user wants
tracked during the latent update, or how they should be updated.** This is the
rule the whole design turns on.

Instead: the user declares whatever arrays they want in a generic `aggregates`
container, and declares how each is updated via a **reversible** update — either
with the `@aggregate` / `@derived_summary` macros (which generate the forward and
reverse update from one expression), or by supplying a generic update function
**together with its reverse**. The package only ever calls those user functions;
it has no idea whether an array holds infected counts, alive counts, or something
else entirely.

Reversibility is what makes iFFBS both correct and cheap: to resample individual
`i`, the sampler **reverses** `i`'s own contribution out of the aggregates, runs
the forward filter / backward sample (so `i` sees leave-one-out statistics), then
**re-applies** `i`'s new contribution. The aggregates stay exactly consistent with
`X` throughout — verified: the incrementally-maintained aggregate equals a
from-scratch recompute exactly (`walkthrough_cattle_check.jl`).

## The reference implementation

**`walkthrough_cattle.jl` is the specification.** It is split into "package code"
and "user code", is runnable, and reproduces the E. coli cattle model. The package
refactor is based on it, and an important end goal is that something very close to
its *user code* section runs against the real package unchanged.

`walkthrough_cattle_check.jl` is the correctness check: it starts iFFBS from a
deliberately wrong all-susceptible `X` and confirms the sampler rebuilds the
epidemic (prevalence 0.0 → ~0.17 against a truth of 0.178), that test-positive
cells are called infected with probability 1 (specificity is 1 in this model), and
that the incremental aggregate matches a full recompute.

`walkthrough.jl` and `walkthrough_sugar.jl` are earlier, non-runnable sketches.

## Planned: a convenience layer for common cases (NOT YET BUILT)

The generic core above deliberately assumes nothing, which means even the simplest
model needs the user to declare its aggregates and rates. **In future we will want
convenience functions/structs for common, simple choices** — e.g. a ready-made
two-state S/I setup, a standard "infected per group" aggregate, common observation
models — so the easy case is one line while the general case stays fully open. An
earlier API (`epidemic_model`/`RateBundle`/`TwoStateSI`/`StateSpace`/
`SimpleEpiTransitionMatrix`) attempted this but baked in assumptions about the
tracked arrays, violating the central design rule; it was deleted in favour of the
walkthrough design. When the convenience layer is rebuilt, it must be a thin,
optional shell OVER the generic core — never a constraint on it.

## The reference model

The first-class example is the **two-state S/I recurrent Markov model** of
Touloupou et al. (2019) (cattle E. coli iFFBS). It is **NOT SIR** — there is no
recovered/removed compartment. States are `{S=0, I=1}` with recurrent S↔I:

- `S -> I`: prob `1 - exp(-(α + β·I₋))`, where `I₋` = number of OTHER infected
  animals in the same pen at time `t` (leave-one-out, frequency-dependent FOI).
- `I -> S` (recovery): prob `1/m`; `I -> I`: `1 - 1/m`. m = mean infectious
  period. In the paper/example, `m = m̃ + 1` (m̃ > 0) keeps `1/m < 1`.

Reference material (read-only, on this dev machine, NOT part of the repo):
- `badgers/ppl/original_cattle_ecoli_iFFBS.qmd` — the Turing prototype this
  package generalizes.
- `badgers/ppl/simulate_data.jl` — the paper's data simulator.
- `badgers/BIID_R/.../JBIIDRjl/Julia/src/{transitions,iFFBS_modular,...}.jl` — the
  full badger SEID framework; the source of the `f(pars,model,data,i,t)` rate
  protocol and (later) the residuals/diagnostics layer to replicate.

## The `model` / `X` / `data` convention

Three distinct things — keep them straight:
- `model` — the model **parameters** (a NamedTuple such as `(; α, β, m, ν, θʳ, θᶠ)`).
- `X` — the **latent state trajectory**, a `Matrix{Int}` indexed `X[t, i]`
  (time × individual), holding 1-based state indices into `state_space`.
- `data` — everything else: the fixed structure (sampling periods, coupling
  structure, transition spec), the user's `derived_summaries`, the user's
  `aggregates` container, and the user's own `extras` (observations, covariates,
  anything) reachable as `data.name`.

Rate functions take `(model, data, i, t)`; derived summaries take
`(model, data, X, s, i, t; reverse=false)`; the observation process takes
`(model, data, X, i, t)` and returns a per-state weight vector.

**The same rule applies to all of these, not just the aggregates.** The package
supplies no observation process (the default is `no_observations` — the honest
default, since it cannot know what you observe), assumes no group structure
(`affected_individuals` is indexed `[t, i]`, so coupling may vary over time; the
fixed-group build is only a convenience default), and assumes no sampling window
(defaults to `1:T` for everyone, overridable per individual). Anything the package
names is a default the user can replace, never a constraint.

## Non-obvious facts

- **The whole latent trajectory `X` is stored as ONE whole-matrix latent** in a
  PracticalBayes `@model` (`X ~ SomeDiscreteMatrixDistribution`, routed to a
  `ValueSlot`, resampled by the iFFBS `AbstractLatentKernel`, read AD-constant in
  the `@addlogprob!` term). This needed a one-line PracticalBayes core fix (skip
  `linked_vec_length` for latent-role sites, now on PracticalBayes master); it is
  NOT the indexed-`X[i,t]~` family approach.
- **Transition probabilities are clamped to `(1e-12, 1-1e-12)`** — a sampler
  exploring parameter space can momentarily propose values (huge FOI, or
  `1/m > 1` when `m < 1`) that would send `log(P[a,b])` to `log(0)`/`log(<0)` and
  crash NUTS with a `DomainError`. Clamping regularizes only the tails; true
  params sit well inside the band. Hence the `m = m̃ + 1` reparameterization.
- **Recovery quality depends on the regime.** With near-saturation (>50%
  prevalence) β and the test sensitivities are poorly identified (a poorly-mixing
  α/β/m produces a wrong latent X, which deflates the conjugate updates). The
  cattle walkthrough uses a lower-transmission regime (~18% prevalence) where all
  six parameters are cleanly identified.
- **`@transitions` arrow parsing:** `S -> I = rate` parses as `Expr(:->, :S,
  block)` where the block holds `Expr(:(=), :I, rate)` — i.e. Julia reads it as
  `S -> (I = rate)`. The macro extracts source/dest/rate accordingly.
- **Aggregates must be consistent with `X` before the first loglik call.** The
  invariant is: aggregates always agree with the current `X`. It is established
  once by `reset_aggregates!` + `apply_derived_summaries!` on the initial `X`, and
  preserved thereafter by iFFBS's reverse→refilter→reapply. `loglik` and the rate
  functions READ the aggregates and never rebuild them.

## Performance: what matters, in order

Measured on the badger model (2384 individuals x 161 timepoints), which is the
package's stress case. 1000 iFFBS sweeps went from **78 hours to 1.4 hours** via
two changes, neither of which was the one that looked obvious:

1. **Concrete types.** `extras`/`aggregates` are `NamedTuple`s (not `Dict{Symbol,Any}`),
   `EpidemicData` is parameterised on them, `getproperty` dispatches on `Val(s)`
   (not `s in fieldnames(T)`, a runtime search that allocates on every access), and
   `rate_fns` is a `Tuple` (not `Vector{Function}`) iterated by recursion in
   `_fill_rates!`. Any of these regressing costs ~5x and is invisible to inference
   checks — `data.age` reports `Matrix{Int64}` either way. **Benchmark, don't infer.**

   **A `Tuple` of callables is necessary but NOT sufficient — it must be iterated
   by RECURSION, never by `for`.** `for f in tup` over a tuple of DISTINCT
   closure types infers the loop variable as their union and dispatches at
   runtime on every call. This bit twice: `rate_fns` (fixed by `_fill_rates!`)
   and, two years of comments later, `derived_summaries` — whose docstring
   *claimed* the Tuple made the hot loop specialise, while the two loops in
   `iffbs_individual!` were **42.7% of a sweep's self time**, both flagged
   `gc? Y dispatch? Y`. Fixed by `apply_summaries!` (2.6x on the sweep, 717 ->
   199 MB). Call summaries ONLY through `apply_summaries!`. Note it takes
   `reverse` POSITIONALLY: a keyword on a call the compiler cannot resolve forces
   the kwarg path and allocates a NamedTuple per call.
2. **`coupled_transitions`.** The user declares which of a neighbour's transitions
   the focal can influence; the sampler skips neighbours whose realised move it
   cannot affect. Exact (verified to 1e-13), and worth ~10x on the badger model.

**The trap in `coupled_transitions`**: the mask must be the CLOSURE — every
transition out of any coupled source state, not just the named one. Probabilities
out of a state sum to one, so influencing `S -> E` necessarily influences `S -> S`.
Masking only the named transition changes the sampler's weights by ~0.25. There are
tests for both properties; do not "simplify" them away.

**Fewer allocations is NOT automatically faster.** Allocation and type stability
are independent axes and trading one for the other can lose. Concretely: the
iFFBS scratch buffer was first held in `const Ref{Any}`, which cut allocation
199 -> 139 MB and made the sweep **26% SLOWER** (0.435 -> 0.510 s) — `Any` made
every buffer access dynamic, reintroducing the exact instability being fixed one
function over. The typed `struct FilterScratch` gave 0.375 s / 85.6 MB. **Measure
each change on its own**: had this been bundled with the `apply_summaries!` fix,
that 2.6x would have masked the regression entirely.

**`_FILTER_SCRATCH` is a module-level `Ref` and is therefore NOT thread-safe.**
That is currently sound because `iffbs!` is single-threaded *by construction* —
individuals within a sweep share mutable aggregate state through the
reverse -> refilter -> reapply invariant, so they cannot be run in parallel at
all. If iFFBS is ever parallelised, this must become per-task state first.

3. **`coupling_trans_mat`.** A separate `TransitionSpec` for the coupling term ONLY
   (`epidemic_data(...; coupling_trans_mat=trans_mat)`, default unchanged). The
   coupling term (`rest_contribution` → `neighbor_logprob`) is never differentiated
   — profiling put it at ~70% of an iFFBS sweep — so it is the one place a rate
   that reads a CACHED, parameter-derived quantity (e.g. a per-`(group,time)` force
   of infection a derived summary maintains) is both safe and worthwhile. Safe
   because `epidemic_loglik` never sees `coupling_trans_mat`, so the cache is
   structurally unreachable from AD — unlike putting a cached rate straight in
   `trans_mat`, which silently zeroes the gradient w.r.t. whatever fed the cache
   while leaving the log-density bit-identical (measured: gradient collapsed to
   the prior's ~±1 instead of +300/-195/-20/+26, no warning at all). Measured
   worth: ~1.19x on the badger model — real, but far short of hoped-for, because
   (see next point) the FOI arithmetic was never the dominant cost inside the
   coupling loop.

4. **`observation_weight`.** `epidemic_obs_loglik(data; observation_process=...,
   observation_weight=...)`. The vector-returning `observation_process` is what the
   SAMPLER needs (the forward filter reads every state's weight); the LIKELIHOOD
   needs exactly one entry, `w[X[t,i]]`. Going via the vector allocates an array
   per `(i,t)` — on badgers **18 MB and ~187k arrays of Duals per call**, making
   the observation term more expensive than the entire transition likelihood.
   `observation_weight(model, data, X, i, t, s)` returns that one entry.
   Measured: primal 0.0148 -> 0.0017 s, **18 MB -> 16 bytes**; gradient 0.226 ->
   0.075 s (3.0x). This is the same trap `transition_prob` already avoids for
   transitions — and it was reintroduced in the observation term the day it was
   written, so watch for it in any new per-`(i,t)` package function.

5. **`observation_process` as a keyword to `epidemic_obs_loglik`.** The seam that
   lets a user keep SOME observation parameters conjugate. The package cannot
   split an observation process itself — it is one opaque function returning a
   weight vector, and nothing in it says which factor belongs to which parameter.
   A user whose process factorises multiplicatively (`w = capture x tests`) gives
   the PRODUCT to `epidemic_data` (so the filter sees everything) and only the
   non-conjugate FACTOR here. Because the weights multiply, the log-likelihood is
   a sum of the factors' contributions, so dropping one drops exactly its term.
   **Keeping a factor in both this likelihood and a conjugate block double-counts
   it** — verify independence (the badger test factor moves by 0.000e+00 when
   `etas` changes).

**GOTCHA: an observation factor's element type depends on the BLOCKING, not the
model.** The same function is called with `Float64` when its parameters are
sampled conjugately and with `ForwardDiff.Dual` when they sit in an HMC block.
Allocate weight vectors as `ones(eltype(model.some_param), n_states)`, never
`ones(Float64, ...)` — the latter works until someone moves that parameter into
HMC, then throws on the write.

**Blocking: the C++ reference's two-gradient split does NOT port to a PPL.**
`badger_ref` splits epidemic vs test parameters into two hand-written gradient
functions over disjoint data. Mirroring that as two Gibbs HMC blocks measured
**23% SLOWER** (4.527 vs 3.663 s/sweep), because PracticalBayes evaluates the
WHOLE model body for every block — two blocks means two full primal evaluations
AND two independent leapfrog trajectories, which costs far more than the saved AD
partials. Use ONE HMC block with conjugate blocks alongside it.

**Match the reference's EXPECTED trajectory length, not its nominal `L`.**
`badger_ref` draws `intL = ceil(runif(0,1)*L)` with `L=30` — uniform on {1..30},
mean 15.5, so ~16.5 gradients per HMC step. A fixed `HMC(L=30)` does 30, i.e.
1.82x the work for no added fidelity. We use a fixed `L=15`. A randomised-L
kernel IS buildable from AdvancedHMC's public API (`Trajectory`, `HMCKernel`,
`FixedNSteps` are exported; `HMCSampler` passes any kernel through — `HMC`/`NUTS`
are just conveniences over that), but `nsteps(τ)` takes no RNG and is called
TWICE per transition, so keeping the simulated and reported `L` in agreement
needs mutable state in the sampler plus an assumption that `refresh` fires once
per transition — an ordering dependency that would fail SILENTLY. Rejected on
those grounds; revisit only if a fixed L shows resonance.

**Where the remaining gap is (2026-07-17 investigation).** *[SUPERSEDED — kept
for the reasoning. The O(n_states)-per-lookup running total described below was
built on 2026-07-19 as the `rest_contribution` keyword, and on 2026-07-20 the
iFFBS sweep went 1.129 -> 0.375 s via the two fixes above. iFFBS is now ~17% of a
badger sweep and is NOT the bottleneck; the gradient is ~83%. Read the following
as history, not as a to-do.]*

After (1)-(3), badger iFFBS is ~1.5-2x the reference's own sweep. Profiling BOTH
sides (not just ours) settled why:

- Our `rest_contribution` is **O(n_states × |affected|) per `(i,t)`**: for every
  candidate state of the focal, loop over every affected neighbour and call
  `neighbor_logprob`. On badgers, `4 × ~70 ≈ 280` neighbour-visits per timepoint.
- The reference's equivalent is **O(n_states) per `(i,t)`**: `logProbRestTotal[s,t]`
  is a running total over ALL individuals, maintained incrementally
  (`updateLogProbRestTotalIndiv!`: subtract the focal's OLD row, recompute just
  that one row, add it back — ~13 array ops/timepoint, no neighbour loop at all).
- **The reference's own `@batch per=thread`/`@batch per=core` (Polyester),
  applied to this update, is close to a wash on this workload** — confirmed by
  disabling all four call sites and re-profiling: single-threaded iFFBS (1.064s)
  was at or below every threaded run measured (0.98-1.86s across repeated runs),
  and the threaded profile was ~90% thread-dispatch/wait/idle-spin, not algorithm
  — a fine-grained-parallelism anti-pattern (a batch region launched 3× PER
  INDIVIDUAL over only 160 iterations each, not once for the whole sweep). So the
  reference's own wall-clock numbers were, all session, an underestimate of what
  its algorithm can do.
- Secondary factor, not yet measured directly: array layout.
  `logProbRest[s,jj,tt]` is `s`-fastest-varying (column-major), so the reference's
  4-state read/write clump is one cache line; our `X[t,i]` is `t`-fastest-varying,
  so the neighbour loop strides across `n_timepoints` per neighbour — a cache
  miss on every one of the ~280 visits.

**Open, NOT implemented: port the same O(n_states)-per-`(i,t)` running-total
shape as a package-side change** (a generic `logProbRest`-equivalent + incremental
patch, replacing `rest_contribution`'s counterfactual recompute). Bigger than
`coupling_trans_mat` (structural, not a keyword); needs invalidation reasoned out
generically from `affected_individuals`/`coupled_transitions` (both already
user-declared, so this is believed tractable — see repro log for the earlier,
now-revisited rejection). Single-threaded first, per the finding above. Needs
sign-off before starting.

- Profile with `ProfileToLLM` (`~/.julia/dev/ProfileToLLM`) rather than guessing;
  it flags runtime dispatch and GC per line, which is how wins 1-2 above were
  found. Guessing had pointed at the coupling, which was not the problem (that
  time). **Sort by TOTAL when deciding what to change, by SELF only when reading
  what one line does** — a self%-sorted profile of the coupling term looked
  completely flat (6.4%) and nearly sent the 2026-07-17 investigation in the
  wrong direction; total% showed the true 69.5%. Same trap nearly repeated on the
  reference's own profile before disabling `@batch` revealed it was measuring
  thread overhead, not the algorithm.

## Known gaps in the current walkthrough design

- `@survival` is parsed but is currently a stub returning `nothing`.
- `@transitions` supports only `:individual` style.
- No convenience layer yet for common/simple choices (see above) — every model
  currently declares its own aggregates and rates.
- **`epidemic_obs_loglik` has no scalar path of its own.** It takes a
  user-supplied `observation_weight`; the package cannot derive one from a
  vector-returning `observation_process`. A model that supplies only the vector
  form silently gets the slow (allocating) path — correct, ~3x slower gradient.
- **The badger example's `phis` is in the HMC block; the reference samples it
  CONJUGATELY** (`CheckSensSpec_` + `rbeta` over susceptible-state test results).
  Not wrong — `phis` is informed either way now that the test factor is in the
  likelihood — but an exact conjugate draw would be cheaper and mix better. The
  derivation is in `MCMCiFFBS_.cpp:849-868` if anyone wants to build the kernel.
- **ESS has never been measured for any of the blocking choices.** The
  conjugate-vs-HMC decision for `etas`/`nu` was made on mixing grounds
  (a conjugate draw is an exact independent sample; an HMC step is correlated
  with an unadapted step size), NOT on data — wall-clock was a wash between them.
  If blocking is revisited, measure ESS/second, not s/sweep.

## Roadmap (per user direction)

- Now: the generic core from `walkthrough_cattle.jl` — user-declared reversible
  aggregates, `@transitions`/`@aggregate` macros, iFFBS. ✓
- Later: convenience functions/structs for common, simple choices (thin shell over
  the generic core, never a constraint on it).
- Later: SEID, or let the user declare which of S/E/I/D/R exist in their system.
- Later: other latent samplers (particle Gibbs/CSMC, blocked Gibbs, MH move-events)
  behind the same `latent!`-shaped role.
- Later: residuals / post-hoc diagnostics (PIT, Cox-Snell, exposure/Sellke,
  infection-link, R_i) reusing the same rate functions — the badger
  `residuals.jl` layer, computed post-hoc from `(model, X, data)` draws.
- Later: continuous-time / spatial, ODE-fit extension.

**Where the performance headroom is now (2026-07-20).** The badger sweep is
~2.0 s: iFFBS ~17%, HMC/AD ~83%. Profiling the gradient shows **zero runtime
dispatch and essentially zero GC** — it is raw floating-point arithmetic, so the
type-stability and allocation classes of win are exhausted there. Two structural
levers remain, both needing sign-off:

1. **A hand-derived gradient seam** (the big one). The C++ reference computes
   ∂logpost analytically in ONE scalar pass (`grad_.cpp`, with closed-form Siler
   derivatives); we run 61-partial forward-mode AD. Shape it like
   `rest_contribution`: an optional seam for a user-supplied gradient, defaulting
   to AD. The package would stay ignorant of what the gradient means.
2. ~~**Keeping rates in log-space**~~ — **INVESTIGATED AND DROPPED.** The claim
   that "the reference never leaves log space" was WRONG. `badger_ref` stores
   `logProbStoSgivenSorE` as a log but **exponentiates it back on every use**
   (`iFFBS_fixedPars.cpp:112-113`), because its forward filter is genuinely in
   probability space: `predProb`/`filtProb` are sums of products (lines 118-124),
   which have no log-space form without logsumexp — and it runs a full logsumexp
   per `(i,t)` in `normTransProbRest` on top. The reference's actual win there is
   caching the group-level FOI per `(g,t)` instead of per `(i,t)`, which we
   already have via `rest_contribution`.

   Log-space rates would help the LIKELIHOOD (`log(P)` becomes a subtraction) but
   hurt the FILTER, which needs real probabilities: `transition_matrix_at!` gives
   the self-transition the leftover mass as `P[a,a] = 1 - rowsum`, and that
   subtraction has NO log-space equivalent — it becomes `log1mexp(logsumexp(...))`
   per row per `(i,t)`, trading one subtraction for two transcendentals. So the
   rates get `exp`'d back on read exactly as the reference does, and the
   transcendentals move from the gradient into the filter roughly one-for-one.
   Filter is ~17% of a sweep vs gradient ~83%, so it is still directionally
   positive, but far below the ~8-10% first estimated — and it would require user
   rate functions to return logs, an API change touching `@transitions`, the
   clamping logic, and every existing model. Not worth the blast radius.

*Do not chase the `log`/`exp` share on the strength of a single profile:* one
noisy run put `log` at 25% total and a cleaner re-run put `_log` at 1.6% self.
Sort by TOTAL to choose what to change, by SELF to read one line, and re-run
before acting.

## Conventions

- git + GitHub (account `EvoArt`), but DO NOT list Claude as a contributor.
- Prefer more inline comments in `src/` than a terse house style (matches the
  PracticalBayes sibling package's convention).
- Julia type annotations on function args: only when needed for dispatch.
