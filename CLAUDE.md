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

## Known gaps in the current walkthrough design

- `@survival` is parsed but is currently a stub returning `nothing`.
- `@transitions` supports only `:individual` style.
- No convenience layer yet for common/simple choices (see above) — every model
  currently declares its own aggregates and rates.

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

## Conventions

- git + GitHub (account `EvoArt`), but DO NOT list Claude as a contributor.
- Prefer more inline comments in `src/` than a terse house style (matches the
  PracticalBayes sibling package's convention).
- Julia type annotations on function args: only when needed for dispatch.
