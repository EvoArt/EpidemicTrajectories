# CLAUDE.md

Guidance for Claude Code (or any agent) working in EpidemicTrajectories.jl.

## What this package is

EpidemicTrajectories.jl builds **discrete-time individual-level epidemic models**
as three reusable, **PPL-agnostic** pure functions:

1. a **simulator** (`simulate_trajectory`, `simulate_chain_binomial`),
2. an **autodiff-friendly likelihood** (`trajectory_loglik`,
   `chain_binomial_loglik`) — usable as an HMC target via `@addlogprob!`, and
3. a **latent-state sampler** (`ffbs_sweep!` / `ffbs_individual!`) — individual
   forward-filtering / backward-sampling (iFFBS).

**It has no dependency on any probabilistic-programming framework.** That is
deliberate: the likelihood drops into a PracticalBayes (or Turing) `@addlogprob!`,
and the iFFBS sampler is exactly what a PracticalBayes `AbstractLatentKernel`'s
`latent_step` calls once per Gibbs sweep. The wiring into PracticalBayes is done
by the user (see `examples/cattle_ecoli_iffbs.jl`) and, later, automated by the
companion package **PracticalEpiBayes.jl** (currently an empty placeholder in
`~/.julia/dev/PracticalEpiBayes`).

## Two idioms, one shared core

Both compile down to the same per-step transition probabilities, so the FFBS
machinery and the likelihood are shared:

- **Functional (iFFBS-paper) style** — you supply rate functions following
  `f(pars, model, data, i, t)`. The extension seam is
  `transition_matrix_at(rb::RateBundle, pars, model, data, i, t) -> Matrix`. This
  is STRICTLY MORE GENERAL than the transition-matrix style: it can express
  per-individual covariates, individual-specific observation, network/spatial
  force of infection, and semi-Markov/history-dependent dynamics. `TwoStateSI` is
  the canonical concrete bundle (the two-state S/I model).

- **Transition-matrix (gemlib-esque) style** — you list `(from, to)` transitions
  with a rate each. Two levels:
  - `SimpleEpiTransitionMatrix` — rates are pure `f(pars, counts, t)` (population
    counts only). Chain-binomial likelihood (`chain_binomial_loglik`). The
    exchangeable, count-sufficient case.
  - `EpiTransitionMatrix` — rates are the full `f(pars, model, data, i, t)`. This
    IS a `RateBundle` (implements `transition_matrix_at`), so it drives the same
    functional machinery (iFFBS, `trajectory_loglik`) — as general as a
    hand-written functional bundle.
  - Both buildable via the `@transitions` macro:
    `@transitions [:individual] StateSpace begin; S -> I = rate; ...; end`.

The two styles converge (give the same posterior) ONLY in the exchangeable,
count-sufficient case — that convergence is a property of the specific model, NOT
a reduction of one style to the other.

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

## The `model` / `data` convention

Kept lightweight so it works with plain NamedTuples/closures (no framework types):
- `model` — any object with `.state_space::StateSpace` and `.rates::RateBundle`
  (and `.pars` for simulation/FFBS, which need concrete parameter values).
- `data` — any object exposing `.states::Matrix{Int}` (individual × time, user
  state codes), `.group::Vector{Int}`, and `.members(data, g)`. Build the default
  with `make_data(states, group)`; extend with `merge(make_data(...), (; ...))`.

## The KEY milestone (done)

`examples/cattle_ecoli_iffbs.jl`: simulate cattle capture-recapture data from
known (α, β, m, ν, θ) and recover them with `Gibbs(NUTS for α/β/m̃, conjugate
Beta kernels for ν/θ, iFFBS kernel for the latent X)` in PracticalBayes. BOTH
styles (functional `TwoStateSI` and transition-matrix `@transitions`) recover all
five parameters within ~1 posterior SD, identically (same core). The example env
develops EpidemicTrajectories + PracticalBayes; run with
`julia --project=examples examples/cattle_ecoli_iffbs.jl`.

## Non-obvious facts

- **The whole latent trajectory `X` is stored as ONE whole-matrix latent** in a
  PracticalBayes `@model` (`X ~ SomeDiscreteMatrixDistribution`, routed to a
  `ValueSlot`, resampled by the iFFBS `AbstractLatentKernel`, read AD-constant in
  the `@addlogprob! trajectory_loglik(...)` term). This needed a one-line
  PracticalBayes core fix (skip `linked_vec_length` for latent-role sites — see
  PracticalBayes branch `epi-2d-latent-support`); it is NOT the indexed-`X[i,t]~`
  family approach.
- **`TwoStateSI` clamps its transition probabilities to `(1e-12, 1-1e-12)`** — a
  sampler exploring parameter space can momentarily propose values (huge FOI, or
  `1/m > 1` when `m < 1`) that would send `log(P[a,b])` in the likelihood to
  `log(0)`/`log(<0)` and crash NUTS with a `DomainError`. Clamping regularizes
  only the tails; true params sit well inside the band.
- **Recovery quality depends on the regime.** With near-saturation (>50%
  prevalence) β and θ are poorly identified (θ recovers low because the poorly-
  mixing α/β/m produce a wrong latent X, which deflates θ's conjugate update).
  The example uses a lower-transmission regime (~18% prevalence) where all five
  parameters are cleanly identified. The FFBS itself is correct in isolation
  (verified: with TRUE params it recovers states at 91% MAP agreement, test-
  positive cells → P(I)=1.0 under specificity-1).
- **`@transitions` arrow parsing:** `S -> I = rate` parses as `Expr(:->, :S,
  block)` where the block holds `Expr(:(=), :I, rate)` — i.e. Julia reads it as
  `S -> (I = rate)`. The macro extracts source/dest/rate accordingly.
- **State encoding:** `StateSpace.codes` is the user's integer codes in dense
  order; `state_index(ss, code)` maps a code to its 1-based dense index. The badger
  SEID uses sparse codes (S=0, E=3, I=1, D=9); the two-state SI uses `{0, 1}`.

## Roadmap (per user direction)

- Now: two-state S/I model, rate components shaped as `f(model,data,i,t)`. ✓
- Later: SEID, or let the user declare which of S/E/I/D/R exist in their system.
- Later: residuals / post-hoc diagnostics (PIT, Cox-Snell, exposure/Sellke,
  infection-link, R_i) reusing the same rate functions — the badger
  `residuals.jl` layer, computed post-hoc from `(pars, X, data)` draws.
- Later: continuous-time / spatial (gemlib HPAI analogue), ODE-fit extension.

## Conventions

- git + GitHub (account `EvoArt`), but DO NOT list Claude as a contributor.
- Prefer more inline comments in `src/` than a terse house style (matches the
  PracticalBayes sibling package's convention).
- Julia type annotations on function args: only when needed for dispatch.
