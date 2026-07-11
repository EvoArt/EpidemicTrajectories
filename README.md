# EpidemicTrajectories.jl

Discrete-time individual-level epidemic models as three reusable, **PPL-agnostic**
pieces:

1. a **simulator**,
2. an **autodiff-friendly likelihood** (usable as an HMC target via `@addlogprob!`), and
3. a **latent-state sampler** — individual forward-filtering / backward-sampling (iFFBS).

No dependency on any probabilistic-programming framework: the likelihood drops
into a [PracticalBayes](https://github.com/EvoArt/PracticalBayes) (or Turing)
`@addlogprob!`, and the iFFBS sampler is exactly what a PracticalBayes
`AbstractLatentKernel`'s `latent_step` calls once per Gibbs sweep.

## Two idioms, one shared core

- **Functional (iFFBS-paper) style** — supply rate functions
  `f(pars, model, data, i, t)`; the seam is `transition_matrix_at`. Strictly more
  general than the transition-matrix style (per-individual covariates, network /
  spatial FOI, semi-Markov dynamics). `TwoStateSI` is the canonical bundle.
- **Transition-matrix (gemlib-esque) style** — list `(from, to)` transitions with
  a rate each, via the `@transitions` macro. `SimpleEpiTransitionMatrix`
  (count-based, chain-binomial) or `EpiTransitionMatrix` (per-individual, itself a
  `RateBundle`).

## The reference model

The two-state **S/I recurrent Markov** model of Touloupou et al. (2019) (cattle
E. coli iFFBS). Not SIR — no recovered compartment; recovery returns `I → S`.

```julia
using EpidemicTrajectories

rates = TwoStateSI()                       # S→I: 1-exp(-(α+β·I₋)); I→S: 1/m
pars  = (; α=0.01, β=0.02, m=6.0)
group = repeat(1:10; inner=8)              # 10 pens of 8 animals
states, data = simulate_trajectory(rng, SI, rates, pars, group, [0.9, 0.1]; n_times=80)

# autodiff-friendly likelihood of the trajectory given params:
model = (; state_space=SI, rates=rates, pars=pars)
trajectory_loglik(pars, model, data)       # -> Real, differentiable in pars
```

See `examples/cattle_ecoli_iffbs.jl` for the full end-to-end fit with
PracticalBayes — `Gibbs(NUTS for α/β/m, conjugate Beta kernels for ν/θ, iFFBS for
the latent state X)` — recovering all five parameters, via **both** styles.

## Status

Early development. Two-state S/I model complete; SEID, spatial/continuous-time,
and a post-hoc residuals/diagnostics layer are planned.
