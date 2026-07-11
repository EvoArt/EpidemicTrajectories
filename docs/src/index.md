# EpidemicTrajectories.jl

Discrete-time, individual-level epidemic models, built from three pieces:

1. a **simulator** that draws hidden state trajectories forward in time,
2. an **automatic-differentiation-friendly likelihood** of the model parameters
   given a state trajectory and observed data, and
3. a **latent-state sampler** — individual forward-filtering, backward-sampling
   (iFFBS) — that draws a new trajectory given the current parameters.

Each is an ordinary Julia function with no dependency on a probabilistic
programming framework, so the likelihood can be used directly as an HMC target
and the iFFBS sampler as a Gibbs latent-variable update.

## Two ways to specify a model

Both produce the same per-step transition probabilities and share the simulator,
likelihood, and iFFBS machinery.

- **Rate functions.** Supply the per-step transition probabilities as functions
  of the parameters, the individual, and the time: `f(pars, model, data, i, t)`.
  This form supports per-individual covariates, individual-specific observation
  models, network or spatial forces of infection, and history-dependent rates.
  `TwoStateSI` is the built-in bundle for the two-state model.

- **State transitions.** List the allowed `(from, to)` transitions and give each
  a rate, with the `@transitions` macro. `SimpleEpiTransitionMatrix` takes rates
  that depend only on compartment counts (a chain-binomial model);
  `EpiTransitionMatrix` takes the full per-individual rate signature.

## Getting started

The [tutorial](tutorials/cattle_iffbs.md) fits a two-state susceptible/infected
model to simulated capture-recapture data with
[PracticalBayes.jl](https://github.com/EvoArt/PracticalBayes), recovering the
transmission parameters, the infectious period, and the test sensitivity while
inferring the hidden infection trajectory.
