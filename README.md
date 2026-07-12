# EpidemicTrajectories.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://evoart.github.io/EpidemicTrajectories/dev/)

Discrete-time, individual-level epidemic models, built from three pieces:

1. a **simulator** that draws hidden state trajectories forward in time,
2. an **automatic-differentiation-friendly likelihood** of the model parameters
   given a state trajectory and observed data, and
3. a **latent-state sampler** — individual forward-filtering, backward-sampling
   (iFFBS) — that draws a new trajectory given the current parameters.

Each is an ordinary Julia function with no dependency on a probabilistic
programming framework, so the likelihood can be used directly as an HMC target
and the iFFBS sampler as a Gibbs latent-variable update. A worked example fitting
a model with [PracticalBayes.jl](https://github.com/EvoArt/PracticalBayes) is in
[`examples/cattle_ecoli_iffbs.jl`](examples/cattle_ecoli_iffbs.jl).

## Two ways to specify a model

Both produce the same per-step transition probabilities and share the simulator,
likelihood, and iFFBS machinery.

- **Rate functions.** Supply the per-step transition probabilities as functions
  of the parameters, the individual, and the time: `f(pars, model, data, i, t)`.
  This form supports per-individual covariates, individual-specific observation
  models, network or spatial forces of infection, and history-dependent rates.

- **State transitions.** List the allowed `(from, to)` transitions and give each
  a rate, using the `@transitions` macro. `SimpleEpiTransitionMatrix` takes rates
  that depend only on compartment counts (a chain-binomial model);
  `EpiTransitionMatrix` takes the full per-individual rate signature.

## Example

The two-state susceptible/infected model of Touloupou et al. (2019): each animal
is susceptible or infected, with recurrent `S → I` and `I → S` transitions (no
recovered compartment).

A model is described in three parts: `model` (the parameters), `X` (the latent
trajectory), and `data` (the observations plus group structure).
`epidemic_model` bundles the fixed structure and returns three functions —
`loglik`, `simulate`, and `latent!` — to plug into a PPL:

```julia
using EpidemicTrajectories, Random

group = repeat(1:10; inner = 8)             # ten pens of eight animals
rams  = DiagnosticTest(; sensitivity = p -> p.θ)

em   = epidemic_model(SI, TwoStateSI(); tests = (rams,))
data = build_data(em, group; observations = (Rmask,), initial_prob = [0.9, 0.1])

pars = (; α = 0.01, β = 0.02, m = 6.0, θ = 0.8)
X    = zeros(Int, 80, 80)                   # latent trajectory (filled by latent!)

em.loglik(pars, X, data)                    # differentiable in `pars`; @addlogprob! this
em.latent!(rng, pars, X, data)              # one iFFBS sweep, resamples X in place
em.simulate(rng, pars, data)                # draw a trajectory and observations
```

The rate functions can also be written with the `@transitions` macro:

Rates are written as bare expressions that refer to the rate-function arguments
by name (`pars`, `model`, `data`, `i`, `t` in the per-individual style):

```julia
si = @transitions :individual SI begin
    S -> I = -expm1(-(pars.α + pars.β *
                      count(j -> j != i && data.states[j, t] == 1,
                            data.members(data, data.group[i]))))
    I -> S = 1 / pars.m
end
```

See [`examples/cattle_ecoli_iffbs.jl`](examples/cattle_ecoli_iffbs.jl) for a full
fit that recovers all model parameters from simulated capture-recapture data.

## Status

Under active development. The two-state susceptible/infected model is complete.
Planned: multi-state (SEID and user-specified compartments), spatial and
continuous-time models, and a post-hoc residual/diagnostic layer.
