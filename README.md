# EpidemicTrajectories.jl

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

```julia
using EpidemicTrajectories, Random

rates = TwoStateSI()                        # S→I: 1 − exp(−(α + β·I₋)); I→S: 1/m
pars  = (; α = 0.01, β = 0.02, m = 6.0)
group = repeat(1:10; inner = 8)             # ten pens of eight animals

states, data = simulate_trajectory(
    Random.default_rng(), SI, rates, pars, group, [0.9, 0.1]; n_times = 80,
)

model = (; state_space = SI, rates = rates, pars = pars)
trajectory_loglik(pars, model, data)        # differentiable in `pars`
```

The same model can be written with the `@transitions` macro:

```julia
si = @transitions :individual SI begin
    S -> I = (pars, model, data, i, t) -> begin
        g = data.group[i]
        I₋ = count(j -> j != i && data.states[j, t] == 1, data.members(data, g))
        -expm1(-(pars.α + pars.β * I₋))
    end
    I -> S = (pars, model, data, i, t) -> 1 / pars.m
end
```

See [`examples/cattle_ecoli_iffbs.jl`](examples/cattle_ecoli_iffbs.jl) for a full
fit that recovers all model parameters from simulated capture-recapture data.

## Status

Under active development. The two-state susceptible/infected model is complete.
Planned: multi-state (SEID and user-specified compartments), spatial and
continuous-time models, and a post-hoc residual/diagnostic layer.
