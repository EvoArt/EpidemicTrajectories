# API

## State spaces

```@docs
StateSpace
SI
SIS
SEID
nstates
state_index
```

## Rate functions

```@docs
RateBundle
transition_matrix_at
TwoStateSI
DiagnosticTest
observation_likelihood
```

## Simulation

```@docs
EpidemicTrajectories.EpiModel
EpidemicTrajectories.make_data
simulate_trajectory
simulate_observations
```

## Likelihood

```@docs
trajectory_loglik
EpidemicTrajectories.observation_loglik
```

## Latent-state sampling

```@docs
ffbs_individual!
ffbs_sweep!
```

## Transition-matrix models

```@docs
SimpleEpiTransitionMatrix
EpiTransitionMatrix
@transitions
incidence_matrix
simulate_chain_binomial
chain_binomial_loglik
```
