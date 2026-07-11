module EpidemicTrajectories

# EpidemicTrajectories.jl — build discrete-time epidemic models as three
# reusable, PPL-agnostic pieces:
#
#   1. a SIMULATOR        (draw a hidden state trajectory forward in time),
#   2. a LIKELIHOOD       (a pure, autodiff-friendly log-density of the
#                          continuous parameters given a fixed state trajectory
#                          and the observed data), and
#   3. a LATENT SAMPLER   (individual forward-filtering / backward-sampling —
#                          iFFBS — that draws a new hidden trajectory given the
#                          current parameters and data).
#
# All three are ordinary Julia functions with no dependency on any
# probabilistic-programming framework. That is deliberate: the likelihood drops
# straight into a PracticalBayes `@addlogprob!` (or Turing's), and the iFFBS
# sampler is exactly what a PracticalBayes `AbstractLatentKernel`'s
# `latent_step` calls once per Gibbs sweep. The wiring into PracticalBayes lives
# in the companion package `PracticalEpiBayes`, not here — this package stays
# usable on its own, with any sampler, or none.
#
# TWO IDIOMS for specifying the same model, sharing one core:
#
#   * FUNCTIONAL style (this file's `RateBundle`): you supply small rate
#     functions — force of infection, progression/recovery rate, survival,
#     test sensitivity — following the signature `f(pars, model, data, i, t)`.
#     This is the "iFFBS paper" way, closest to bespoke per-individual
#     capture-recapture models (see `rates.jl`, `iffbs.jl`).
#
#   * TRANSITION-MATRIX style (`transition_matrix.jl`, gemlib-esque): you supply
#     a state-transition structure (an incidence/stoichiometry matrix + a rate
#     for each transition) and get simulate + log-likelihood for free. Good for
#     compartmental (population-level) chain-binomial models.
#
# Both compile down to the same per-step transition probabilities, so the FFBS
# machinery and the likelihood are shared.

using Random: Random, AbstractRNG
using Distributions: Distributions
using StatsFuns: StatsFuns, logsumexp, log1mexp
using LinearAlgebra: LinearAlgebra

include("compartments.jl")
include("rates.jl")
include("simulate.jl")
include("likelihood.jl")
include("iffbs.jl")
include("transition_matrix.jl")

# Compartment / state-space
export StateSpace, SI, SIS, SEID, nstates, state_index

# Functional (iFFBS-paper) style
export RateBundle, transition_matrix_at, TwoStateSI

# Observation model (imperfect diagnostic tests)
export DiagnosticTest, observation_likelihood

# Simulation, likelihood, latent sampling
export simulate_trajectory, simulate_observations
export trajectory_loglik
export ffbs_individual!, ffbs_sweep!

# Transition-matrix (gemlib-esque) style
export SimpleEpiTransitionMatrix, EpiTransitionMatrix, StateTransitionModel
export @transitions, incidence_matrix, chain_binomial_loglik, simulate_chain_binomial

end # module EpidemicTrajectories
