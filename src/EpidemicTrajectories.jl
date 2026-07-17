module EpidemicTrajectories

# EpidemicTrajectories.jl — build discrete-time individual-level epidemic models,
# and from one model spec generate the three things needed to fit them:
#
#   1. a SIMULATOR       — draw a state trajectory forward in time,
#   2. a LIKELIHOOD      — autodiff-friendly in the parameters, for an HMC target,
#   3. a LATENT SAMPLER  — resample the whole hidden trajectory (currently iFFBS).
#
# All three are ordinary Julia functions with no dependency on any probabilistic
# programming framework. The likelihood drops into a PracticalBayes (or Turing)
# `@addlogprob!`; the latent sampler is what a PracticalBayes latent kernel calls
# once per Gibbs sweep — outside every gradient call, which is the whole reason
# this package exists (see CLAUDE.md).
#
# THE CENTRAL DESIGN RULE: the package never assumes what arrays (if any) the user
# wants tracked during the latent update, or how they update. The user declares
# whatever they like in `data.aggregates`, with REVERSIBLE updates written via
# `@aggregate` / `@derived_summary` (or by hand, supplying the reverse). The latent
# sampler reverses an individual's contribution, refilters, and re-applies it — so
# the aggregates stay exactly consistent with the trajectory, and the individual
# being resampled automatically sees leave-one-out statistics. See aggregates.jl.

using Random: Random, AbstractRNG
using Distributions: Distributions
using StatsFuns: StatsFuns, logsumexp, log1mexp
using LinearAlgebra: LinearAlgebra

# Include order note: `data.jl` defines the `EpidemicData` type that the other
# files annotate their arguments with, so it comes before them. Function BODIES
# resolve at call time, so `data.jl` may still call into `transitions.jl`.
include("spec.jl")
include("aggregates.jl")
include("data.jl")
include("transitions.jl")
include("iffbs.jl")
include("build.jl")

# Model specification
export TransitionSpec, @transitions, @survival

# User-declared aggregates and their reversible updates
export @aggregate, AggregateSpec, AggregateDeclaration, allocate_aggregates
export reset_aggregates!, apply_derived_summaries!

# The data container
export EpidemicData, epidemic_data, members, build_affected_individuals_from_groups
export no_observations

# Transition matrices and the coupling term
export transition_matrix_at, make_rest_contribution, no_rest_contribution
export coupled_transition_mask
export make_neighbor_logprob_from_transitions

# The latent sampler
export iffbs!, iffbs_individual!, forward_filter, backward_sample!

# What a model spec generates
export epidemic_simulator, epidemic_loglik, epidemic_latent_sampler

end # module EpidemicTrajectories
