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
using Statistics: std, var
using Printf: @printf, @sprintf

# For BatchedArchive and the leave-future-out backends: a stdlib, so it adds no
# dependency to the four-package core. Trajectories go to disc in Int8 batches
# rather than into memory or a chain object, which on a model of any size is the
# only workable route (residuals.jl).
using Serialization: Serialization, serialize, deserialize

# Include order note: `data.jl` defines the `EpidemicData` type that the other
# files annotate their arguments with, so it comes before them. Function BODIES
# resolve at call time, so `data.jl` may still call into `transitions.jl`.
include("spec.jl")
include("aggregates.jl")
include("data.jl")
include("transitions.jl")
include("iffbs.jl")
include("build.jl")

# After build.jl: `iffbs_mh.jl` calls `epidemic_conditional_loglik` as a default
# argument, and its `IFFBSMHSampler` is what `epidemic_latent_sampler(; mh=true)`
# returns. Function BODIES resolve at call time, so the mutual reference is fine
# either way round; this order just reads better.
include("iffbs_mh.jl")
include("residuals.jl")
include("attribution.jl")
include("diagnostics.jl")

# Leave-future-out cross-validation. `truncate.jl` first: the rest take a
# truncated `EpidemicData`.
include("truncate.jl")
include("lfo.jl")
include("lfo_score.jl")
# `lfo_backend.jl` before `lfo_run.jl`: the driver dispatches on the backend
# types and writes worker output through `_item_file`.
include("lfo_backend.jl")
include("lfo_run.jl")
include("lfo_psis.jl")
include("lfo_adapt.jl")
# Model specification
export TransitionSpec, @transitions, @survival

# User-declared aggregates and their reversible updates
export @aggregate, AggregateSpec, AggregateDeclaration, allocate_aggregates
export reset_aggregates!, apply_derived_summaries!, apply_summaries!

# The data container
export EpidemicData, epidemic_data, members, build_affected_individuals_from_groups
export no_observations

# Transition matrices and the coupling term
export transition_matrix_at, transition_prob, make_rest_contribution, no_rest_contribution
export coupled_transition_mask
export make_neighbor_logprob_from_transitions

# The latent sampler
export iffbs!, iffbs_individual!, forward_filter, backward_sample!

# iFFBS as a PROPOSAL, corrected by Metropolis-Hastings (iffbs_mh.jl). The
# proposal names the chain the FILTER runs; the conditional target names the chain
# the LIKELIHOOD scores; when they differ, the MH step is what makes the sweep
# valid. `check_iffbs_exact` reports whether they differ at all.
export IFFBSProposal, iffbs_proposal, uncorrected_proposal, markov_proposal
export backward_logq, backward_sample_logq!
export epidemic_conditional_loglik, default_step_logprob
export iffbs_mh!, iffbs_mh_individual!, check_iffbs_exact
export MHStats, IFFBSMHSampler, acceptance_rate, identical_rate, reset_stats!

# What a model spec generates
export epidemic_simulator, epidemic_loglik, epidemic_obs_loglik, epidemic_latent_sampler

# Trajectory summaries: residuals and other post-hoc, per-draw, per-individual
# quantities computed from a sampled `X`. The FOURTH artefact generated from one
# model spec — the one that says whether the fit was any good. See residuals.jl.
export TrajectorySummary, @residual, WaitingTimeResidual, LeftTruncatedSurvivalResidual
export trajectory_summaries, SummaryResult, residual_values, draw_values
export randomized_pit, discrete_product_cdf, cumulative_hazard_cdf
export first_entry, state_code, transition_hazard
export archive_draw, post_hoc_draws, aggregate_synced_draws

# Source attribution: the infection-link residual (Lau et al. 2014 §2.2.2), the
# endogenous/exogenous FOI split, and case reproduction numbers. All three rest on
# a DECLARED decomposition of the force of infection. See attribution.jl.
export SourceAttributionResidual, FOIRatioSummary
export case_reproduction_numbers, summarize_population
# Online collection (never stores X) and disc-batched archiving (bounded memory).
export SummaryCollector, collect_summaries!, finish
export BatchedArchive, archive_push!, archive_close!, archived_draws

# Checks and plots. These are STUBS whose implementations live in weak-dependency
# extensions: `using HypothesisTests` enables the tests, a Makie backend the plots.
# Neither package enters the core dependency list. See diagnostics.jl.
export uniformity_test, pvalue_distribution, pi_05
export residual_plot, residual_plot!, residual_qq, residual_qq!
export pvalue_plot, pvalue_plot!, residual_panel

# Leave-future-out cross-validation: declared truncation of the data, the
# granularity axis the score is aggregated over, the per-window score, the
# driver, PSIS smoothing, the execution backends, and the adaptation diagnostic.
export truncation, truncate_data, lfo_cutoffs, TruncationPlan, TruncationRule
export Granularity, Joint, Pointwise, ByGroup, cell_of, aggregate_cells
export LFOResult, WindowResult, elpd, compare, cutoffs, n_informative
export forward_simulate, score_window, survival_constrained
export LFOSpec, lfo_cv
export psis_smooth, psis_ess
export LocalBackend, SlurmArray, SweepHandle, sweep_status, work_item, write_sbatch
export resubmit, collect_sweep
export AdaptTrace, record!, check_adaptation, flat_likelihood_range

end # module EpidemicTrajectories
