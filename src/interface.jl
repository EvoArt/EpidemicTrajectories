# The high-level entry point, in the three-part vocabulary a PPL model uses:
#
#   model : the model PARAMETERS (a NamedTuple of the sampled/fixed rates etc.).
#   X     : the latent state trajectory (individuals × time), sampled separately.
#   data  : everything ELSE the model and latent update need — the observations
#           passed into the PPL, plus any per-sweep mutable bookkeeping a latent
#           update maintains (e.g. an infected-per-group count). Built ONCE with
#           `build_data` and passed in; a latent update may mutate its bookkeeping
#           in place across sweeps.
#
# `epidemic_model` bundles a model's fixed *structure* (state space, rate bundle,
# diagnostic tests) and returns three functions with the `(model, X, data)`
# signature:
#
#   em = epidemic_model(SI, TwoStateSI(); tests = (rams,))
#   data = build_data(em, group; observations = (Rmask,), initial_prob = init)
#
#   em.loglik(model, X, data)        # log-likelihood of X — @addlogprob! this
#   em.simulate(rng, model, data)    # draw a trajectory (and observations)
#   em.latent!(rng, model, X, data)  # in-place iFFBS update of X (and bookkeeping)

"""
    EpidemicData

The `data` object in the `(model, X, data)` split: the observations a PPL model
is conditioned on, plus the fixed group structure and any mutable per-sweep
bookkeeping a latent update maintains. Built by [`build_data`](@ref); passed to
the [`EpidemicModel`](@ref) closures. Distinct from the latent trajectory `X`
(passed separately) and from `model` (the parameters).

Fields:
- `group` — group/pen index per individual.
- `members` — `members(g)` gives the individuals in group `g`.
- `observations` — tuple of diagnostic-test result matrices (aligned with the
  model's `tests`), each `n_ind × n_times`, entries `-1`/`0`/`1`.
- `initial_prob` — probability vector over states at `t = 1` (or `nothing`, then
  supplied per call).
- `n_ind`, `n_times`.
"""
struct EpidemicData{GT,MB,OT,IP}
    group::GT
    members::MB   # members(g) -> individuals in group g
    observations::OT
    initial_prob::IP
    n_ind::Int
    n_times::Int
end

"""
    EpidemicModel

Returned by [`epidemic_model`](@ref). Holds a model's fixed *structure* (state
space, rate bundle, diagnostic tests) and exposes three callables in the
`(model, X, data)` vocabulary — `model` is the parameters, `X` the latent
trajectory, `data` an [`EpidemicData`](@ref):

- `loglik(model, X, data) -> Real` — log-likelihood of trajectory `X`.
  Autodiff-friendly in `model`; `X` and `data` are fixed. Add to a PPL's log
  density, e.g. `@addlogprob! em.loglik(pars, X, data)`.
- `simulate(rng, model, data) -> (states, observations)` — draw a hidden
  trajectory and observed test results.
- `latent!(rng, model, X, data) -> X` — one iFFBS Gibbs sweep, resampling `X` in
  place (and updating any bookkeeping in `data`).
"""
struct EpidemicModel{SS<:StateSpace,RB<:RateBundle,TT,LK,SM,LT}
    state_space::SS
    rates::RB
    tests::TT
    loglik::LK
    simulate::SM
    latent!::LT
end

"""
    epidemic_model(state_space, rates; tests=(), coupling=true) -> EpidemicModel

Bundle a discrete-time epidemic model's fixed structure — the [`StateSpace`](@ref),
the [`RateBundle`](@ref) (e.g. [`TwoStateSI`](@ref) or an
[`EpiTransitionMatrix`](@ref)), and any diagnostic [`DiagnosticTest`](@ref)s —
and return an [`EpidemicModel`](@ref). Pair it with a [`build_data`](@ref) object
and call the returned closures in the `(model, X, data)` vocabulary.

`coupling` is passed to the iFFBS sampler (include the between-individual
coupling term; default `true`).

# Example
```julia
em = epidemic_model(SI, TwoStateSI(); tests = (rams,))
data = build_data(em, group; observations = (Rmask,), initial_prob = init)

# in a PPL model body, after `X ~ TrajectoryLatent(...)`:
@addlogprob! em.loglik((; α, β, m), X, data)
```
"""
function epidemic_model(state_space::StateSpace, rates::RateBundle;
                        tests=(), coupling::Bool=true)
    tests_t = Tuple(tests)

    # `loglik(model, X, data)`: score trajectory X under the parameters `model`.
    loglik = function (model, X, data::EpidemicData)
        d = _pack(X, data)
        m = (; state_space=state_space, rates=rates, pars=model)
        return trajectory_loglik(model, m, d)
    end

    # `simulate(rng, model, data)`: draw a trajectory and (if tests) test data.
    simulate = function (rng, model, data::EpidemicData; initial_prob=data.initial_prob, n_times=data.n_times)
        initial_prob === nothing &&
            throw(ArgumentError("provide `initial_prob` (in build_data or as a keyword)"))
        states, _ = simulate_trajectory(rng, state_space, rates, model, data.group, initial_prob; n_times=n_times)
        obs = isempty(tests_t) ? () : simulate_observations(rng, tests_t, model, state_space, states)
        return (states=states, observations=obs)
    end

    # `latent!(rng, model, X, data)`: one iFFBS sweep, resampling X in place.
    latent! = function (rng, model, X, data::EpidemicData; initial_prob=data.initial_prob)
        initial_prob === nothing &&
            throw(ArgumentError("provide `initial_prob` (in build_data or as a keyword)"))
        d = _pack(X, data)
        m = (; state_space=state_space, rates=rates, pars=model)
        ffbs_sweep!(rng, m, d, tests_t, data.observations; initial_prob=initial_prob, coupling=coupling)
        return X
    end

    return EpidemicModel(state_space, rates, tests_t, loglik, simulate, latent!)
end

# Assemble the lightweight `data` object the lower-level functions expect from a
# latent trajectory `X` plus the persistent `EpidemicData` — `X` is the current
# `states`, and `group`/`members` come from `data`. Kept internal; the public
# API is the `(model, X, data)` split above.
@inline function _pack(X, data::EpidemicData)
    return (; states=X, group=data.group, members=(_, g) -> data.members(g))
end

"""
    build_data(em::EpidemicModel, group; observations=(), initial_prob=nothing,
               n_times=nothing) -> EpidemicData

Build the `data` object for a model: the observations the PPL is conditioned on
plus the fixed group structure (and room for any per-sweep bookkeeping a latent
update maintains). Built once and passed to `em.loglik`/`em.simulate`/`em.latent!`.

- `group`: group/pen index per individual (length `n_ind`).
- `observations`: tuple of result matrices aligned with the model's `tests`,
  each `n_ind × n_times`, entries `-1` (missing) / `0` / `1`.
- `initial_prob`: length-`nstates` probability vector over states at `t = 1`, or
  `nothing` to supply per call.
- `n_times`: inferred from `observations`; pass explicitly if `observations` is
  empty.
"""
function build_data(::EpidemicModel, group;
                    observations=(), initial_prob=nothing,
                    n_times::Union{Int,Nothing}=nothing)
    group = collect(Int, group)
    n_ind = length(group)
    obs_t = Tuple(observations)
    nt = n_times === nothing ?
        (isempty(obs_t) ?
            throw(ArgumentError("pass `n_times` when `observations` is empty")) :
            size(first(obs_t), 2)) :
        n_times
    groups = sort!(unique(group))
    members_by_group = Dict(g => findall(==(g), group) for g in groups)
    members(g) = members_by_group[g]
    return EpidemicData(group, members, obs_t, initial_prob, n_ind, nt)
end
