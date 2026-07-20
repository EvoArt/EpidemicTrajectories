# The badger observation process, FACTORED — the worked example of
# `epidemic_obs_loglik`'s `observation_process` keyword.
#
# WHY. `epidemic_loglik` covers the starting state and the transitions only. It
# never calls `observation_process` (that is used solely by the iFFBS forward
# filter), so with `@addlogprob! loglik(...)` alone the observation parameters
# thetas/rhos/phis/etas receive NO likelihood information — verified: halving all
# four changes the log density by exactly 0.000e+00 (examples/check_grad_zeros.jl).
# That is a genuine model error, not just a performance quirk.
#
# THE FACTORISATION. badger_observations is a PRODUCT of two independent factors:
#
#   w(state) = capture_factor(state) * test_factor(state)
#
#   capture_factor : eta / (1-eta) on the alive states (S,E,I), and the
#                    "seen alive => cannot be dead" zero on D. Depends on `etas`.
#   test_factor    : the per-test specificity/sensitivity product, applied only
#                    at capture times. Depends on `thetas`, `rhos`, `phis`.
#
# This mirrors badger_ref's ObsProcess_.cpp exactly, which builds
#     corrector.row = {eta, eta, eta, 0}      // capture
#     corrector(tt,0) *= productIfSuscep;     // tests multiplied IN
# — a product, so the log-likelihood is a SUM of the two factors' contributions
# and either can be dropped independently.
#
# WHY SPLIT. The C++ reference does NOT put all observation parameters in HMC:
#     thetas, rhos -> HMC              (HMC_thetas_rhos)
#     phis         -> conjugate Gibbs  (CheckSensSpec_ + rbeta)
#     etas         -> conjugate Gibbs
# Passing only the TEST factor to `epidemic_obs_loglik` keeps `etas` conjugate
# (its factor never enters the differentiated density, so there is no
# double-counting against the EtaKernel), while giving thetas/rhos the likelihood
# they were missing.
#
# NOTE ON `phis`. `phis` appears in the test factor, so with the test factor in
# the likelihood, `phis` DOES get likelihood information here. That differs from
# the C++, which samples phis conjugately. Keeping phis in HMC is a valid (if
# less efficient) choice — it is informed either way. Splitting phis out too
# would need a third factor and a CheckSensSpec_-equivalent conjugate kernel;
# see the `obs_test_no_phi` variant below for the piece that would enable it.

using EpidemicTrajectories
using Distributions
using Random

isdefined(@__MODULE__, :BADGER_STATES) || include(joinpath(@__DIR__, "badger_model.jl"))

## ---------------------------------------------------------------------------
## The two factors
## ---------------------------------------------------------------------------

"""
    badger_obs_capture(model, data, X, i, t)

The CAPTURE factor of the badger observation process: `eta` on the alive states
when the individual was seen, `1 - eta` when it was not, and the hard zero on
`D` when it was seen (a badger seen alive cannot be dead). Depends only on
`model.etas`.
"""
function badger_obs_capture(model, data, X, i, t)
    # Element type must follow `etas`, not be hard-coded Float64: when `etas` is
    # an HMC parameter (the `naive` blocking) it arrives as a ForwardDiff.Dual,
    # and writing a Dual into a Float64 array throws. Hard-coding Float64 here
    # worked only as long as `etas` was conjugate and therefore always Float64.
    w = ones(eltype(model.etas), data.n_states)
    eta = model.etas[data.season[t]]
    if data.capture[t, i] == 0
        w[1] = w[2] = w[3] = 1 - eta
        w[4] = one(eltype(w))
    else
        w[1] = w[2] = w[3] = eta
        w[4] = zero(eltype(w))
    end
    w
end

"""
    badger_obs_tests(model, data, X, i, t)

The TEST factor: the per-test product of specificity/sensitivity terms, applied
only at capture times (an uncaptured badger is not tested, so the factor is 1).
Depends on `model.thetas`, `model.rhos`, `model.phis` — and NOT on `model.etas`.

This is what gets passed to `epidemic_obs_loglik`, so that `etas` can stay in its
conjugate Gibbs block without being double-counted.
"""
function badger_obs_tests(model, data, X, i, t)
    w = ones(_obs_eltype(model), data.n_states)
    data.capture[t, i] == 0 && return w          # not seen: not tested
    for j in 1:size(data.tests, 3)
        x = data.tests[t, i, j]
        (x == 0 || x == 1) || continue           # not tested with this one
        θ, ρ, φ = model.thetas[j], model.rhos[j], model.phis[j]
        w[1] *= x == 1 ? (1 - φ) : φ             # S: specificity
        w[2] *= x == 1 ? (θ * ρ) : (1 - θ * ρ)   # E: reduced sensitivity
        w[3] *= x == 1 ? θ : (1 - θ)             # I: full sensitivity
    end
    w
end

# The weights must carry the parameter element type so this differentiates: under
# ForwardDiff `model.thetas` is a Vector{<:Dual}, and a Float64 `w` would silently
# drop the partials (or fail to convert). Mirrors the package's `_param_eltype`.
#
# Only the TEST parameters appear in `badger_obs_tests`, so only those are
# promoted here; `badger_obs_capture` follows `eltype(model.etas)` separately.
# Which of them is a Dual depends on the BLOCKING (etas is Float64 when
# conjugate, Dual when in an HMC block), so neither may be hard-coded.
_obs_eltype(model) = promote_type(eltype(model.thetas), eltype(model.rhos), eltype(model.phis))

## ---------------------------------------------------------------------------
## Scalar (allocation-free) counterparts, for `epidemic_obs_loglik`
## ---------------------------------------------------------------------------
#
# The likelihood needs ONE entry of the weight vector — the state the individual
# is actually in — so the vector-returning functions above allocate an array per
# (i,t) and discard all but one element. On the badger model that is ~187k array
# allocations per likelihood call, each of Duals under AD.
#
# These compute exactly that one entry. Same relationship `transition_prob` has
# to `transition_matrix_at` in the package: identical numbers, no allocation.
# Verified entry-for-entry against the vector versions in check_obs_split.jl.

"""
    badger_obs_tests_weight(model, data, X, i, t, s)

Scalar form of [`badger_obs_tests`](@ref): the test-factor weight for state `s`
alone, without building the weight vector.
"""
@inline function badger_obs_tests_weight(model, data, X, i, t, s)
    T = _obs_eltype(model)
    data.capture[t, i] == 0 && return one(T)     # not seen: not tested
    s == _D_CODE && return one(T)                # dead: test factor is 1
    w = one(T)
    @inbounds for j in 1:size(data.tests, 3)
        x = data.tests[t, i, j]
        (x == 0 || x == 1) || continue           # not tested with this one
        if s == _S_CODE
            φ = model.phis[j]
            w *= x == 1 ? (1 - φ) : φ            # S: specificity
        elseif s == _E_CODE
            θρ = model.thetas[j] * model.rhos[j]
            w *= x == 1 ? θρ : (1 - θρ)          # E: reduced sensitivity
        else                                     # _I_CODE
            θ = model.thetas[j]
            w *= x == 1 ? θ : (1 - θ)            # I: full sensitivity
        end
    end
    w
end

"""
    badger_obs_capture_weight(model, data, X, i, t, s)

Scalar form of [`badger_obs_capture`](@ref).
"""
@inline function badger_obs_capture_weight(model, data, X, i, t, s)
    eta = model.etas[data.season[t]]
    T = typeof(eta)
    if data.capture[t, i] == 0
        return s == _D_CODE ? one(T) : (1 - eta)
    else
        return s == _D_CODE ? zero(T) : eta      # seen alive: cannot be dead
    end
end

"""
    badger_obs_split_weight(model, data, X, i, t, s)

Scalar form of [`badger_observations_split`](@ref) — the product of the capture
and test factors at state `s`.
"""
@inline function badger_obs_split_weight(model, data, X, i, t, s)
    badger_obs_capture_weight(model, data, X, i, t, s) *
        badger_obs_tests_weight(model, data, X, i, t, s)
end

"""
    badger_observations_split(model, data, X, i, t)

The FULL observation process, as the product of the two factors above. Given to
`epidemic_data` as `observation_process` so the iFFBS forward filter still sees
the complete weights — only the LIKELIHOOD gets the reduced (test-only) factor.

Verified identical to `badger_observations` (examples/check_obs_split.jl).
"""
function badger_observations_split(model, data, X, i, t)
    badger_obs_capture(model, data, X, i, t) .* badger_obs_tests(model, data, X, i, t)
end

## ---------------------------------------------------------------------------
## Assembling it
## ---------------------------------------------------------------------------

"""
    badger_data_obssplit(dir; brock_changepoint=BROCK_CHANGEPOINT)

Same model and same fast `rest_contribution` coupling as
[`badger_data_reststotal`](@ref), but with the factored observation process, so
the observation likelihood can be split between `@addlogprob!` and the conjugate
`etas` block.
"""
function badger_data_obssplit(dir; brock_changepoint=BROCK_CHANGEPOINT)
    b = badger_data_reststotal(dir; brock_changepoint=brock_changepoint)
    d = b.raw

    # Rebuild with the split observation process. Everything else is identical to
    # badger_data_reststotal — same transitions, aggregates, summaries, coupling.
    affected = Matrix{Vector{Int}}(undef, d.n_timepoints, d.n_individuals)
    by_group = [Int[] for _ in 1:d.n_groups, _ in 1:d.n_timepoints]
    for t in 1:d.n_timepoints, i in 1:d.n_individuals
        g = d.social_group[i, t]
        g > 0 && push!(by_group[g, t], i)
    end
    for t in 1:d.n_timepoints, i in 1:d.n_individuals
        g = d.social_group[i, t]
        affected[t, i] = g == 0 ? Int[] : [j for j in by_group[g, t] if j != i]
    end

    aggregates = (; n_infectious=zeros(Int, d.n_groups, d.n_timepoints),
                    n_alive=zeros(Int, d.n_groups, d.n_timepoints),
                    nSE=zeros(Int, d.n_groups, d.n_timepoints),
                    nSS=zeros(Int, d.n_groups, d.n_timepoints))

    data = epidemic_data(;
        n_individuals=d.n_individuals,
        n_timepoints=d.n_timepoints,
        trans_mat=badger_transitions(),
        starting_state=badger_starting_state,
        observation_process=badger_observations_split,   # full product, for iFFBS
        aggregates=aggregates,
        derived_summaries=(_summary_n_infectious_rt, _summary_n_alive_rt,
                           _summary_nSE_rt, _summary_nSS_rt),
        rest_contribution=make_badger_rest_contribution(),
        sampling_period=d.sampling_period,
        affected_individuals=affected,
        coupled_transitions=[(:S, :E)],
        state_space=BADGER_STATES,
        social_group=d.social_group,
        age=d.age,
        capture=d.capture,
        capt_effort=d.capt_effort,
        tests=d.tests,
        season=d.season,
        birth_time=d.birth_time,
        last_capture_time=d.last_capture_time,
        nu_times=d.nu_times,
        K=Float64(d.K),
        k=d.k,
        n_groups=d.n_groups,
    )
    (; data, raw=d)
end
