# The badger model again, but with the force of infection CACHED per (group, time)
# — a power-user variant, written to measure whether caching it is worth it.
#
# The base model (badger_model.jl) recomputes the FOI inside `badger_infection` for
# every individual, even though it depends only on the individual's GROUP and TIME:
# with 2384 badgers over 34 groups, the same `(g, t)` FOI is recomputed ~70 times
# over. That redundancy sits inside the coupling term, which profiling puts at ~70%
# of an iFFBS sweep (rest_contribution -> neighbor_logprob -> transition_prob).
#
# WHERE THE CACHE IS ALLOWED TO GO, and why it is only half the model:
#
#   trans_mat          -> badger_infection         (honest: recomputes from params)
#   coupling_trans_mat -> badger_infection_cached  (reads foi[g,t])
#
# The cached rate goes to `coupling_trans_mat` ONLY. `trans_mat` is what
# `epidemic_loglik` differentiates for the HMC gradient, and a Float64 FOI computed
# outside the AD call is a CONSTANT to the AD backend. Measured, on this very file
# when it (wrongly) used the cached rate for both: the gradient w.r.t.
# lambda/alpha/beta/q collapsed from +300.8/-195.2/-20.9/+26.0 to ~±1 — the PRIOR
# gradient, with the likelihood's contribution gone — while the log-density stayed
# bit-identical (-81919.0229379290 either way). Nothing warns you; the fit just
# never moves those four parameters. `coupling_trans_mat` is safe from this
# structurally: the gradient never reaches the coupling term.
#
# Note the counts (n_infectious/n_alive) are NOT the same hazard: they are Int
# functions of X, genuinely constant w.r.t. the parameters, so the base model reads
# them inside the differentiated rate quite safely. It is caching a
# PARAMETER-DERIVED quantity that breaks AD.
#
# Everything here is still user code — the package supplies the seam
# (`coupling_trans_mat`) and knows nothing about FOI or groups. It uses the VERBOSE
# fallback rather than @aggregate because @aggregate's updates must be reversible
# arithmetic (`+=`/`-=` pairs), and an FOI cache is not accumulated — it is
# RECOMPUTED from the counts. That is the case the fallback exists for.
#
# Why the cache needs no invalidation: the summary recomputes from whatever `model`
# it is handed, so it is always consistent with the caller's current parameters.
# There is no stale window, even though the parameters change every Gibbs sweep
# while the counts change within one.

using EpidemicTrajectories
using Distributions
using Random

# Reuses badger_model.jl's own pieces (BADGER_STATES, siler_survival,
# badger_progression, badger_starting_state, badger_observations, load_badger_data)
# — only the S -> E rate and the aggregate bookkeeping differ. Include it first if
# it isn't already loaded; `isdefined` keeps this safe when a caller (e.g.
# badger_fit.jl, or bench_gradient.jl via it) has already pulled it in.
isdefined(@__MODULE__, :BADGER_STATES) || include(joinpath(@__DIR__, "badger_model.jl"))

## ---------------------------------------------------------------------------
## The cached rate
## ---------------------------------------------------------------------------

# Reads the FOI the summaries maintain, instead of recomputing it from the counts
# and parameters. Compare badger_model.jl's `badger_infection`, which does the
# arithmetic per individual.
function badger_infection_cached(model, data, i, t)
    g = data.social_group[i, t]
    g == 0 && return 0.0                       # not present: no exposure
    return data.aggregates.foi[g, t]
end

# The coupling term's spec. Identical to badger_transitions() except that S -> E
# reads the cached FOI. NOT for `trans_mat` — see the warning at the top.
function badger_transitions_cached()
    @transitions BADGER_STATES begin
        @survival siler_survival death=:D
        S -> E = badger_infection_cached
        E -> I = badger_progression
    end
end

## ---------------------------------------------------------------------------
## The summaries: two reversible counts, then the FOI recomputed from them
## ---------------------------------------------------------------------------

# The count updates, hand-written to match what @aggregate would generate for
# badger_model.jl's `n_infectious`/`n_alive` (a `+=` forward, a `-=` reverse).
const _S_CODE = findfirst(==(:S), BADGER_STATES)
const _I_CODE = findfirst(==(:I), BADGER_STATES)
const _D_CODE = findfirst(==(:D), BADGER_STATES)

function _summary_n_infectious(model, data, X, s, i, t; reverse=false)
    g = data.social_group[i, t]
    g > 0 || return nothing
    contrib = (s == _I_CODE)
    if reverse
        data.aggregates.n_infectious[g, t] -= contrib
    else
        data.aggregates.n_infectious[g, t] += contrib
    end
    nothing
end

function _summary_n_alive(model, data, X, s, i, t; reverse=false)
    g = data.social_group[i, t]
    g > 0 || return nothing
    contrib = (s != _D_CODE)
    if reverse
        data.aggregates.n_alive[g, t] -= contrib
    else
        data.aggregates.n_alive[g, t] += contrib
    end
    nothing
end

# The cache itself. Recomputed (not accumulated) from the counts the two summaries
# above have just updated, so `reverse` does the same work as forward — after
# either, `foi[g,t]` agrees with `n_infectious[g,t]`/`n_alive[g,t]` and `model`.
#
# Only the ONE group this individual is in at this time can have changed, so this
# is O(1) per call, not O(n_groups): that is the whole point, since the alternative
# (`badger_infection`) is O(1) per INDIVIDUAL and there are ~70 individuals per
# group.
#
# Must be ordered AFTER the count summaries in `derived_summaries` — it reads what
# they write.
function _summary_foi(model, data, X, s, i, t; reverse=false)
    g = data.social_group[i, t]
    g > 0 || return nothing
    I = data.aggregates.n_infectious[g, t]
    M = data.aggregates.n_alive[g, t]
    if M == 0
        data.aggregates.foi[g, t] = 0.0
        return nothing
    end
    α_g = model.lambda * model.alpha[g]
    foi = α_g + model.beta * I / ((M / data.K)^model.q)
    data.aggregates.foi[g, t] = -expm1(-foi)   # 1 - exp(-foi), as badger_infection
    nothing
end

## ---------------------------------------------------------------------------
## Assembling it
## ---------------------------------------------------------------------------

"""
    badger_data_foicache(dir; brock_changepoint=BROCK_CHANGEPOINT)

Same as [`badger_data`](@ref) but with the FOI cached per `(group, time)`. See the
comments at the top of this file.
"""
function badger_data_foicache(dir; brock_changepoint=BROCK_CHANGEPOINT)
    d = load_badger_data(dir; brock_changepoint=brock_changepoint)

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

    # The verbose fallback: our own storage, our own summaries. `foi` is Float64
    # because it holds a probability, not a count.
    aggregates = (; n_infectious=zeros(Int, d.n_groups, d.n_timepoints),
                    n_alive=zeros(Int, d.n_groups, d.n_timepoints),
                    foi=zeros(Float64, d.n_groups, d.n_timepoints))

    data = epidemic_data(;
        n_individuals=d.n_individuals,
        n_timepoints=d.n_timepoints,
        # The whole point: the honest rate is differentiated, the cached one is
        # only ever reached by the coupling term. Swapping these silently breaks
        # the gradient — see the top of this file.
        trans_mat=badger_transitions(),
        coupling_trans_mat=badger_transitions_cached(),
        starting_state=badger_starting_state,
        observation_process=badger_observations,
        aggregates=aggregates,
        derived_summaries=(_summary_n_infectious, _summary_n_alive, _summary_foi),
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
