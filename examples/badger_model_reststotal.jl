# The badger model again, coupling term replaced by a running total — the
# `rest_contribution` keyword's worked example, matching the reference
# implementation's own O(group)-per-sweep-refresh + O(1)-per-lookup structure
# instead of the default O(n_states x |affected|)-per-(i,t) recompute.
#
# HOW IT WORKS (mirrors badger_ref's logProbStoSgivenSorE / logProbRest /
# logProbRestTotal). Only S -> E is coupled, so a neighbour's realised move is
# sensitive to the focal only if that neighbour is SUSCEPTIBLE, and only through
# the group's (I, M) counts feeding the FOI. There are exactly three distinct
# (I, M) cases across the focal's four candidate states:
#
#   focal candidate | I (infectious in group)     | M (alive in group)
#   S               | I_minus                     | M_minus + 1
#   E               | I_minus                     | M_minus + 1   (same as S)
#   I               | I_minus + 1                 | M_minus + 1
#   D               | I_minus                     | M_minus       (dead: not alive)
#
# where I_minus / M_minus are the counts EXCLUDING the focal. So per (group, time)
# there are three FOI scenarios: SorE, I, D.
#
# The decomposition (derived and verified against the brute-force coupling term —
# see the block above `make_badger_rest_contribution` for the full reasoning): a
# susceptible neighbour's ONLY scenario-c-dependent contribution is log(foi_c) if
# it moves S->E, log(1-foi_c) if S->S; S->D and the individual survival factor are
# constant across c and cancel when rest_contribution normalises. So all that must
# be maintained is TWO INTEGER COUNTS per (group, time):
#
#   nSE[g, t] : number of susceptible individuals moving S->E
#   nSS[g, t] : number of susceptible individuals moving S->S
#
# These are ordinary reversible integer aggregates (like n_infectious/n_alive) —
# order-independent, no float staleness. The three scenario FOIs are cheap to
# recompute on demand from n_infectious/n_alive, so nothing else is cached.
#
# rest_contribution(i, t) then returns, per candidate s -> case c(s):
#   nSE * log(foi_c) + nSS * log(1 - foi_c)
# — O(n_states), no neighbour loop. Because iFFBS reverses the focal out of the
# aggregates before forward_filter, nSE/nSS already EXCLUDE the focal, so no
# self-subtraction is needed.
#
# (An earlier version cached a per-neighbour float `restRow` and a group float
# `restTotal`; it was fragile — a per-individual rebuild left the total reflecting
# stale counts, and it mis-scored S->D moves as S->S. The integer-count
# decomposition above is both simpler and correct.)

using EpidemicTrajectories
using Distributions
using Random

isdefined(@__MODULE__, :BADGER_STATES) || include(joinpath(@__DIR__, "badger_model.jl"))

## ---------------------------------------------------------------------------
## Constants
## ---------------------------------------------------------------------------

const _CASE_SorE = 1
const _CASE_I = 2
const _CASE_D = 3
const _N_CASES = 3

if !isdefined(@__MODULE__, :_S_CODE)
    const _S_CODE = findfirst(==(:S), BADGER_STATES)
    const _E_CODE = findfirst(==(:E), BADGER_STATES)
    const _I_CODE = findfirst(==(:I), BADGER_STATES)
    const _D_CODE = findfirst(==(:D), BADGER_STATES)
end

# Which of the three FOI scenarios candidate state code `s` induces.
@inline function _foi_case(s)
    s == _I_CODE && return _CASE_I
    s == _D_CODE && return _CASE_D
    return _CASE_SorE
end

# Group FOI (prob S->E over one step) under scenario `c`, from leave-focal-out
# counts. Same arithmetic as badger_infection.
@inline function _grp_foi(model, data, g, t, c, I_minus, M_minus)
    I, M = if c == _CASE_I
        I_minus + 1, M_minus + 1
    elseif c == _CASE_D
        I_minus, M_minus
    else
        I_minus, M_minus + 1
    end
    M == 0 && return 0.0
    α_g = model.lambda * model.alpha[g]
    foi = α_g + model.beta * I / ((M / data.K)^model.q)
    return -expm1(-foi)
end

## ---------------------------------------------------------------------------
## The decomposition that makes this O(1) per lookup, and correct
## ---------------------------------------------------------------------------
#
# A susceptible neighbour j's realised one-step move contributes
# log P(move | scenario c) to candidate c's total. The MOVES a susceptible
# individual can make and their scenario-c dependence:
#
#   S -> E : prob is (survival_j) * foi_c        -> c-dependent part is log(foi_c)
#   S -> S : prob is (survival_j) * (1 - foi_c)  -> c-dependent part is log(1 - foi_c)
#   S -> D : prob is (1 - survival_j)            -> DOES NOT depend on c at all
#
# (S -> I is impossible in one step; a badger must pass through E.) The survival
# factor is individual and age-dependent, NOT coupled to the focal — so it is
# constant across the focal's candidate states c, and cancels when
# rest_contribution normalises (`logw .-= maximum(logw)`). Likewise the S -> D
# term is entirely c-independent and cancels. So the ONLY thing that survives
# normalisation, per group and time, is:
#
#   restTotal_c = nSE * log(foi_c) + nSS * log(1 - foi_c)
#
# where nSE / nSS are just COUNTS of susceptible neighbours moving S->E / S->S.
# Those counts are order-independent integer aggregates — a susceptible individual
# doing S->E increments nSE, S->S increments nSS — so they maintain robustly under
# @aggregate's reverse/forward, with none of the staleness a rebuilt float total
# had. The three grpFOI scenarios are cheap to recompute on demand (3 numbers), so
# they aren't cached at all; rest_contribution computes them from the current
# leave-focal-out counts.

# nSE / nSS: susceptible-neighbour move counts per (group, time). Reversible
# integer aggregates, exactly like n_infectious / n_alive.
function _summary_nSE_rt(model, data, X, s, i, t; reverse=false)
    g = data.social_group[i, t]
    g > 0 || return nothing
    t == data.n_timepoints && return nothing
    # `s` is the candidate state being applied to i at t; count i as an S->E mover
    # only if that candidate is S AND i's realised NEXT state is E.
    contrib = (s == _S_CODE) && (X[t + 1, i] == _E_CODE)
    reverse ? (data.aggregates.nSE[g, t] -= contrib) :
              (data.aggregates.nSE[g, t] += contrib)
    nothing
end
function _summary_nSS_rt(model, data, X, s, i, t; reverse=false)
    g = data.social_group[i, t]
    g > 0 || return nothing
    t == data.n_timepoints && return nothing
    contrib = (s == _S_CODE) && (X[t + 1, i] == _S_CODE)
    reverse ? (data.aggregates.nSS[g, t] -= contrib) :
              (data.aggregates.nSS[g, t] += contrib)
    nothing
end

## ---------------------------------------------------------------------------
## The custom rest_contribution
## ---------------------------------------------------------------------------

function make_badger_rest_contribution()
    function rest_contribution(model, data, X, i, t, n_states, affected_override=nothing)
        t == data.n_timepoints && return ones(n_states)
        g = data.social_group[i, t]
        g == 0 && return ones(n_states)

        # Leave-focal-out counts (the focal was reversed out before forward_filter).
        I_minus = data.aggregates.n_infectious[g, t]
        M_minus = data.aggregates.n_alive[g, t]

        # nSE / nSS include EVERY susceptible individual currently in the group,
        # which — since the focal was reversed out of the aggregates — already
        # EXCLUDES the focal. So no self-subtraction is needed here (unlike the
        # float version): the focal simply isn't in these counts.
        nSE = data.aggregates.nSE[g, t]
        nSS = data.aggregates.nSS[g, t]

        logw = zeros(Float64, n_states)
        @inbounds for s in 1:n_states
            c = _foi_case(s)
            foi = _grp_foi(model, data, g, t, c, I_minus, M_minus)
            logw[s] = nSE * log(max(foi, 1e-12)) + nSS * log(max(1.0 - foi, 1e-12))
        end
        logw .-= maximum(logw)
        exp.(logw)
    end
    rest_contribution
end

## ---------------------------------------------------------------------------
## Assembling it
## ---------------------------------------------------------------------------

"""
    badger_data_reststotal(dir; brock_changepoint=BROCK_CHANGEPOINT)

Same model as [`badger_data`](@ref) but with the coupling term computed from a
running total (see the top of this file). Identical posterior; faster iFFBS sweep.
"""
function badger_data_reststotal(dir; brock_changepoint=BROCK_CHANGEPOINT)
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

    # All small integer/scalar aggregates — no per-individual float arrays, no
    # `members` list. nSE/nSS are the susceptible-neighbour move counts the
    # decomposition above needs.
    aggregates = (; n_infectious=zeros(Int, d.n_groups, d.n_timepoints),
                    n_alive=zeros(Int, d.n_groups, d.n_timepoints),
                    nSE=zeros(Int, d.n_groups, d.n_timepoints),
                    nSS=zeros(Int, d.n_groups, d.n_timepoints))

    epidemic_data(;
        n_individuals=d.n_individuals,
        n_timepoints=d.n_timepoints,
        trans_mat=badger_transitions(),
        starting_state=badger_starting_state,
        observation_process=badger_observations,
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
    ) |> data -> (; data, raw=d)
end

# n_infectious / n_alive count summaries, hand-written (verbose fallback, since we
# mix them with the nSE/nSS summaries). Same as badger_model_foicache.jl's.
function _summary_n_infectious_rt(model, data, X, s, i, t; reverse=false)
    g = data.social_group[i, t]
    g > 0 || return nothing
    contrib = (s == _I_CODE)
    reverse ? (data.aggregates.n_infectious[g, t] -= contrib) :
              (data.aggregates.n_infectious[g, t] += contrib)
    nothing
end
function _summary_n_alive_rt(model, data, X, s, i, t; reverse=false)
    g = data.social_group[i, t]
    g > 0 || return nothing
    contrib = (s != _D_CODE)
    reverse ? (data.aggregates.n_alive[g, t] -= contrib) :
              (data.aggregates.n_alive[g, t] += contrib)
    nothing
end
