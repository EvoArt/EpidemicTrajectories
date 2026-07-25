# Stage 4 — + custom O(n_states) coupling (the "reststotal" running total). The
# fully hand-optimized model. This is the same fit as badger_siler.jl.
#
# The default coupling (stages 1-3) recomputes, for each candidate focal state, how
# probable every affected neighbour's realised move is — O(n_states × |affected|)
# per (i,t). The reference instead keeps per-(group,time) running totals of the S→E
# and S→S moves (`nSE`, `nSS`) plus the group alive/infected counts, so the coupling
# weight is a closed form `nSE·log(foi) + nSS·log(1-foi)` in those totals —
# O(n_states) per (i,t), with no neighbour loop at all.
#
# The price is user code: two extra reversible summaries (`nSE`, `nSS`) and a custom
# `rest_contribution`. It stays in user code on purpose — the formula assumes a
# single coupled transition (S→E) with a per-group frequency-dependent FOI, a
# modelling choice the generic core does not bake in. `epidemic_data(;
# rest_contribution=…)` is the drop-in seam.

include(joinpath(@__DIR__, "badger_common.jl"))

# --- The reststotal coupling: FOI cases, extra summaries, custom rest term -----

const _CASE_SorE = 1
const _CASE_I = 2
const _CASE_D = 3

@inline function _foi_case(s)
    s == _I_CODE && return _CASE_I
    s == _D_CODE && return _CASE_D
    return _CASE_SorE
end

# The group FOI with the focal placed in a candidate state (the reference's Plus1).
@inline function _grp_foi(model, data, g, t, c, I_minus, M_minus)
    I, M = if c == _CASE_I
        I_minus + 1, M_minus + 1
    elseif c == _CASE_D
        I_minus, M_minus
    else
        I_minus, M_minus + 1
    end
    M == 0 && return 0.0
    lambda_g = model.lambda * model.alpha[g]
    foi = lambda_g + model.beta * I / ((M / data.K)^model.q)
    return -expm1(-foi)
end

function _summary_n_infectious_rt(model, data, X, s, i, t, reverse=false)
    g = data.social_group[i, t]
    g > 0 || return nothing
    contrib = (s == _I_CODE)
    reverse ? (data.aggregates.n_infectious[g, t] -= contrib) :
              (data.aggregates.n_infectious[g, t] += contrib)
    nothing
end

function _summary_n_alive_rt(model, data, X, s, i, t, reverse=false)
    g = data.social_group[i, t]
    g > 0 || return nothing
    contrib = (s != _D_CODE)
    reverse ? (data.aggregates.n_alive[g, t] -= contrib) :
              (data.aggregates.n_alive[g, t] += contrib)
    nothing
end

function _summary_nSE_rt(model, data, X, s, i, t, reverse=false)
    g = data.social_group[i, t]
    g > 0 || return nothing
    t == data.n_timepoints && return nothing
    contrib = (s == _S_CODE) && (X[t + 1, i] == _E_CODE)
    reverse ? (data.aggregates.nSE[g, t] -= contrib) :
              (data.aggregates.nSE[g, t] += contrib)
    nothing
end

function _summary_nSS_rt(model, data, X, s, i, t, reverse=false)
    g = data.social_group[i, t]
    g > 0 || return nothing
    t == data.n_timepoints && return nothing
    contrib = (s == _S_CODE) && (X[t + 1, i] == _S_CODE)
    reverse ? (data.aggregates.nSS[g, t] -= contrib) :
              (data.aggregates.nSS[g, t] += contrib)
    nothing
end

function make_badger_rest_contribution()
    function rest_contribution(model, data, X, i, t, n_states, affected_override=nothing)
        t == data.n_timepoints && return ones(n_states)
        g = data.social_group[i, t]
        g == 0 && return ones(n_states)
        I_minus = data.aggregates.n_infectious[g, t]
        M_minus = data.aggregates.n_alive[g, t]
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

# ------------------------------------------------------------------------------

function build(dir)
    d = load_badger_data(dir)
    # Plain-NamedTuple aggregates (verbose fallback) — the two @aggregate counts PLUS
    # the two reststotal running totals, with the four hand-written summaries.
    aggregates = (; n_infectious=zeros(Int, d.n_groups, d.n_timepoints),
                    n_alive=zeros(Int, d.n_groups, d.n_timepoints),
                    nSE=zeros(Int, d.n_groups, d.n_timepoints),
                    nSS=zeros(Int, d.n_groups, d.n_timepoints))
    data = epidemic_data(;
        n_individuals=d.n_individuals, n_timepoints=d.n_timepoints,
        trans_mat=badger_transitions(),
        starting_state=badger_starting_state,
        observation_process=badger_observations,
        aggregates=aggregates,
        derived_summaries=(_summary_n_infectious_rt, _summary_n_alive_rt,
                           _summary_nSE_rt, _summary_nSS_rt),
        rest_contribution=make_badger_rest_contribution(),   # <-- the O(n_states) coupling
        sampling_period=d.sampling_period,
        affected_individuals=badger_affected_lists(d),
        coupled_transitions=[(:S, :E)],
        state_space=BADGER_STATES,
        social_group=d.social_group, age=d.age, capture=d.capture,
        capt_effort=d.capt_effort, tests=d.tests, season=d.season,
        birth_time=d.birth_time, last_capture_time=d.last_capture_time,
        nu_times=d.nu_times, K=Float64(d.K), k=d.k, n_groups=d.n_groups)
    (; data, raw=d)
end

const STAGE = "reststotal"
const (data, raw) = build(DATA_DIR())

const loglik = epidemic_loglik(data; entry_time=raw.first_capture_time,
                               survival=siler_survival)
const obs_loglik = epidemic_obs_loglik(data;
                                 observation_process=badger_obs_tests,
                                 observation_weight=badger_obs_tests_weight)

if get(ENV, "BADGER_BENCH", "0") == "1"
    time_stage(STAGE, data, raw, loglik, obs_loglik)
elseif get(ENV, "BADGER_NO_RUN", "0") != "1"
    n = parse(Int, get(ENV, "BADGER_SWEEPS", "50"))
    println("[$STAGE] ", data.n_individuals, " badgers x ", data.n_timepoints, " t")
    chn, elapsed = run_badger_fit(data, raw, loglik, obs_loglik; n_sweeps=n)
    report(chn)
end
