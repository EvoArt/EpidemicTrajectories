
using EpidemicTrajectories
using PracticalBayes
using Distributions
using Random
using AdvancedHMC: HMC, Leapfrog, DiagEuclideanMetric
using ADTypes: AutoPolyesterForwardDiff
using PolyesterForwardDiff
using StableRNGs: StableRNG
using Statistics: mean, std, median
using Dates
using JLD2: @save
using CSV
using DataFrames
import AbstractMCMC

const ADTYPE = AutoPolyesterForwardDiff(; chunksize=nothing, tag=nothing)


const BADGER_STATES = [:S, :E, :I, :D]
REF_STATE_CODE = Dict(0 => 1, 3 => 2, 1 => 3, 9 => 4)
BROCK_CHANGEPOINT_RAW = 80
BROCK_CHANGEPOINT = 101

function load_badger_data(dir; brock_changepoint=BROCK_CHANGEPOINT, known_sex_only=true)
    rd(f) = CSV.read(joinpath(dir, f), DataFrame)

    dims = rd("dimensions.csv")
    m, maxt = dims[1, :m], dims[1, :maxt]
    n_groups, n_tests = dims[1, :G], dims[1, :numTests]
    n_seasons, n_nu_times = dims[1, :numSeasons], dims[1, :numNuTimes]

    Xinit_raw = Matrix(rd("Xinit.csv"))
    test_mat = Matrix{Float64}(rd("TestMat.csv"))
    capt_hist = Matrix{Int}(rd("CaptHist.csv"))
    capt_effort = Matrix{Int}(rd("CaptEffort.csv"))
    birth_time = vec(Matrix{Int}(rd("birthTimes.csv")))
    start_period = vec(Matrix{Int}(rd("startSamplingPeriod.csv")))
    end_period = vec(Matrix{Int}(rd("endSamplingPeriod.csv")))
    nu_times = vec(Matrix{Int}(rd("nuTimes.csv")))
    sex_raw = vec(Matrix{Int}(rd("sex.csv")))
    K = rd("Kay.csv")[1, :K]
    k = rd("k.csv")[1, :k]

    keep = known_sex_only ? findall(!=(0), sex_raw) : collect(1:m)
    old_to_new = zeros(Int, m)
    for (new_i, old_i) in enumerate(keep)
        old_to_new[old_i] = new_i
    end

    Xinit_raw = Xinit_raw[keep, :]
    capt_hist = capt_hist[keep, :]
    birth_time = birth_time[keep]
    start_period = start_period[keep]
    end_period = end_period[keep]
    sex = Int[sex_raw[i] == 1 ? 1 : 0 for i in keep]

    kept_rows = [old_to_new[Int(test_mat[r, 2])] != 0 for r in 1:size(test_mat, 1)]
    test_mat = test_mat[kept_rows, :]
    for r in 1:size(test_mat, 1)
        test_mat[r, 2] = old_to_new[Int(test_mat[r, 2])]
    end
    m = length(keep)

    X_init = Matrix{Int}(undef, maxt, m)
    for i in 1:m, t in 1:maxt
        code = Xinit_raw[i, t]
        X_init[t, i] = code == -10 ? 1 : REF_STATE_CODE[Int(code)]
    end

    social_group = zeros(Int, m, maxt)
    for i in 1:m
        rows = findall(==(Float64(i)), @view test_mat[:, 2])
        isempty(rows) && continue
        times_i = test_mat[rows, 1]
        groups_i = test_mat[rows, 3]
        g = Int(groups_i[argmin(times_i)])
        for t in max(1, birth_time[i]):maxt
            at_t = findfirst(==(Float64(t)), times_i)
            at_t === nothing || (g = Int(groups_i[at_t]))
            social_group[i, t] = g
        end
    end

    age = fill(-10, m, maxt)
    for i in 1:m, t in max(1, birth_time[i]):maxt
        age[i, t] = t - birth_time[i]
    end

    tests = fill(-1, maxt, m, n_tests)
    for r in 1:size(test_mat, 1)
        t = Int(test_mat[r, 1])
        i = Int(test_mat[r, 2])
        (1 <= t <= maxt && 1 <= i <= m) || continue
        for j in 1:n_tests
            v = test_mat[r, 3 + j]
            (isnan(v) || v == -10) && continue
            tests[t, i, j] = Int(v)
        end
    end
    tests = apply_brock_changepoint!(tests, brock_changepoint)

    capture = permutedims(capt_hist, (2, 1))

    last_capture_time = [findlast(==(1), @view capture[:, i]) for i in 1:m]
    last_capture_time = [lc === nothing ? 0 : lc for lc in last_capture_time]

    first_capture_time = [findfirst(==(1), @view capture[:, i]) for i in 1:m]
    first_capture_time = [fc === nothing ? max(birth_time[i], 1) : fc for (i, fc) in enumerate(first_capture_time)]

    cam_raw = Matrix{Int}(rd("capturesAfterMonit.csv"))
    cam_rows = Tuple{Int,Int}[]
    for r in 1:size(cam_raw, 1)
        oid = cam_raw[r, 1]
        (1 <= oid <= length(old_to_new) && old_to_new[oid] != 0) || continue
        nid = old_to_new[oid]
        tcap = cam_raw[r, 2]
        push!(cam_rows, (nid, tcap))
        last_capture_time[nid] = max(last_capture_time[nid], tcap)
    end

    season = make_season_vec(n_seasons, 1, maxt)
    sampling_period = [(start_period[i], end_period[i]) for i in 1:m]

    return (; n_individuals=m, n_timepoints=maxt, n_groups, n_tests, n_seasons,
            n_nu_times, X_init, social_group, age, capture, capt_effort, tests,
            sampling_period, birth_time, last_capture_time, first_capture_time,
            captures_after_monit=cam_rows, season, nu_times, sex, K, k)
end

function apply_brock_changepoint!(tests, changepoint;
                                  raw=BROCK_CHANGEPOINT_RAW, brock1=1, brock2=2)
    changepoint == raw && return tests
    window = changepoint > raw ? (raw:(changepoint - 1)) : (changepoint:(raw - 1))
    for t in window
        1 <= t <= size(tests, 1) || continue
        for i in axes(tests, 2)
            tests[t, i, brock1], tests[t, i, brock2] = tests[t, i, brock2], tests[t, i, brock1]
        end
    end
    tests
end

function make_season_vec(n_seasons, season_start, maxt)
    v = ones(Int, maxt)
    v[1] = season_start
    for t in 2:maxt
        v[t] = v[t-1] < n_seasons ? v[t-1] + 1 : 1
    end
    v
end


function badger_aggregates(n_groups, n_timepoints)
    @aggregate BADGER_STATES begin
        @array n_infectious Int (n_groups, n_timepoints)
        @array n_alive Int (n_groups, n_timepoints)
        if data.social_group[i, t] > 0
            n_infectious[data.social_group[i, t], t] += (state == :I)
        end
        if data.social_group[i, t] > 0
            n_alive[data.social_group[i, t], t] += (state != :D)
        end
    end
end

function badger_infection(model, data, i, t)
    g = data.social_group[i, t]
    g == 0 && return 0.0
    I = data.aggregates.n_infectious[g, t]
    M = data.aggregates.n_alive[g, t]
    M == 0 && return 0.0
    lambda_g = model.lambda * model.alpha[g]
    foi = lambda_g + model.beta * I / ((M / data.K)^model.q)
    return -expm1(-foi)
end

function erlang_cdf_at_1(k, tau)
    return 1.0 - exp(-tau) * sum((tau^j) / factorial(j) for j in 0:(k - 1))
end

badger_progression(model, data, i, t) = erlang_cdf_at_1(data.k, model.tau / data.k)

function siler_survival(model, data, i, t)
    age = t >= 1 ? data.age[i, t] : data.age[i, 1] + (t - 1)
    age < 0 && return 1.0
    a1, b1, a2, b2, c1 = model.a1, model.b1, model.a2, model.b2, model.c1
    y1 = b2 * (age - 1); y2 = b2 * age
    late = -exp(y1) * expm1(y2 - y1)
    z1 = -b1 * (age - 1); z2 = -b1 * age
    early = exp(z1) * expm1(z2 - z1)
    s = exp(-c1 + (a2 / b2) * late + (a1 / b1) * early)
    return t <= data.last_capture_time[i] ? 1.0 : s
end

function badger_transitions()
    @transitions BADGER_STATES begin
        @survival siler_survival death=:D
        S -> E = badger_infection
        E -> I = badger_progression
    end
end

function badger_starting_state(model, data, X, i, t)
    p = zeros(Float64, data.n_states)
    start_time = data.sampling_period[i][1]
    nuE = nuI = 0.0
    if data.birth_time[i] < start_time
        idx = findfirst(==(start_time), data.nu_times)
        if idx !== nothing
            nuE, nuI = model.nu[idx, 1], model.nu[idx, 2]
        end
    end
    p[1] = 1.0 - nuE - nuI
    p[2] = nuE
    p[3] = nuI
    p[4] = 0.0
    p
end

function badger_observations(model, data, X, i, t)
    w = ones(Float64, data.n_states)
    eta = model.etas[data.season[t]]
    if data.capture[t, i] == 0
        w[1] = w[2] = w[3] = 1 - eta
        w[4] = 1.0
        return w
    end
    w[1] = w[2] = w[3] = eta
    w[4] = 0.0
    for j in 1:size(data.tests, 3)
        x = data.tests[t, i, j]
        (x == 0 || x == 1) || continue
        theta, rho, phi = model.thetas[j], model.rhos[j], model.phis[j]
        w[1] *= x == 1 ? (1 - phi) : phi
        w[2] *= x == 1 ? (theta * rho) : (1 - theta * rho)
        w[3] *= x == 1 ? theta : (1 - theta)
    end
    w
end

function badger_data(dir; brock_changepoint=BROCK_CHANGEPOINT)
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
    data = epidemic_data(;
        n_individuals=d.n_individuals,
        n_timepoints=d.n_timepoints,
        trans_mat=badger_transitions(),
        starting_state=badger_starting_state,
        observation_process=badger_observations,
        aggregates=badger_aggregates(d.n_groups, d.n_timepoints),
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

function badger_initial_params(d; rng=Random.default_rng())
    (; tau=5.0,
       alpha=fill(0.001, d.n_groups),
       lambda=rand(rng, Gamma(1, 1 / 100)),
       beta=rand(rng, Gamma(1, 1 / 100)),
       q=rand(rng, Beta(1, 1)),
       c1=rand(rng, Gamma(1, 1 / 100)),
       a1=rand(rng, Gamma(1, 1 / 100)),
       b1=rand(rng, Gamma(1, 1 / 100)),
       a2=rand(rng, Gamma(1, 1 / 100)),
       b2=rand(rng, Gamma(1, 1 / 100)),
       thetas=rand(rng, Uniform(0.5, 1), d.n_tests),
       rhos=rand(rng, Uniform(0.2, 0.8), d.n_tests),
       phis=rand(rng, Uniform(0.7, 1), d.n_tests),
       etas=rand(rng, Beta(1, 1), d.n_seasons),
       nuE=[p[2] for p in eachcol(rand(rng, Dirichlet([8.0, 1.0, 1.0]), d.n_nu_times))],
       nuI=[p[3] for p in eachcol(rand(rng, Dirichlet([8.0, 1.0, 1.0]), d.n_nu_times))])
end


function gompertz_makeham_survival(model, data, i, t)
    age = t >= 1 ? data.age[i, t] : data.age[i, 1] + (t - 1)
    age < 0 && return 1.0
    a2, b2, c1 = model.a2, model.b2, model.c1
    y1 = b2 * (age - 1); y2 = b2 * age
    late = -exp(y1) * expm1(y2 - y1)
    s = exp(-c1 + (a2 / b2) * late)
    return t <= data.last_capture_time[i] ? 1.0 : s
end

function badger_transitions_gompertz()
    @transitions BADGER_STATES begin
        @survival gompertz_makeham_survival death=:D
        S -> E = badger_infection
        E -> I = badger_progression
    end
end


function erlang_cdf_at_1_meantime(k, theta)
    x_over_theta = 1.0 / theta
    exp_term = exp(-x_over_theta)
    sum_val = one(x_over_theta)
    term = one(x_over_theta)
    for i in 1:(k - 1)
        term *= x_over_theta / i
        sum_val += term
    end
    return 1.0 - sum_val * exp_term
end

badger_progression_meantime(model, data, i, t) =
    erlang_cdf_at_1_meantime(data.k, model.tau / data.k)

function gompertz_makeham_survival_tp1(model, data, i, t)
    tt = t + 1
    age = tt <= data.n_timepoints ? data.age[i, tt] : data.age[i, data.n_timepoints] + (tt - data.n_timepoints)
    age < 0 && return 1.0
    a2, b2, c1 = model.a2, model.b2, model.c1
    y1 = b2 * (age - 1); y2 = b2 * age
    late = -exp(y1) * expm1(y2 - y1)
    return exp(-c1 + (a2 / b2) * late)
end

function siler_survival_tp1(model, data, i, t)
    tt = t + 1
    age = tt <= data.n_timepoints ? data.age[i, tt] : data.age[i, data.n_timepoints] + (tt - data.n_timepoints)
    age < 0 && return 1.0
    a1, b1, a2, b2, c1 = model.a1, model.b1, model.a2, model.b2, model.c1
    y1 = b2 * (age - 1); y2 = b2 * age
    late = -exp(y1) * expm1(y2 - y1)
    z1 = -b1 * (age - 1); z2 = -b1 * age
    early = exp(z1) * expm1(z2 - z1)
    return exp(-c1 + (a2 / b2) * late + (a1 / b1) * early)
end

function badger_transitions_meantime_siler_tp1()
    @transitions BADGER_STATES begin
        @survival siler_survival_tp1 death=:D
        S -> E = badger_infection
        E -> I = badger_progression_meantime
    end
end

function badger_transitions_meantime_gompertz()
    @transitions BADGER_STATES begin
        @survival gompertz_makeham_survival_tp1 death=:D
        S -> E = badger_infection
        E -> I = badger_progression_meantime
    end
end

function badger_obs_capture_deathban(model, data, X, i, t)
    w = ones(eltype(model.etas), data.n_states)
    eta = model.etas[data.season[t]]
    seen_alive = t <= data.last_capture_time[i]
    if data.capture[t, i] == 0
        w[1] = w[2] = w[3] = 1 - eta
        w[4] = seen_alive ? zero(eltype(w)) : one(eltype(w))
    else
        w[1] = w[2] = w[3] = eta
        w[4] = zero(eltype(w))
    end
    w
end

function badger_observations_deathban(model, data, X, i, t)
    badger_obs_capture_deathban(model, data, X, i, t) .* badger_obs_tests(model, data, X, i, t)
end


const _CASE_SorE = 1
const _CASE_I = 2
const _CASE_D = 3
const _N_CASES = 3
const _S_CODE = findfirst(==(:S), BADGER_STATES)
const _E_CODE = findfirst(==(:E), BADGER_STATES)
const _I_CODE = findfirst(==(:I), BADGER_STATES)
const _D_CODE = findfirst(==(:D), BADGER_STATES)

@inline function _foi_case(s)
    s == _I_CODE && return _CASE_I
    s == _D_CODE && return _CASE_D
    return _CASE_SorE
end

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


function badger_obs_capture(model, data, X, i, t)
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

function badger_obs_tests(model, data, X, i, t)
    w = ones(_obs_eltype(model), data.n_states)
    data.capture[t, i] == 0 && return w
    for j in 1:size(data.tests, 3)
        x = data.tests[t, i, j]
        (x == 0 || x == 1) || continue
        theta, rho, phi = model.thetas[j], model.rhos[j], model.phis[j]
        w[1] *= x == 1 ? (1 - phi) : phi
        w[2] *= x == 1 ? (theta * rho) : (1 - theta * rho)
        w[3] *= x == 1 ? theta : (1 - theta)
    end
    w
end

_obs_eltype(model) = promote_type(eltype(model.thetas), eltype(model.rhos), eltype(model.phis))

@inline function badger_obs_tests_weight(model, data, X, i, t, s)
    T = _obs_eltype(model)
    data.capture[t, i] == 0 && return one(T)
    s == _D_CODE && return one(T)
    w = one(T)
    @inbounds for j in 1:size(data.tests, 3)
        x = data.tests[t, i, j]
        (x == 0 || x == 1) || continue
        if s == _S_CODE
            phi = model.phis[j]
            w *= x == 1 ? (1 - phi) : phi
        elseif s == _E_CODE
            theta_rho = model.thetas[j] * model.rhos[j]
            w *= x == 1 ? theta_rho : (1 - theta_rho)
        else
            theta = model.thetas[j]
            w *= x == 1 ? theta : (1 - theta)
        end
    end
    w
end

@inline function badger_obs_capture_weight(model, data, X, i, t, s)
    eta = model.etas[data.season[t]]
    T = typeof(eta)
    if data.capture[t, i] == 0
        return s == _D_CODE ? one(T) : (1 - eta)
    else
        return s == _D_CODE ? zero(T) : eta
    end
end

@inline function badger_obs_split_weight(model, data, X, i, t, s)
    badger_obs_capture_weight(model, data, X, i, t, s) *
        badger_obs_tests_weight(model, data, X, i, t, s)
end

function badger_observations_split(model, data, X, i, t)
    badger_obs_capture(model, data, X, i, t) .* badger_obs_tests(model, data, X, i, t)
end

function badger_data_obssplit(dir; brock_changepoint=BROCK_CHANGEPOINT,
                              trans_mat=badger_transitions(),
                              observation_process=badger_observations_split)
    b = badger_data_reststotal(dir; brock_changepoint=brock_changepoint)
    d = b.raw
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
        trans_mat=trans_mat,
        starting_state=badger_starting_state,
        observation_process=observation_process,
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


DATA_DIR = let
    base1 = joinpath(pwd(), "badger_ref", "RData2")
    base2 = joinpath(pwd(), "..", "badger_ref", "RData2")
    isdir(base1) ? base1 : base2
end

const (data, raw) = badger_data_obssplit(DATA_DIR;
                         trans_mat=badger_transitions_meantime_siler_tp1(),
                         observation_process=badger_observations_deathban)
const G = raw.n_groups
const NT = raw.n_tests
const NS = raw.n_seasons
const NNU = raw.n_nu_times

println("Badger base model (reststotal coupling): ", data.n_individuals, " badgers x ",
        data.n_timepoints, " timepoints x ", G, " groups, ", NT, " tests")
println("Brock changepoint fixed at t=", BROCK_CHANGEPOINT, " (not inferred)")

const loglik = epidemic_loglik(data; entry_time=raw.first_capture_time,
                               survival=siler_survival_tp1)
const obs_loglik = epidemic_obs_loglik(data;
                                 observation_process=badger_obs_tests,
                                 observation_weight=badger_obs_tests_weight)

const latent! = epidemic_latent_sampler(data)

struct TrajectoryLatent <: Distributions.DiscreteMatrixDistribution
    n_time::Int
    n_ind::Int
end
Base.size(d::TrajectoryLatent) = (d.n_time, d.n_ind)
Distributions.logpdf(::TrajectoryLatent, X::AbstractMatrix) = 0.0
Distributions.rand(rng::AbstractRNG, d::TrajectoryLatent) = fill(1, d.n_time, d.n_ind)

_pars(c) = (; tau=c.values.tau, alpha=c.values.alpha, lambda=c.values.lambda,
            beta=c.values.beta, q=c.values.q,
            a1=c.values.a1, b1=c.values.b1, a2=c.values.a2, b2=c.values.b2, c1=c.values.c1,
            thetas=c.values.thetas, rhos=c.values.rhos, phis=c.values.phis,
            etas=c.values.etas, nu=c.values.nu)

struct iFFBSKernel{F} <: PracticalBayes.AbstractLatentKernel
    latent!::F
end
PracticalBayes.latent_step(rng, k::iFFBSKernel, block_names, c::PracticalBayes.ModelConditional) = begin
    X = copy(c.values.X)
    k.latent!(rng, _pars(c), X)
    (; X=X)
end

struct EtaKernel{D} <: PracticalBayes.AbstractLatentKernel
    a::Float64
    b::Float64
    data::D
end
EtaKernel(a, b) = EtaKernel(a, b, data)

function PracticalBayes.latent_step(rng, k::EtaKernel, block_names, c::PracticalBayes.ModelConditional)
    X = c.values.X
    d = k.data
    caught = zeros(Int, NS)
    available = zeros(Int, NS)
    @inbounds for i in 1:d.n_individuals
        for t in 1:d.n_timepoints
            g = d.social_group[i, t]
            (g > 0 && d.capt_effort[g, t] == 1) || continue
            X[t, i] == 4 && continue
            s = d.season[t]
            available[s] += 1
            caught[s] += d.capture[t, i] == 1
        end
    end
    etas = zeros(Float64, NS)
    for s in 1:NS
        etas[s] = rand(rng, Beta(k.a + caught[s], k.b + max(available[s] - caught[s], 0)))
    end
    (; etas=etas)
end

struct NuKernel{D} <: PracticalBayes.AbstractLatentKernel
    hp::Vector{Float64}
    data::D
end
NuKernel(hp) = NuKernel(hp, data)

function PracticalBayes.latent_step(rng, k::NuKernel, block_names, c::PracticalBayes.ModelConditional)
    X = c.values.X
    d = k.data
    nu = Matrix{Float64}(undef, NNU, 2)
    for (idx, nt) in enumerate(d.nu_times)
        counts = zeros(Int, 3)
        @inbounds for i in 1:d.n_individuals
            start_time = d.sampling_period[i][1]
            (start_time == nt && d.birth_time[i] < start_time) || continue
            s = X[nt, i]
            s <= 3 && (counts[s] += 1)
        end
        p = rand(rng, Dirichlet(counts .+ k.hp))
        nu[idx, 1], nu[idx, 2] = p[2], p[3]
    end
    (; nu=nu)
end

struct NuSimplex <: Distributions.DiscreteMatrixDistribution
    n_nu::Int
    hp::Vector{Float64}
end
Base.size(d::NuSimplex) = (d.n_nu, 2)
Distributions.logpdf(::NuSimplex, nu::AbstractMatrix) = 0.0
function Distributions.rand(rng::AbstractRNG, d::NuSimplex)
    nu = Matrix{Float64}(undef, d.n_nu, 2)
    for i in 1:d.n_nu
        p = rand(rng, Dirichlet(d.hp))
        nu[i, 1], nu[i, 2] = p[2], p[3]
    end
    nu
end

@model function badger_base(data, n_time, n_ind, n_groups, n_tests, n_seasons, n_nu, loglik_fn, obs_loglik_fn)
    tau ~ Exponential(TAU_PRIOR_SCALE)
    alpha ~ PracticalBayes.filldist(Exponential(1.0), n_groups)
    lambda ~ Exponential(1.0)
    beta ~ Exponential(1.0)
    q ~ Beta(1, 1)

    c1 ~ Exponential(1.0)
    a1 ~ Exponential(1.0)
    b1 ~ Exponential(1.0)
    a2 ~ Exponential(1.0)
    b2 ~ Exponential(1.0)

    thetas ~ PracticalBayes.filldist(Beta(1, 1), n_tests)
    rhos ~ PracticalBayes.filldist(Beta(1, 1), n_tests)
    phis ~ PracticalBayes.filldist(Beta(1, 1), n_tests)

    etas ~ PracticalBayes.filldist(Beta(1, 1), n_seasons)
    nu ~ NuSimplex(n_nu, [1.0, 1.0, 1.0])

    X ~ TrajectoryLatent(n_time, n_ind)

    pars = (; tau=tau, alpha=alpha, lambda=lambda, beta=beta, q=q,
            a1=a1, b1=b1, a2=a2, b2=b2, c1=c1,
            thetas=thetas, rhos=rhos, phis=phis,
            etas=etas, nu=nu)
    @addlogprob! loglik_fn(pars, data, X) + obs_loglik_fn(pars, data, X)
end

const TAU_PRIOR_SCALE = 100.0
const HMC_L = 15
const HMC_EPS = vcat(
    0.002,
    fill(0.2, G),
    0.01,
    0.05,
    0.05,
    0.005,
    0.02,
    0.05,
    0.001,
    0.001,
    fill(0.005, NT),
    fill(0.005, NT),
    fill(0.005, NT),
)

function make_hmc_block(eps_vec, L)
    HMC(L; integrator=Leapfrog(1.0), metric=DiagEuclideanMetric(eps_vec .^ 2))
end

function run_badger_fit(n_sweeps; n_burn=0, seed=13)
    m = badger_base(data, data.n_timepoints, data.n_individuals, G, NT, NS, NNU,
                    loglik, obs_loglik)

    hmc_kernel = make_hmc_block(HMC_EPS, HMC_L)
    n_adapts = 0

    outdir = joinpath(pwd(), "outputs")
    mkpath(outdir)
    x_path = joinpath(outdir, "badger-siler-fixed-hmc-$(n_sweeps)iter-X.jld2")
    x_flush_every = 500

    spl = Gibbs(
        (:tau, :alpha, :lambda, :beta, :q, :c1, :a1, :b1, :a2, :b2,
         :thetas, :rhos, :phis) => hmc_kernel,
        :etas => EtaKernel(1.0, 1.0),
        :nu => NuKernel([1.0, 1.0, 1.0]),
        :X => iFFBSKernel(latent!),
    )

    init0 = badger_initial_params(raw; rng=StableRNG(seed))
    X0 = copy(raw.X_init)
    init = (; X=X0, tau=init0.tau, alpha=init0.alpha, lambda=init0.lambda,
            beta=init0.beta, q=init0.q, c1=init0.c1, a1=init0.a1, b1=init0.b1,
            a2=init0.a2, b2=init0.b2, thetas=init0.thetas, rhos=init0.rhos,
            phis=init0.phis, etas=init0.etas,
            nu=hcat(init0.nuE, init0.nuI))

    reset_aggregates!(data)
    apply_derived_summaries!(init0, data, X0)

    println("\nGibbs: fixed-eps HMC(L=$HMC_L), Siler survival (fixed) + conjugate(etas, nu) + iFFBS(X, reststotal coupling)")
    println("adtype=", ADTYPE)
    println("sweeps=$n_sweeps burn=$n_burn seed=$seed")
    t0 = time()
    chn = AbstractMCMC.sample(StableRNG(seed), m, spl, n_sweeps;
                              init=init, adtype=ADTYPE, n_adapts=n_adapts, discard_initial=n_burn,
                              save_states=(X=(x_path, x_flush_every),))
    elapsed = time() - t0
    println("done in ", round(elapsed / 60, digits=1), " min (", round(elapsed, digits=1), " s)")
    chn, elapsed
end

function report(chn)
    println("\n=== Posterior summary ===")
    for name in (:tau, :lambda, :beta, :q, :c1, :a1, :b1, :a2, :b2)
        v = vec(chn[name])
        println(rpad(string(name), 8), " mean ", rpad(round(mean(v); digits=5), 10),
                " sd ", round(std(v); digits=5))
    end
    for name in (:thetas, :rhos, :phis, :etas)
        mm = chn[name]
        println(rpad(string(name), 8), " means ", round.(vec(mean(reduce(hcat, mm); dims=2)); digits=3))
    end
    am = vec(mean(reduce(hcat, chn[:alpha]); dims=2))
    println(rpad("alpha", 8), " ", length(am), " groups; posterior means: ",
            "min ", round(minimum(am); digits=4), " median ", round(median(am); digits=4),
            " max ", round(maximum(am); digits=4))
    println(rpad("", 8), " first 5: ", round.(am[1:min(5, length(am))]; digits=4))
end

function save_run(chn, elapsed, n_sweeps; tag)
    outdir = joinpath(pwd(), "outputs")
    mkpath(outdir)
    stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
    path = joinpath(outdir, "badger-siler-fixed-hmc-$tag-$stamp.jld2")
    @save path chn elapsed
    timing_path = joinpath(outdir, "badger-siler-fixed-hmc-$tag-$stamp-timing.txt")
    open(timing_path, "w") do io
        println(io, "n_sweeps=", n_sweeps)
        println(io, "elapsed_seconds=", elapsed)
        println(io, "elapsed_minutes=", elapsed / 60)
        println(io, "elapsed_hours=", elapsed / 3600)
    end
    println("saved: ", path)
    println("timing: ", timing_path)
    path
end


chn, elapsed = run_badger_fit(1000; n_burn=0, seed=13)
report(chn)
save_run(chn, elapsed, 1000; tag="1000iter")
println("1000-sweep timing: ", round(elapsed, digits=1), " s (",
        round(elapsed / 60, digits=2), " min); ",
        round(elapsed / 1000, digits=3), " s/sweep")
