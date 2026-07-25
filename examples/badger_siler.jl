# Badger SEID fit — the clean, single-path version.
#
# This is the badger stress-test model (2384 badgers × 161 timepoints × groups):
# an S/E/I/D framework with a Siler survival, a frequency-dependent force of
# infection, an Erlang E→I progression, and a multi-test capture/observation
# process. It fits with PracticalBayes' Gibbs sampler:
#
#   * the epidemic + Siler continuous parameters  → one HMC block,
#   * the per-season capture probabilities `etas`  → a conjugate Beta kernel,
#   * the entry-cohort mixing `nu`                 → a conjugate Dirichlet kernel,
#   * the latent trajectory `X`                    → the iFFBS kernel.
#
# The three Gibbs-kernel conveniences (`iffbs_kernel`, `capture_prob_kernel`,
# `initial_state_kernel`) come from PracticalEpiBayes — the sibling package that
# glues EpidemicTrajectories' PPL-agnostic pieces to PracticalBayes — so this
# script hand-writes none of that boilerplate.
#
# The coupling term uses the reference's O(n_states)-per-(i,t) "reststotal"
# running-total shape (`make_badger_rest_contribution` + the `_rt` derived
# summaries below), NOT the package default `make_rest_contribution`
# (O(n_states × |affected|)). It is kept as user code on purpose: the formula
# `nSE·log(foi) + nSS·log(1-foi)` is specific to a single coupled transition
# (S→E) with a per-group frequency-dependent FOI, which is a modelling choice, not
# something the generic core should assume. `epidemic_data(; rest_contribution=…)`
# is the seam that makes this a drop-in replacement.

using EpidemicTrajectories
using PracticalBayes
using PracticalEpiBayes
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


# ---------------------------------------------------------------------------
# State space and data loading
# ---------------------------------------------------------------------------

const BADGER_STATES = [:S, :E, :I, :D]
const _S_CODE = findfirst(==(:S), BADGER_STATES)
const _E_CODE = findfirst(==(:E), BADGER_STATES)
const _I_CODE = findfirst(==(:I), BADGER_STATES)
const _D_CODE = findfirst(==(:D), BADGER_STATES)

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


# ---------------------------------------------------------------------------
# Rates, survival and transitions
# ---------------------------------------------------------------------------

# S→E force of infection. Reads the per-group alive/infected counts off the
# aggregates. Those are leave-one-out during the focal's own iFFBS filter, but the
# package re-adds the focal transiently (via `focal_self_contribution=true`, the
# default) so this sees the reference's `M` (not `M-1`) denominator — this
# function needs no `+1` of its own.
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

# E→I progression: the Erlang(k) CDF at one step, parameterised by the mean time τ.
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

badger_progression(model, data, i, t) = erlang_cdf_at_1_meantime(data.k, model.tau / data.k)

# Siler survival for the t→t+1 step, evaluated at the age at t+1 (the destination
# time). Death is impossible while the individual is still being captured
# (t ≤ last_capture_time), so survival is 1 there; the `epidemic_loglik` entry
# gate uses this same function to divide the survival factor back out pre-entry.
function siler_survival(model, data, i, t)
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

function badger_transitions()
    @transitions BADGER_STATES begin
        @survival siler_survival death=:D
        S -> E = badger_infection
        E -> I = badger_progression
    end
end

# The entry-cohort mixing: a newly-entering individual starts S/E/I with the drawn
# `nu` proportions (the conjugate `nu` kernel resamples these each sweep).
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


# ---------------------------------------------------------------------------
# Observation process
# ---------------------------------------------------------------------------
#
# The observation weight factorises as capture × tests. The FILTER needs the full
# product (a per-state vector); the LIKELIHOOD needs only the test factor (capture
# is informed conjugately via `etas`), and only the ONE entry `w[X[t,i]]`. So the
# script supplies three pieces:
#
#   * `badger_observations` — the full vector product, for the filter;
#   * `badger_obs_tests`     — the test factor as a vector, the obs-likelihood's
#                              `observation_process`;
#   * `badger_obs_tests_weight` — the same test factor as a scalar `w[s]`, the
#                              obs-likelihood's `observation_weight` (no per-(i,t)
#                              array — see the `epidemic_obs_loglik` perf note).
#
# `etas` appears only in the capture factor, which is dropped from the likelihood,
# so `etas` stays purely conjugate (verified: the test factor moves by 0 when
# `etas` changes).

_obs_eltype(model) = promote_type(eltype(model.thetas), eltype(model.rhos), eltype(model.phis))

# Capture factor, with the death-ban: once seen alive (t ≤ last_capture_time) the
# individual cannot be dead, so state D gets weight 0 while it is still monitored.
function badger_obs_capture(model, data, X, i, t)
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

# The scalar counterpart of `badger_obs_tests`: just the entry for state `s`, so
# the likelihood never allocates a weight vector per (i, t).
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

# The full weight the filter sees.
badger_observations(model, data, X, i, t) =
    badger_obs_capture(model, data, X, i, t) .* badger_obs_tests(model, data, X, i, t)


# ---------------------------------------------------------------------------
# The "reststotal" coupling term (user code — the reference's O(n_states) shape)
# ---------------------------------------------------------------------------
#
# For each candidate focal state, the coupling asks: how probable are this group's
# realised S→E / S→S moves at time t? The reference maintains, per (group, time),
# running totals `nSE` (number of S→E moves) and `nSS` (S→S moves) plus the group
# alive/infected counts, so the coupling weight is a closed form in those totals —
# O(n_states) per (i, t), not the default's neighbour loop. The `_summary_*_rt`
# functions below are the reversible updates that keep those totals consistent with
# X; `make_badger_rest_contribution` reads them.

const _CASE_SorE = 1
const _CASE_I = 2
const _CASE_D = 3

@inline function _foi_case(s)
    s == _I_CODE && return _CASE_I
    s == _D_CODE && return _CASE_D
    return _CASE_SorE
end

# The group FOI with the focal placed in a candidate state: I/D/other shift the
# infected (I) and alive (M) counts by the reference's `Plus1` rule.
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


# ---------------------------------------------------------------------------
# Assemble the EpidemicData
# ---------------------------------------------------------------------------

function build_badger_data(dir; brock_changepoint=BROCK_CHANGEPOINT)
    d = load_badger_data(dir; brock_changepoint=brock_changepoint)

    # Who each individual affects at each time: its groupmates (leave-one-out).
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

    # Plain-NamedTuple aggregates (the reststotal running totals) with the matching
    # hand-written reversible summaries — the "verbose fallback" to `@aggregate`.
    aggregates = (; n_infectious=zeros(Int, d.n_groups, d.n_timepoints),
                    n_alive=zeros(Int, d.n_groups, d.n_timepoints),
                    nSE=zeros(Int, d.n_groups, d.n_timepoints),
                    nSS=zeros(Int, d.n_groups, d.n_timepoints))

    data = epidemic_data(;
        n_individuals=d.n_individuals,
        n_timepoints=d.n_timepoints,
        trans_mat=badger_transitions(),
        starting_state=badger_starting_state,
        observation_process=badger_observations,
        aggregates=aggregates,
        derived_summaries=(_summary_n_infectious_rt, _summary_n_alive_rt,
                           _summary_nSE_rt, _summary_nSS_rt),
        rest_contribution=make_badger_rest_contribution(),
        # focal_self_contribution defaults to true: `badger_infection` reads the
        # leave-one-out aggregates and does NOT add its own `+1`, so the package
        # must re-insert the focal. (The reststotal `_grp_foi` above handles the
        # focal only for the COUPLING term / neighbours, not the focal's own rate.)
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

const (data, raw) = build_badger_data(DATA_DIR)
const G = raw.n_groups
const NT = raw.n_tests
const NS = raw.n_seasons
const NNU = raw.n_nu_times

println("Badger model (reststotal coupling): ", data.n_individuals, " badgers x ",
        data.n_timepoints, " timepoints x ", G, " groups, ", NT, " tests")
println("Brock changepoint fixed at t=", BROCK_CHANGEPOINT, " (not inferred)")


# ---------------------------------------------------------------------------
# The three generated functions and the PracticalBayes model
# ---------------------------------------------------------------------------

# `epidemic_loglik` scores the starting state + transitions, conditioning on entry
# (survival divided back out before first capture); `epidemic_obs_loglik` scores
# only the TEST factor (capture is conjugate via `etas`), through the scalar
# `observation_weight` so it never allocates a per-(i,t) vector.
const loglik = epidemic_loglik(data; entry_time=raw.first_capture_time,
                               survival=siler_survival)
const obs_loglik = epidemic_obs_loglik(data;
                                 observation_process=badger_obs_tests,
                                 observation_weight=badger_obs_tests_weight)
const latent! = epidemic_latent_sampler(data)

# A placeholder matrix distribution for the `nu` block (owned by the conjugate
# Dirichlet kernel). `logpdf ≡ 0`; `rand` returns a valid entry-mixing so the
# model body evaluates cleanly during Gibbs-coverage validation.
struct NuPlaceholder <: Distributions.DiscreteMatrixDistribution
    n_nu::Int
end
Base.size(d::NuPlaceholder) = (d.n_nu, 2)
Distributions.logpdf(::NuPlaceholder, ::AbstractMatrix) = 0.0
Distributions.rand(::AbstractRNG, d::NuPlaceholder) = fill(0.05, d.n_nu, 2)

@model function badger_base(data, n_time, n_ind, n_groups, n_tests, n_seasons, n_nu,
                            loglik_fn, obs_loglik_fn)
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

    # `etas` and `nu` are placeholders here (logpdf ≡ 0): their real draws come from
    # the conjugate Gibbs kernels below, which own these blocks' values. `nu` needs
    # a placeholder whose `rand` is a VALID entry-mixing (each row a sub-simplex
    # `[P(E) P(I)]` with `P(E)+P(I) < 1`), because PracticalBayes evaluates the
    # model body with the placeholder's own `rand` during Gibbs-coverage validation,
    # before the conjugate kernel or `init` takes over. `TrajectoryLatent`'s
    # all-ones `rand` would make `badger_starting_state` return `1-1-1 < 0`.
    etas ~ PracticalBayes.filldist(Beta(1, 1), n_seasons)
    nu ~ NuPlaceholder(n_nu)

    X ~ TrajectoryLatent(n_time, n_ind)

    pars = (; tau, alpha, lambda, beta, q, a1, b1, a2, b2, c1,
            thetas, rhos, phis, etas, nu)
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

make_hmc_block(eps_vec, L) =
    HMC(L; integrator=Leapfrog(1.0), metric=DiagEuclideanMetric(eps_vec .^ 2))


# ---------------------------------------------------------------------------
# Gibbs kernels (from PracticalEpiBayes) and the fit driver
# ---------------------------------------------------------------------------

# The iFFBS trajectory kernel. `params` maps the Gibbs state to the parameter
# NamedTuple the rate functions expect.
const kX = iffbs_kernel(latent!;
    params = v -> (; tau=v.tau, alpha=v.alpha, lambda=v.lambda, beta=v.beta, q=v.q,
                   a1=v.a1, b1=v.b1, a2=v.a2, b2=v.b2, c1=v.c1,
                   thetas=v.thetas, rhos=v.rhos, phis=v.phis, etas=v.etas, nu=v.nu))

# Per-season capture probability `etas`, conjugate Beta. Available = trapped this
# season and not dead; caught = actually captured.
const k_etas = capture_prob_kernel(:etas;
    caught=data.capture, effort=data.capt_effort, group=data.social_group,
    index=data.season, dead_state=_D_CODE, n=NS, prior=(1, 1))

# Entry-cohort mixing `nu`, conjugate Dirichlet. Eligible at cohort time nt =
# entered at nt and born before it; the default collect gives the n_nu × 2
# [P(E) P(I)] matrix `badger_starting_state` indexes.
const k_nu = initial_state_kernel(:nu;
    at=raw.nu_times,
    eligible=(X, d, i, nt) -> d.sampling_period[i][1] == nt && d.birth_time[i] < nt,
    states=(_S_CODE, _E_CODE, _I_CODE), prior=[1.0, 1.0, 1.0], n=NNU)

function run_badger_fit(n_sweeps; n_burn=0, seed=13)
    m = badger_base(data, data.n_timepoints, data.n_individuals, G, NT, NS, NNU,
                    loglik, obs_loglik)

    spl = Gibbs(
        (:tau, :alpha, :lambda, :beta, :q, :c1, :a1, :b1, :a2, :b2,
         :thetas, :rhos, :phis) => make_hmc_block(HMC_EPS, HMC_L),
        :etas => k_etas,
        :nu => k_nu,
        :X => kX,
    )

    # "Without init params": the only value we seed is the latent trajectory `X`
    # (from the reference's X_init). Every continuous / conjugate block
    # self-initialises — the placeholder distributions' `rand` gives each a valid
    # starting value, which the HMC and conjugate kernels then take over. The
    # invariant setup uses the initial X; the reststotal summaries read only X and
    # data, so no real parameters are needed for it.
    X0 = copy(raw.X_init)
    reset_aggregates!(data)
    apply_derived_summaries!((;), data, X0)

    outdir = joinpath(pwd(), "outputs")
    mkpath(outdir)
    x_path = joinpath(outdir, "badger-siler-hmc-$(n_sweeps)iter-X.jld2")

    println("\nGibbs: fixed-eps HMC(L=$HMC_L), Siler survival + conjugate(etas, nu) " *
            "+ iFFBS(X, reststotal coupling)")
    println("adtype=", ADTYPE)
    println("sweeps=$n_sweeps burn=$n_burn seed=$seed")
    t0 = time()
    chn = AbstractMCMC.sample(StableRNG(seed), m, spl, n_sweeps;
                              init=(; X=X0), adtype=ADTYPE, n_adapts=0,
                              discard_initial=n_burn,
                              save_states=(X=(x_path, 500),))
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
    path = joinpath(outdir, "badger-siler-hmc-$tag-$stamp.jld2")
    @save path chn elapsed
    open(joinpath(outdir, "badger-siler-hmc-$tag-$stamp-timing.txt"), "w") do io
        println(io, "n_sweeps=", n_sweeps)
        println(io, "elapsed_seconds=", elapsed)
        println(io, "elapsed_minutes=", elapsed / 60)
        println(io, "elapsed_hours=", elapsed / 3600)
    end
    println("saved: ", path)
    path
end


# Run when executed as a script (skipped when the file is `include`d for a smoke
# test or benchmark that only wants the definitions). `BADGER_SWEEPS` overrides the
# sweep count.
if get(ENV, "BADGER_NO_RUN", "0") != "1"
    n = parse(Int, get(ENV, "BADGER_SWEEPS", "1000"))
    chn, elapsed = run_badger_fit(n; n_burn=0, seed=13)
    report(chn)
    save_run(chn, elapsed, n; tag="$(n)iter")
    println("$n-sweep timing: ", round(elapsed, digits=1), " s (",
            round(elapsed / 60, digits=2), " min); ",
            round(elapsed / n, digits=3), " s/sweep")
end
