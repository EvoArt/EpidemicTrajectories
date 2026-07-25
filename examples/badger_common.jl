# Shared badger scaffolding for the staged optimization scripts.
#
# The staged scripts (badger_naive.jl → badger_coupled.jl → badger_obsweight.jl →
# badger_reststotal.jl) each `include` this file, then differ ONLY in how they build
# the aggregates, the coupling term, and the observation likelihood — so the
# progression from fully-naive package code to the fully-hand-optimized model is
# readable as a diff, and each stage can be timed/profiled on its own.
#
# Everything model-INDEPENDENT lives here: the data loader, the rate / survival /
# observation functions, the affected-individual lists, the PracticalBayes model
# skeleton and Gibbs kernels, and the run/report helpers. Nothing here chooses an
# optimization; that is the staged scripts' job.

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
    for r in 1:size(cam_raw, 1)
        oid = cam_raw[r, 1]
        (1 <= oid <= length(old_to_new) && old_to_new[oid] != 0) || continue
        nid = old_to_new[oid]
        last_capture_time[nid] = max(last_capture_time[nid], cam_raw[r, 2])
    end

    season = make_season_vec(n_seasons, 1, maxt)
    sampling_period = [(start_period[i], end_period[i]) for i in 1:m]

    return (; n_individuals=m, n_timepoints=maxt, n_groups, n_tests, n_seasons,
            n_nu_times, X_init, social_group, age, capture, capt_effort, tests,
            sampling_period, birth_time, last_capture_time, first_capture_time,
            season, nu_times, K, k)
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

# Who each individual affects at each time: its groupmates (leave-one-out). Shared
# by every stage that declares `affected_individuals` explicitly.
function badger_affected_lists(d)
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
    affected
end


# ---------------------------------------------------------------------------
# Rates, survival, transitions, starting state
# ---------------------------------------------------------------------------

# S→E force of infection. Reads the per-group alive/infected counts off the
# aggregates (leave-one-out during the focal's own filter; the package re-adds the
# focal via `focal_self_contribution=true`, so this needs no `+1` of its own).
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

# Siler survival for the t→t+1 step, at the age at t+1 (the destination time).
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

# The @aggregate declaration for the group alive/infected counts (all stages share
# these; the reststotal stage adds nSE/nSS on top via hand-written summaries). The
# `@array` dims must be literals/locals at expansion, so take them as arguments.
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


# ---------------------------------------------------------------------------
# Observation process (capture × tests, with the death-ban)
# ---------------------------------------------------------------------------

_obs_eltype(model) = promote_type(eltype(model.thetas), eltype(model.rhos), eltype(model.phis))

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

# The scalar counterpart of `badger_obs_tests` (stage 3+): just the entry for state
# `s`, so the likelihood never allocates a weight vector per (i, t).
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
# The PracticalBayes model + Gibbs kernels + run/report helpers
# ---------------------------------------------------------------------------

# `nu` placeholder (owned by the conjugate Dirichlet kernel). `rand` must be a valid
# entry-mixing, since PracticalBayes evaluates the model body with it during
# Gibbs-coverage validation, before the kernel or `init` takes over.
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

    etas ~ PracticalBayes.filldist(Beta(1, 1), n_seasons)
    nu ~ NuPlaceholder(n_nu)

    X ~ TrajectoryLatent(n_time, n_ind)

    pars = (; tau, alpha, lambda, beta, q, a1, b1, a2, b2, c1,
            thetas, rhos, phis, etas, nu)
    @addlogprob! loglik_fn(pars, data, X) + obs_loglik_fn(pars, data, X)
end

const TAU_PRIOR_SCALE = 100.0
const HMC_L = 15

# Per-stage HMC step sizes: one leading `tau`, `n_groups` alphas, then the scalar
# epidemic + Siler params, then the three length-`n_tests` test-parameter blocks.
function badger_hmc_eps(n_groups, n_tests)
    vcat(0.002, fill(0.2, n_groups), 0.01, 0.05, 0.05, 0.005, 0.02, 0.05,
         0.001, 0.001, fill(0.005, n_tests), fill(0.005, n_tests), fill(0.005, n_tests))
end

make_hmc_block(eps_vec, L) =
    HMC(L; integrator=Leapfrog(1.0), metric=DiagEuclideanMetric(eps_vec .^ 2))

# The Gibbs kernels that are identical across stages (they depend only on `data`).
badger_iffbs_kernel(latent!) = iffbs_kernel(latent!;
    params = v -> (; tau=v.tau, alpha=v.alpha, lambda=v.lambda, beta=v.beta, q=v.q,
                   a1=v.a1, b1=v.b1, a2=v.a2, b2=v.b2, c1=v.c1,
                   thetas=v.thetas, rhos=v.rhos, phis=v.phis, etas=v.etas, nu=v.nu))

badger_etas_kernel(data, n_seasons) = capture_prob_kernel(:etas;
    caught=data.capture, effort=data.capt_effort, group=data.social_group,
    index=data.season, dead_state=_D_CODE, n=n_seasons, prior=(1, 1))

badger_nu_kernel(nu_times, n_nu) = initial_state_kernel(:nu;
    at=nu_times,
    eligible=(X, d, i, nt) -> d.sampling_period[i][1] == nt && d.birth_time[i] < nt,
    states=(_S_CODE, _E_CODE, _I_CODE), prior=[1.0, 1.0, 1.0], n=n_nu)

DATA_DIR() = let
    base1 = joinpath(pwd(), "badger_ref", "RData2")
    base2 = joinpath(pwd(), "..", "badger_ref", "RData2")
    isdir(base1) ? base1 : base2
end

# Run `n_sweeps` Gibbs sweeps of the assembled model. `data`, `raw`, `loglik`,
# `obs_loglik` are the per-stage pieces the caller has already built.
function run_badger_fit(data, raw, loglik, obs_loglik; n_sweeps, n_burn=0, seed=13)
    G, NT, NS, NNU = raw.n_groups, raw.n_tests, raw.n_seasons, raw.n_nu_times
    m = badger_base(data, data.n_timepoints, data.n_individuals, G, NT, NS, NNU,
                    loglik, obs_loglik)

    spl = Gibbs(
        (:tau, :alpha, :lambda, :beta, :q, :c1, :a1, :b1, :a2, :b2,
         :thetas, :rhos, :phis) => make_hmc_block(badger_hmc_eps(G, NT), HMC_L),
        :etas => badger_etas_kernel(data, NS),
        :nu => badger_nu_kernel(raw.nu_times, NNU),
        :X => badger_iffbs_kernel(epidemic_latent_sampler(data)),
    )

    X0 = copy(raw.X_init)
    reset_aggregates!(data)
    apply_derived_summaries!((;), data, X0)

    println("sweeps=$n_sweeps burn=$n_burn seed=$seed  adtype=", ADTYPE)
    t0 = time()
    chn = AbstractMCMC.sample(StableRNG(seed), m, spl, n_sweeps;
                              init=(; X=X0), adtype=ADTYPE, n_adapts=0,
                              discard_initial=n_burn)
    elapsed = time() - t0
    println("done in ", round(elapsed / 60, digits=2), " min (",
            round(elapsed, digits=1), " s); ", round(elapsed / n_sweeps, digits=3), " s/sweep")
    chn, elapsed
end

# Time the two things the optimizations move: one iFFBS latent sweep (at fixed
# params) and one full Gibbs sweep (HMC + conjugate + iFFBS). Prints a machine-
# readable line the staged benchmark driver greps for.
#
# Compilation is measured SEPARATELY, never folded into the per-sweep average: each
# timer runs only AFTER a discarded compile/warm-up run, and the full-sweep average
# is over `nfull` STEADY-STATE sweeps (default 100 — enough that a stray GC pause or
# a one-off allocation does not dominate). The Gibbs `sample` compile is the whole
# HMC-leapfrog + forward-AD-gradient stack and takes tens of seconds on its own, so
# amortising it over a handful of sweeps (as a naive timing does) can inflate the
# reported s/sweep several-fold — that is a measurement artefact, not the model.
function time_stage(stage, data, raw, loglik, obs_loglik;
                    nffbs=20, nfull=parse(Int, get(ENV, "BADGER_NFULL", "100")), seed=13)
    G, NT, NS, NNU = raw.n_groups, raw.n_tests, raw.n_seasons, raw.n_nu_times
    pars = (; tau=5.0, alpha=fill(0.5, G), lambda=0.5, beta=0.3, q=0.2,
            a1=0.6, b1=1.0, a2=3.0, b2=1.7, c1=0.45,
            thetas=fill(0.3, NT), rhos=fill(0.5, NT), phis=fill(0.5, NT),
            etas=fill(0.3, NS), nu=fill(0.05, NNU, 2))

    # (a) iFFBS sweep, in isolation. Compile once (discarded), then average nffbs.
    X = copy(raw.X_init)
    reset_aggregates!(data)
    apply_derived_summaries!((;), data, X)
    rng = StableRNG(seed)
    iffbs!(pars, data, X, rng)                       # compile — NOT timed
    t_ffbs = @elapsed for _ in 1:nffbs
        iffbs!(pars, data, X, rng)
    end
    t_ffbs /= nffbs

    # (b) full Gibbs sweep. Two separate `sample` calls: the first compiles the
    # whole sampler (discarded), the second is the timed steady-state run.
    m = badger_base(data, data.n_timepoints, data.n_individuals, G, NT, NS, NNU,
                    loglik, obs_loglik)
    spl = Gibbs(
        (:tau, :alpha, :lambda, :beta, :q, :c1, :a1, :b1, :a2, :b2,
         :thetas, :rhos, :phis) => make_hmc_block(badger_hmc_eps(G, NT), HMC_L),
        :etas => badger_etas_kernel(data, NS),
        :nu => badger_nu_kernel(raw.nu_times, NNU),
        :X => badger_iffbs_kernel(epidemic_latent_sampler(data)),
    )
    fit_once(nsweeps) = begin
        X0 = copy(raw.X_init)
        reset_aggregates!(data)
        apply_derived_summaries!((;), data, X0)
        AbstractMCMC.sample(StableRNG(seed), m, spl, nsweeps;
                            init=(; X=X0), adtype=ADTYPE, n_adapts=0, discard_initial=0)
    end
    t_compile = @elapsed fit_once(2)                 # compile — NOT counted
    t_full = @elapsed fit_once(nfull)
    t_full /= nfull

    println("STAGE_TIMING stage=$stage ffbs=", round(t_ffbs, digits=4),
            " full=", round(t_full, digits=4),
            " nfull=", nfull, " compile_s=", round(t_compile, digits=1))
    (; stage, t_ffbs, t_full)
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
end
