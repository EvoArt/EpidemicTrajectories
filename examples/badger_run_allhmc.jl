# Standalone badger SEID fit — EVERYTHING IN HMC. No conjugate kernels: the capture
# probabilities `etas` (Beta prior) and the entry mixing `nu` (Dirichlet prior) are
# sampled by HMC alongside the epidemic/Siler/test parameters. Only the latent
# trajectory X stays with iFFBS. Because capture is now differentiated, ONE
# observation function (capture x tests) serves both the filter and the likelihood —
# no split, no NuPlaceholder. Uses AdaptiveHMC (learns mass matrix + step size).
# Saves the chain to JLD2 and CSV.

import Pkg
Pkg.activate(@__DIR__)
for (name, path) in (("EpidemicTrajectories", dirname(@__DIR__)),
                     ("PracticalBayes", joinpath(homedir(), ".julia", "dev", "PracticalBayes")),
                     ("PracticalEpiBayes", joinpath(homedir(), ".julia", "dev", "PracticalEpiBayes")))
    haskey(Pkg.project().dependencies, name) || Pkg.develop(path=path)
end
for pkg in ("Distributions", "ADTypes", "PolyesterForwardDiff",
            "StableRNGs", "CSV", "DataFrames", "JLD2", "AbstractMCMC", "Dates", "Statistics")
    haskey(Pkg.project().dependencies, pkg) || Pkg.add(pkg)
end
Pkg.instantiate()

using EpidemicTrajectories, PracticalBayes, PracticalEpiBayes
using Distributions, Random, StableRNGs, CSV, DataFrames, JLD2, Dates
using ADTypes: AutoPolyesterForwardDiff
using PolyesterForwardDiff
using Statistics: mean, std, median
import AbstractMCMC

const N_SWEEPS = 1000                 # total sweeps (adapt + sampled)
const N_ADAPT = 500                   # AdaptiveHMC warm-up; learns mass matrix + step size
const DATA_DIR = let a = joinpath(pwd(), "badger_ref", "RData2"),
                     b = joinpath(pwd(), "..", "badger_ref", "RData2")
    isdir(a) ? a : b
end

const STATES = [:S, :E, :I, :D]
const S, E, I, D = 1, 2, 3, 4
const BROCK_RAW, BROCK = 80, 101
const REF_CODE = Dict(0 => 1, 3 => 2, 1 => 3, 9 => 4)


# --- data ---------------------------------------------------------------------

function load_data(dir)
    rd(f) = CSV.read(joinpath(dir, f), DataFrame)
    dims = rd("dimensions.csv")
    m, maxt = dims[1, :m], dims[1, :maxt]
    n_groups, n_tests = dims[1, :G], dims[1, :numTests]
    n_seasons, n_nu = dims[1, :numSeasons], dims[1, :numNuTimes]

    Xinit = Matrix(rd("Xinit.csv"))
    test_mat = Matrix{Float64}(rd("TestMat.csv"))
    capt_hist = Matrix{Int}(rd("CaptHist.csv"))
    capt_effort = Matrix{Int}(rd("CaptEffort.csv"))
    birth = vec(Matrix{Int}(rd("birthTimes.csv")))
    start_p = vec(Matrix{Int}(rd("startSamplingPeriod.csv")))
    end_p = vec(Matrix{Int}(rd("endSamplingPeriod.csv")))
    nu_times = vec(Matrix{Int}(rd("nuTimes.csv")))
    sex = vec(Matrix{Int}(rd("sex.csv")))
    K = rd("Kay.csv")[1, :K]
    k = rd("k.csv")[1, :k]

    keep = findall(!=(0), sex)
    old_to_new = zeros(Int, m)
    for (new_i, old_i) in enumerate(keep)
        old_to_new[old_i] = new_i
    end
    Xinit = Xinit[keep, :]; capt_hist = capt_hist[keep, :]
    birth = birth[keep]; start_p = start_p[keep]; end_p = end_p[keep]

    rows = [old_to_new[Int(test_mat[r, 2])] != 0 for r in 1:size(test_mat, 1)]
    test_mat = test_mat[rows, :]
    for r in 1:size(test_mat, 1)
        test_mat[r, 2] = old_to_new[Int(test_mat[r, 2])]
    end
    m = length(keep)

    X_init = Matrix{Int}(undef, maxt, m)
    for i in 1:m, t in 1:maxt
        c = Xinit[i, t]
        X_init[t, i] = c == -10 ? 1 : REF_CODE[Int(c)]
    end

    social_group = zeros(Int, m, maxt)
    for i in 1:m
        rs = findall(==(Float64(i)), @view test_mat[:, 2])
        isempty(rs) && continue
        ts, gs = test_mat[rs, 1], test_mat[rs, 3]
        g = Int(gs[argmin(ts)])
        for t in max(1, birth[i]):maxt
            at = findfirst(==(Float64(t)), ts)
            at === nothing || (g = Int(gs[at]))
            social_group[i, t] = g
        end
    end

    age = fill(-10, m, maxt)
    for i in 1:m, t in max(1, birth[i]):maxt
        age[i, t] = t - birth[i]
    end

    tests = fill(-1, maxt, m, n_tests)
    for r in 1:size(test_mat, 1)
        t, i = Int(test_mat[r, 1]), Int(test_mat[r, 2])
        (1 <= t <= maxt && 1 <= i <= m) || continue
        for j in 1:n_tests
            v = test_mat[r, 3 + j]
            (isnan(v) || v == -10) && continue
            tests[t, i, j] = Int(v)
        end
    end
    if BROCK != BROCK_RAW                      # brock-test changepoint swap
        for t in BROCK_RAW:(BROCK - 1), i in axes(tests, 2)
            tests[t, i, 1], tests[t, i, 2] = tests[t, i, 2], tests[t, i, 1]
        end
    end

    capture = permutedims(capt_hist, (2, 1))
    last_cap = [something(findlast(==(1), @view capture[:, i]), 0) for i in 1:m]
    first_cap = [something(findfirst(==(1), @view capture[:, i]), max(birth[i], 1)) for i in 1:m]
    cam = Matrix{Int}(rd("capturesAfterMonit.csv"))
    for r in 1:size(cam, 1)
        oid = cam[r, 1]
        (1 <= oid <= length(old_to_new) && old_to_new[oid] != 0) || continue
        last_cap[old_to_new[oid]] = max(last_cap[old_to_new[oid]], cam[r, 2])
    end

    season = ones(Int, maxt)
    for t in 2:maxt
        season[t] = season[t-1] < n_seasons ? season[t-1] + 1 : 1
    end

    (; n_individuals=m, n_timepoints=maxt, n_groups, n_tests, n_seasons, n_nu,
     X_init, social_group, age, capture, capt_effort, tests,
     sampling_period=[(start_p[i], end_p[i]) for i in 1:m],
     birth_time=birth, last_capture_time=last_cap, first_capture_time=first_cap,
     season, nu_times, K, k)
end


# --- model functions ----------------------------------------------------------

function infection(model, data, i, t)
    g = data.social_group[i, t]
    g == 0 && return 0.0
    Inf_, M = data.aggregates.n_infectious[g, t], data.aggregates.n_alive[g, t]
    M == 0 && return 0.0
    foi = model.lambda * model.alpha[g] + model.beta * Inf_ / ((M / data.K)^model.q)
    -expm1(-foi)
end

function progression(model, data, i, t)                  # Erlang(k) CDF at 1 step
    x = data.k / model.tau
    s, term = 1.0, 1.0
    for j in 1:(data.k - 1)
        term *= x / j
        s += term
    end
    1.0 - s * exp(-x)
end

function survival(model, data, i, t)                     # Siler, at age(t+1)
    tt = t + 1
    age = tt <= data.n_timepoints ? data.age[i, tt] :
          data.age[i, data.n_timepoints] + (tt - data.n_timepoints)
    age < 0 && return 1.0
    y1, y2 = model.b2 * (age - 1), model.b2 * age
    z1, z2 = -model.b1 * (age - 1), -model.b1 * age
    late = -exp(y1) * expm1(y2 - y1)
    early = exp(z1) * expm1(z2 - z1)
    exp(-model.c1 + (model.a2 / model.b2) * late + (model.a1 / model.b1) * early)
end

transitions() = @transitions STATES begin
    @survival survival death=:D
    S -> E = infection
    E -> I = progression
end

function starting_state(model, data, X, i, t)
    p = zeros(eltype(model.nu), data.n_states)   # nu is now HMC-sampled (Dual under AD)
    st = data.sampling_period[i][1]
    nuE = nuI = zero(eltype(model.nu))
    if data.birth_time[i] < st
        idx = findfirst(==(st), data.nu_times)
        idx === nothing || ((nuE, nuI) = (model.nu[idx, 1], model.nu[idx, 2]))
    end
    p[1] = 1 - nuE - nuI; p[2] = nuE; p[3] = nuI
    p
end

aggregates(ng, nt) = @aggregate STATES begin
    @array n_infectious Int (ng, nt)
    @array n_alive Int (ng, nt)
    if data.social_group[i, t] > 0
        n_infectious[data.social_group[i, t], t] += (state == :I)
    end
    if data.social_group[i, t] > 0
        n_alive[data.social_group[i, t], t] += (state != :D)
    end
end

# One observation function: capture x tests. Since etas is now HMC-sampled, capture
# belongs in the differentiated likelihood, so the filter and the likelihood use the
# same scalar `obs_weight` — the package auto-derives the filter's per-state vector.
obs_eltype(model) = promote_type(eltype(model.etas), eltype(model.thetas),
                                 eltype(model.rhos), eltype(model.phis))

function obs_weight(model, data, X, i, t, s)
    T = obs_eltype(model)
    eta = model.etas[data.season[t]]
    if data.capture[t, i] == 0
        s == D && return t <= data.last_capture_time[i] ? zero(T) : one(T)
        w = one(T) * (1 - eta)                   # no test data when uncaptured
        return w
    end
    s == D && return zero(T)
    w = eta
    for j in 1:size(data.tests, 3)
        x = data.tests[t, i, j]
        (x == 0 || x == 1) || continue
        if s == S
            ϕ = model.phis[j]; w *= x == 1 ? (1 - ϕ) : ϕ
        elseif s == E
            θρ = model.thetas[j] * model.rhos[j]; w *= x == 1 ? θρ : (1 - θρ)
        else
            θ = model.thetas[j]; w *= x == 1 ? θ : (1 - θ)
        end
    end
    w
end

function build(dir)
    d = load_data(dir)
    affected = Matrix{Vector{Int}}(undef, d.n_timepoints, d.n_individuals)
    by_group = [Int[] for _ in 1:d.n_groups, _ in 1:d.n_timepoints]
    for t in 1:d.n_timepoints, i in 1:d.n_individuals
        g = d.social_group[i, t]; g > 0 && push!(by_group[g, t], i)
    end
    for t in 1:d.n_timepoints, i in 1:d.n_individuals
        g = d.social_group[i, t]
        affected[t, i] = g == 0 ? Int[] : [j for j in by_group[g, t] if j != i]
    end
    data = epidemic_data(;
        n_individuals=d.n_individuals, n_timepoints=d.n_timepoints,
        trans_mat=transitions(), starting_state=starting_state,
        observation_weight=obs_weight,           # capture x tests; filter vector auto-derived
        aggregates=aggregates(d.n_groups, d.n_timepoints),
        sampling_period=d.sampling_period, affected_individuals=affected,
        coupled_transitions=[(:S, :E)],
        state_space=STATES,
        social_group=d.social_group, age=d.age, capture=d.capture,
        capt_effort=d.capt_effort, tests=d.tests, season=d.season,
        birth_time=d.birth_time, last_capture_time=d.last_capture_time,
        nu_times=d.nu_times, K=Float64(d.K), k=d.k, n_groups=d.n_groups)
    (; data, raw=d)
end


# --- assemble and fit ---------------------------------------------------------

const (data, raw) = build(DATA_DIR)
const G, NT, NS, NNU = raw.n_groups, raw.n_tests, raw.n_seasons, raw.n_nu

const loglik = epidemic_loglik(data; entry_time=raw.first_capture_time, survival=survival)
const obs_loglik = epidemic_obs_loglik(data)     # full capture x tests (etas is HMC-sampled)
const latent! = epidemic_latent_sampler(data)

@model function badger(data, n_time, n_ind, n_groups, n_tests, n_seasons, n_nu, ll, oll)
    tau ~ Exponential(100.0)
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
    etas ~ PracticalBayes.filldist(Beta(1, 1), n_seasons)          # HMC (was conjugate)
    # nu: each entry-cohort is a 3-simplex [P(S),P(E),P(I)]; HMC samples it via the
    # simplex bijector. Reshape to the n_nu x 2 [P(E) P(I)] `starting_state` reads.
    nu_cols ~ PracticalBayes.filldist(Dirichlet([1.0, 1.0, 1.0]), n_nu)   # 3 x n_nu
    nu = permutedims(nu_cols[2:3, :])                                     # n_nu x 2
    X ~ TrajectoryLatent(n_time, n_ind)
    pars = (; tau, alpha, lambda, beta, q, a1, b1, a2, b2, c1, thetas, rhos, phis, etas, nu)
    @addlogprob! ll(pars, data, X) + oll(pars, data, X)
end

# One HMC block for every continuous parameter (nu_cols and etas included); iFFBS
# for X. AdaptiveHMC learns the mass matrix + step size over the first N_ADAPT sweeps.
spl = Gibbs(
    (:tau, :alpha, :lambda, :beta, :q, :c1, :a1, :b1, :a2, :b2,
     :thetas, :rhos, :phis, :etas, :nu_cols) => AdaptiveHMC(0.8; n_leapfrog=15),
    :X => iffbs_kernel(latent!; params = v -> (; v.tau, v.alpha, v.lambda, v.beta, v.q,
                                               v.a1, v.b1, v.a2, v.b2, v.c1,
                                               v.thetas, v.rhos, v.phis, v.etas,
                                               nu=permutedims(v.nu_cols[2:3, :]))),
)

X0 = copy(raw.X_init)
reset_aggregates!(data)
apply_derived_summaries!((;), data, X0)

# Explicit starting values near the plausible regime (diffuse priors make a
# prior-draw start land far out). etas and nu_cols are HMC-sampled here too, so they
# get initial values: nu_cols as uniform 3-simplex columns.
init = (; X=X0, tau=5.0, alpha=fill(0.001, G), lambda=0.01, beta=0.01, q=0.5,
        c1=0.01, a1=0.01, b1=0.01, a2=0.01, b2=0.01,
        thetas=fill(0.75, NT), rhos=fill(0.5, NT), phis=fill(0.85, NT),
        etas=fill(0.5, NS), nu_cols=fill(1/3, 3, NNU))

m = badger(data, data.n_timepoints, data.n_individuals, G, NT, NS, NNU, loglik, obs_loglik)

println("Badger fit (all-HMC): $(data.n_individuals) badgers x $(data.n_timepoints) t, ",
        "$G groups, $N_SWEEPS sweeps ($N_ADAPT adapt)")
t0 = time()
chn = AbstractMCMC.sample(StableRNG(13), m, spl, N_SWEEPS;
                          init=init, n_adapts=N_ADAPT, discard_initial=N_ADAPT,
                          adtype=AutoPolyesterForwardDiff(; chunksize=nothing, tag=nothing),
                          save_states=(X=:buffer,))
elapsed = time() - t0


# --- report and save ----------------------------------------------------------

println("\ninference time: $(round(elapsed / 60, digits=1)) min ",
        "($(round(elapsed, digits=1)) s, $(round(elapsed / N_SWEEPS, digits=3)) s/sweep)")
println("=== posterior means (sd), $(N_SWEEPS - N_ADAPT) post-warmup draws ===")
for name in (:tau, :lambda, :beta, :q, :c1, :a1, :b1, :a2, :b2)
    v = vec(chn[name])
    println(rpad(name, 8), round(mean(v); digits=4), "  (", round(std(v); digits=4), ")")
end
for name in (:thetas, :rhos, :phis, :etas)
    println(rpad(name, 8), round.(vec(mean(reduce(hcat, chn[name]); dims=2)); digits=3))
end
am = vec(mean(reduce(hcat, chn[:alpha]); dims=2))
println(rpad("alpha", 8), "$(length(am)) groups: min $(round(minimum(am); digits=3)) ",
        "median $(round(median(am); digits=3)) max $(round(maximum(am); digits=3))")

function chain_to_df(chn, scalars, vectors)
    df = DataFrame()
    for name in scalars
        df[!, name] = vec(chn[name])
    end
    for name in vectors
        mat = reduce(hcat, chn[name])
        for c in 1:size(mat, 1)
            df[!, "$(name)[$c]"] = mat[c, :]
        end
    end
    df
end

df = chain_to_df(chn,
                 (:tau, :lambda, :beta, :q, :c1, :a1, :b1, :a2, :b2),
                 (:alpha, :thetas, :rhos, :phis, :etas))

outdir = joinpath(@__DIR__, "outputs")
mkpath(outdir)
stamp = Dates.format(now(), "yyyymmdd-HHMMSS")
jld2_path = joinpath(outdir, "badger-allhmc-$stamp.jld2")
csv_path = joinpath(outdir, "badger-allhmc-$stamp.csv")
JLD2.@save jld2_path chn elapsed
CSV.write(csv_path, df)
println("\nsaved:\n  $jld2_path\n  $csv_path")
