# End-to-end: truncate -> fit -> score -> collect, on a small SIR-with-death
# model where the truth is known.
#
# The gates that matter:
#  * the sweep runs and produces finite scores under every granularity,
#  * the cache is keyed on the POSTERIOR, so a rescore reuses fits,
#  * a constraint passed without its weight is REFUSED,
#  * the true model outscores a deliberately wrong one.

function _sid_run(; n_ind = 12, n_t = 16)
    state_space = [:S, :I, :D]
    aggs = @aggregate state_space begin
        @array n_infected Int (2, n_t)
        n_infected[data.group[i], t] += (state == :I)
    end
    spec = @transitions state_space begin
        S -> I = (model, data, i, t) ->
            -expm1(-(model.α + model.β * data.aggregates[:n_infected][data.group[i], t]))
        S -> D = (model, data, i, t) -> model.μ
        I -> S = (model, data, i, t) -> 1 / model.m
        I -> D = (model, data, i, t) -> model.μ
    end
    group = repeat(1:2; inner = n_ind ÷ 2)
    data = epidemic_data(;
        n_individuals = n_ind, n_timepoints = n_t, group = group,
        trans_mat = spec, aggregates = aggs,
        starting_state = (model, data, X, i, t) -> [0.85, 0.15, 0.0],
        last_seen = fill(n_t, n_ind),
        sex = repeat([1, 2], n_ind ÷ 2))
    (; data, n_ind, n_t)
end

# A "fit" that returns fixed parameter sets and the observed trajectory. Standing
# in for MCMC: the driver only cares about the shape of what comes back, and this
# keeps the test fast and deterministic.
function _fake_fit(truth, X_obs, n_draws)
    (train, cutoff) -> begin
        draws = [truth for _ in 1:n_draws]
        Xs = [copy(X_obs) for _ in 1:n_draws]
        (draws, Xs)
    end
end

@testset "lfo_cv: end to end" begin
    s = _sid_run()
    truth = (; α = 0.02, β = 0.03, m = 4.0, μ = 0.02)
    X_obs = epidemic_simulator(s.data)(StableRNG(1), truth)

    plan = truncation(clamp = (:last_seen,), keep = (:sex,))
    spec = LFOSpec(fit = _fake_fit(truth, X_obs, 8),
                   cell_logdensity = (m, d, X, i, t) -> -0.3,
                   plan = plan,
                   is_informative = (d, i, t) -> true)

    res = lfo_cv(spec, s.data; L = 8, M = 2,
                 granularity = (Pointwise(), ByGroup(s.data.group), Joint()),
                 verbose = false)

    @test length(res.windows) == length(lfo_cutoffs(s.data; L = 8, M = 2))
    @test Set(res.granularities) == Set([:pointwise, :by_group, :joint])
    for g in res.granularities
        @test isfinite(elpd(res, g))
    end
    # With a CONSTANT per-cell density and no -Inf, every draw agrees, so each
    # cell's weighted average is just that constant and all three granularities
    # sum to the same total. That is a sharp check on the normaliser: without the
    # per-cell `log S` subtraction, pointwise would be inflated over joint by
    # n_cells * log S and this equality would fail by a wide margin.
    @test isapprox(elpd(res, :pointwise), elpd(res, :joint); rtol = 1e-10)
    @test isapprox(elpd(res, :by_group), elpd(res, :joint); rtol = 1e-10)
    @test n_informative(res) > 0
    @test all(w.n_draws == 8 for w in res.windows)
end

@testset "lfo_cv: cache is keyed on the posterior, not the scorer" begin
    s = _sid_run()
    truth = (; α = 0.02, β = 0.03, m = 4.0, μ = 0.02)
    X_obs = epidemic_simulator(s.data)(StableRNG(2), truth)

    n_fits = Ref(0)
    counting_fit = (train, cutoff) -> begin
        n_fits[] += 1
        ([truth for _ in 1:4], [copy(X_obs) for _ in 1:4])
    end

    dir = mktempdir()
    spec = LFOSpec(fit = counting_fit,
                   cell_logdensity = (m, d, X, i, t) -> -0.3,
                   plan = truncation(clamp = (:last_seen,), keep = (:sex,)))

    r1 = lfo_cv(spec, s.data; L = 12, M = 2, cache = dir, verbose = false)
    first_count = n_fits[]
    @test first_count == length(r1.windows)

    # Rescoring under a DIFFERENT granularity must not refit: granularity changes
    # the score, never the posterior.
    r2 = lfo_cv(spec, s.data; L = 12, M = 2, granularity = (Joint(),),
                cache = dir, verbose = false)
    @test n_fits[] == first_count          # no new fits at all
    @test length(r2.windows) == length(r1.windows)
end

@testset "lfo_cv: a constraint without its weight is refused" begin
    s = _sid_run()
    truth = (; α = 0.02, β = 0.03, m = 4.0, μ = 0.02)
    X_obs = epidemic_simulator(s.data)(StableRNG(3), truth)
    plan = truncation(clamp = (:last_seen,), keep = (:sex,))

    # This is the -1.14 nat bias in constructor form: the package refuses rather
    # than quietly producing a biased estimate.
    bad = LFOSpec(fit = _fake_fit(truth, X_obs, 4),
                  cell_logdensity = (m, d, X, i, t) -> -0.3,
                  plan = plan, constrain = (i, t) -> true)
    @test_throws ArgumentError lfo_cv(bad, s.data; L = 12, M = 2, verbose = false)

    # both halves together is fine
    con, w = survival_constrained((i, t) -> i <= 2)
    ok = LFOSpec(fit = _fake_fit(truth, X_obs, 4),
                 cell_logdensity = (m, d, X, i, t) -> -0.3,
                 plan = plan, constrain = con, survival_weight = w)
    r = lfo_cv(ok, s.data; L = 12, M = 2, verbose = false)
    @test isfinite(elpd(r, :pointwise))
end

@testset "lfo_cv: repeated granularity is rejected" begin
    s = _sid_run()
    truth = (; α = 0.02, β = 0.03, m = 4.0, μ = 0.02)
    X_obs = epidemic_simulator(s.data)(StableRNG(4), truth)
    spec = LFOSpec(fit = _fake_fit(truth, X_obs, 4),
                   cell_logdensity = (m, d, X, i, t) -> -0.3,
                   plan = truncation(clamp = (:last_seen,), keep = (:sex,)))
    @test_throws ArgumentError lfo_cv(spec, s.data; L = 12, M = 2,
                                      granularity = (Joint(), Joint()), verbose = false)
end

@testset "lfo_cv: the better density wins, and compare says so" begin
    # A minimal discrimination check: two "models" differing only in the density
    # they assign. The one that fits better must win, and `compare` must report a
    # positive margin with a finite SE.
    s = _sid_run()
    truth = (; α = 0.02, β = 0.03, m = 4.0, μ = 0.02)
    X_obs = epidemic_simulator(s.data)(StableRNG(5), truth)
    plan = truncation(clamp = (:last_seen,), keep = (:sex,))

    good = LFOSpec(fit = _fake_fit(truth, X_obs, 6), plan = plan,
                   cell_logdensity = (m, d, X, i, t) -> -0.2)
    poor = LFOSpec(fit = _fake_fit(truth, X_obs, 6), plan = plan,
                   cell_logdensity = (m, d, X, i, t) -> -0.9)

    rg = lfo_cv(good, s.data; L = 8, M = 2, verbose = false)
    rp = lfo_cv(poor, s.data; L = 8, M = 2, verbose = false)

    @test elpd(rg, :pointwise) > elpd(rp, :pointwise)
    c = compare(rg, rp; granularity = :pointwise)
    @test c.diff > 0
    @test c.n_windows == length(rg.windows)
end
