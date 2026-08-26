# Pareto-smoothed importance weights.
#
# The gate that matters most is the degenerate case: when most ratios are -Inf a
# naive implementation returns all-NaN weights, which silently voids the window
# and once got written up as a substantive finding.

@testset "psis_smooth: well-behaved ratios" begin
    rng = StableRNG(1)
    logr = randn(rng, 400) .* 0.3          # tight ratios: an easy case
    w, k = psis_smooth(logr)

    @test length(w) == length(logr)
    @test all(isfinite, w)
    @test k < 0.7                          # comfortably reusable
    @test psis_ess(w) > 100                # and not concentrated on a few draws
end

@testset "_gpd_fit recovers a KNOWN shape (pins the sign)" begin
    # REGRESSION. Zhang & Stephens profile over b = -k/sigma, so the estimator's
    # mean of log1p(-b*x) is -k and khat is its NEGATIVE. Getting that backwards
    # is SILENT: sigma still comes out right (it is k/b, and both flip together),
    # the weights are still finite and plausible, and only the DIAGNOSTIC is
    # inverted -- so a heavy tail reports as healthy and no refit is triggered.
    #
    # Comparing two of our own outputs would not have caught it. This draws from
    # a GPD of known shape and pins the absolute value.
    gpd_rand(rng, k, sigma, n) =
        [k == 0 ? -sigma * log(rand(rng)) : sigma * (rand(rng)^(-k) - 1) / k for _ in 1:n]
    rng = StableRNG(99)
    for ktrue in (0.1, 0.3, 0.5, 0.7, 1.0)
        x = gpd_rand(rng, ktrue, 1.0, 5000)
        k, sigma = EpidemicTrajectories._gpd_fit(x)
        @test k > 0                                    # sign, first and foremost
        @test isapprox(k, ktrue; atol = 0.1)
        @test isapprox(sigma, 1.0; atol = 0.15)
    end
end

@testset "psis_smooth: heavy tails raise k-hat" begin
    rng = StableRNG(2)
    tight = psis_smooth(randn(rng, 400) .* 0.2)[2]
    heavy = psis_smooth(randn(rng, 400) .* 3.0)[2]
    @test heavy > tight                    # the diagnostic responds to the problem
end

@testset "psis_smooth: mostly -Inf ratios do NOT give NaN weights" begin
    # REGRESSION, and the important one. Under a naive proposal most
    # forward-simulated paths kill someone the data prove was alive, so most
    # ratios are exactly -Inf. Taking the tail cutoff over ALL ratios then gives
    # -Inf, every exceedance becomes -Inf - (-Inf) = NaN, and every weight is
    # NaN -- so the window scores -Inf whatever the granularity.
    #
    # That looked like the finding "the naive proposal collapses every window",
    # not like a bug. The tell was a run totalling exactly 0.00 with every window
    # non-finite, which then BEAT every finite competitor.
    logr = fill(-Inf, 88)
    logr[1:7] .= [-1.0, -1.2, -0.8, -2.0, -1.5, -0.9, -1.1]

    w, k = psis_smooth(logr)
    @test !any(isnan, w)                   # the actual regression
    @test count(isfinite, w) == 7          # exactly the usable draws survive
    @test all(!isfinite, w[8:end])
    @test isfinite(psis_ess(w))

    # Fewer than five usable draws: flag with k = Inf, which forces a refit,
    # rather than fabricating a tail estimate from nothing.
    few = fill(-Inf, 50); few[1:3] .= [-1.0, -1.1, -0.9]
    w2, k2 = psis_smooth(few)
    @test k2 == Inf
    @test !any(isnan, w2)
    @test count(isfinite, w2) == 3

    # and all--Inf is handled without NaN too
    w3, k3 = psis_smooth(fill(-Inf, 20))
    @test !any(isnan, w3)
    @test k3 == Inf
    @test psis_ess(w3) == 0.0
end

@testset "psis_ess: bounds and degenerate inputs" begin
    # flat weights: every draw counts equally
    @test psis_ess(zeros(50)) ≈ 50
    # one draw dominating: ESS near 1
    w = fill(-50.0, 50); w[1] = 0.0
    @test psis_ess(w) < 1.01
    @test psis_ess(Float64[]) == 0.0
    @test psis_ess(fill(-Inf, 10)) == 0.0
end

@testset "psis_smooth: weights feed aggregate_cells without NaN" begin
    # The weights are consumed by aggregate_cells, so the degenerate case has to
    # survive the whole path, not just the smoother.
    logr = fill(-Inf, 40); logr[1:6] .= -1.0 .* (1:6)
    w, _ = psis_smooth(logr)
    lp = [Dict{Any,Float64}(1 => -0.5, 2 => -0.7) for _ in 1:40]
    total = aggregate_cells(w, lp)
    @test !isnan(total)
    @test isfinite(total)
end

@testset "psis_smooth: input validation" begin
    @test_throws ArgumentError psis_smooth(Float64[])
end
