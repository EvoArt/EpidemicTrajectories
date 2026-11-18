using Distributions: Poisson, logpdf

@testset "spatial captures" begin
    # four detectors, the third at a latent position; two occasions
    xy = [0.0 1.0 NaN 0.0; 0.0 0.0 NaN 1.0]
    active = Bool[1 1; 1 0; 1 1; 0 1]
    caps = [(1, 1, 1, 2), (1, 1, 3, 1), (2, 2, 4, 1)]
    sc = spatial_captures(; detector_xy = xy, active, captures = caps, n_individuals = 3, latent_detectors = [3])
    Xu = reshape([0.5, 0.5], 2, 1)
    pos(j) = j == 3 ? (0.5, 0.5) : (xy[1, j], xy[2, j])
    k(c, j, σ) = exp(-((c[1] - pos(j)[1])^2 + (c[2] - pos(j)[2])^2) / (2σ^2))
    c, σ, r = (0.2, 0.1), 0.6, 1.3
    on1 = [1, 2, 3]; g = sum(k(c, j, σ) for j in on1)
    # hazard: caught with 1 - exp(-r g), then 2 at detector 1 and 1 at detector 3
    want = log(1 - exp(-r * g)) + 2log(k(c, 1, σ) / g) + log(k(c, 3, σ) / g)
    @test capture_loglik(sc, 1, 1, c, σ, r; detectors = Xu) ≈ want
    @test capture_loglik(sc, 2, 1, c, σ, r; detectors = Xu) ≈ -r * g
    # poisson: independent counts at the active detectors
    n1 = Dict(1 => 2, 3 => 1)
    @test capture_loglik(sc, 1, 1, c, σ, r; detectors = Xu, form = :poisson) ≈
          sum(logpdf(Poisson(r * k(c, j, σ)), get(n1, j, 0)) for j in on1)
    @test caught(sc, 1, 1) && !caught(sc, 1, 2) && caught(sc, 2, 2)
    @test trapping(sc, 1) && trapping(sc, 2)
    # starting values: individual 1 caught at mapped detector 1 on occasion 1 only
    S = starting_centres(sc, identity, 2; bounds = (0, 1, 0, 1))
    @test S[:, 1, 1] == [0.0, 0.0] && S[:, 2, 1] == [0.0, 0.0]
    @test S[:, 2, 2] == [0.0, 1.0]
    X = starting_detectors(sc, S, identity; bounds = (0, 1, 0, 1))
    @test X[:, 1] ≈ [0.001, 0.001]
end
