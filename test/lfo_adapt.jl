# Adaptation diagnostics.
#
# The case these exist for is a REAL one, reproduced here from a badger fit: a
# changepoint bounded in 1:160 whose proposal sd grew to 939 over 5000 sweeps
# while acceptance never fell below 0.77. Every summary of that fit looked
# healthy -- high acceptance reads as "moving freely" -- but the parameter was
# frozen, because at that width nearly every proposal was out of range and
# silently discarded.

@testset "check_adaptation: a healthy adapter is quiet" begin
    tr = AdaptTrace(:theta)
    for (s, a, sd) in zip(100:100:1000,
                          [0.9, 0.7, 0.55, 0.48, 0.45, 0.44, 0.43, 0.44, 0.45, 0.44],
                          [1.0, 1.4, 1.8, 2.0, 2.1, 2.1, 2.2, 2.2, 2.1, 2.2])
        record!(tr, s, a, sd)
    end
    r = check_adaptation(tr)
    @test r.ok
    @test isempty(r.warnings)
    @test r.final_acceptance ≈ 0.44
end

@testset "check_adaptation: the xi divergence is caught" begin
    # The real trace, abridged: acceptance pinned high, sd growing without bound.
    tr = AdaptTrace(:xi)
    for (s, a, sd) in zip(100:100:5000,
                          vcat(fill(0.95, 20), fill(0.9, 20), fill(0.8, 10)),
                          [8.8 * 1.1^i for i in 0:49])
        record!(tr, s, a, sd)
    end
    r = check_adaptation(tr; support = (1, 160))

    @test !r.ok
    msg = join(r.warnings, " ")
    # each of the three pathologies must be named specifically
    @test occursin("acceptance never fell", msg)
    @test occursin("FLAT likelihood", msg)          # says WHY, under LFO
    @test occursin("proposal scale grew", msg)
    @test occursin("exceeds the support", msg)
    @test r.final_scale > 160                        # sd past the whole support
end

@testset "check_adaptation: uncounted out-of-range proposals are flagged" begin
    # This is what lets the other two pathologies persist unseen: a proposal
    # rejected for being illegal is a REJECTION, and a kernel that treats it as a
    # no-op starves its own adapter.
    tr = AdaptTrace(:xi)
    for s in 100:100:500
        record!(tr, s, 0.42, 2.0; n_out_of_range = 90, n_proposed = 100)
    end
    r = check_adaptation(tr)
    @test !r.ok
    @test occursin("out of range", join(r.warnings, " "))
    @test occursin("REJECTIONS", join(r.warnings, " "))
end

@testset "check_adaptation: empty and single-point traces" begin
    @test check_adaptation(AdaptTrace(:x)).ok       # nothing recorded: no verdict
    tr = AdaptTrace(:x); record!(tr, 100, 0.44, 1.0)
    r = check_adaptation(tr)
    @test r.ok                                       # one point cannot show growth
    @test r.scale_ratio ≈ 1.0
end

@testset "flat_likelihood_range names the truncated extras" begin
    # A clamped or filtered extra carries no information past the cutoff BY
    # CONSTRUCTION -- so any likelihood windowed by one is flat out there. Saying
    # so before a sweep runs beats discovering it after a 2-hour fit.
    plan = truncation(clamp = (:last_seen,), filter = (:events,),
                      copy = (:tests,), keep = (:sex,))
    flat = flat_likelihood_range(plan)
    @test Set(flat) == Set([:last_seen, :events])
    @test !(:tests in flat)      # copied: same values, still informative
    @test !(:sex in flat)        # kept: no time information at all
end
