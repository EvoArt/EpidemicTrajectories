# The leave-future-out core: granularity, cell aggregation, and the result object.
#
# The gates that matter:
#  * `Joint()` reproduces an unbucketed scorer EXACTLY. This is the degenerate
#    case that catches most aggregation bugs, so it is asserted to 1e-12.
#  * a -Inf in one cell costs only that cell under `Pointwise`, and the whole
#    window under `Joint`. That difference is the entire reason the axis exists.
#  * `compare` refuses the one comparison that is always invalid.

using StatsFuns: logsumexp

# A reference scorer with no bucketing at all: one self-normalised weighted
# average over draws of the summed log density. This is what `Joint()` must
# reproduce. Note the normaliser -- see the `log S` regression test below.
_unbucketed(logw, per_draw_total) = logsumexp(logw .+ per_draw_total) - logsumexp(logw)

"Random per-draw, per-cell log densities plus their per-draw sums."
function _fake_cells(rng, S, N, M)
    lp = [Dict{Any,Float64}() for _ in 1:S]
    tot = zeros(S)
    for s in 1:S, i in 1:N, m in 1:M
        v = -rand(rng) * 3
        lp[s][(i, m)] = v
        tot[s] += v
    end
    lp, tot
end

@testset "granularity: cell_of maps to the right bucket" begin
    g = ByGroup([1, 1, 2, 2, 3])
    @test cell_of(Joint(), 3, 2) == cell_of(Joint(), 1, 1)      # one bucket, always
    @test cell_of(Pointwise(), 3, 2) == (3, 2)
    @test cell_of(Pointwise(), 3, 2) != cell_of(Pointwise(), 3, 1)
    @test cell_of(g, 1, 2) == cell_of(g, 2, 2)                  # same group, same step
    @test cell_of(g, 1, 2) != cell_of(g, 3, 2)                  # different group
    @test cell_of(g, 1, 1) != cell_of(g, 1, 2)                  # different step
end

@testset "Joint reproduces an unbucketed scorer to 1e-12" begin
    rng = StableRNG(11)
    S, N, M = 40, 12, 2
    _, tot = _fake_cells(rng, S, N, M)
    logw = zeros(S)

    # everything in one bucket, exactly as Joint() charges it
    joint_lp = [Dict{Any,Float64}(1 => tot[s]) for s in 1:S]
    @test isapprox(aggregate_cells(logw, joint_lp), _unbucketed(logw, tot); atol = 1e-12)

    # and with non-trivial importance weights
    w = randn(rng, S)
    @test isapprox(aggregate_cells(w, joint_lp), _unbucketed(w, tot); atol = 1e-12)
end

@testset "granularity changes the score only by where the log is taken" begin
    rng = StableRNG(22)
    S, N, M = 60, 10, 2
    lp, tot = _fake_cells(rng, S, N, M)
    logw = zeros(S)

    pw = aggregate_cells(logw, lp)
    jt = aggregate_cells(logw, [Dict{Any,Float64}(1 => tot[s]) for s in 1:S])

    @test isfinite(pw) && isfinite(jt)
    # Positively correlated cells (the normal case) make E[prod] > prod E, so
    # joint scores HIGHER. The old claim that Jensen forces pointwise >= joint is
    # wrong; the direction is set by the correlation across cells.
    @test pw != jt
end

@testset "a -Inf cell costs one cell pointwise, the window jointly" begin
    rng = StableRNG(33)
    S, N, M = 30, 8, 2
    lp, tot = _fake_cells(rng, S, N, M)
    logw = zeros(S)
    finite_pw = aggregate_cells(logw, lp)

    # ONE draw cannot explain ONE cell.
    lp[1][(1, 1)] = -Inf
    tot[1] = -Inf
    pw = aggregate_cells(logw, lp)
    jt = aggregate_cells(logw, [Dict{Any,Float64}(1 => tot[s]) for s in 1:S])

    @test isfinite(pw)                     # the other 29 draws still cover that cell
    @test isfinite(jt)                     # and the other draws still cover the window
    @test pw < finite_pw                   # but it did cost something

    # EVERY draw fails that one cell: no draw can explain the observation, so the
    # window is -Inf under both. That is honest, not a bug.
    for s in 1:S; lp[s][(1, 1)] = -Inf; end
    @test aggregate_cells(logw, lp) == -Inf
end

@testset "aggregate_cells: the per-cell normaliser is not optional" begin
    # REGRESSION. Each cell is a self-normalised weighted average over draws, so
    # logsumexp_s(logw) must be subtracted PER CELL -- log S with flat weights.
    # Omit it and every cell gains log S, so a pointwise total (many cells) is
    # inflated by n_cells * log S over a joint total (one cell). Both totals stay
    # finite and plausible; only the comparison between them is destroyed.
    #
    # Pinned against a hand-computed value rather than against another
    # aggregation: comparing two of our own estimators to each other would have
    # missed this entirely.
    S, C = 4, 3
    v = -0.5
    lp = [Dict{Any,Float64}(c => v for c in 1:C) for _ in 1:S]
    logw = zeros(S)
    # every draw gives every cell exactly `v`, so each cell's average IS `v`,
    # and the window is C * v. No log S anywhere.
    @test isapprox(aggregate_cells(logw, lp), C * v; atol = 1e-12)

    # and the same holds for one cell, which is the Joint() case
    @test isapprox(aggregate_cells(logw, [Dict{Any,Float64}(1 => v) for _ in 1:S]),
                   v; atol = 1e-12)

    # non-flat weights: still a weighted average, so a constant cell is unchanged
    w = [log(1.0), log(3.0), log(2.0), log(4.0)]
    @test isapprox(aggregate_cells(w, lp), C * v; atol = 1e-12)
end

@testset "aggregate_cells: shape errors" begin
    lp = [Dict{Any,Float64}(1 => -1.0) for _ in 1:3]
    @test_throws ArgumentError aggregate_cells(zeros(2), lp)
end

# ---------------------------------------------------------------------------
# result object
# ---------------------------------------------------------------------------

function _fake_result(; cutoffs = 1:10, base = -100.0, shift = 0.0, M = 2,
                        grans = [:joint, :pointwise])
    ws = WindowResult[]
    rng = StableRNG(7)
    for t in cutoffs
        e = Dict(g => base + shift + randn(rng) for g in grans)
        push!(ws, WindowResult(t, e, 50,
                               Dict(g => 50 for g in grans),
                               Dict(g => 10 for g in grans),
                               17, 1.0, 0.1))
    end
    LFOResult(ws, collect(grans), first(cutoffs), M, Dict{Symbol,Any}())
end

@testset "LFOResult: totals and diagnostics" begin
    r = _fake_result()
    @test length(cutoffs(r)) == 10
    @test elpd(r, :joint) ≈ sum(w.elpd[:joint] for w in r.windows)
    @test elpd(r, Joint()) == elpd(r, :joint)          # type or name, same answer
    @test n_informative(r) == 170                      # 17 per window x 10
    @test_throws ArgumentError elpd(r, :nonesuch)
end

@testset "compare: refuses the invalid comparisons" begin
    a = _fake_result(shift = 0.0)
    b = _fake_result(shift = -5.0)

    c = compare(a, b; granularity = :pointwise)
    @test c.n_windows == 10
    @test c.diff > 0                                    # a is better by construction
    @test isfinite(c.se) && isfinite(c.se_indep)
    # overlapping windows make the naive SE optimistic; the independent-subset SE
    # is the conservative reading and must not be smaller
    @test c.se_indep >= c.se || isnan(c.se_indep)

    # a granularity neither result has
    @test_throws ArgumentError compare(a, b; granularity = :by_group)

    # different M is not comparable
    b4 = _fake_result(shift = -5.0, M = 4)
    @test_throws ArgumentError compare(a, b4; granularity = :pointwise)

    # no shared cutoffs
    far = _fake_result(cutoffs = 100:105)
    @test_throws ArgumentError compare(a, far; granularity = :pointwise)
end

@testset "compare: only shared cutoffs are used" begin
    a = _fake_result(cutoffs = 1:10)
    b = _fake_result(cutoffs = 1:6)
    c = compare(a, b; granularity = :joint)
    @test c.n_windows == 6          # NOT 10 -- a 10-window total vs a 6-window
end                                 # total is not a comparison
