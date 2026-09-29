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
    @test cell_of(ByIndividual(), 3, 2) == cell_of(ByIndividual(), 3, 1)
    @test cell_of(ByIndividual(), 3, 2) != cell_of(ByIndividual(), 4, 2)
    @test cell_of(g, 1, 2) == cell_of(g, 2, 2)                  # same group, same step
    @test cell_of(g, 1, 2) != cell_of(g, 3, 2)                  # different group
    @test cell_of(g, 1, 1) != cell_of(g, 1, 2)                  # different step
end

@testset "animal-history score keeps temporal dependence" begin
    # One animal survives each interval with probability q. Observing it alive
    # twice has probability q^2; the two marginal alive probabilities multiply
    # to q^3. Each posterior draw is a complete future history.
    q = 0.8
    S = 100
    raw = [s <= 64 ? (true, true) : s <= 80 ? (true, false) : (false, false)
           for s in 1:S]
    animal = [Dict{Any,Float64}(1 => (a && b ? 0.0 : -Inf)) for (a, b) in raw]
    steps = [Dict{Any,Float64}((1, 1) => (a ? 0.0 : -Inf),
                                (1, 2) => (b ? 0.0 : -Inf)) for (a, b) in raw]
    @test aggregate_cells(zeros(S), animal) ≈ log(q^2)
    @test aggregate_cells(zeros(S), steps) ≈ log(q^3)
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

# ---------------------------------------------------------------------------
# model weights
# ---------------------------------------------------------------------------
#
# The gates that matter here:
#  * identical models split evenly, and pseudo-BMA is exactly a softmax of the
#    totals -- both checked against a hand-computed value, not against the
#    other method.
#  * stacking actually maximises its own objective, so it must beat both equal
#    weights and the best single model on the mixture log-score.
#  * a window where ANY model is non-finite is dropped rather than kept, since
#    keeping it would let one model's degeneracy set the weights.

"An LFOResult with exactly the per-window scores given."
function _result_from(scores; cuts = nothing, M = 1, g = :joint)
    cs = cuts === nothing ? (1:length(scores)) : cuts
    ws = [WindowResult(c, Dict(g => s), 100, Dict(g => 100), Dict(g => 1),
                       10, 0.0, 0.0) for (c, s) in zip(cs, scores)]
    LFOResult(ws, [g], first(cs), M, Dict{Symbol,Any}())
end

_mixlog(lp, w) = sum(log(sum(w[k] * exp(lp[t, k]) for k in axes(lp, 2)))
                     for t in axes(lp, 1))

@testset "model_weights: identical models split evenly" begin
    a = _result_from(fill(-1.0, 8)); b = _result_from(fill(-1.0, 8))
    for m in (:stacking, :pseudo_bma)
        r = model_weights(Dict(:a => a, :b => b); granularity = :joint, method = m)
        @test r.weights ≈ [0.5, 0.5] atol = 1e-6
    end
end

@testset "model_weights: pseudo-BMA is a softmax of the totals" begin
    # totals -3 and -6, so the weights are e^3/(1+e^3) and 1/(1+e^3).
    r = model_weights(Dict(:a => _result_from(fill(-1.0, 3)),
                           :b => _result_from(fill(-2.0, 3)));
                      granularity = :joint, method = :pseudo_bma)
    want = exp(3) / (1 + exp(3))
    ia = findfirst(==("a"), r.names)
    @test r.weights[ia] ≈ want atol = 1e-8
    @test sum(r.weights) ≈ 1
end

@testset "model_weights: stacking maximises the mixture score" begin
    # Alternating experts: each model is far better in half the windows, so a
    # mixture beats either alone and stacking must keep both.
    lp = [0.0 -10.0; -10.0 0.0; 0.0 -10.0; -10.0 0.0; 0.0 -9.0]
    r = model_weights(Dict(:a => _result_from(lp[:, 1]),
                           :b => _result_from(lp[:, 2]));
                      granularity = :joint)
    ia = findfirst(==("a"), r.names)
    w = ia == 1 ? r.weights : reverse(r.weights)
    @test minimum(w) > 0.2                        # both experts kept
    K = size(lp, 2)
    @test _mixlog(lp, w) >= _mixlog(lp, fill(1 / K, K)) - 1e-8
    @test _mixlog(lp, w) >= maximum(_mixlog(lp, [i == k ? 1.0 : 0.0 for i in 1:K])
                                    for k in 1:K) - 1e-8

    # ... where pseudo-BMA, seeing only the totals, collapses onto one.
    rb = model_weights(Dict(:a => _result_from(lp[:, 1]),
                            :b => _result_from(lp[:, 2]));
                       granularity = :joint, method = :pseudo_bma)
    @test minimum(rb.weights) < 0.05
end

@testset "model_weights: a uniformly dominated model gets no weight" begin
    r = model_weights(Dict(:good => _result_from(fill(0.0, 5)),
                           :bad  => _result_from(fill(-50.0, 5)));
                      granularity = :joint)
    @test r.weights[findfirst(==("bad"), r.names)] < 1e-3
end

@testset "model_weights: non-finite windows are dropped, not hidden" begin
    r = model_weights(Dict(:a => _result_from([-1.0, -Inf, -1.0]),
                           :b => _result_from([-2.0, -2.0, -2.0]));
                      granularity = :joint)
    @test r.n_dropped == 1
    @test r.n_windows == 2
    # every window unusable is an error, not a silent uniform answer
    @test_throws ArgumentError model_weights(
        Dict(:a => _result_from([-Inf, -Inf]), :b => _result_from([-1.0, -1.0]));
        granularity = :joint)
end

@testset "model_weights: only shared cutoffs, and the usual refusals" begin
    r = model_weights(Dict(:a => _result_from(fill(-1.0, 3); cuts = [1, 2, 3]),
                           :b => _result_from(fill(-2.0, 3); cuts = [2, 3, 4]));
                      granularity = :joint)
    @test r.n_windows == 2

    @test_throws ArgumentError model_weights(
        Dict(:a => _result_from([-1.0]), :b => _result_from([-2.0]));
        granularity = :pointwise)                      # nobody ran it
    @test_throws ArgumentError model_weights(
        Dict(:a => _result_from([-1.0])); granularity = :joint)   # one model
    @test_throws ArgumentError model_weights(
        Dict(:a => _result_from([-1.0, -1.0]; M = 1),
             :b => _result_from([-2.0, -2.0]; M = 4));
        granularity = :joint)                          # different M
    @test_throws ArgumentError model_weights(
        Dict(:a => _result_from([-1.0]; cuts = [1]),
             :b => _result_from([-2.0]; cuts = [99]));
        granularity = :joint)                          # no shared cutoffs
    @test_throws ArgumentError model_weights(
        Dict(:a => _result_from([-1.0]), :b => _result_from([-2.0]));
        granularity = :joint, method = :nonesuch)
end

@testset "model_weights: the matrix method matches the LFOResult one" begin
    # Per-window scores survive a session where an LFOResult does not, so the
    # matrix method is what a cluster pipeline gathers with. It must agree
    # exactly with the object method on the same numbers.
    lp = [0.0 -10.0; -10.0 0.0; 0.0 -10.0; -10.0 0.0; 0.0 -9.0]
    a = model_weights(Dict(:a => _result_from(lp[:, 1]),
                           :b => _result_from(lp[:, 2])); granularity = :joint)
    b = model_weights(lp; names = ["a", "b"])
    ia = findfirst(==("a"), a.names); ib = findfirst(==("a"), b.names)
    @test a.weights[ia] ≈ b.weights[ib] atol = 1e-8
    @test b.n_windows == 5 && b.n_dropped == 0

    # and it drops non-finite rows the same way
    c = model_weights([0.0 -1.0; -Inf -1.0; 0.0 -1.0]; names = ["a", "b"])
    @test c.n_dropped == 1 && c.n_windows == 2

    @test_throws ArgumentError model_weights(reshape([1.0, 2.0], 2, 1))
    @test_throws ArgumentError model_weights(lp; names = ["only_one"])
end
