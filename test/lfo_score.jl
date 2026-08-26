# Scoring one leave-future-out window.
#
# The gates that matter:
#  * the absorbing state is DERIVED from the declared transitions, and a model
#    without one gets `nothing` rather than a frozen real state,
#  * a `-Inf` from one individual is charged to that individual's cell and the
#    loop continues -- returning early would silently re-impose joint scoring,
#  * the survival constraint and its weight cover exactly the same individuals.

# An S/I model with NO absorbing state.
function _si_no_death(; n_t = 8)
    state_space = [:S, :I]
    aggs = @aggregate state_space begin
        @array n_infected Int (1, n_t)
        n_infected[1, t] += (state == :I)
    end
    spec = @transitions state_space begin
        S -> I = (model, data, i, t) -> 0.1
        I -> S = (model, data, i, t) -> 0.2
    end
    epidemic_data(; n_individuals = 4, n_timepoints = n_t, group = ones(Int, 4),
                  trans_mat = spec, aggregates = aggs,
                  starting_state = (model, data, X, i, t) -> [0.9, 0.1])
end

# An S/I/D model where D is absorbing.
function _sid(; n_t = 8)
    state_space = [:S, :I, :D]
    aggs = @aggregate state_space begin
        @array n_infected Int (1, n_t)
        n_infected[1, t] += (state == :I)
    end
    spec = @transitions state_space begin
        S -> I = (model, data, i, t) -> 0.1
        S -> D = (model, data, i, t) -> 0.05
        I -> S = (model, data, i, t) -> 0.2
        I -> D = (model, data, i, t) -> 0.1
    end
    epidemic_data(; n_individuals = 4, n_timepoints = n_t, group = ones(Int, 4),
                  trans_mat = spec, aggregates = aggs,
                  starting_state = (model, data, X, i, t) -> [0.9, 0.1, 0.0])
end

@testset "absorbing state is derived, not assumed" begin
    @test EpidemicTrajectories._absorbing_state(_si_no_death()) === nothing
    @test EpidemicTrajectories._absorbing_state(_sid()) == 3      # :D
end

@testset "forward_simulate: shape, determinism, and absorption" begin
    data = _sid()
    X = fill(1, data.n_timepoints, data.n_individuals)
    model = (;)

    Xf = forward_simulate(StableRNG(1), model, data, X, 4, 2)
    @test size(Xf) == size(X)
    @test Xf[1:4, :] == X[1:4, :]                     # the past is untouched
    @test X == fill(1, size(X)...)                    # and the input is not modified
    @test forward_simulate(StableRNG(1), model, data, X, 4, 2) == Xf   # same seed

    # once absorbed, always absorbed
    Xd = copy(X); Xd[4, 1] = 3
    Xf2 = forward_simulate(StableRNG(2), model, data, Xd, 4, 2)
    @test Xf2[5, 1] == 3 && Xf2[6, 1] == 3
end

@testset "a -Inf individual costs its own cell, not the window" begin
    data = _sid()
    X = fill(1, data.n_timepoints, data.n_individuals)
    model = (;)

    # individual 1 is inadmissible everywhere; everyone else is fine
    cld(model, data, Xf, i, t) = i == 1 ? -Inf : -0.5

    cells, _ = score_window(model, data, X, 4, 2, Pointwise();
                            cell_logdensity = cld, rng = StableRNG(3))
    bad = [v for (k, v) in cells if k[1] == 1]
    good = [v for (k, v) in cells if k[1] != 1]
    @test !isempty(bad) && all(==(-Inf), bad)          # charged where it belongs
    @test !isempty(good) && all(isfinite, good)        # and nowhere else

    # under Joint the same failure takes the whole window -- that contrast is the
    # entire reason the granularity axis exists
    jcells, _ = score_window(model, data, X, 4, 2, Joint();
                             cell_logdensity = cld, rng = StableRNG(3))
    @test all(==(-Inf), values(jcells))
end

@testset "n_informative is reported, and counts only what the user says" begin
    data = _sid()
    X = fill(1, data.n_timepoints, data.n_individuals)
    cld(model, data, Xf, i, t) = -0.5

    # nothing informative
    _, n0 = score_window((;), data, X, 4, 2, Pointwise();
                         cell_logdensity = cld, rng = StableRNG(4),
                         is_informative = (d, i, t) -> false)
    @test n0 == 0

    # only individual 2, only at the first step
    _, n1 = score_window((;), data, X, 4, 2, Pointwise();
                         cell_logdensity = cld, rng = StableRNG(4),
                         is_informative = (d, i, t) -> i == 2 && t == 5)
    @test n1 == 1
end

@testset "survival_constrained: constraint and weight are co-gated" begin
    data = _sid()
    X = fill(1, data.n_timepoints, data.n_individuals)
    model = (;)
    # individuals 1 and 2 are known present throughout the window
    known(i, t) = i <= 2
    constrain, wfun = survival_constrained(known)

    # the constraint holds: neither 1 nor 2 may be absorbed
    Xf = forward_simulate(StableRNG(5), model, data, X, 4, 2; constrain)
    @test all(Xf[t, i] != 3 for t in 5:6, i in 1:2)

    # the weight covers EXACTLY the constrained individuals -- this is the
    # co-gating rule, and breaking it is a bias that does not shrink with n_sim
    w = wfun(model, data, Xf, 4, 2, Pointwise())
    covered = sort(unique(k[1] for k in keys(w)))
    @test covered == [1, 2]
    @test all(<=(0.0), values(w))          # log of a probability

    # every weight lands in the same cell the density did
    cells, _ = score_window(model, data, X, 4, 2, Pointwise();
                            cell_logdensity = (m, d, Xf, i, t) -> -0.5,
                            rng = StableRNG(5), constrain = constrain,
                            survival_weight = wfun)
    @test issubset(keys(w), keys(cells))
end

@testset "no absorbing state: the constraint is a no-op, not an error" begin
    data = _si_no_death()
    X = fill(1, data.n_timepoints, data.n_individuals)
    constrain, wfun = survival_constrained((i, t) -> true)
    # must not throw, and must produce no weights: there is no death to forbid
    Xf = forward_simulate(StableRNG(6), (;), data, X, 4, 2; constrain)
    @test size(Xf) == size(X)
    @test isempty(wfun((;), data, Xf, 4, 2, Pointwise()))
end
