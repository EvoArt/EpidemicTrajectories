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

@testset "forward_simulate updates coupled population summaries" begin
    state_space = [:S, :I]
    aggs = @aggregate state_space begin
        @array n_infected Int (1, 6)
        n_infected[1, t] += (state == :I)
    end
    spec = @transitions state_space begin
        S -> I = (model, data, i, t) ->
            data.aggregates.n_infected[1, t] > 0 ? 1.0 : 0.0
    end
    data = epidemic_data(; n_individuals = 2, n_timepoints = 6,
                         group = ones(Int, 2), trans_mat = spec,
                         aggregates = aggs,
                         starting_state = (model, data, X, i, t) -> [1.0, 0.0])
    X = fill(1, 6, 2)
    X[4, 1] = 2

    Xf = forward_simulate(StableRNG(9), (;), data, X, 4, 1)
    @test Xf[5, 2] == 2
    @test data.aggregates.n_infected[1, 5] == 2
end

@testset "future entrants are initialized but not scored" begin
    state_space = [:S, :I, :D]
    aggs = @aggregate state_space begin
        @array n_infected Int (1, 7)
        n_infected[1, t] += (state == :I)
    end
    spec = @transitions state_space begin
        S -> I = (model, data, i, t) -> 0.0
        S -> D = (model, data, i, t) -> 0.0
        I -> S = (model, data, i, t) -> 0.0
        I -> D = (model, data, i, t) -> 0.0
    end
    data = epidemic_data(;
        n_individuals=2, n_timepoints=7, group=ones(Int, 2),
        trans_mat=spec, aggregates=aggs, sampling_period=[(1, 7), (5, 7)],
        starting_state=(model, data, X, i, t) ->
            i == 2 ? [0.0, 1.0, 0.0] : [1.0, 0.0, 0.0])
    X = fill(1, 7, 2)

    Xf = forward_simulate(StableRNG(72), (;), data, X, 4, 2)
    @test Xf[5, 2] == 2              # drawn at entry, not advanced from X[4,2]

    cells, n_inf = score_window((;), data, X, 4, 2, Pointwise();
        cell_logdensity=(m, d, path, i, t) -> -0.5,
        is_informative=(d, i, t) -> true, rng=StableRNG(72))
    @test all(k[1] == 1 for k in keys(cells))
    @test n_inf == 2

    # The survival proposal must use the same cutoff cohort as the density.
    constrain, wfun = survival_constrained((i, t) -> true)
    Xc = forward_simulate(StableRNG(73), (;), data, X, 4, 2; constrain)
    w = wfun((;), data, Xc, 4, 2, Pointwise())
    @test all(k[1] == 1 for k in keys(w))
end

@testset "entrants = false fixes the forecast population at the cutoff" begin
    # Individual 2 enters at 5 already infectious, and individual 1 is infected
    # only if someone else is. Simulated in, it infects 1; left out, it cannot.
    state_space = [:S, :I]
    aggs = @aggregate state_space begin
        @array n_infected Int (1, 7)
        n_infected[1, t] += (state == :I)
    end
    spec = @transitions state_space begin
        S -> I = (model, data, i, t) -> data.aggregates.n_infected[1, t] > 0 ? 1.0 : 0.0
    end
    data = epidemic_data(;
        n_individuals=2, n_timepoints=7, group=ones(Int, 2),
        trans_mat=spec, aggregates=aggs, sampling_period=[(1, 7), (5, 7)],
        starting_state=(model, data, X, i, t) -> i == 2 ? [0.0, 1.0] : [1.0, 0.0])
    X = fill(1, 7, 2); X[6:7, 2] .= 2           # placeholders the fit never saw
    with = forward_simulate(StableRNG(1), (;), data, X, 4, 3)
    without = forward_simulate(StableRNG(1), (;), data, X, 4, 3; entrants = false)
    @test with[6, 1] == 2
    @test all(==(1), without[5:7, 1])
    @test without[5:7, 2] == X[5:7, 2]           # left alone, not simulated
end

@testset "an individual not yet known is neither scored nor simulated" begin
    # Individual 2 is present from t = 1 but not known until t = 6. At cutoff 4
    # the fit never sampled it, so X holds a placeholder: here, infectious.
    # Counting that placeholder would make individual 1 certain to be infected.
    state_space = [:S, :I, :D]
    aggs = @aggregate state_space begin
        @array n_infected Int (1, 7)
        n_infected[1, t] += (state == :I)
    end
    spec = @transitions state_space begin
        S -> I = (model, data, i, t) -> data.aggregates.n_infected[1, t] > 0 ? 1.0 : 0.0
        S -> D = (model, data, i, t) -> 0.0
        I -> S = (model, data, i, t) -> 0.0
        I -> D = (model, data, i, t) -> 0.0
    end
    data = epidemic_data(;
        n_individuals=2, n_timepoints=7, group=ones(Int, 2),
        trans_mat=spec, aggregates=aggs,
        starting_state=(model, data, X, i, t) -> [1.0, 0.0, 0.0])
    X = fill(1, 7, 2)
    X[:, 2] .= 2
    known = [1, 6]

    Xf = forward_simulate(StableRNG(91), (;), data, X, 4, 2; known)
    @test Xf[5, 1] == 1 && Xf[6, 1] == 1
    @test Xf[5:6, 2] == X[5:6, 2]            # not stepped

    # the default reproduces the old behaviour: the placeholder infects
    Xd = forward_simulate(StableRNG(91), (;), data, X, 4, 2)
    @test Xd[5, 1] == 2

    cells, _ = score_window((;), data, X, 4, 2, Pointwise();
        cell_logdensity=(m, d, path, i, t) -> path[t, i] == 1 ? 0.0 : -Inf,
        rng=StableRNG(91), known)
    @test all(k[1] == 1 for k in keys(cells))
    @test all(==(0.0), values(cells))

    constrain, wfun = survival_constrained((i, t) -> true)
    Xc = forward_simulate(StableRNG(92), (;), data, X, 4, 2; constrain, known)
    w = wfun((;), data, Xc, 4, 2, Pointwise(); known)
    @test all(k[1] == 1 for k in keys(w))
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

@testset "individual scoring keeps an animal's future steps together" begin
    data = _sid()
    X = fill(1, data.n_timepoints, data.n_individuals)
    cld(model, data, Xf, i, t) = i == 1 && t == 6 ? -Inf : -0.5
    cells, _ = score_window((;), data, X, 4, 2, ByIndividual();
                            cell_logdensity = cld, rng = StableRNG(3))
    @test cells[1] == -Inf
    @test all(isfinite(v) for (i, v) in cells if i != 1)
    @test length(cells) == data.n_individuals
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

@testset "known_present can read the end of the window it is in" begin
    data = _sid()
    X = fill(1, data.n_timepoints, data.n_individuals)
    seen = Set{Int}()
    kp = (i, t, last_t) -> (push!(seen, last_t); i == 1)
    constrain, wfun = survival_constrained(kp)
    Xf = forward_simulate(StableRNG(5), (;), data, X, 4, 3; constrain)
    wfun((;), data, Xf, 4, 3, Joint())
    @test seen == Set([7])                          # t_star + M, and nothing else
    @test all(Xf[t, 1] != 3 for t in 5:7)
    # the window end is clipped to the series like the window itself
    empty!(seen)
    wfun((;), data, forward_simulate(StableRNG(5), (;), data, X, 6, 5; constrain),
         6, 5, Joint())
    @test seen == Set([data.n_timepoints])
end

@testset "overlapping windows each see their own end" begin
    # stride < M puts a step in several windows with different ends. Scored in
    # one run, every window must equal the same window scored alone -- the
    # forward seeds depend only on the draw and the cutoff, so exactly -- and
    # equal a run whose constraint has that window's end written in.
    data = _sid()
    caught = Dict(1 => (6, 8), 2 => (5,), 3 => (8,))
    seen(i, t, last_t) = any(u -> t <= u <= last_t, get(caught, i, ()))
    S = 50
    fit = (train, t) -> (fill((;), S), [fill(1, data.n_timepoints, 4) for _ in 1:S])
    cld = (m, d, X, i, t) -> X[t, i] == 3 ? log(0.9) : log(0.4)
    function run(kp, cutoffs)
        constrain, survival_weight = survival_constrained(kp)
        spec = LFOSpec(fit = fit, plan = truncation(), cell_logdensity = cld,
                       constrain = constrain, survival_weight = survival_weight,
                       n_sim = 4, seed = 3)
        res = lfo_cv(spec, data; L = 3, M = 3, stride = 1, cutoffs = cutoffs,
                     granularity = Joint(), verbose = false)
        Dict(w.cutoff => w.elpd[:joint] for w in res.windows)
    end
    together = run(seen, [3, 4, 5])
    for c in (3, 4, 5)
        @test together[c] == run(seen, [c])[c]
        @test together[c] == run((i, t, _) -> seen(i, t, c + 3), [c])[c]
    end
    # and the end matters: the window at 3 ends at 6, so individual 3's only
    # capture, at 8, must not keep it alive there; reading past the end would
    # change the score
    @test together[3] != run((i, t, _) -> seen(i, t, 8), [3])[3]
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
