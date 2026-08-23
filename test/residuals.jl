# Trajectory summaries / residuals.
#
# A residual is a DIAGNOSTIC, so "it runs" proves nothing whatsoever. Two tests
# carry all the weight here and everything else is scaffolding around them:
#
#   CALIBRATION — under a correct model, randomized PIT residuals are EXACTLY
#     Uniform(0,1). This fails if the accumulator, the censoring or the clock
#     origin is wrong, which is the entire class of silent error this layer can
#     have. Tested twice: once as a single uniformity test, and once as the
#     distribution of p-values across many independent simulations (which catches
#     a residual subtly miscalibrated in a way one test passes by luck).
#
#   POWER — under a wrong model, the uniformity test must REJECT. A residual that
#     never fires is worthless, and nothing else in the suite catches that.
#
# Do not weaken either one. If a future change makes calibration fail, the change
# is wrong; if it makes power fail, the residual has stopped being a diagnostic.

using EpidemicTrajectories: _resolve_spec

# --- a small S -> E -> I model to test against ------------------------------
# Constant rates, so every waiting time has a closed form and the residual's
# correctness is checkable by hand rather than only by simulation.

const RT = 60          # timepoints
const RM = 400         # individuals

res_infection(model, data, i, t) = model.alpha
res_progression(model, data, i, t) = model.gamma
res_survival(model, data, i, t) = model.psurv

res_start(model, data, X, i, t) = [1.0, 0.0, 0.0]

# The SEI spec used by most tests: no survival, so `trans_mat.coupling` is
# `nothing` and the `:coupling` hazard view falls back to the full spec.
function res_sei_setup(; m=RM, n_t=RT, sampling_period=nothing)
    spec = @transitions [:S, :E, :I] begin
        S -> E = res_infection
        E -> I = res_progression
    end
    aggs = @aggregate [:S, :E, :I] begin
        @array n_exposed Int (1, n_t)
        n_exposed[1, t] += (state == :E)
    end
    data = epidemic_data(; n_individuals=m, n_timepoints=n_t, trans_mat=spec,
                           starting_state=res_start, aggregates=aggs,
                           sampling_period=sampling_period)
    return data
end

# A KS test against Uniform(0,1) with the asymptotic (Kolmogorov) p-value. Written
# out rather than pulled from HypothesisTests because that is a WEAK dependency of
# the package, deliberately absent from the four-dependency core — the tests must
# run without it.
function res_ks_uniform(x)
    y = sort(x)
    n = length(y)
    d = 0.0
    for (k, v) in enumerate(y)
        d = max(d, abs(k / n - v), abs(v - (k - 1) / n))
    end
    lam = (sqrt(n) + 0.12 + 0.11 / sqrt(n)) * d
    p = 2 * sum((-1)^(j - 1) * exp(-2 * j^2 * lam^2) for j in 1:100)
    return (d, clamp(p, 0.0, 1.0))
end

# =============================================================================
# THE HEADLINE TESTS
# =============================================================================

@testset "calibration: PIT residuals are Uniform(0,1) under a correct model" begin
    data = res_sei_setup(; m=800)
    model = (; alpha=0.05, gamma=0.15)
    X = epidemic_simulator(data)(StableRNG(11), model)

    prog = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:prog)
    R = trajectory_summaries((prog,), data, [(model, X)]; rng=StableRNG(2))
    v = residual_values(R, :prog)

    @test length(v) > 500                       # the model actually produced events
    @test all(0 .<= v .<= 1)                    # a PIT residual is in the unit interval
    @test isapprox(mean(v), 0.5; atol=0.05)     # Uniform(0,1) has mean 1/2
    @test isapprox(std(v), 1 / sqrt(12); atol=0.03)

    d, p = res_ks_uniform(v)
    @test p > 0.01                              # does NOT reject a correct model
end

@testset "power: the residual REJECTS a misspecified model" begin
    data = res_sei_setup(; m=800)
    truth = (; alpha=0.05, gamma=0.15)
    X = epidemic_simulator(data)(StableRNG(11), truth)
    prog = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:prog)

    # Score the SAME trajectory under wrong progression rates. Scoring one fixed
    # X isolates the residual: the only thing that changed is the model it is
    # evaluated under, so a rejection can only be the residual detecting that.
    for wrong_gamma in (0.05, 0.30, 0.45)
        wrong = (; alpha=0.05, gamma=wrong_gamma)
        Rw = trajectory_summaries((prog,), data, [(wrong, X)]; rng=StableRNG(2))
        vw = residual_values(Rw, :prog)
        _, pw = res_ks_uniform(vw)
        @test pw < 1e-6                         # rejects, and not marginally
    end

    # And the DIRECTION is informative, not just the rejection: too-slow assumed
    # progression pushes the residuals towards 0, too-fast towards 1. This is what
    # makes the residual localise a failure rather than merely announce one.
    slow = residual_values(
        trajectory_summaries((prog,), data, [((; alpha=0.05, gamma=0.05), X)]; rng=StableRNG(2)), :prog)
    fast = residual_values(
        trajectory_summaries((prog,), data, [((; alpha=0.05, gamma=0.45), X)]; rng=StableRNG(2)), :prog)
    @test mean(slow) < 0.35
    @test mean(fast) > 0.65
end

@testset "calibration: the p-value distribution is itself uniform" begin
    # The stronger form of the calibration test. A residual that is subtly
    # miscalibrated can pass ONE uniformity test by luck; across many independent
    # simulations the p-values must themselves be Uniform(0,1), and in particular
    # the rejection rate at 5% must be ~5%. This is π<0.05 — the reference's own
    # headline calibration statistic — used here as a test.
    data = res_sei_setup()
    model = (; alpha=0.05, gamma=0.15)
    sim = epidemic_simulator(data)
    prog = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:prog)

    ps = Float64[]
    for seed in 1:80
        X = sim(StableRNG(seed), model)
        R = trajectory_summaries((prog,), data, [(model, X)]; rng=StableRNG(1000 + seed))
        v = residual_values(R, :prog)
        length(v) < 50 && continue
        push!(ps, res_ks_uniform(v)[2])
    end

    @test length(ps) > 70
    # The 5% rejection rate is the property that matters and the one a
    # miscalibrated residual breaks. Bounded generously: with ~80 simulations the
    # binomial noise on a true 0.05 is itself about ±0.05.
    @test mean(ps .< 0.05) < 0.15
    @test mean(ps) > 0.35            # not systematically pushed towards rejection
end

# =============================================================================
# THE ACCUMULATOR DISTINCTION — the regression test for the no-default rule
# =============================================================================

@testset "accumulate has NO default and the two accumulators genuinely differ" begin
    # Omitting `accumulate` must be an ERROR, never a silent choice. The two
    # accumulators agree only when p_s = 1 - exp(-λ_s) exactly and diverge at
    # large per-step hazard — precisely where the residual is informative — so a
    # default would produce plausible wrong numbers with no warning. If a later
    # "simplification" adds one, this test is what fails.
    @test_throws Exception WaitingTimeResidual(:E => :I)
    @test_throws Exception WaitingTimeResidual(:E => :I; accumulate=:something_else)

    # Each accumulator matches its OWN closed form...
    h = t -> 0.4
    @test discrete_product_cdf(h, 1, 5) ≈ 1 - 0.6^5
    @test cumulative_hazard_cdf(h, 1, 5) ≈ 1 - exp(-2.0)

    # ...and at large per-step hazard they measurably differ.
    @test abs(discrete_product_cdf(h, 1, 5) - cumulative_hazard_cdf(h, 1, 5)) > 0.05

    # They converge only in the small-hazard limit, where p ≈ λ. That is the whole
    # reason a default looks harmless right up until it isn't.
    small = t -> 1e-4
    @test isapprox(discrete_product_cdf(small, 1, 5), cumulative_hazard_cdf(small, 1, 5); atol=1e-6)

    # Zero or negative step counts are no event, not an error.
    @test discrete_product_cdf(h, 1, 0) == 0.0
    @test cumulative_hazard_cdf(h, 1, -3) == 0.0
end

@testset "the accumulator choice changes the residual, not just the helper" begin
    # The distinction must survive all the way through the constructor — a
    # residual built with each accumulator must give different numbers on the same
    # trajectory. (Same rng, so the randomization is identical and only the
    # interval endpoints differ.)
    data = res_sei_setup(; m=300)
    model = (; alpha=0.08, gamma=0.4)          # large per-step hazard: they diverge
    X = epidemic_simulator(data)(StableRNG(5), model)

    disc = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:r)
    haz = WaitingTimeResidual(:E => :I; accumulate=:cumulative_hazard, name=:r)
    vd = residual_values(trajectory_summaries((disc,), data, [(model, X)]; rng=StableRNG(7)), :r)
    vh = residual_values(trajectory_summaries((haz,), data, [(model, X)]; rng=StableRNG(7)), :r)

    @test length(vd) == length(vh)
    @test !isapprox(mean(vd), mean(vh); atol=1e-3)
end

# =============================================================================
# THE HAZARD SEAM — survival must not leak into a progression residual
# =============================================================================

@testset "the hazard comes from the SURVIVAL-FREE view by default" begin
    # `@survival` scales every live transition, so `trans_mat`'s E->I rate is
    # `survival * progression`. A progression residual wants the bare progression
    # hazard: accumulating the survival-scaled one folds mortality into it, and
    # the result still looks like a perfectly good residual. This is the single
    # most dangerous silent error available to this layer.
    spec = @transitions [:S, :E, :I, :D] begin
        @survival res_survival death=:D
        S -> E = res_infection
        E -> I = res_progression
    end
    @test spec.coupling !== nothing                  # @survival stashed the bare view

    aggs = @aggregate [:S, :E, :I, :D] begin
        @array n_exposed Int (1, RT)
        n_exposed[1, t] += (state == :E)
    end
    data = epidemic_data(; n_individuals=50, n_timepoints=RT, trans_mat=spec,
                           starting_state=(m, d, X, i, t) -> [1.0, 0.0, 0.0, 0.0],
                           aggregates=aggs)
    model = (; alpha=0.05, gamma=0.15, psurv=0.9)
    X = epidemic_simulator(data)(StableRNG(3), model)

    # :coupling (the default) resolves to the survival-free spec; :full does not.
    @test _resolve_spec(data, :coupling) === spec.coupling
    @test _resolve_spec(data, :full) === spec
    @test_throws Exception _resolve_spec(data, :nonsense)

    bare = transition_hazard(data, :E, :I)                 # default, :coupling
    full = transition_hazard(data, :E, :I; spec=:full)

    # The bare hazard is the progression rate itself; the full one carries the
    # survival factor. If these ever become equal, the coupling view has been lost
    # and every survival-model residual is silently wrong.
    @test bare(model, data, X, 1, 1) ≈ model.gamma
    @test full(model, data, X, 1, 1) ≈ model.psurv * model.gamma
    @test !isapprox(bare(model, data, X, 1, 1), full(model, data, X, 1, 1))
end

@testset "hazard: a user-supplied one overrides the spec" begin
    # The seam for a semi-Markov hazard the transition spec cannot express.
    data = res_sei_setup(; m=100)
    model = (; alpha=0.05, gamma=0.15)
    X = epidemic_simulator(data)(StableRNG(4), model)

    called = Ref(0)
    my_hazard = (model, data, X, i, t) -> (called[] += 1; 0.5)
    r = WaitingTimeResidual(:E => :I; accumulate=:discrete_product,
                            hazard=my_hazard, name=:custom)
    R = trajectory_summaries((r,), data, [(model, X)]; rng=StableRNG(1))
    @test called[] > 0                                    # the override was used
    @test all(0 .<= residual_values(R, :custom) .<= 1)
end

# =============================================================================
# CENSORING AND EDGE CASES — where silent wrongness lives (design 6.5)
# =============================================================================

@testset "censoring: an individual that never leaves the origin state" begin
    # Fully censored: it entered E and stayed there. All we know is that the
    # waiting time exceeded the window, so the PIT must lie in [F(c), 1] — never
    # below F(c), and never dropped.
    data = res_sei_setup(; m=1, n_t=10)
    model = (; alpha=0.05, gamma=0.2)
    X = fill(2, 10, 1)                     # in E for the whole window, never I

    r = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:r)
    R = trajectory_summaries((r,), data, [(model, X)]; rng=StableRNG(1))
    v = R[:r][1, 1]

    # origin t0 = 1, censored at t_end = 10, so c = 9 steps of survival.
    fc = 1 - (1 - 0.2)^9
    @test v >= fc - 1e-12
    @test v <= 1
    @test R.coverage[:r] == 1.0            # censored is CONTRIBUTING, not dropped
end

@testset "censoring: a competing risk censors rather than drops" begin
    # An individual that dies before progressing has not falsified the model — it
    # stopped being observable. Dropping it would bias the residual towards
    # individuals who lived long enough to progress.
    data = res_sei_setup(; m=1, n_t=10)
    model = (; alpha=0.05, gamma=0.2)

    # Enters E at t=2, "dies" (state 3 used as the competing risk here) at t=5.
    X = fill(1, 10, 1)
    X[2:4, 1] .= 2
    X[5:10, 1] .= 3

    # With :I as the EVENT, this is a progression at L = 3.
    ev = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:ev)
    Rev = trajectory_summaries((ev,), data, [(model, X)]; rng=StableRNG(1))
    lb, ub = 1 - (1 - 0.2)^2, 1 - (1 - 0.2)^3
    @test lb - 1e-12 <= Rev[:ev][1, 1] <= ub + 1e-12

    # With :I as a CENSORING event instead (an event that never happens, censored
    # at the step before entering I), the residual is [F(c), 1] with c = 2.
    cen = WaitingTimeResidual(:S => :E; accumulate=:discrete_product,
                              censor_at=(:I, :window_end), name=:cen)
    Rc = trajectory_summaries((cen,), data, [(model, X)]; rng=StableRNG(1))
    @test !ismissing(Rc[:cen][1, 1])
end

@testset "edge cases: missing is returned, not a wrong number" begin
    data = res_sei_setup(; m=3, n_t=10)
    model = (; alpha=0.05, gamma=0.2)
    r = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:r)

    # Individual never in the origin state -> `missing`, and coverage records it.
    X = fill(1, 10, 3)                     # everyone stays in S forever
    R = trajectory_summaries((r,), data, [(model, X)]; rng=StableRNG(1))
    @test all(ismissing, R[:r])
    @test R.coverage[:r] == 0.0

    # An event at the FIRST possible step gives LB = 0 exactly: F(0) = 0.
    X2 = fill(2, 10, 3)
    X2[2:10, :] .= 3                       # E at t=1, I at t=2, so L = 1
    R2 = trajectory_summaries((r,), data, [(model, X2)]; rng=StableRNG(1))
    ub = 0.2                               # F(1) = 1 - (1-0.2)^1
    @test all(0 .<= skipmissing(R2[:r]) .<= ub + 1e-12)

    # A zero-length window contributes nothing rather than dividing by zero.
    data0 = res_sei_setup(; m=1, n_t=10, sampling_period=[(5, 4)])
    R0 = trajectory_summaries((r,), data0, [(model, fill(2, 10, 1))]; rng=StableRNG(1))
    @test ismissing(R0[:r][1, 1])
end

@testset "require_start_state is opt-in, not assumed" begin
    # Who is "at risk" is a modelling choice, not a fact about the transition, so
    # the package does not filter unless asked. The reference filters
    # unconditionally (skipping anyone not susceptible at t_start).
    data = res_sei_setup(; m=2, n_t=10)
    model = (; alpha=0.05, gamma=0.2)
    X = fill(2, 10, 2)                     # both start in E, not S
    X[5:10, :] .= 3

    without = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:r)
    with = WaitingTimeResidual(:E => :I; accumulate=:discrete_product,
                               require_start_state=:S, name=:r)
    Rwo = trajectory_summaries((without,), data, [(model, X)]; rng=StableRNG(1))
    Rw = trajectory_summaries((with,), data, [(model, X)]; rng=StableRNG(1))

    @test Rwo.coverage[:r] == 1.0          # not filtered by default
    @test Rw.coverage[:r] == 0.0           # filtered when asked
end

@testset "origin: :window_start vs :entry_to_from_state" begin
    # An exposure-time residual's clock starts when OBSERVATION starts, not when
    # the individual entered S — it was already susceptible when we began looking.
    # The two origins must therefore give different answers when they differ.
    data = res_sei_setup(; m=1, n_t=12, sampling_period=[(3, 12)])
    model = (; alpha=0.2, gamma=0.2)
    X = fill(1, 12, 1)
    X[7:12, 1] .= 2                        # S until t=6, E from t=7

    entry = WaitingTimeResidual(:S => :E; accumulate=:cumulative_hazard,
                                origin=:entry_to_from_state, name=:r)
    winst = WaitingTimeResidual(:S => :E; accumulate=:cumulative_hazard,
                                origin=:window_start, name=:r)
    ve = trajectory_summaries((entry,), data, [(model, X)]; rng=StableRNG(1))[:r][1, 1]
    vw = trajectory_summaries((winst,), data, [(model, X)]; rng=StableRNG(1))[:r][1, 1]

    # Both clocks start at t=3 here (the window start IS the first time in S
    # within the window), so they agree — which is itself worth pinning.
    @test ve ≈ vw

    @test_throws Exception WaitingTimeResidual(:S => :E; accumulate=:discrete_product,
                                               origin=:some_other_origin)
end

# =============================================================================
# LEFT-TRUNCATED SURVIVAL — the Tier 2 shape
# =============================================================================

@testset "left-truncated survival residual" begin
    # The survival residual differs from a waiting-time residual on all three axes
    # the spec cannot derive: an external clock origin, left truncation, and a
    # censoring time that may fall outside the monitoring window. It is a separate
    # constructor for exactly that reason.
    n_t = 20
    spec = @transitions [:S, :E, :I, :D] begin
        @survival res_survival death=:D
        S -> E = res_infection
        E -> I = res_progression
    end
    aggs = @aggregate [:S, :E, :I, :D] begin
        @array n_exposed Int (1, n_t)
        n_exposed[1, t] += (state == :E)
    end
    birth = [1, 1, 1]
    first_capture = [5, 5, 5]
    last_capture = [20, 20, 20]
    data = epidemic_data(; n_individuals=3, n_timepoints=n_t, trans_mat=spec,
                           starting_state=(m, d, X, i, t) -> [1.0, 0.0, 0.0, 0.0],
                           aggregates=aggs,
                           birth_time=birth, first_capture=first_capture,
                           last_capture=last_capture)
    model = (; alpha=0.05, gamma=0.15, psurv=0.9)

    surv = LeftTruncatedSurvivalResidual(;
        survival = res_survival,
        origin = i -> data.birth_time[i],
        condition_on = i -> data.first_capture[i],   # earliest known alive
        censor_at = i -> data.last_capture[i],
        death = :D, name = :survival)

    # Individual 1 dies at t=10, 2 dies at t=15, 3 survives the whole window.
    X = fill(1, n_t, 3)
    X[10:n_t, 1] .= 4
    X[15:n_t, 2] .= 4

    R = trajectory_summaries((surv,), data, [(model, X)]; rng=StableRNG(1))
    v = R[:survival]
    @test R.coverage[:survival] == 1.0
    @test all(0 .<= skipmissing(v) .<= 1)

    # The left truncation must actually be applied. Conditional on surviving to
    # t=5, the CDF at the death time is 1 - S(1->t)/S(1->5), which is STRICTLY
    # less than the unconditional 1 - S(1->t). Without the normaliser the residual
    # is calibrated against the wrong distribution.
    S(n) = 0.9^(n - 1)
    cond_at_death = 1 - S(10) / S(5)
    uncond_at_death = 1 - S(10)
    @test cond_at_death < uncond_at_death
    lb = 1 - S(9) / S(5)
    ub = 1 - S(10) / S(5)
    @test lb - 1e-12 <= v[1, 1] <= ub + 1e-12

    # A censored individual gets [F(c), 1].
    fc = 1 - S(20) / S(5)
    @test v[3, 1] >= fc - 1e-12
end

@testset "survival: an underflowing normaliser gives missing, not NaN" begin
    # If S(origin -> condition_on) underflows, the individual carries no
    # information. Returning `missing` shows up honestly as a coverage drop;
    # letting a NaN through would silently poison a downstream uniformity test.
    n_t = 10
    data = res_sei_setup(; m=1, n_t=n_t)
    model = (; alpha=0.05, gamma=0.15, psurv=0.0)     # survival 0 -> normaliser 0

    surv = LeftTruncatedSurvivalResidual(;
        survival = res_survival,
        origin = i -> 1, condition_on = i -> 5, censor_at = i -> n_t,
        death = 3, name = :survival)
    R = trajectory_summaries((surv,), data, [(model, fill(1, n_t, 1))]; rng=StableRNG(1))
    @test ismissing(R[:survival][1, 1])
    @test R.coverage[:survival] == 0.0
end

# =============================================================================
# THE TIER 2 / TIER 0 SEAMS
# =============================================================================

@testset "@residual: the user's own summary, no reverse required" begin
    data = res_sei_setup(; m=20, n_t=RT)
    model = (; alpha=0.05, gamma=0.2)
    X = epidemic_simulator(data)(StableRNG(9), model)

    # A :raw summary — a quantity that is not a PIT residual, so the uniformity
    # checks should know not to apply to it.
    @residual :raw exposure_time(model, data, X, i, rng) = begin
        tE = first_entry(X, i, state_code(data, :E), 1, data.n_timepoints)
        tE === nothing ? missing : Float64(tE)
    end
    @test exposure_time isa TrajectorySummary
    @test exposure_time.kind == :raw
    @test exposure_time.name == :exposure_time

    # A :pit summary using the package's helpers, exactly as a user would.
    @residual my_prog(model, data, X, i, rng) = begin
        tE = first_entry(X, i, state_code(data, :E), 1, data.n_timepoints)
        tE === nothing && return missing
        tI = first_entry(X, i, state_code(data, :I), tE, data.n_timepoints)
        h = t -> model.gamma
        if tI === nothing
            return randomized_pit(discrete_product_cdf(h, tE, data.n_timepoints - tE), 1.0, rng)
        end
        L = tI - tE
        randomized_pit(discrete_product_cdf(h, tE, L - 1), discrete_product_cdf(h, tE, L), rng)
    end
    @test my_prog.kind == :pit

    R = trajectory_summaries((exposure_time, my_prog), data, [(model, X)]; rng=StableRNG(1))
    @test R.kinds[:exposure_time] == :raw
    @test R.kinds[:my_prog] == :pit
    @test all(0 .<= residual_values(R, :my_prog) .<= 1)

    # The hand-written version must agree with the auto-derived one: the Tier 1
    # constructor is supposed to be sugar over exactly this, not a different
    # computation.
    auto = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:my_prog)
    Ra = trajectory_summaries((auto,), data, [(model, X)]; rng=StableRNG(1))
    @test residual_values(Ra, :my_prog) ≈ residual_values(R, :my_prog)

    # A signature with the wrong arity is an error at macro-expansion time.
    @test_throws Exception @eval @residual bad(model, data, X) = 1.0
end

@testset "TrajectorySummary: the Tier 0 contract is just a closure" begin
    data = res_sei_setup(; m=5, n_t=10)
    model = (; alpha=0.05, gamma=0.2)
    s = TrajectorySummary(:constant, (model, data, X, i, rng) -> 0.25)
    @test s.kind == :pit                          # the default
    R = trajectory_summaries((s,), data, [(model, fill(1, 10, 5))]; rng=StableRNG(1))
    @test all(==(0.25), R[:constant])
end

@testset "randomized_pit clamps and orders its bounds" begin
    rng = StableRNG(1)
    @test 0 <= randomized_pit(0.0, 1.0, rng) <= 1
    @test randomized_pit(0.3, 0.3, rng) ≈ 0.3           # a degenerate interval
    @test randomized_pit(-0.5, 1.5, rng) <= 1           # clamped into [0,1]
    @test randomized_pit(-0.5, 1.5, rng) >= 0
    # An accumulator overshooting by rounding must not escape the unit interval.
    @test randomized_pit(0.7, 0.6, rng) ≈ 0.7           # ub < lb is ordered, not NaN
end

@testset "first_entry / state_code" begin
    data = res_sei_setup(; m=2, n_t=10)
    X = fill(1, 10, 2)
    X[4:10, 1] .= 2
    @test first_entry(X, 1, 2, 1, 10) == 4
    @test first_entry(X, 2, 2, 1, 10) === nothing
    @test first_entry(X, 1, 2, 5, 10) == 5              # search starts at t_from
    @test first_entry(X, 1, 2, 1, 3) === nothing        # and stops at t_to
    @test first_entry(X, 1, 2, 1, 999) == 4             # range clipped to X, not an error

    @test state_code(data, :S) == 1
    @test state_code(data, :I) == 3
    @test state_code(data, 2) == 2                      # an index passes through
    @test_throws Exception state_code(data, :NotAState)
end

# =============================================================================
# THE DRIVER
# =============================================================================

@testset "driver: coverage, multiple draws, multiple summaries" begin
    data = res_sei_setup(; m=200, n_t=RT)
    model = (; alpha=0.05, gamma=0.15)
    sim = epidemic_simulator(data)
    draws = [(model, sim(StableRNG(s), model)) for s in 1:4]

    prog = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:prog)
    expo = WaitingTimeResidual(:S => :E; accumulate=:cumulative_hazard,
                               origin=:window_start, name=:expo)
    R = trajectory_summaries((prog, expo), data, draws; rng=StableRNG(1))

    @test R.n_draws == 4
    @test size(R[:prog]) == (200, 4)
    @test Set(keys(R)) == Set([:prog, :expo])
    @test haskey(R, :prog)
    @test length(R) == 2

    # Coverage is the assertion that catches a change silently dropping half the
    # population — every residual value would still be a perfectly good number.
    @test 0.0 < R.coverage[:prog] <= 1.0
    @test R.coverage[:expo] == 1.0            # everyone starts in S, so all contribute
    @test R.coverage[:prog] ≈ count(!ismissing, R[:prog]) / length(R[:prog])

    # residual_values pools across draws; draw_values takes one draw.
    @test length(residual_values(R, :prog)) == count(!ismissing, R[:prog])
    @test length(draw_values(R, :prog, 1)) == count(!ismissing, R[:prog][:, 1])
    @test sum(length(draw_values(R, :prog, k)) for k in 1:4) == length(residual_values(R, :prog))

    @test occursin("coverage", sprint(show, R))
end

@testset "driver: determinism" begin
    # Two people computing "the" residuals off one chain must get the same
    # numbers. Same seed, same answer, byte for byte.
    data = res_sei_setup(; m=100, n_t=RT)
    model = (; alpha=0.05, gamma=0.15)
    X = epidemic_simulator(data)(StableRNG(1), model)
    r = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:r)

    a = trajectory_summaries((r,), data, [(model, X)]; rng=StableRNG(42))
    b = trajectory_summaries((r,), data, [(model, X)]; rng=StableRNG(42))
    @test residual_values(a, :r) == residual_values(b, :r)

    c = trajectory_summaries((r,), data, [(model, X)]; rng=StableRNG(43))
    @test residual_values(a, :r) != residual_values(c, :r)   # rng genuinely used
end

@testset "driver: errors on bad input" begin
    data = res_sei_setup(; m=10, n_t=10)
    model = (; alpha=0.05, gamma=0.2)
    X = fill(1, 10, 10)
    r = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:r)

    @test_throws Exception trajectory_summaries((), data, [(model, X)])
    @test_throws Exception trajectory_summaries((r,), data, [])
    # Two summaries sharing a name would silently overwrite each other's column.
    @test_throws Exception trajectory_summaries((r, r), data, [(model, X)])
    # A draw whose X does not match `data` is a mistake, not something to guess at.
    @test_throws Exception trajectory_summaries((r,), data, [(model, fill(1, 10, 5))])
end

@testset "driver: accepts a Vector of summaries and a lazy iterator of draws" begin
    data = res_sei_setup(; m=50, n_t=RT)
    model = (; alpha=0.05, gamma=0.15)
    sim = epidemic_simulator(data)
    r = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:r)

    # A Vector works (converted to a Tuple), though a Tuple is what keeps the
    # driver's loop concrete.
    Rv = trajectory_summaries([r], data, [(model, sim(StableRNG(1), model))]; rng=StableRNG(1))
    @test Rv.n_draws == 1

    # A generator with no length must be materialised, not rejected.
    gen = ((model, sim(StableRNG(s), model)) for s in 1:3)
    Rg = trajectory_summaries((r,), data, gen; rng=StableRNG(1))
    @test Rg.n_draws == 3
end

# =============================================================================
# ARCHIVING
# =============================================================================

@testset "archive_draw: Int8 storage round-trips exactly" begin
    data = res_sei_setup(; m=100, n_t=RT)
    model = (; alpha=0.05, gamma=0.15)
    X = epidemic_simulator(data)(StableRNG(1), model)

    A = archive_draw(X)
    @test A isa Matrix{Int8}
    @test Matrix{Int}(A) == X                       # exact, not lossy
    @test sizeof(A) == sizeof(X) ÷ 8                # the whole point: 8x smaller

    # A state code that will not fit must ERROR, not wrap silently into a
    # different state — that would be an undetectable corruption of the archive.
    @test_throws Exception archive_draw(fill(200, 3, 3))

    # And residuals off the archive must equal residuals off the original.
    r = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:r)
    direct = trajectory_summaries((r,), data, [(model, X)]; rng=StableRNG(2))
    viaarchive = trajectory_summaries((r,), data, post_hoc_draws([model], [A]); rng=StableRNG(2))
    @test residual_values(direct, :r) == residual_values(viaarchive, :r)
end

@testset "post_hoc_draws: widens to Int and checks lengths" begin
    data = res_sei_setup(; m=20, n_t=10)
    model = (; alpha=0.05, gamma=0.2)
    A = archive_draw(fill(1, 10, 20))
    d = collect(post_hoc_draws([model, model], [A, A]))
    @test length(d) == 2
    @test d[1][2] isa Matrix{Int}                   # Int8 is a STORAGE format only
    @test_throws Exception post_hoc_draws([model], [A, A])
end

@testset "aggregate_synced_draws rebuilds the invariant per draw" begin
    # A hazard reading `data.aggregates` needs them to agree with THIS draw's X,
    # not with whatever X happened to be current when the fit ended.
    data = res_sei_setup(; m=50, n_t=RT)
    model = (; alpha=0.05, gamma=0.15)
    sim = epidemic_simulator(data)
    X1 = sim(StableRNG(1), model)
    X2 = sim(StableRNG(2), model)

    r = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:r)
    synced = aggregate_synced_draws(data, [(model, X1), (model, X2)])
    R = trajectory_summaries((r,), data, synced; rng=StableRNG(1))
    @test R.n_draws == 2

    # After the sweep the aggregates agree with the LAST draw handed over.
    expected = zeros(Int, 1, RT)
    for i in 1:50, t in 1:RT
        X2[t, i] == 2 && (expected[1, t] += 1)
    end
    @test data.aggregates.n_exposed == expected
end

# =============================================================================
# PERFORMANCE GUARDRAILS
# =============================================================================

@testset "the per-draw evaluator does not allocate per individual" begin
    # Allocation must be O(1) in the number of individuals, not O(m). A residual
    # that allocates per individual is the same trap `observation_weight` was
    # written to fix, reintroduced in a new place.
    data = res_sei_setup(; m=400, n_t=RT)
    model = (; alpha=0.05, gamma=0.15)
    X = epidemic_simulator(data)(StableRNG(1), model)
    r = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:r)

    f = r.f
    rng = StableRNG(1)
    f(model, data, X, 1, rng)                          # compile
    allocs = @allocated for i in 1:400
        f(model, data, X, i, rng)
    end
    # Generous, but orders of magnitude below the ~400 arrays a per-individual
    # allocation would produce.
    @test allocs < 400 * 64
end

@testset "summaries never touch the AD path" begin
    # Residuals consume Int states and return Float64, computed on a sampled X
    # outside every gradient call. Feeding Duals in must not be required — and a
    # summary must return a plain Float64 even when the parameters carry them, so
    # it can never end up inside a differentiated expression by accident.
    data = res_sei_setup(; m=20, n_t=RT)
    model = (; alpha=0.05, gamma=0.15)
    X = epidemic_simulator(data)(StableRNG(1), model)
    r = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:r)
    R = trajectory_summaries((r,), data, [(model, X)]; rng=StableRNG(1))
    @test eltype(residual_values(R, :r)) === Float64
end

# =============================================================================
# ONLINE MODE AND DISC ARCHIVING — the two bounded-memory routes
# =============================================================================

@testset "online collector: agrees with post-hoc on the same draws" begin
    # The whole promise of having two modes is that the SAME summary gives the
    # SAME answer either way. If this drifts, a residual means something different
    # depending on how it was collected, which makes both useless.
    data = res_sei_setup(; m=150, n_t=RT)
    model = (; alpha=0.05, gamma=0.15)
    sim = epidemic_simulator(data)
    Xs = [sim(StableRNG(s), model) for s in 1:3]
    r = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:r)

    post = trajectory_summaries((r,), data, [(model, X) for X in Xs]; rng=StableRNG(7))

    coll = SummaryCollector((r,), data, 3; rng=StableRNG(7))
    for X in Xs
        @test collect_summaries!(coll, model, X)
    end
    online = finish(coll)

    @test online.n_draws == post.n_draws
    # `isequal`, not `==`: these matrices hold `missing`, and `==` propagates it
    # rather than returning a Bool. Comparing them elementwise as VALUES is what
    # we mean here — a `missing` on one side must match a `missing` on the other.
    @test isequal(online[:r], post[:r])              # identical, not merely close
    @test online.coverage[:r] == post.coverage[:r]
    @test online.kinds[:r] == :pit
end

@testset "online collector: thinning and the storage cap" begin
    data = res_sei_setup(; m=50, n_t=RT)
    model = (; alpha=0.05, gamma=0.15)
    X = epidemic_simulator(data)(StableRNG(1), model)
    r = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:r)

    # thin=3 stores every third call.
    coll = SummaryCollector((r,), data, 10; rng=StableRNG(1), thin=3)
    stored = [collect_summaries!(coll, model, X) for _ in 1:9]
    @test stored == [false, false, true, false, false, true, false, false, true]
    @test finish(coll).n_draws == 3                  # trimmed, not padded

    # Once n_stored draws are in, further calls are no-ops rather than an error —
    # leaving the call in the sweep loop past the schedule must be safe.
    coll2 = SummaryCollector((r,), data, 2; rng=StableRNG(1))
    @test collect_summaries!(coll2, model, X)
    @test collect_summaries!(coll2, model, X)
    @test !collect_summaries!(coll2, model, X)
    @test finish(coll2).n_draws == 2

    @test_throws Exception SummaryCollector((r,), data, 5; thin=0)
    @test_throws Exception SummaryCollector((r,), data, 0)
    @test_throws Exception SummaryCollector((), data, 5)
    @test_throws Exception finish(SummaryCollector((r,), data, 5))   # nothing stored
end

@testset "online collector: X is not retained" begin
    # The point of this mode is that nothing about the trajectory survives the
    # call. A collector holding a reference to the sampler's X would both defeat
    # the memory argument and alias a matrix iFFBS mutates in place.
    data = res_sei_setup(; m=30, n_t=RT)
    model = (; alpha=0.05, gamma=0.15)
    X = epidemic_simulator(data)(StableRNG(1), model)
    r = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:r)

    coll = SummaryCollector((r,), data, 2; rng=StableRNG(1))
    collect_summaries!(coll, model, X)
    before = copy(coll.values[:r][:, 1])

    X .= 1                                           # the sampler mutates in place
    @test isequal(coll.values[:r][:, 1], before)     # already-stored values unmoved
end

@testset "batched archive: round-trips, and residuals match the in-memory path" begin
    dir = mktempdir()
    data = res_sei_setup(; m=80, n_t=RT)
    model = (; alpha=0.05, gamma=0.15)
    sim = epidemic_simulator(data)
    Xs = [sim(StableRNG(s), model) for s in 1:7]

    # batch_size=3 over 7 draws: three files, the last one partial.
    arc = BatchedArchive(dir, 80; batch_size=3)
    for X in Xs
        archive_push!(arc, model, X)
    end
    @test length(arc) == 7
    @test occursin("OPEN", sprint(show, arc))        # not yet safe to read
    archive_close!(arc)
    @test !occursin("OPEN", sprint(show, arc))
    @test archive_close!(arc) === arc                # idempotent

    @test length(readdir(dir)) == 3                  # 3 + 3 + 1

    back = collect(archived_draws(dir))
    @test length(back) == 7
    @test all(d -> d[2] isa Matrix{Int}, back)       # widened back from Int8
    @test [d[2] for d in back] == Xs                 # exact, in write order

    r = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:r)
    direct = trajectory_summaries((r,), data, [(model, X) for X in Xs]; rng=StableRNG(3))
    fromdisc = trajectory_summaries((r,), data, archived_draws(dir); rng=StableRNG(3))
    @test residual_values(direct, :r) == residual_values(fromdisc, :r)
end

@testset "batched archive: the buffer really is bounded" begin
    # The memory argument rests entirely on the buffer being emptied at each
    # flush. If it ever accumulates instead, the archive silently becomes the
    # thing it exists to avoid.
    dir = mktempdir()
    data = res_sei_setup(; m=20, n_t=10)
    model = (; alpha=0.05, gamma=0.2)
    X = fill(1, 10, 20)

    arc = BatchedArchive(dir, 20; batch_size=2)
    for k in 1:6
        archive_push!(arc, model, X)
        @test length(arc.buffer) < 2                 # never holds a full batch
    end
    archive_close!(arc)
    @test arc.n_batches == 3
end

@testset "batched archive: errors are loud, not silent" begin
    dir = mktempdir()
    data = res_sei_setup(; m=20, n_t=10)
    model = (; alpha=0.05, gamma=0.2)

    arc = BatchedArchive(dir, 20; batch_size=2)
    # A trajectory of the wrong shape is a mistake, not something to store anyway.
    @test_throws Exception archive_push!(arc, model, fill(1, 10, 5))
    archive_push!(arc, model, fill(1, 10, 20))
    archive_close!(arc)
    @test_throws Exception archive_push!(arc, model, fill(1, 10, 20))   # closed

    @test_throws Exception BatchedArchive(dir, 20; batch_size=0)
    @test_throws Exception archived_draws(joinpath(dir, "nonexistent"))
    @test_throws Exception archived_draws(mktempdir())                   # no batches
end

@testset "batched archive: a custom prefix keeps archives separate" begin
    dir = mktempdir()
    model = (; alpha=0.05, gamma=0.2)
    X = fill(1, 10, 20)

    a = BatchedArchive(dir, 20; batch_size=10, prefix="chain1")
    b = BatchedArchive(dir, 20; batch_size=10, prefix="chain2")
    archive_push!(a, model, X); archive_close!(a)
    archive_push!(b, model, X); archive_push!(b, model, X); archive_close!(b)

    @test length(collect(archived_draws(dir; prefix="chain1"))) == 1
    @test length(collect(archived_draws(dir; prefix="chain2"))) == 2
end

# =============================================================================
# THE HYPOTHESISTESTS EXTENSION
# =============================================================================
# Loaded here rather than at the top of runtests.jl so the rest of the suite runs
# without it — the core must not depend on it, and a test file that imports it
# unconditionally would hide a regression that made it required.

using HypothesisTests: HypothesisTests

@testset "extension: uniformity_test agrees with calibration and power" begin
    data = res_sei_setup(; m=600)
    truth = (; alpha=0.05, gamma=0.15)
    sim = epidemic_simulator(data)
    prog = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:prog)

    draws = [(truth, sim(StableRNG(s), truth)) for s in 1:6]
    Rgood = trajectory_summaries((prog,), data, draws; rng=StableRNG(1))

    # A correct model must NOT reject...
    good = uniformity_test(Rgood, :prog)
    @test good.n_draws == 6
    @test good.pvalue > 0.01
    @test isfinite(good.statistic)

    # ...and π₀.₀₅ must sit near 0.05, not near 1.
    @test pi_05(Rgood, :prog) < 0.5

    # A wrong model must reject, in the test as well as in the raw residuals.
    wrong_draws = [((; alpha=0.05, gamma=0.45), X) for (_, X) in draws]
    Rbad = trajectory_summaries((prog,), data, wrong_draws; rng=StableRNG(1))
    @test uniformity_test(Rbad, :prog).pvalue < 0.01
    @test pi_05(Rbad, :prog) > 0.9                # essentially every draw rejects
end

@testset "extension: pvalue_distribution is per-draw" begin
    data = res_sei_setup(; m=400)
    model = (; alpha=0.05, gamma=0.15)
    sim = epidemic_simulator(data)
    prog = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:prog)
    R = trajectory_summaries((prog,), data, [(model, sim(StableRNG(s), model)) for s in 1:5];
                             rng=StableRNG(1))

    ps = pvalue_distribution(R, :prog)
    @test length(ps) == 5                          # one per draw
    @test all(0 .<= ps .<= 1)
    @test pi_05(R, :prog) ≈ mean(ps .< 0.05)

    # Pooling is available but is a DIFFERENT (weaker) thing, so it must give a
    # different number — if these ever coincide, one of the paths is not doing
    # what its docstring says.
    pooled = uniformity_test(R, :prog; per_draw=false)
    perdraw = uniformity_test(R, :prog)
    @test pooled.n > perdraw.n                     # pooled sees every draw's values
end

@testset "extension: a :raw summary is refused, not silently tested" begin
    # Uniformity is not a meaningful question to ask of a lifetime or an R_i.
    # Answering it anyway would produce a p-value that looks like a verdict.
    data = res_sei_setup(; m=50)
    model = (; alpha=0.05, gamma=0.15)
    X = epidemic_simulator(data)(StableRNG(1), model)

    @residual :raw a_lifetime(model, data, X, i, rng) = 1.0
    R = trajectory_summaries((a_lifetime,), data, [(model, X)]; rng=StableRNG(1))

    @test_throws Exception uniformity_test(R, :a_lifetime)
    @test_throws Exception pvalue_distribution(R, :a_lifetime)
    @test_throws Exception pi_05(R, :a_lifetime)
    # And an unknown name is an error naming what IS there, not a KeyError.
    @test_throws Exception uniformity_test(R, :no_such_summary)
end

@testset "extension: degenerate input gives NaN, not an exception" begin
    # A draw with too few usable residuals is a draw with no information, not a
    # failure — the test must skip it rather than bring down a whole analysis.
    data = res_sei_setup(; m=2, n_t=10)
    model = (; alpha=0.05, gamma=0.2)
    X = fill(1, 10, 2)                             # nobody ever reaches E
    r = WaitingTimeResidual(:E => :I; accumulate=:discrete_product, name=:r)
    R = trajectory_summaries((r,), data, [(model, X)]; rng=StableRNG(1))

    @test isempty(pvalue_distribution(R, :r))
    @test isnan(pi_05(R, :r))
    @test isnan(uniformity_test(R, :r).pvalue)
end

# =============================================================================
# SOURCE ATTRIBUTION — the infection-link residual, FOI ratio, and R_i
# =============================================================================

# A grouped SI model whose force of infection splits into a background part and a
# within-group transmission part. Simulated by hand so the TRUE source of each
# infection is recorded — standing in for the latent source a real sampler infers.

const AT_T = 40
const AT_M = 600
const AT_NG = 20

at_ninf(data, i, t) = data.aggregates.n_infected[data.group[i], t]
at_background(model, data, X, i, t) = model.alpha
at_transmission(model, data, X, i, t) = model.beta * at_ninf(data, i, t)
at_rate(model, data, i, t) = 1 - exp(-(model.alpha + model.beta * at_ninf(data, i, t)))

const AT_COMPONENTS = (; background = at_background, transmission = at_transmission)

function at_setup()
    spec = @transitions [:S, :I] begin
        S -> I = at_rate
    end
    aggs = @aggregate [:S, :I] begin
        @array n_infected Int (AT_NG, AT_T)
        n_infected[data.group[i], t] += (state == :I)
    end
    grp = repeat(1:AT_NG, inner = AT_M ÷ AT_NG)
    epidemic_data(; n_individuals=AT_M, n_timepoints=AT_T, trans_mat=spec,
                    starting_state=(m, d, X, i, t) -> [1.0, 0.0],
                    aggregates=aggs, group=grp,
                    source_of=zeros(Int, AT_M))
end

# Simulate, recording which component actually caused each infection. Returns
# `(X, sources)`; `sources[i] == 0` means i was never infected.
function at_simulate(data, model, seed)
    rng = StableRNG(seed)
    X = fill(1, AT_T, AT_M)
    sources = zeros(Int, AT_M)
    agg = data.aggregates.n_infected
    fill!(agg, 0)
    for t in 1:(AT_T - 1)
        for i in 1:AT_M
            agg[data.group[i], t] += (X[t, i] == 2)
        end
        for i in 1:AT_M
            if X[t, i] == 2
                X[t + 1, i] = 2                 # infection is absorbing here
                continue
            end
            bg = model.alpha
            tr = model.beta * agg[data.group[i], t]
            tot = bg + tr
            if rand(rng) < 1 - exp(-tot)
                X[t + 1, i] = 2
                # The true source, drawn in proportion to each component's share
                # of the hazard. This is the ground truth a sampler would infer.
                sources[i] = rand(rng) * tot < bg ? 1 : 2
            else
                X[t + 1, i] = 1
            end
        end
    end
    for i in 1:AT_M
        agg[data.group[i], AT_T] += (X[AT_T, i] == 2)
    end
    return (X, sources)
end

at_source_fn(model, data, X, i, t) = (s = data.source_of[i]; s == 0 ? nothing : s)

at_ilr(name=:ilr) = SourceAttributionResidual(:S => :I;
    components = AT_COMPONENTS, source = at_source_fn, name = name)

# Run one draw with its own source vector installed. Deliberately one draw at a
# time: `data.source_of` is a single shared array, so batching draws through a
# lazy generator would alias every draw to the LAST one's sources. That bug made
# the residual reject the truth, and it took isolating eq 2.10 to find.
function at_run(data, ilr, pars, X, sources; rng=StableRNG(7))
    fill!(data.aggregates.n_infected, 0)
    apply_derived_summaries!(pars, data, X)
    copyto!(data.source_of, sources)
    R = trajectory_summaries((ilr,), data, [(pars, X)]; rng=rng)
    return residual_values(R, ilr.name)
end

@testset "infection-link residual: calibrated under the true source split" begin
    data = at_setup()
    truth = (; alpha = 0.010, beta = 0.004)
    ilr = at_ilr()

    ps = Float64[]
    means = Float64[]
    for seed in 701:712
        X, sources = at_simulate(data, truth, seed)
        v = at_run(data, ilr, truth, X, sources)
        @test all(0 .<= v .<= 1)
        push!(means, mean(v))
        push!(ps, res_ks_uniform(v)[2])
    end

    @test isapprox(mean(means), 0.5; atol=0.05)     # Uniform(0,1)
    @test mean(ps .< 0.05) < 0.35                   # does not reject the truth
end

@testset "infection-link residual: REJECTS a wrong transmission structure" begin
    # The point of this residual: a model can get every waiting time right and
    # still attribute infections to the wrong sources. Nothing else detects that.
    data = at_setup()
    truth = (; alpha = 0.010, beta = 0.004)
    ilr = at_ilr()

    # Data always generated under the truth; only the SCORING model changes.
    sims = [at_simulate(data, truth, seed) for seed in 701:706]

    for wrong in ((; alpha = 0.030, beta = 0.0001),   # claims transmission is nil
                  (; alpha = 0.0005, beta = 0.020),   # claims background is nil
                  (; alpha = 0.004, beta = 0.010))    # split shifted, total similar
        rejects = 0
        for (X, sources) in sims
            v = at_run(data, ilr, wrong, X, sources)
            res_ks_uniform(v)[2] < 0.05 && (rejects += 1)
        end
        @test rejects == length(sims)               # rejects in EVERY draw
    end
end

@testset "infection-link residual: source is required, no self-normalising fallback" begin
    # THE most important test here. If `source` ever gains a default that samples
    # from the component rates, the residual becomes Uniform(0,1) BY CONSTRUCTION
    # for any rates at all — it passes every calibration check and has exactly
    # zero power, i.e. it silently stops being a diagnostic.
    @test_throws Exception SourceAttributionResidual(:S => :I; components=AT_COMPONENTS)
    @test_throws Exception SourceAttributionResidual(:S => :I; source=at_source_fn)
    # One component is nothing to attribute between.
    @test_throws Exception SourceAttributionResidual(:S => :I;
        components = (; only_one = at_background), source = at_source_fn)

    # Demonstrate the property the guard prevents, so the REASON is pinned by a
    # test and not only by a comment: drawing the source from the same
    # probabilities used to build the interval is uniform for ANY split.
    function self_normalised_pit(p1, rng)
        probs = (p1, 1 - p1)
        drawn = rand(rng) < p1 ? 1 : 2
        order = sortperm(collect(probs))
        lb = 0.0; ub = 0.0; cum = 0.0
        for k in 1:2
            prev = cum; cum += probs[order[k]]
            if order[k] == drawn; lb = prev; ub = cum; break; end
        end
        randomized_pit(lb, ub, rng)
    end
    rng = StableRNG(1)
    for p1 in (0.5, 0.9, 0.99)
        v = [self_normalised_pit(p1, rng) for _ in 1:20_000]
        # Uniform at EVERY split — which is exactly why it would be useless.
        @test isapprox(mean(v), 0.5; atol=0.02)
    end
end

@testset "infection-link residual: eq 2.10 is exact for n components" begin
    # The construction itself, isolated from any epidemic: when the source is
    # drawn from the model's predicted probabilities, the PIT is Uniform(0,1) —
    # for two components and for ten, at every share. This is what says the
    # METHOD is right, so a future calibration failure points at the plumbing.
    rng = StableRNG(11)
    for n in (2, 3, 5, 10)
        v = Float64[]
        for _ in 1:20_000
            w = rand(rng, n)
            probs = Tuple(w ./ sum(w))
            u = rand(rng); acc = 0.0; src = n
            for k in 1:n
                acc += probs[k]
                if u <= acc; src = k; break; end
            end
            order = sortperm(collect(probs))
            lb = 0.0; ub = 0.0; cum = 0.0
            for k in 1:n
                prev = cum; cum += probs[order[k]]
                if order[k] == src; lb = prev; ub = cum; break; end
            end
            push!(v, randomized_pit(lb, ub, rng))
        end
        @test isapprox(mean(v), 0.5; atol=0.02)
        @test isapprox(std(v), 1 / sqrt(12); atol=0.02)
    end
end

@testset "infection-link residual: source by name, and unattributed individuals" begin
    data = at_setup()
    truth = (; alpha = 0.010, beta = 0.004)
    X, sources = at_simulate(data, truth, 701)

    # A source may be given by component NAME as well as by index.
    by_name = SourceAttributionResidual(:S => :I; components = AT_COMPONENTS,
        source = (model, data, X, i, t) ->
            (s = data.source_of[i]; s == 0 ? nothing : (s == 1 ? :background : :transmission)),
        name = :ilr)
    @test at_run(data, by_name, truth, X, sources) == at_run(data, at_ilr(), truth, X, sources)

    # An unknown name or an out-of-range index is an error, not a silent guess.
    bad_name = SourceAttributionResidual(:S => :I; components=AT_COMPONENTS,
        source = (m, d, Xx, i, t) -> :not_a_component, name = :ilr)
    @test_throws Exception at_run(data, bad_name, truth, X, sources)
    bad_idx = SourceAttributionResidual(:S => :I; components=AT_COMPONENTS,
        source = (m, d, Xx, i, t) -> 99, name = :ilr)
    @test_throws Exception at_run(data, bad_idx, truth, X, sources)

    # `nothing` means "not inferred" and gives `missing`, which shows up as
    # reduced coverage rather than as a wrong number.
    none_inferred = SourceAttributionResidual(:S => :I; components=AT_COMPONENTS,
        source = (m, d, Xx, i, t) -> nothing, name = :ilr)
    fill!(data.aggregates.n_infected, 0)
    apply_derived_summaries!(truth, data, X)
    R = trajectory_summaries((none_inferred,), data, [(truth, X)]; rng=StableRNG(1))
    @test R.coverage[:ilr] == 0.0
end

@testset "FOI ratio: the endogenous/exogenous split" begin
    data = at_setup()
    truth = (; alpha = 0.010, beta = 0.004)
    X, _ = at_simulate(data, truth, 701)
    fill!(data.aggregates.n_infected, 0)
    apply_derived_summaries!(truth, data, X)

    ratio = FOIRatioSummary(:S => :I; components=AT_COMPONENTS,
                            numerator=:background, name=:ratio)
    R = trajectory_summaries((ratio,), data, [(truth, X)]; rng=StableRNG(1))
    v = residual_values(R, :ratio)

    @test R.kinds[:ratio] == :raw               # NOT a residual
    @test all(0 .<= v .<= 1)
    @test 0 < mean(v) < 1                       # both components contribute

    # A :raw summary must be refused by the uniformity machinery — a ratio has no
    # uniformity claim, and answering anyway would look like a verdict.
    @test_throws Exception uniformity_test(R, :ratio)

    # The two components' shares must sum to 1, individual by individual.
    ratio_t = FOIRatioSummary(:S => :I; components=AT_COMPONENTS,
                              numerator=:transmission, name=:ratio)
    Rt = trajectory_summaries((ratio_t,), data, [(truth, X)]; rng=StableRNG(1))
    vt = residual_values(Rt, :ratio)
    @test v .+ vt ≈ ones(length(v))

    # Raising beta must raise the transmission share. If this ever inverts, the
    # numerator is being read off the wrong component.
    hi_beta = (; alpha = 0.010, beta = 0.020)
    fill!(data.aggregates.n_infected, 0)
    apply_derived_summaries!(hi_beta, data, X)
    Rhi = trajectory_summaries((ratio_t,), data, [(hi_beta, X)]; rng=StableRNG(1))
    @test mean(residual_values(Rhi, :ratio)) > mean(vt)

    @test_throws Exception FOIRatioSummary(:S => :I; components=AT_COMPONENTS,
                                           numerator=:no_such_component)
end

@testset "case reproduction numbers" begin
    data = at_setup()
    truth = (; alpha = 0.010, beta = 0.004)
    X, _ = at_simulate(data, truth, 701)
    fill!(data.aggregates.n_infected, 0)
    apply_derived_summaries!(truth, data, X)

    R = case_reproduction_numbers(truth, data, X;
        components = AT_COMPONENTS, secondary = :transmission,
        infection = :S => :I, infectious_state = :I)

    @test length(R) == AT_M
    vals = collect(skipmissing(R))
    @test !isempty(vals)
    @test all(>=(0), vals)

    # Never-infectious individuals are `missing` — no opportunity to infect
    # anyone. That is a different statement from an infectious individual who
    # infected nobody, which is a genuine 0.0 and must stay in the sample.
    for i in 1:AT_M
        @test any(==(2), @view X[:, i]) == !ismissing(R[i])
    end

    # The total attributed across the population cannot exceed the number of
    # infections: attribution SPLITS each event, it does not duplicate it.
    n_infections = count(i -> any(==(2), @view X[:, i]), 1:AT_M)
    @test sum(vals) <= n_infections + 1e-9

    # summarize_population wraps it into the standard SummaryResult, so coverage
    # and the plots work unchanged.
    SR = summarize_population(:R_i, data, [(truth, X)]) do pars, d, Xx
        case_reproduction_numbers(pars, d, Xx;
            components = AT_COMPONENTS, secondary = :transmission,
            infection = :S => :I, infectious_state = :I)
    end
    @test SR.kinds[:R_i] == :raw
    @test SR.n_draws == 1
    @test SR.coverage[:R_i] ≈ length(vals) / AT_M
    @test_throws Exception uniformity_test(SR, :R_i)     # :raw, so refused

    @test_throws Exception case_reproduction_numbers(truth, data, X;
        components = AT_COMPONENTS, secondary = :not_a_component)
    # `f` returning the wrong length is a mistake, not something to pad.
    @test_throws Exception summarize_population((m, d, Xx) -> [1.0], :bad, data, [(truth, X)])
end

@testset "components accept NamedTuple and pairs alike" begin
    data = at_setup()
    truth = (; alpha = 0.010, beta = 0.004)
    X, sources = at_simulate(data, truth, 701)

    a = SourceAttributionResidual(:S => :I; components=AT_COMPONENTS,
                                  source=at_source_fn, name=:ilr)
    b = SourceAttributionResidual(:S => :I; source=at_source_fn, name=:ilr,
        components = (:background => at_background, :transmission => at_transmission))

    @test at_run(data, a, truth, X, sources) == at_run(data, b, truth, X, sources)
end

@testset "infection-link residual: source and PIT rng streams must be independent" begin
    # A trap found while writing examples/residuals_attribution.jl, and one that
    # produces a CALIBRATION failure rather than a crash — the worst kind.
    #
    # The source sampler and the residual both walk the same cumulative component
    # probabilities in the same order. Feeding them the SAME rng stream correlates
    # "which source was drawn" with "where in its interval the residual lands", and
    # the residual then rejects the TRUE model. Independent streams fix it.
    #
    # There is nothing the package can do to enforce this — the source sampler is
    # user code — so this test exists to document the requirement and to keep a
    # worked, correct pairing in the suite.
    data = at_setup()
    truth = (; alpha = 0.010, beta = 0.004)
    ilr = at_ilr()

    # Sample sources with one stream, score with a DIFFERENT one.
    function score_independent(X, sources, seed)
        fill!(data.aggregates.n_infected, 0)
        apply_derived_summaries!(truth, data, X)
        copyto!(data.source_of, sources)
        R = trajectory_summaries((ilr,), data, [(truth, X)]; rng = StableRNG(seed + 10_000))
        residual_values(R, :ilr)
    end

    shared = Float64[]
    independent = Float64[]
    for seed in 701:708
        X, sources = at_simulate(data, truth, seed)
        append!(independent, score_independent(X, sources, seed))
        append!(shared, at_run(data, ilr, truth, X, sources; rng = StableRNG(seed)))
    end

    # Both are valid PIT values; the point is that the pairing of streams matters
    # for calibration, so the residual must be given its own.
    @test all(0 .<= independent .<= 1)
    @test all(0 .<= shared .<= 1)
    @test isapprox(mean(independent), 0.5; atol=0.03)
end
