# The badger bovine-TB model, written as user code against EpidemicTrajectories.
#
# Reproduces the base model of badger_ref/run_base_exp.jl: four states with
# mortality, a density-scaled force of infection, an age-dependent survival curve,
# time-varying social groups, six diagnostic tests with three-way state-dependent
# accuracy, and a capture process.
#
# Everything here is user code. The package supplies @transitions/@aggregate, the
# simulator/likelihood/latent-sampler builders, and nothing else — it has no idea
# what a badger, a social group, or a Brock test is.
#
# Differs from the reference in one agreed way: the Brock changepoint is fixed at
# t=101 rather than inferred (see badger_data.jl and badger_repro_log.md).

using EpidemicTrajectories
using Distributions
using Random
using Statistics: mean

include(joinpath(@__DIR__, "badger_data.jl"))

## ---------------------------------------------------------------------------
## The group statistics the force of infection needs
## ---------------------------------------------------------------------------

# Two arrays, declared with their updates. The package allocates them and knows
# nothing of what they mean; it just runs these updates forwards, and in reverse
# when the sampler needs to take an individual back out.
#
# Both are indexed by the individual's group AT TIME t — badgers move between
# groups, so this is `social_group[i, t]`, not a fixed `group[i]`. A group of 0
# means "not present", and the guard skips it.
function badger_aggregates(n_groups, n_timepoints)
    @aggregate BADGER_STATES begin
        @array n_infectious Int (n_groups, n_timepoints)
        @array n_alive Int (n_groups, n_timepoints)
        if data.social_group[i, t] > 0
            n_infectious[data.social_group[i, t], t] += (state == :I)
        end
        if data.social_group[i, t] > 0
            n_alive[data.social_group[i, t], t] += (state != :D)
        end
    end
end

## ---------------------------------------------------------------------------
## Rates
## ---------------------------------------------------------------------------

# Force of infection: a per-group external term plus density-scaled transmission
# from the infectious badgers in the group. `K` is a fixed scaling constant and
# `q` the density-dependence exponent — at q=0 this is frequency-independent, at
# q=1 fully density-dependent.
function badger_infection(model, data, i, t)
    g = data.social_group[i, t]
    g == 0 && return 0.0                       # not present: no exposure
    I = data.aggregates.n_infectious[g, t]
    M = data.aggregates.n_alive[g, t]
    M == 0 && return 0.0

    α_g = model.lambda * model.alpha[g]
    foi = α_g + model.beta * I / ((M / data.K)^model.q)
    return -expm1(-foi)                        # 1 - exp(-foi)
end

# Progression E -> I. With k=1 the Erlang collapses to an exponential; the
# reference keeps the general form, so we do too.
function erlang_cdf_at_1(k, tau)
    return 1.0 - exp(-tau) * sum((tau^j) / factorial(j) for j in 0:(k - 1))
end

badger_progression(model, data, i, t) = erlang_cdf_at_1(data.k, model.tau / data.k)

# Siler survival: high infant mortality falling with age, plus senescence rising
# with age, plus a constant hazard.
function siler_survival(model, data, i, t)
    age = t >= 1 ? data.age[i, t] : data.age[i, 1] + (t - 1)
    age < 0 && return 1.0                      # not yet born

    a1, b1, a2, b2, c1 = model.a1, model.b1, model.a2, model.b2, model.c1
    y1 = b2 * (age - 1); y2 = b2 * age
    late = -exp(y1) * expm1(y2 - y1)
    z1 = -b1 * (age - 1); z2 = -b1 * age
    early = exp(z1) * expm1(z2 - z1)
    s = exp(-c1 + (a2 / b2) * late + (a1 / b1) * early)

    # A badger seen alive later cannot have died yet: survival is certain until
    # its last capture. This is data, not a rate — the reference does the same.
    return t <= data.last_capture_time[i] ? 1.0 : s
end

function badger_transitions()
    @transitions BADGER_STATES begin
        @survival siler_survival death=:D
        S -> E = badger_infection
        E -> I = badger_progression
    end
end

## ---------------------------------------------------------------------------
## Starting state
## ---------------------------------------------------------------------------

# A badger born during the study starts susceptible. One already alive when the
# study began starts from the mixing (1-nuE-nuI, nuE, nuI) for its start time —
# an exact match into nu_times, with no mixing if it isn't one of them.
function badger_starting_state(model, data, X, i, t)
    p = zeros(Float64, data.n_states)
    start_time = data.sampling_period[i][1]
    nuE = nuI = 0.0
    if data.birth_time[i] < start_time
        idx = findfirst(==(start_time), data.nu_times)
        if idx !== nothing
            nuE, nuI = model.nuE[idx], model.nuI[idx]
        end
    end
    p[1] = 1.0 - nuE - nuI      # S
    p[2] = nuE                  # E
    p[3] = nuI                  # I
    p[4] = 0.0                  # D — never start dead
    p
end

## ---------------------------------------------------------------------------
## Observation process: capture, then tests
## ---------------------------------------------------------------------------

# What we observe and how it relates to the states. Two parts:
#
#  * capture — seen at time t means the badger was alive (dead weight 0) and the
#    alive states carry the season's capture probability; not seen means the
#    alive states carry (1 - eta) and dead carries 1.
#  * tests — only at capture times, and three-way by true state: a susceptible
#    badger's result is governed by SPECIFICITY, an exposed one's by a reduced
#    sensitivity (theta*rho), an infectious one's by full sensitivity (theta).
function badger_observations(model, data, X, i, t)
    w = ones(Float64, data.n_states)
    eta = model.etas[data.season[t]]

    if data.capture[t, i] == 0
        w[1] = w[2] = w[3] = 1 - eta
        w[4] = 1.0
        return w                                # not seen: no tests
    end

    w[1] = w[2] = w[3] = eta
    w[4] = 0.0                                  # seen alive: cannot be dead

    for j in 1:size(data.tests, 3)
        x = data.tests[t, i, j]
        (x == 0 || x == 1) || continue          # not tested with this one
        θ, ρ, φ = model.thetas[j], model.rhos[j], model.phis[j]
        w[1] *= x == 1 ? (1 - φ) : φ            # S: specificity
        w[2] *= x == 1 ? (θ * ρ) : (1 - θ * ρ)  # E: reduced sensitivity
        w[3] *= x == 1 ? θ : (1 - θ)            # I: full sensitivity
    end
    w
end

## ---------------------------------------------------------------------------
## Assembling it
## ---------------------------------------------------------------------------

"""
    badger_data(dir; brock_changepoint=BROCK_CHANGEPOINT)

Load the badger data and build the `EpidemicData` for the base model.

Everything the model's own functions need beyond the parameters — the social
groups, ages, captures, tests, seasons, `K`, `k` — is passed as extras and reached
as `data.name`. The package stores them without looking inside.
"""
function badger_data(dir; brock_changepoint=BROCK_CHANGEPOINT)
    d = load_badger_data(dir; brock_changepoint=brock_changepoint)

    # Who each badger's state affects: its groupmates at that time. Because
    # membership moves, this genuinely differs from one timepoint to the next.
    affected = Matrix{Vector{Int}}(undef, d.n_timepoints, d.n_individuals)
    by_group = [Int[] for _ in 1:d.n_groups, _ in 1:d.n_timepoints]
    for t in 1:d.n_timepoints, i in 1:d.n_individuals
        g = d.social_group[i, t]
        g > 0 && push!(by_group[g, t], i)
    end
    for t in 1:d.n_timepoints, i in 1:d.n_individuals
        g = d.social_group[i, t]
        affected[t, i] = g == 0 ? Int[] : [j for j in by_group[g, t] if j != i]
    end

    data = epidemic_data(;
        n_individuals=d.n_individuals,
        n_timepoints=d.n_timepoints,
        trans_mat=badger_transitions(),
        starting_state=badger_starting_state,
        observation_process=badger_observations,
        aggregates=badger_aggregates(d.n_groups, d.n_timepoints),
        sampling_period=d.sampling_period,
        affected_individuals=affected,
        # The ONLY way one badger affects another is by contributing to its force
        # of infection, which only the S -> E rate reads. A neighbour that died,
        # progressed, or stayed put would have done so regardless of the focal's
        # state, so the sampler can skip it exactly rather than rebuild its whole
        # transition matrix once per candidate state.
        coupled_transitions=[(:S, :E)],
        state_space=BADGER_STATES,
        # my own data, reachable as data.name in the functions above
        social_group=d.social_group,
        age=d.age,
        capture=d.capture,
        tests=d.tests,
        season=d.season,
        birth_time=d.birth_time,
        last_capture_time=d.last_capture_time,
        nu_times=d.nu_times,
        K=Float64(d.K),
        k=d.k,
        n_groups=d.n_groups,
    )
    (; data, raw=d)
end

"""
    badger_initial_params(d; rng=Random.default_rng())

A starting point for the parameters, matching the reference's initial draws.
"""
function badger_initial_params(d; rng=Random.default_rng())
    (; tau=5.0,
       alpha=fill(0.001, d.n_groups),
       lambda=rand(rng, Gamma(1, 1 / 100)),
       beta=rand(rng, Gamma(1, 1 / 100)),
       q=rand(rng, Beta(1, 1)),
       c1=rand(rng, Gamma(1, 1 / 100)),
       a1=rand(rng, Gamma(1, 1 / 100)),
       b1=rand(rng, Gamma(1, 1 / 100)),
       a2=rand(rng, Gamma(1, 1 / 100)),
       b2=rand(rng, Gamma(1, 1 / 100)),
       thetas=rand(rng, Uniform(0.5, 1), d.n_tests),
       rhos=rand(rng, Uniform(0.2, 0.8), d.n_tests),
       phis=rand(rng, Uniform(0.7, 1), d.n_tests),
       etas=rand(rng, Beta(1, 1), d.n_seasons),
       nuE=rand(rng, Uniform(0.05, 0.2), d.n_nu_times),
       nuI=rand(rng, Uniform(0.05, 0.2), d.n_nu_times))
end
