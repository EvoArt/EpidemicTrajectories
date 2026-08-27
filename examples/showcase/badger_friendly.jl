# The badger bTB model, written the easy way.
#
# 2384 badgers x 161 quarters. Four states, S -> E -> I, with death reachable from
# any of them. Six imperfect diagnostic tests, capture-recapture observation,
# social groups that badgers move between, and age-dependent mortality.
#
# This script is the FRIENDLY build: describe the model, let the package do
# everything else. Every performance decision is left at its default. It is not
# the fast version -- `badger_power.jl` is the same model with the optimisations
# turned on, and asserts an identical log density -- but it is the one to read
# first, and for many models it is fast enough to stop at.
#
# Run:  julia --project=examples examples/showcase/badger_friendly.jl
#       BADGER_SWEEPS=200 julia --project=examples ...   (default is 20)

using EpidemicTrajectories
using PracticalBayes
using PracticalEpiBayes
using Distributions
using Random
using StableRNGs: StableRNG
using Statistics: mean, std
import AbstractMCMC

include(joinpath(@__DIR__, "badger_data.jl"))

# ---------------------------------------------------------------------------
# 1. The states
# ---------------------------------------------------------------------------
# The order fixes the encoding: state k in the trajectory means STATES[k].

const STATES = [:S, :E, :I, :D]

# ---------------------------------------------------------------------------
# 2. The data
# ---------------------------------------------------------------------------

raw = load_badger_data(DATA_DIR())

println("badger data: $(raw.n_individuals) individuals x $(raw.n_timepoints) quarters, " *
        "$(raw.n_groups) groups, $(raw.n_tests) tests")

# ---------------------------------------------------------------------------
# 3. What to track while the trajectory is resampled
# ---------------------------------------------------------------------------
# The force of infection needs to know how many groupmates are infectious, and
# how many are alive. Declare the arrays and how they update; the package
# allocates them, keeps them in step with the trajectory, and -- because the
# updates are reversible -- can take one individual back out again, which is
# exactly the leave-one-out count a force of infection wants.
#
# The package attaches no meaning to these. They are whatever your rates need.

# (`@array` dims are read as locals at macro-expansion time, so they are bound
# here rather than interpolated.)
n_groups, n_timepoints = raw.n_groups, raw.n_timepoints

aggregates = @aggregate STATES begin
    @array n_infectious Int (n_groups, n_timepoints)
    @array n_alive      Int (n_groups, n_timepoints)

    if data.social_group[i, t] > 0
        n_infectious[data.social_group[i, t], t] += (state == :I)
    end
    if data.social_group[i, t] > 0
        n_alive[data.social_group[i, t], t] += (state != :D)
    end
end

# ---------------------------------------------------------------------------
# 4. The rates
# ---------------------------------------------------------------------------
# Ordinary functions of (model, data, i, t). `model` holds the parameters,
# `data` everything else -- including the aggregates above.

# S -> E. A group-level baseline plus a density-dependent term, with `q`
# interpolating between frequency- and density-dependence.
function infection(model, data, i, t)
    g = data.social_group[i, t]
    g == 0 && return 0.0
    infectious, alive = data.aggregates.n_infectious[g, t], data.aggregates.n_alive[g, t]
    alive == 0 && return 0.0
    foi = model.lambda * model.alpha[g] +
          model.beta * infectious / ((alive / data.K)^model.q)
    return -expm1(-foi)
end

# E -> I. The Erlang(k, tau/k) CDF at one step, so `tau` is the mean latent
# period in quarters and `k` its shape.
function progression(model, data, i, t)
    x = data.k / model.tau
    s = term = 1.0
    for j in 1:(data.k - 1)
        term *= x / j
        s += term
    end
    return 1.0 - s * exp(-x)
end

# Gompertz-Makeham mortality, evaluated at the age the badger would reach:
#     P(survive | age) = exp(-c1 + (a2/b2) * (exp(b2*(age-1)) - exp(b2*age)))
function survival(model, data, i, t)
    tt = t + 1
    age = tt <= data.n_timepoints ? data.age[i, tt] :
          data.age[i, data.n_timepoints] + (tt - data.n_timepoints)
    age < 0 && return 1.0
    y1, y2 = model.b2 * (age - 1), model.b2 * age
    return exp(-model.c1 + (model.a2 / model.b2) * (-exp(y1) * expm1(y2 - y1)))
end

# ---------------------------------------------------------------------------
# 5. The transitions
# ---------------------------------------------------------------------------
# Two moves and a survival declaration. `@survival` scales every non-death rate
# by `survival` and gives EVERY live state a transition to `:D` with the
# remaining mass -- including `I -> D`, which is never written here. The
# self-transitions (S->S, E->E, I->I) get the leftover probability
# automatically.

transitions = @transitions STATES begin
    @survival survival death=:D
    S -> E = infection
    E -> I = progression
end

# ---------------------------------------------------------------------------
# 6. The observation process
# ---------------------------------------------------------------------------
# Capture x tests. A badger that was not caught is not tested; one caught later
# cannot have been dead now.
#
# Each READING is scored once, via the repeat-capture multiplicities -- see the
# note in badger_data.jl. theta is sensitivity in I, theta*rho in E, and phi is
# specificity in S.

function observation(model, data, X, i, t)
    T = promote_type(eltype(model.thetas), eltype(model.rhos),
                     eltype(model.phis), eltype(model.etas))
    w = ones(T, data.n_states)

    eta = model.etas[data.season[t]]
    if data.capture[t, i] == 0
        w[1] = w[2] = w[3] = 1 - eta
        w[4] = t <= data.last_capture_time[i] ? zero(T) : one(T)
        return w                       # not caught, so not tested
    end
    w[1] = w[2] = w[3] = eta
    w[4] = zero(T)

    for j in 1:size(data.tests, 3)
        nneg, npos = data.n_neg[t, i, j], data.n_pos[t, i, j]
        (nneg == 0 && npos == 0) && continue
        theta, rho, phi = model.thetas[j], model.rhos[j], model.phis[j]
        theta_rho = theta * rho
        if npos > 0
            w[1] *= (1 - phi)^npos; w[2] *= theta_rho^npos; w[3] *= theta^npos
        end
        if nneg > 0
            w[1] *= phi^nneg; w[2] *= (1 - theta_rho)^nneg; w[3] *= (1 - theta)^nneg
        end
    end
    return w
end

# Where a badger starts, for one already alive when watching began.
function starting_state(model, data, X, i, t)
    p = zeros(Float64, data.n_states)
    start = data.sampling_period[i][1]
    nuE = nuI = 0.0
    if data.birth_time[i] < start
        idx = findfirst(==(start), data.nu_times)
        idx === nothing || ((nuE, nuI) = (model.nu[idx, 1], model.nu[idx, 2]))
    end
    p[1] = 1.0 - nuE - nuI; p[2] = nuE; p[3] = nuI
    return p
end

# ---------------------------------------------------------------------------
# 7. Assemble
# ---------------------------------------------------------------------------
# Everything the package does not name is yours: pass it as a keyword and read it
# back as `data.whatever` inside any of the functions above.

data = epidemic_data(;
    n_individuals   = raw.n_individuals,
    n_timepoints    = raw.n_timepoints,
    trans_mat       = transitions,
    starting_state  = starting_state,
    aggregates      = aggregates,
    observation_process = observation,
    sampling_period = raw.sampling_period,
    # Badgers move between groups, so who each one affects varies with time.
    affected_individuals = badger_affected_lists(raw),
    # your own arrays, reachable as data.<name>
    social_group = raw.social_group, age = raw.age, capture = raw.capture,
    capt_effort = raw.capt_effort, tests = raw.tests,
    n_neg = raw.n_neg, n_pos = raw.n_pos, season = raw.season,
    birth_time = raw.birth_time, last_capture_time = raw.last_capture_time,
    nu_times = raw.nu_times, K = Float64(raw.K), k = raw.k)

# One spec gives all three artefacts. The entry gate says a badger is known alive
# at first capture but its disease dynamics before then were not watched: the
# likelihood scores the disease moves pre-entry and divides the survival factor
# back out.
loglik     = epidemic_loglik(data; entry_time = raw.first_capture_time,
                                   survival   = survival)
obs_loglik = epidemic_obs_loglik(data)
latent!    = epidemic_latent_sampler(data)

# ---------------------------------------------------------------------------
# 8. The model
# ---------------------------------------------------------------------------

# `nu` is owned by a conjugate kernel below, so its prior is a placeholder: the
# kernel supplies the real draw. `rand` must still return a valid mixing, since
# the model body is evaluated with it during Gibbs validation.
struct NuPrior <: Distributions.DiscreteMatrixDistribution
    n_nu::Int
end
Base.size(d::NuPrior) = (d.n_nu, 2)
Distributions.logpdf(::NuPrior, ::AbstractMatrix) = 0.0
Distributions.rand(::Random.AbstractRNG, d::NuPrior) = fill(0.05, d.n_nu, 2)

@model function badger(data, loglik_fn, obs_loglik_fn, n_groups, n_tests,
                       n_seasons, n_nu, n_time, n_ind)
    tau    ~ Exponential(100.0)
    alpha  ~ PracticalBayes.filldist(Exponential(1.0), n_groups)
    lambda ~ Exponential(1.0)
    beta   ~ Exponential(1.0)
    q      ~ Beta(1, 1)

    c1 ~ Exponential(1.0)          # Makeham constant
    a2 ~ Exponential(1.0)          # Gompertz scale
    b2 ~ Exponential(1.0)          # Gompertz rate

    thetas ~ PracticalBayes.filldist(Beta(1, 1), n_tests)     # sensitivity in I
    rhos   ~ PracticalBayes.filldist(Beta(1, 1), n_tests)     # E-sensitivity multiplier
    phis   ~ PracticalBayes.filldist(Beta(1, 1), n_tests)     # specificity in S
    etas   ~ PracticalBayes.filldist(Beta(1, 1), n_seasons)   # capture probability
    nu     ~ NuPrior(n_nu)                     # entry mixing

    # The whole trajectory is ONE latent block: held constant during the
    # gradients, resampled once per Gibbs sweep by iFFBS. That is the entire
    # reason this package exists.
    X ~ TrajectoryLatent(n_time, n_ind)

    pars = (; tau, alpha, lambda, beta, q, c1, a2, b2, thetas, rhos, phis, etas, nu)

    # BOTH halves. `loglik` covers the starting state and the transitions;
    # `obs_loglik` covers the observations. Omitting the second is a silent bug:
    # every test parameter would then be sampled from its prior.
    @addlogprob! loglik_fn(pars, data, X) + obs_loglik_fn(pars, data, X)
end

# ---------------------------------------------------------------------------
# 9. Fit
# ---------------------------------------------------------------------------
# One NUTS block for everything continuous; conjugate kernels where a closed
# form exists (an exact independent draw beats a correlated HMC step); iFFBS for
# the trajectory.

model = badger(data, loglik, obs_loglik, raw.n_groups, raw.n_tests,
               raw.n_seasons, raw.n_nu_times, raw.n_timepoints, raw.n_individuals)

sampler = Gibbs(
    (:tau, :alpha, :lambda, :beta, :q, :c1, :a2, :b2,
     :thetas, :rhos, :phis) => NUTS(0.8),

    :etas => capture_prob_kernel(:etas;
        caught = raw.capture, effort = raw.capt_effort, group = raw.social_group,
        index  = raw.season, dead_state = 4, n = raw.n_seasons),

    :nu => initial_state_kernel(:nu;
        at       = raw.nu_times,
        eligible = (X, d, i, nt) -> d.sampling_period[i][1] == nt &&
                                    d.birth_time[i] < nt,
        states = (1, 2, 3), prior = [1.0, 1.0, 1.0], n = raw.n_nu_times),

    # `params` names the parameters the rate functions want; the kernel selects
    # them out of the Gibbs state.
    :X => iffbs_kernel(latent!;
        params = (:tau, :alpha, :lambda, :beta, :q, :c1, :a2, :b2,
                  :thetas, :rhos, :phis, :etas, :nu)))

n_sweeps = parse(Int, get(ENV, "BADGER_SWEEPS", "20"))

# The aggregates must agree with the STARTING trajectory before the first
# likelihood call. Established once here; the sampler preserves it thereafter.
X0 = copy(raw.X_init)
reset_aggregates!(data)
apply_derived_summaries!((;), data, X0)

chain = AbstractMCMC.sample(StableRNG(13), model, sampler, n_sweeps;
                            init = (; X = X0),
                            # X is 2384 x 161 per sweep: keep it live for
                            # conditioning, but out of the chain object.
                            save_states = (X = :buffer,))

println("\n=== posterior summary ($n_sweeps sweeps) ===")
for name in (:tau, :lambda, :beta, :q, :c1, :a2, :b2)
    v = vec(chain[name])
    println(rpad(name, 8), " mean ", rpad(round(mean(v); digits = 4), 10),
            " sd ", round(std(v); digits = 4))
end
