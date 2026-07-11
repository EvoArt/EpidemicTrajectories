# Functional ("iFFBS paper") model specification.
#
# A model is described by a `RateBundle`: a small set of functions that, given
# the current parameters, the model object, the data, an individual `i`, and a
# time `t`, return the per-step transition probabilities for individual `i` at
# time `t`. This mirrors the convention in the badger iFFBS framework, where
# every rate follows the signature `f(pars, model, data, i, t)` and the SAME
# functions drive both the HMC likelihood and the FFBS latent-state sampler —
# the single-source-of-truth property that makes the two consistent.
#
# The core method a bundle must supply is `transition_matrix_at`: it returns the
# `n_states x n_states` matrix `P` where `P[a, b]` is the probability of moving
# from dense state `a` at time `t` to dense state `b` at time `t+1`, for
# individual `i`. Everything downstream (simulate, likelihood, FFBS) is written
# once against `transition_matrix_at` and works for any `RateBundle`.

"""
    RateBundle

Abstract supertype for a functional model specification. A concrete subtype
must implement

    transition_matrix_at(rb, pars, model, data, i, t) -> AbstractMatrix

returning the per-individual, per-step transition-probability matrix `P` with
`P[a, b] = Prob(state a at t -> state b at t+1)` in the dense state ordering of
the model's `StateSpace`. `pars` is whatever parameter object the model uses
(a NamedTuple is typical); `model` and `data` are user-defined carriers of the
fixed structure and observations; `i`, `t` are the individual and time indices.

The returned matrix's element type should follow `eltype(pars)` so the same
function differentiates cleanly under AD (for the HMC likelihood) and runs in
plain `Float64` (for the FFBS sampler) — see [`TwoStateSI`](@ref) for the
canonical implementation.
"""
abstract type RateBundle end

"""
    transition_matrix_at(rb::RateBundle, pars, model, data, i, t) -> AbstractMatrix

The one method every [`RateBundle`](@ref) must implement. See `RateBundle`'s
docstring for the contract. Must be defined for each concrete bundle type.
"""
function transition_matrix_at end

# ---------------------------------------------------------------------------
# The canonical two-state S/I bundle: the iFFBS-paper cattle model.
# ---------------------------------------------------------------------------

"""
    TwoStateSI(; foi_external, foi_within, recovery)

The functional-style rate bundle for the two-state `{S, I}` recurrent Markov
model of the iFFBS paper (Touloupou et al. 2019). Each animal in a pen is either
susceptible (`S`) or infected (`I`); transitions from `t` to `t+1` are:

- `S -> I` with probability `1 - exp(-α - β·I₋)`, where `I₋` is the number of
  OTHER infected animals in the same pen at time `t` (frequency/density-dependent
  within-pen transmission), `α` the external (primary) force of infection, and
  `β` the within-pen transmission rate.
- `I -> S` (recovery) with probability `1/m`, where `m` is the mean infectious
  period; `I -> I` (stay infected) with probability `1 - 1/m`. There is no
  removed compartment — a recovered animal is susceptible again.

The three fields are functions extracting the relevant scalar from `pars` (they
default to reading NamedTuple fields `α`, `β`, `m`):

- `foi_external(pars) -> α`
- `foi_within(pars)   -> β`
- `recovery(pars)     -> 1/m`  (returns the per-step recovery PROBABILITY)

`transition_matrix_at(::TwoStateSI, pars, model, data, i, t)` needs `data` to
expose, for the pen/group containing individual `i`:
- `data.states::AbstractArray{<:Integer}` — the current hidden-state trajectory
  (used to count other infected animals `I₋` at time `t`);
- `data.group[i]` / `data.members(data, g)` — group membership, so `I₋` counts
  only same-pen animals and EXCLUDES `i` itself (leave-one-out force of
  infection, the key to the FFBS coupling term).

See the package docs' iFFBS example for a concrete `data` object.
"""
Base.@kwdef struct TwoStateSI{Fα,Fβ,Fr} <: RateBundle
    foi_external::Fα = pars -> pars.α
    foi_within::Fβ = pars -> pars.β
    recovery::Fr = pars -> 1 / pars.m
end

# Count infected animals in individual `i`'s group at time `t`, EXCLUDING `i`
# itself. This leave-one-out count is what makes the frequency-dependent force of
# infection correct inside FFBS: when we resample individual `i`'s whole
# trajectory, the pressure it feels from its penmates must not include its own
# (currently-being-resampled) infection status. `data.members(data, g)` returns
# the individuals in group `g`; `data.group[i]` is `i`'s group.
@inline function _other_infected(data, ss::StateSpace, i, t, infected_code)
    g = data.group[i]
    n = 0
    for j in data.members(data, g)
        j == i && continue
        @inbounds data.states[j, t] == infected_code && (n += 1)
    end
    return n
end

function transition_matrix_at(rb::TwoStateSI, pars, model, data, i, t)
    ss = model.state_space::StateSpace
    T = eltype(pars)
    α = T(rb.foi_external(pars))
    β = T(rb.foi_within(pars))
    rec = T(rb.recovery(pars))

    # dense indices for S and I
    S = state_index(ss, 0)
    I = state_index(ss, 1)

    # leave-one-out infected count among penmates at time t
    I_minus = _other_infected(data, ss, i, t, 1)

    # S -> I infection probability (frequency/density-dependent within pen).
    # `foi = α + β·I₋ ≥ 0` for non-negative α, β, so `pStoI = 1 - exp(-foi)`
    # is automatically in [0, 1). We still clamp both recovery and infection
    # probabilities strictly inside (0, 1): a sampler (NUTS) exploring the
    # parameter space can momentarily propose values (e.g. a recovery rate
    # `1/m > 1` when `m < 1`, or a huge `foi`) that would push a probability to
    # exactly 0 or 1 and make `log(P[a,b])` in the likelihood hit `log(0) = -Inf`
    # or, worse, a negative argument. Clamping keeps the log-density finite and
    # differentiable everywhere; the true parameters sit well inside the band, so
    # this only regularizes the tails, it doesn't bias the fit. (Model authors who
    # constrain `m ≥ 1`, e.g. via an `m = m̃ + 1` reparameterization as in the
    # iFFBS paper, never reach the clamp for recovery.)
    eps = T(1e-12)
    foi = α + β * I_minus
    pStoI = clamp(-expm1(-foi), eps, one(T) - eps)   # 1 - exp(-foi), numerically stable
    pStoS = one(T) - pStoI

    # I -> S recovery / I -> I persistence
    pItoS = clamp(rec, eps, one(T) - eps)
    pItoI = one(T) - pItoS

    P = zeros(T, 2, 2)
    @inbounds begin
        P[S, S] = pStoS
        P[S, I] = pStoI
        P[I, S] = pItoS
        P[I, I] = pItoI
    end
    return P
end

# ---------------------------------------------------------------------------
# Observation model: imperfect diagnostic tests.
# ---------------------------------------------------------------------------

"""
    DiagnosticTest(; sensitivity, specificity=(pars -> 1.0), positive_code=1)

An imperfect diagnostic test applied to individuals over time. `sensitivity(pars)`
returns P(test positive | truly infected); `specificity(pars)` returns
P(test negative | truly susceptible) (default 1, i.e. no false positives, as in
the iFFBS-paper cattle tests). `positive_code` is the infected state's user code.

Used to build the per-state observation likelihood via [`observation_likelihood`](@ref):
given a test result `r` for an individual at a time, it returns the vector of
likelihoods `[P(r | state a) for a in dense states]`, the "corrector" term the
FFBS filter multiplies into its predicted state probabilities.

A test result convention (matching the iFFBS-paper data): a NEGATIVE value (e.g.
`-1`) means "not tested / result missing" and contributes likelihood 1 to every
state (no information); `1` means test positive, `0` means test negative.
"""
Base.@kwdef struct DiagnosticTest{Fse,Fsp}
    sensitivity::Fse
    specificity::Fsp = pars -> 1.0
    positive_code::Int = 1
end

"""
    observation_likelihood(tests, pars, ss::StateSpace, results) -> Vector

Given a tuple/vector of [`DiagnosticTest`](@ref)s and their `results` for one
individual at one time (each result `-1`/missing, `0` negative, or `1` positive),
return the length-`nstates(ss)` vector whose entry `a` is the joint likelihood of
all the results if the individual were truly in dense state `a`. Tests are
assumed conditionally independent given the true state.

For the two-state `{S, I}` model with a test of sensitivity `θ` and specificity
`1`: a positive result gives likelihood `[0, θ]` (impossible if susceptible,
probability `θ` if infected); a negative gives `[1, 1-θ]`; a missing result gives
`[1, 1]`.
"""
function observation_likelihood(tests, pars, ss::StateSpace, results)
    N = nstates(ss)
    T = promote_type(eltype(pars), Float64)
    lik = ones(T, N)
    for (test, r) in zip(tests, results)
        r < 0 && continue  # missing / not tested: no information
        se = T(test.sensitivity(pars))
        sp = T(test.specificity(pars))
        pos_idx = state_index(ss, test.positive_code)
        for a in 1:N
            if a == pos_idx
                # truly infected: P(positive)=se, P(negative)=1-se
                lik[a] *= (r == 1 ? se : (one(T) - se))
            else
                # truly not infected: P(positive)=1-sp, P(negative)=sp
                lik[a] *= (r == 1 ? (one(T) - sp) : sp)
            end
        end
    end
    return lik
end
