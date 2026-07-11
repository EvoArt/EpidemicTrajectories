# Forward simulation of hidden state trajectories and observed test data.
#
# `simulate_trajectory` steps the model forward in time using the SAME
# `transition_matrix_at` that drives the likelihood and FFBS — so a simulated
# dataset is guaranteed self-consistent with the model being fit.
#
# The `model`/`data` convention used throughout the package (kept deliberately
# lightweight so it works with plain NamedTuples/closures, no framework types):
#
#   model : any object with fields `.state_space::StateSpace` and `.rates::RateBundle`.
#           (test observation models are passed separately to the observation
#           functions.)
#   data  : any object exposing
#             .states  :: Matrix{Int}   (n_individuals x n_times) hidden states
#             .group   :: Vector{Int}   group/pen membership per individual
#             .members(data, g)         iterable of individuals in group g
#           plus whatever a particular RateBundle reads. `simulate_trajectory`
#           WRITES into `data.states`, so pass a fresh mutable matrix.

"""
    EpiModel(state_space, rates)

Minimal carrier bundling a [`StateSpace`](@ref) with a [`RateBundle`](@ref).
This is the `model` argument the simulate / likelihood / FFBS functions expect
(they read `model.state_space` and `model.rates`). You can use any object with
those two fields instead — `EpiModel` is just the convenient default.
"""
struct EpiModel{SS<:StateSpace,RB<:RateBundle}
    state_space::SS
    rates::RB
end

"""
    make_data(states, group; members=default_members)

Build the lightweight `data` object the package functions expect, from a hidden
state matrix `states` (`n_individuals x n_times`, user state codes) and a `group`
vector (group index per individual). `members(data, g)` defaults to "all
individuals whose `group` equals `g`" (precomputed for speed). Additional fields
(e.g. test results) can be attached by using a NamedTuple built with
`merge(make_data(...), (; test_results=...))`-style extension, or by supplying
your own object with the required fields.
"""
function make_data(states::AbstractMatrix{<:Integer}, group::AbstractVector{<:Integer};
                   extra...)
    groups = sort!(unique(group))
    members_by_group = Dict(g => findall(==(g), group) for g in groups)
    members(data, g) = members_by_group[g]
    return (; states=states, group=collect(Int, group), members=members, extra...)
end

"""
    simulate_trajectory!(rng, model, data, initial_prob; n_times)

Simulate a hidden state trajectory forward in time IN PLACE into `data.states`
(which must be `n_individuals x n_times`). At `t = 1`, each individual is drawn
from `initial_prob` (a length-`nstates` probability vector over dense states).
For `t = 2:n_times`, each individual is advanced one step by sampling from the
row of its per-step transition matrix (`transition_matrix_at`) corresponding to
its current dense state.

Because transition matrices can depend on the states of OTHER individuals at
time `t` (frequency-dependent transmission), the states at `t` must all be known
before advancing to `t+1` — this function fills column `t+1` from column `t`,
left to right, which is exactly the discrete-time Markov assumption.

Returns `data.states`.
"""
function simulate_trajectory!(rng::AbstractRNG, model, data, initial_prob; n_times=size(data.states, 2))
    ss = model.state_space
    X = data.states
    n_ind = size(X, 1)
    cum0 = cumsum(initial_prob)
    # t = 1: draw from the initial distribution
    for i in 1:n_ind
        u = rand(rng)
        a = findfirst(≥(u), cum0)
        X[i, 1] = ss.codes[a === nothing ? length(cum0) : a]
    end
    # t = 2 .. n_times: advance one Markov step per individual
    for t in 1:(n_times - 1)
        for i in 1:n_ind
            a = state_index(ss, X[i, t])
            P = transition_matrix_at(model.rates, _sim_pars(model), model, data, i, t)
            row = @view P[a, :]
            u = rand(rng)
            c = zero(eltype(row))
            b = length(row)
            for k in eachindex(row)
                c += row[k]
                if u ≤ c
                    b = k
                    break
                end
            end
            X[i, t + 1] = ss.codes[b]
        end
    end
    return X
end

# Simulation needs concrete parameter values; `transition_matrix_at` takes them
# as its `pars` argument. We stash them on the model for simulation convenience
# via a `sim_pars` field if present, else require the caller to pass them. To
# keep the signature simple we read `model.pars` when simulating.
_sim_pars(model) = model.pars

"""
    simulate_trajectory(rng, state_space, rates, pars, group, initial_prob; n_times)

Convenience wrapper: allocate a fresh `states` matrix and simulate a trajectory
for the given parameters. Returns `(states, data)` where `states` is the
`n_individuals x n_times` hidden-state matrix and `data` is the `data` object
(so downstream likelihood/FFBS calls can reuse it). `group` gives each
individual's group index.

# Example (two-state S/I, one pen of 8 animals, 99 days)
```julia
ss = SI
rates = TwoStateSI()
pars = (; α=0.009, β=0.01, m=9.0)
group = ones(Int, 8)                 # all 8 animals in pen 1
init = [0.9, 0.1]                    # 10% infected at t=1 (ν = 0.1)
states, data = simulate_trajectory(rng, ss, rates, pars, group, init; n_times=99)
```
"""
function simulate_trajectory(rng::AbstractRNG, state_space::StateSpace, rates::RateBundle,
                             pars, group::AbstractVector{<:Integer}, initial_prob; n_times)
    n_ind = length(group)
    states = zeros(Int, n_ind, n_times)
    data = make_data(states, group)
    model = (; state_space=state_space, rates=rates, pars=pars)
    simulate_trajectory!(rng, model, data, initial_prob; n_times=n_times)
    return states, data
end

"""
    simulate_observations(rng, tests, pars, state_space, states) -> Vector{Matrix{Int}}

Given simulated hidden `states` and a tuple of [`DiagnosticTest`](@ref)s, draw
observed test results for every individual and time. Returns one
`n_individuals x n_times` `Int` matrix of results per test, with `1` = positive,
`0` = negative (an individual truly in the infected state tests positive with the
test's sensitivity; a truly-uninfected individual tests positive with probability
`1 - specificity`). No missingness is introduced here — mask untested times
afterward by setting entries to `-1`.
"""
function simulate_observations(rng::AbstractRNG, tests, pars, state_space::StateSpace, states::AbstractMatrix{<:Integer})
    n_ind, n_t = size(states)
    out = Matrix{Int}[]
    for test in tests
        R = zeros(Int, n_ind, n_t)
        se = test.sensitivity(pars)
        sp = test.specificity(pars)
        pos_code = test.positive_code
        for t in 1:n_t, i in 1:n_ind
            if states[i, t] == pos_code
                R[i, t] = rand(rng) < se ? 1 : 0
            else
                R[i, t] = rand(rng) < (1 - sp) ? 1 : 0
            end
        end
        push!(out, R)
    end
    return out
end
