# Individual forward-filtering, backward-sampling (iFFBS).
#
# Resamples one individual's whole state trajectory from its exact conditional
# given the parameters, the observations, and every other individual's trajectory.
# Sweeping over all individuals is a valid Gibbs update of the entire latent `X`.
#
# The aggregates are kept consistent with `X` throughout by reversing the focal
# individual's contribution before refiltering and re-applying it afterwards — see
# `aggregates.jl` for why that reversibility is the design's foundation. It is also
# what makes the statistics the focal individual sees leave-one-out (excluding
# itself) without any special-casing.

initialise_forward_filter(model, data::EpidemicData, X, i, t) = data.starting_state(model, data, X, i, t)

"""
    forward_filter(xᵢ, start_sampling, end_sampling, model, data, X, i)

The forward pass for individual `i`: at each time, push the previous filtered
distribution through `i`'s transition matrix, then multiply in the observation
likelihood and the coupling term. Returns the filtered distributions and the
cached transition matrices (reused by the backward pass).
"""
function forward_filter(xᵢ, start_sampling, end_sampling, model, data::EpidemicData, X, i)
    probs = zeros(Float64, length(xᵢ), data.n_states)
    trans_cache = Vector{Matrix{Float64}}(undef, length(xᵢ))

    t0 = start_sampling
    base = initialise_forward_filter(model, data, X, i, t0)
    obs = data.observation_process(model, data, X, i, t0)
    affected = data.affected_individuals === nothing ? nothing : data.affected_individuals[t0, i]
    rest = data.rest_contribution(model, data, X, i, t0, data.n_states, affected)
    init = base .* obs .* rest
    probs[1, :] .= init ./ sum(init)
    trans_cache[1] = zeros(Float64, data.n_states, data.n_states)

    for j in 2:length(xᵢ)
        t = start_sampling + j - 1
        tp = t - 1
        trans = transition_matrix_at(data.trans_mat, model, data, X, i, tp)
        trans_cache[j] = trans
        pred = trans' * view(probs, j - 1, :)
        obs_w = data.observation_process(model, data, X, i, t)
        affected = data.affected_individuals === nothing ? nothing : data.affected_individuals[t, i]
        rest_w = data.rest_contribution(model, data, X, i, t, data.n_states, affected)
        unnorm = pred .* obs_w .* rest_w
        z = sum(unnorm)
        probs[j, :] .= z > 0 ? unnorm ./ z : fill(1.0 / data.n_states, data.n_states)
    end

    probs, trans_cache
end

"""
    backward_sample!(probs, trans_cache, xᵢ, start_sampling, end_sampling, model, data, X, i, rng)

The backward pass: draw the final state from its filtered distribution, then walk
backwards drawing each state from its filtered distribution reweighted by the
already-sampled next state. Writes into `xᵢ` (a view into `X`).
"""
function backward_sample!(probs, trans_cache, xᵢ, start_sampling, end_sampling, model, data::EpidemicData, X, i, rng)
    n_t = length(xᵢ)
    xᵢ[n_t] = _sample_categorical(rng, view(probs, n_t, :))

    for j in (n_t - 1):-1:1
        trans = trans_cache[j + 1]
        bnext = xᵢ[j + 1]
        cond = view(probs, j, :) .* view(trans, :, bnext)
        z = sum(cond)
        w = z > 0 ? cond ./ z : fill(1.0 / data.n_states, data.n_states)
        xᵢ[j] = _sample_categorical(rng, w)
    end
    nothing
end

"""
    iffbs_individual!(model, data, X, i, rng)

Resample individual `i`'s whole trajectory in place.

Reverses `i`'s contribution out of the aggregates, runs the forward filter and
backward sample, then re-applies `i`'s new contribution — leaving the aggregates
exactly consistent with the updated `X`, and giving `i` leave-one-out statistics
while it is being resampled.
"""
function iffbs_individual!(model, data::EpidemicData, X, i, rng)
    start_sampling, end_sampling = data.sampling_period[i]
    xᵢ = @view X[start_sampling:end_sampling, i]

    for t in start_sampling:end_sampling
        for ds in data.derived_summaries
            ds(model, data, X, X[t, i], i, t; reverse=true)
        end
    end

    probs, trans_cache = forward_filter(xᵢ, start_sampling, end_sampling, model, data, X, i)
    backward_sample!(probs, trans_cache, xᵢ, start_sampling, end_sampling, model, data, X, i, rng)

    for t in start_sampling:end_sampling
        for ds in data.derived_summaries
            ds(model, data, X, X[t, i], i, t)
        end
    end

    nothing
end

"""
    iffbs!(model, data, X, rng) -> X

One full iFFBS sweep: resample every individual's trajectory in turn, each
conditioning on the others' current trajectories.

Assumes the aggregates already agree with `X` on entry (see
[`apply_derived_summaries!`](@ref)) and preserves that invariant on exit, so a
sweep never rebuilds them.
"""
function iffbs!(model, data::EpidemicData, X, rng)
    for i in 1:data.n_individuals
        iffbs_individual!(model, data, X, i, rng)
    end
    X
end
