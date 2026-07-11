# Individual forward-filtering, backward-sampling (iFFBS).
#
# Resamples ONE individual's entire hidden-state trajectory `X[i, :]` from its
# exact full conditional given the current parameters, the observed test data,
# and every OTHER individual's fixed trajectory. Sweeping this over all
# individuals is a valid Gibbs update of the whole latent state `X`.
#
# This is a direct generalization of the iFFBS-paper FFBS (see the package's
# reference `original_cattle_ecoli_iFFBS` example) to the `RateBundle` protocol,
# so it works for any two-state (and, later, multi-state) model whose per-step
# dynamics are given by `transition_matrix_at`. The three ingredients of the
# forward filter for the focal individual `i` at each time `t`:
#
#   1. PREDICTION: push last step's filtered distribution through individual i's
#      own transition matrix `P_i(t-1)` (which depends on the OTHER penmates'
#      states at t-1, held fixed).
#   2. OBSERVATION likelihood ("corrector"): multiply by the per-state
#      likelihood of i's observed test results at t (imperfect diagnostics).
#   3. COUPLING likelihood: the crucial iFFBS term. Because i's state at t feeds
#      into its PENMATES' force of infection for their t -> t+1 transitions,
#      the penmates' (fixed, observed-as-latent) transitions carry information
#      about i's state at t. For each candidate state of i at t, we compute the
#      probability the penmates would make exactly the transitions they do, and
#      weight by it. Omitting this term samples from the wrong conditional (it
#      corresponds to the "no coupling" approximation the paper also studies).
#
# Backward sampling then draws `X[i, T]` from the final filtered distribution and
# walks backward, at each step drawing `X[i, t]` from the filtered distribution
# reweighted by the sampled `X[i, t+1]` and i's own transition matrix.

"""
    ffbs_individual!(rng, model, data, i, tests, results;
                     initial_prob, coupling=true)

Resample individual `i`'s full hidden-state trajectory in place into
`data.states[i, :]`, from its exact full conditional given the current
parameters (`model.pars`), the observed `results`, and every other individual's
current trajectory. Implements forward-filtering/backward-sampling with an
imperfect-test observation likelihood and (optionally) the penmate COUPLING term.

Arguments:
- `model`: object with `.state_space`, `.rates`, and `.pars` (current parameter
  values — FFBS runs in plain `Float64`, never under AD).
- `data`: exposes `.states`, `.group`, `.members` (see [`make_data`](@ref)).
- `i`: focal individual index.
- `tests`, `results`: tuple of [`DiagnosticTest`](@ref)s and their result
  matrices (see [`observation_loglik`](@ref) for the format).
- `initial_prob`: length-`nstates` probability vector over dense states at `t=1`.
- `coupling`: include the penmate coupling likelihood (default `true`). Set
  `false` to reproduce the paper's no-coupling variant / for models where an
  individual's state does not enter others' transition rates.

Returns the newly sampled dense-state vector for individual `i`.
"""
function ffbs_individual!(rng::AbstractRNG, model, data, i, tests, results;
                          initial_prob, coupling=true)
    ss = model.state_space::StateSpace
    pars = model.pars
    N = nstates(ss)
    X = data.states
    n_t = size(X, 2)

    predicted = Matrix{Float64}(undef, N, n_t)   # p(state at t | data up to t-1)
    filtered = Matrix{Float64}(undef, N, n_t)    # p(state at t | data up to t)

    # --- t = 1 ---
    @views predicted[:, 1] .= initial_prob
    obs1 = _obs_lik_vec(tests, pars, ss, results, i, 1)
    coup1 = coupling ? _coupling_lik_vec(model, data, i, 1, ss) : ones(N)
    w = @views predicted[:, 1] .* obs1 .* coup1
    @views filtered[:, 1] .= w ./ sum(w)

    # --- forward filter, t = 2 .. n_t ---
    for t in 2:n_t
        # individual i's own transition matrix for (t-1 -> t) depends on penmates
        # at t-1 (i excluded — leave-one-out FOI), held fixed during this sweep.
        Pprev = transition_matrix_at(model.rates, pars, model, data, i, t - 1)
        for b in 1:N
            s = 0.0
            @inbounds for a in 1:N
                s += Pprev[a, b] * filtered[a, t - 1]
            end
            predicted[b, t] = s
        end
        obs = _obs_lik_vec(tests, pars, ss, results, i, t)
        coup = coupling ? _coupling_lik_vec(model, data, i, t, ss) : ones(N)
        wt = @views predicted[:, t] .* obs .* coup
        z = sum(wt)
        @views filtered[:, t] .= (z > 0 ? wt ./ z : fill(1 / N, N))
    end

    # --- backward sample ---
    sampled = Vector{Int}(undef, n_t)      # dense indices
    sampled[n_t] = _sample_categorical(rng, @view filtered[:, n_t])
    for t in (n_t - 1):-1:1
        # p(X_t = a | X_{t+1} = sampled[t+1], data up to t)
        #   ∝ filtered[a, t] * P_i(a -> sampled[t+1]; at time t)
        Pt = transition_matrix_at(model.rates, pars, model, data, i, t)
        bnext = sampled[t + 1]
        w = Vector{Float64}(undef, N)
        @inbounds for a in 1:N
            w[a] = filtered[a, t] * Pt[a, bnext]
        end
        sampled[t] = _sample_categorical(rng, w)
    end

    # write the sampled trajectory back (as user state codes)
    @inbounds for t in 1:n_t
        X[i, t] = ss.codes[sampled[t]]
    end
    return @view X[i, :]
end

# Observation likelihood vector for individual i at time t (length nstates).
@inline function _obs_lik_vec(tests, pars, ss::StateSpace, results, i, t)
    rs = ntuple(k -> results[k][i, t], length(results))
    return observation_likelihood(tests, pars, ss, rs)
end

# Coupling likelihood: for each candidate dense state `a` of focal individual i
# at time t, the probability the OTHER penmates make exactly the (t -> t+1)
# transitions their fixed trajectories show, GIVEN i is in state `a` at t.
# Individual i's state enters penmates' transitions only through the count of
# infected animals in the pen; so we temporarily set i's state and recompute each
# penmate's realized transition probability.
#
# At the final time there is no t -> t+1 transition, so coupling is uninformative
# (returns all-ones).
function _coupling_lik_vec(model, data, i, t, ss::StateSpace)
    N = nstates(ss)
    X = data.states
    n_t = size(X, 2)
    if t == n_t
        return ones(N)
    end
    pars = model.pars
    g = data.group[i]
    penmates = data.members(data, g)

    saved = X[i, t]  # restore afterward
    lik = ones(Float64, N)
    for a in 1:N
        X[i, t] = ss.codes[a]  # hypothesize i in dense state a at time t
        prob = 1.0
        for j in penmates
            j == i && continue
            aj = state_index(ss, X[j, t])
            bj = state_index(ss, X[j, t + 1])
            Pj = transition_matrix_at(model.rates, pars, model, data, j, t)
            @inbounds prob *= Pj[aj, bj]
        end
        lik[a] = prob
    end
    X[i, t] = saved  # restore i's original state
    return lik
end

@inline function _sample_categorical(rng::AbstractRNG, w)
    z = sum(w)
    u = rand(rng) * z
    c = 0.0
    @inbounds for k in eachindex(w)
        c += w[k]
        u ≤ c && return k
    end
    return length(w)
end

"""
    ffbs_sweep!(rng, model, data, tests, results; initial_prob, coupling=true)

One full iFFBS Gibbs sweep of the entire latent state: resample every
individual's trajectory in turn via [`ffbs_individual!`](@ref), in index order
(each resample conditions on the just-updated trajectories of earlier
individuals — a systematic-scan Gibbs sweep). Mutates `data.states` in place and
returns it.

This is the function a PracticalBayes `AbstractLatentKernel`'s `latent_step`
calls once per outer Gibbs sweep (see the `PracticalEpiBayes` companion package),
handing back `(; X = copy(data.states))`.
"""
function ffbs_sweep!(rng::AbstractRNG, model, data, tests, results; initial_prob, coupling=true)
    n_ind = size(data.states, 1)
    for i in 1:n_ind
        ffbs_individual!(rng, model, data, i, tests, results; initial_prob=initial_prob, coupling=coupling)
    end
    return data.states
end
