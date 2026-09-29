# Exact marginalisation over the hidden trajectories of independent individuals.
#
# When no individual's transitions or observations depend on another's hidden
# state, the population given the parameters is a product of independent hidden
# Markov models, one per individual. Each one's trajectory can then be summed
# out exactly by the forward algorithm, at O(T K^2) per individual, and the
# parameters fitted against the marginal likelihood with no latent block at all.
#
# Four operations live here, and they are different things:
#
#   * the forward algorithm gives p(y | theta), summing over every hidden path;
#   * the filtered distribution p(x_t | y_1:t, theta) falls out of the same pass;
#   * backward sampling draws whole paths from p(x | y, theta);
#   * (Viterbi would give the single most probable path -- not a likelihood, and
#     not provided).
#
# A collapsed parameter fit needs only the first. It is deterministic, so running
# it inside every gradient call is integration, not the stochastic latent update
# the rest of the package keeps outside the gradient.
#
# The factors are exactly those of `epidemic_loglik + epidemic_obs_loglik` with
# `X` summed out: the starting state and the observation at each individual's
# window start, then one transition and one observation per later timepoint in
# its window. No 1e-12 guards: an observation no state can explain gives `-Inf`,
# where the complete-path likelihood's guard would return a large finite number.

"""
    require_independent(data; caller="the marginal likelihood") -> data

Refuse a model whose DECLARED structure couples individuals.

Exact per-individual marginalisation is valid only when no individual's
transitions or observations depend on another's hidden state. The package can
check what the model declares about that, and does:

  * no individual has anyone in `affected_individuals`, or
  * `coupled_transitions = []` was given, declaring that no neighbour's
    transition can be influenced by the focal.

A model with neighbours and no such declaration (the default for a grouped
model) is refused, with a message saying which to change.

What it cannot check is what an opaque rate or observation function actually
reads. Passing this check is the caller's attestation that:

  * rates `(model, data, i, t)` read no aggregate or other quantity built from
    the latent trajectory, only parameters, covariates and fixed structure;
  * `starting_state` and the observation functions read nothing from `X`
    except, at most, cells they are handed but never depend on;
  * transitions are first-order: no path, sojourn-time or history dependence
    beyond fixed covariates and OBSERVED history.

[`check_independent`](@ref) adds a numeric tripwire for the first two.
"""
function require_independent(data::EpidemicData; caller = "the marginal likelihood")
    aff = data.affected_individuals
    aff === nothing && return data
    mask = data.coupled_mask
    # An all-false mask is an explicit statement that no neighbour move depends
    # on the focal, which is exactly the independence being asked about.
    (mask !== nothing && !any(mask)) && return data
    k = findfirst(!isempty, aff)
    k === nothing && return data
    t, i = Tuple(k)
    throw(ArgumentError("""
        $caller needs individuals that are independent given the parameters, but
        individual $i has neighbours at t = $t and the model does not declare
        that none of their transitions depend on it. Either give every
        individual an empty neighbour list (e.g. one group per individual), or
        pass `coupled_transitions = []` to `epidemic_data` if the model really
        has no between-individual coupling. A genuinely coupled model cannot be
        marginalised individual by individual; use the latent sampler."""))
end

# One set of working buffers per call, in the parameters' element type. Held per
# call rather than module-wide so that concurrent chains, and gradient calls
# interleaved with plain ones, never share state. `from`/`to` are the declared
# transitions' state codes and `r` their rates at one step: the forward step
# moves probability along those alone rather than through a dense K x K matrix.
struct _HMMWork{T}
    alpha::Vector{T}
    buf::Vector{T}
    w::Vector{T}
    rowsum::Vector{T}
    r::Vector{T}
    from::Vector{Int}
    to::Vector{Int}
end
function _HMMWork(::Type{T}, data::EpidemicData) where {T}
    K = data.n_states
    tr = data.trans_mat.transitions
    from = [_state_index(data, f) for (f, _) in tr]
    to = [_state_index(data, s) for (_, s) in tr]
    _HMMWork{T}(zeros(T, K), zeros(T, K), zeros(T, K), zeros(T, K), zeros(T, length(tr)),
                from, to)
end

# The per-state observation weights at (i, t), from the scalar form when the
# model has one (no allocation) and the vector form otherwise.
@inline function _obs_weights!(w, model, data::EpidemicData, X, i, t)
    ow = data.observation_weight
    if ow === nothing
        v = data.observation_process(model, data, X, i, t)
        @inbounds for s in eachindex(w)
            w[s] = v[s]
        end
    else
        _fill_obs!(w, ow, model, data, X, i, t)
    end
    w
end
@inline function _fill_obs!(w, ow, model, data, X, i, t)
    @inbounds for s in eachindex(w)
        w[s] = ow(model, data, X, i, t, s)
    end
end
# Shared values once per (i, t) rather than once per state.
@inline function _fill_obs!(w, ow::SharedObservationWeight, model, data, X, i, t)
    sh = ow.shared(model, data, i, t)
    @inbounds for s in eachindex(w)
        w[s] = ow.weight(model, data, X, i, t, s, sh)
    end
end

# alpha <- alpha' P_t, along the declared transitions only. Entry for entry the
# same probabilities as `transition_matrix_at!` (each rate clamped, each row
# closed by `_close_row`), so the same model, only without building P.
function _propagate!(work::_HMMWork, model, data::EpidemicData, X, i, t)
    alpha, buf, rowsum, r, from, to = work.alpha, work.buf, work.rowsum, work.r, work.from, work.to
    tm = data.trans_mat
    sh = _step_shared(tm.shared, model, data, i, t)
    _rates!(r, tm.rate_fns, 1, from, to, tm.shared, sh, model, data, X, i, t)
    fill!(rowsum, zero(eltype(rowsum)))
    @inbounds for k in eachindex(r)
        rowsum[from[k]] += r[k]
    end
    @inbounds for a in eachindex(alpha)
        scale, self = _close_row(rowsum[a])
        rowsum[a] = scale * alpha[a]       # now the mass leaving a, per unit rate
        buf[a] = self * alpha[a]
    end
    @inbounds for k in eachindex(r)
        buf[to[k]] += rowsum[from[k]] * r[k]
    end
    copyto!(alpha, buf)
end

@inline _rates!(r, ::Tuple{}, k, from, to, shf, sh, model, data, X, i, t) = nothing
@inline function _rates!(r, rates::Tuple, k, from, to, shf, sh, model, data, X, i, t)
    @inbounds r[k] = clamp(_call_rate_with_focal(first(rates), shf, sh, model, data, X, i, t,
                                                 from[k], to[k]), 1e-12, 1 - 1e-12)
    _rates!(r, Base.tail(rates), k + 1, from, to, shf, sh, model, data, X, i, t)
end

# Forward-filter individual `i` over `lo:hi`, accumulating the log normalisers.
#
# `start = true` initialises from the starting state at `lo`; `start = false`
# continues from the filtered distribution already in `work.alpha` (at `lo - 1`)
# by propagating it first. Continuing through held-out observations evaluates
# their JOINT density by the chain rule -- it does not refit anything.
#
# Returns `(loglik, zero_at)`: `zero_at` is the first time no state could
# explain the observation (and `loglik` is then -Inf), or 0.
function _hmm_forward!(work::_HMMWork{T}, model, data::EpidemicData, X, i::Int,
                       lo::Int, hi::Int, start::Bool) where {T}
    alpha, w = work.alpha, work.w
    K = length(alpha)
    ll = zero(T)
    if start
        p0 = data.starting_state(model, data, X, i, lo)
        @inbounds for s in 1:K
            alpha[s] = p0[s]
        end
    end
    for t in lo:hi
        if !(start && t == lo)
            # alpha_t(b) = sum_a alpha_{t-1}(a) P_{t-1}(a, b). The step t-1 -> t
            # uses the matrix at t-1, as `transition_prob` does in the
            # complete-path likelihood.
            _propagate!(work, model, data, X, i, t - 1)
        end
        _obs_weights!(w, model, data, X, i, t)
        z = zero(T)
        @inbounds for s in 1:K
            alpha[s] *= w[s]
            z += alpha[s]
        end
        # `!(z > 0)` also catches NaN. A structural zero is reported, never
        # papered over with a uniform restart as the sampler's filter does.
        (z > 0) || return (zero(T) - Inf, t)
        @inbounds for s in 1:K
            alpha[s] /= z
        end
        ll += log(z)
    end
    (ll, 0)
end

# A read-only stand-in for the `X` argument the model's callbacks take. An
# eligible model never reads it (see `require_independent`); it exists only to
# satisfy the signatures, and `check_independent` perturbs it to prove that.
_scaffold(data::EpidemicData) = ones(Int, data.n_timepoints, data.n_individuals)

function _check_scaffold(X, data::EpidemicData)
    size(X) == (data.n_timepoints, data.n_individuals) || throw(DimensionMismatch(
        "the model is $(data.n_timepoints) x $(data.n_individuals) but the " *
        "marginal likelihood was built for $(size(X, 1)) x $(size(X, 2))"))
    X
end

@inline function _window(data::EpidemicData, i::Int)
    f, l = data.sampling_period[i]
    (f, min(l, data.n_timepoints))
end

"""
    epidemic_marginal_loglik(data; threads=1) -> marginal_loglik

Build the MARGINAL likelihood of a model whose individuals are independent given
the parameters: `marginal_loglik(model, data) -> Real`, the log probability of
the observations with every hidden trajectory summed out exactly.

It replaces BOTH halves of the complete-data likelihood,
`epidemic_loglik(data)(model, data, X) + epidemic_obs_loglik(data)(model, data, X)`,
and takes no `X`: a collapsed model has no latent block. In a PracticalBayes
model it is the one likelihood term:

```julia
mll = epidemic_marginal_loglik(data)
@model function collapsed(data, mll)
    phi ~ Beta(2, 2)
    p   ~ Beta(2, 2)
    @addlogprob! mll((; phi, p), data)
end
```

and any continuous sampler (NUTS, HMC) fits it. The forward recursion runs
inside every gradient call; that is deterministic integration, so the gradient
is exact, unlike a stochastic latent update inside the gradient.

`data` is an argument of the returned function, as for [`epidemic_loglik`](@ref),
so the same function scores a truncated copy for leave-future-out refits.

Autodiff-friendly in `model`: every buffer takes the parameters' element type
(see `_param_eltype`), is allocated per call, and nothing shared is mutated, so
calls are reentrant across chains and threads. Tested under ForwardDiff.

Returns `-Inf` when some individual's observations have probability zero under
every hidden path. [`hmm_filter`](@ref) says which individual and when.

`threads > 1` splits the individuals into that many contiguous blocks of about
equal work (individual-timepoints), runs each as a task and sums the blocks in
order, so the value does not depend on scheduling. It suits plain and
forward-mode (Dual) evaluation. Reverse-mode backends that do not follow tasks
(Mooncake) need the default, `threads = 1`.

## Scope

Independent, finite-state, first-order individuals: time- and
individual-varying transitions, starting distributions and observation weights,
irregular intervals folded into the rates, absorbing states and missing
observations (an all-ones weight). Refused, via [`require_independent`](@ref):
declared coupling. Not supported: the `entry_time` survival gate and a
semi-Markov `step_logprob`, both of which change the per-step factor.

The observation model is the FULL one `data` carries. A conjugate split of the
observation factor (the `observation_process` keyword of
[`epidemic_obs_loglik`](@ref)) has no meaning once `X` is integrated out:
there is no sampled `X` left for a conjugate kernel to condition on.
"""
function epidemic_marginal_loglik(data::EpidemicData; threads::Int = 1)
    require_independent(data; caller = "epidemic_marginal_loglik")
    threads >= 1 || throw(ArgumentError("threads must be at least 1, got $threads"))
    X0 = _scaffold(data)
    function marginal_loglik(model, data::EpidemicData)
        _check_scaffold(X0, data)
        T = _param_eltype(model)
        threads == 1 && return _marginal_block(model, data, X0, T, 1:data.n_individuals)
        tasks = [Threads.@spawn(_marginal_block(model, data, X0, T, b))
                 for b in _work_blocks(data, threads)]
        ll = zero(T)
        for tk in tasks
            ll += fetch(tk)::T
        end
        ll
    end
    marginal_loglik
end

function _marginal_block(model, data::EpidemicData, X, ::Type{T}, individuals) where {T}
    work = _HMMWork(T, data)
    ll = zero(T)
    for i in individuals
        f, hi = _window(data, i)
        f <= hi || continue
        lli, _ = _hmm_forward!(work, model, data, X, i, f, hi, true)
        ll += lli
        isfinite(lli) || return ll
    end
    ll
end

# Contiguous ranges of individuals holding about equal numbers of timepoints.
function _work_blocks(data::EpidemicData, n::Int)
    cum = cumsum(max(0, hi - f + 1) for (f, hi) in (_window(data, i) for i in 1:data.n_individuals))
    total = isempty(cum) ? 0 : cum[end]
    edges = [0; [searchsortedfirst(cum, total * k / n) for k in 1:(n - 1)]; data.n_individuals]
    [(edges[k] + 1):edges[k + 1] for k in 1:n if edges[k] < edges[k + 1]]
end

"""
    hmm_logliks(model, data) -> Vector

Each individual's own marginal log likelihood, in the parameters' element type:
the terms [`epidemic_marginal_loglik`](@ref) sums. Zero for an individual with
an empty sampling period.
"""
function hmm_logliks(model, data::EpidemicData; X = _scaffold(data))
    require_independent(data; caller = "hmm_logliks")
    T = _param_eltype(model)
    work = _HMMWork(T, data)
    out = zeros(T, data.n_individuals)
    for i in 1:data.n_individuals
        f, hi = _window(data, i)
        f <= hi || continue
        out[i], _ = _hmm_forward!(work, model, data, X, i, f, hi, true)
    end
    out
end

"""
    hmm_filter(model, data, i; upto) -> NamedTuple

Forward-filter individual `i` from the start of its sampling period to `upto`
(default: the end of its period).

Returns `(; loglik, filtered, zero_at)`: the log probability of its
observations over that stretch, the filtered state distribution
`p(x_upto | y_first:upto)`, and `zero_at`, the first time at which no state
could explain its observation (0 if none). A non-zero `zero_at` is the
diagnostic for a `-Inf` marginal likelihood: it names the individual and the
occasion rather than leaving a bare `-Inf`.
"""
function hmm_filter(model, data::EpidemicData, i::Int;
                    upto::Int = _window(data, i)[2], X = _scaffold(data))
    f, hi = _window(data, i)
    hi = min(hi, upto)
    T = _param_eltype(model)
    work = _HMMWork(T, data)
    f <= hi || return (; loglik = zero(T), filtered = fill(T(NaN), data.n_states),
                         zero_at = 0)
    ll, z = _hmm_forward!(work, model, data, X, i, f, hi, true)
    (; loglik = ll, filtered = copy(work.alpha), zero_at = z)
end

"""
    hmm_sample!(rng, model, data, X; individuals=1:n_individuals) -> X

Draw each listed individual's hidden trajectory from its exact posterior
`p(x_i | y_i, theta)`, writing it into `X` over the individual's sampling
period; cells outside that period are left alone.

Forward filtering, then backward sampling. It is what turns a collapsed fit's
parameter draws into trajectory draws after the fact: one call per parameter
draw gives a joint posterior sample of `(theta, X)`, e.g. for trajectory
summaries or to feed a simulation-based scorer. It is never part of the
likelihood.

Errors on an individual whose observations have zero probability under
`theta`: there is no path to sample.
"""
function hmm_sample!(rng::AbstractRNG, model, data::EpidemicData, X;
                     individuals = 1:data.n_individuals)
    require_independent(data; caller = "hmm_sample!")
    K = data.n_states
    Xs = _scaffold(data)
    work = _HMMWork(Float64, data)
    P = zeros(Float64, K, K)
    rowsum = zeros(Float64, K)
    p = zeros(Float64, K)
    for i in individuals
        f, hi = _window(data, i)
        f <= hi || continue
        n = hi - f + 1
        alphas = Matrix{Float64}(undef, n, K)
        # One-step forward passes, keeping every filtered distribution for the
        # backward pass.
        _, z = _hmm_forward!(work, model, data, Xs, i, f, f, true)
        z == 0 || error("hmm_sample!: individual $i has no path consistent with its data (t = $z)")
        alphas[1, :] .= work.alpha
        for j in 2:n
            t = f + j - 1
            _, z = _hmm_forward!(work, model, data, Xs, i, t, t, false)
            z == 0 || error("hmm_sample!: individual $i has no path consistent with its data (t = $z)")
            alphas[j, :] .= work.alpha
        end
        X[hi, i] = _sample_categorical(rng, view(alphas, n, :))
        for j in (n - 1):-1:1
            t = f + j - 1
            transition_matrix_at!(P, rowsum, data.trans_mat, model, data, Xs, i, t)
            nxt = X[t + 1, i]
            tot = 0.0
            @inbounds for a in 1:K
                p[a] = alphas[j, a] * P[a, nxt]
                tot += p[a]
            end
            p ./= tot
            X[t, i] = _sample_categorical(rng, p)
        end
    end
    X
end

"""
    hmm_forecast_logliks(model, data, t_star, M; known) -> NamedTuple

Exact per-individual log predictive densities for one leave-future-out window,
given the parameters:

    log p(y_i,(t*+1):(t*+M) | y_i,1:t*, theta)

for every individual in the cohort fixed at `t_star` (known by then, see
[`known_times`](@ref), and with a scored occasion in the window).

Each individual is filtered through its training observations only, so its
state at the cutoff is integrated over rather than fixed, then the same
recursion continues through the window and only the window's log normalisers
are kept. That is the whole-block density by the chain rule, so it integrates
over the cutoff state and every future path together. Future entrants are not
scored, and cannot affect the cohort in an independent model.

Returns `(; logp, zero_at)`, both `Dict`s keyed by individual. An individual
whose block has probability zero under `theta` gets `logp = -Inf` and a
`zero_at` entry: the occasion no state could explain, negated if it lies in
the TRAINING history (the draw then contradicts data it was fitted to, which
is a bug in the fit, not a forecast failure).

Combining these over posterior draws is where the predictive target is chosen:
summing over individuals inside the average over draws gives the joint score,
averaging each individual separately gives the individual-history score. They
differ even here, because shared parameter uncertainty makes individuals'
forecasts dependent. See [`lfo_cv`](@ref) with `scorer = ExactHMM()`.
"""
function hmm_forecast_logliks(model, data::EpidemicData, t_star::Int, M::Int;
                              known = first.(data.sampling_period),
                              X = _scaffold(data))
    K = data.n_states
    work = _HMMWork(Float64, data)
    logp = Dict{Int,Float64}()
    zero_at = Dict{Int,Int}()
    last_t = min(t_star + M, data.n_timepoints)
    for i in 1:data.n_individuals
        f, l = _window(data, i)
        (known[i] <= t_star && f <= t_star) || continue
        hi = min(last_t, l)
        hi > t_star || continue        # nothing of this individual's is scored
        _, z = _hmm_forward!(work, model, data, X, i, f, t_star, true)
        if z != 0
            logp[i] = -Inf
            zero_at[i] = -z
            continue
        end
        lp, z = _hmm_forward!(work, model, data, X, i, t_star + 1, hi, false)
        logp[i] = lp
        z == 0 || (zero_at[i] = z)
    end
    (; logp, zero_at)
end

"""
    check_independent(model, data; atol=1e-8) -> data

[`require_independent`](@ref), plus a numeric tripwire for what the structural
check cannot see: a rate that reads a latent aggregate, or a callback that
reads `X`.

Each individual's marginal log likelihood is computed twice at `model`, once
with every callback handed one stand-in trajectory and the aggregates built
from it, once with a different one. An eligible model gives identical answers,
because nothing it computes should read either. A difference names the first
individual affected.

This is a test, not a proof: a dependence that happens to vanish on both
stand-ins escapes it. It restores `data`'s aggregates before returning, but it
does write to them in between, so do not call it while a sampler is using the
same `data`.
"""
function check_independent(model, data::EpidemicData; atol = 1e-8)
    require_independent(data; caller = "check_independent")
    K = data.n_states
    saved = [copy(v) for v in values(data.aggregates)]
    XA = _scaffold(data)
    XB = [mod1(t + 2i, K) for t in 1:data.n_timepoints, i in 1:data.n_individuals]
    la = lb = Float64[]
    try
        reset_aggregates!(data); apply_derived_summaries!(model, data, XA)
        la = Float64.(hmm_logliks(model, data; X = XA))
        reset_aggregates!(data); apply_derived_summaries!(model, data, XB)
        lb = Float64.(hmm_logliks(model, data; X = XB))
    finally
        for (v, s) in zip(values(data.aggregates), saved)
            copyto!(v, s)
        end
    end
    for i in eachindex(la)
        same = la[i] == lb[i] || isapprox(la[i], lb[i]; atol = atol, rtol = 0)
        same || throw(ArgumentError("""
            individual $i's marginal log likelihood changes with the trajectory
            the callbacks are handed ($(la[i]) vs $(lb[i])). Some rate,
            starting-state or observation function reads the latent state or an
            aggregate built from it, so the individuals are not independent and
            cannot be marginalised one at a time."""))
    end
    data
end
