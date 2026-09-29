# Exact forward filtering over the JOINT hidden state of a whole population.
#
# For a coupled model the individuals cannot be summed out one at a time: each
# one's transition probabilities depend on the others' states. The joint state
# can still be summed out, at K^N states per timepoint, which is hopeless for a
# real population and trivial for four or five individuals. That is what this
# is for: an absolute reference against which estimators of coupled predictive
# densities can be checked, not a way to fit anything.
#
# The joint transition from state x at t is the product over individuals of
# their own transition probabilities, each evaluated with the aggregates built
# from x -- exactly the rates the simulator and the likelihood see.

const _JOINT_MAX_STATES = 50_000

struct _JointSpace
    K::Int
    N::Int
    states::Vector{Vector{Int}}     # joint state index -> per-individual states
end

function _joint_space(data::EpidemicData)
    K, N = data.n_states, data.n_individuals
    n = K^N
    n <= _JOINT_MAX_STATES || throw(ArgumentError(
        "the joint state space has $K^$N = $n states; this reference is for tiny " *
        "populations only (at most $_JOINT_MAX_STATES states)"))
    for (i, (f, l)) in enumerate(data.sampling_period)
        (f == 1 && l >= data.n_timepoints) || throw(ArgumentError(
            "the joint reference needs every individual observed over the whole " *
            "series; individual $i has sampling period ($f, $l)"))
    end
    _JointSpace(K, N, [collect(Tuple(c)) for c in vec(CartesianIndices(ntuple(_ -> K, N)))])
end

# Transition matrix of the joint chain from t to t+1, rates evaluated with the
# aggregates built from each source state.
function _joint_transition(model, data::EpidemicData, sp::_JointSpace, X, t::Int)
    n = length(sp.states)
    Pj = zeros(Float64, n, n)
    Ps = [zeros(Float64, sp.K, sp.K) for _ in 1:sp.N]
    rowsum = zeros(Float64, sp.K)
    for (a, x) in enumerate(sp.states)
        X[t, :] .= x
        reset_aggregates!(data)
        apply_derived_summaries!(model, data, X)
        for i in 1:sp.N
            transition_matrix_at!(Ps[i], rowsum, data.trans_mat, model, data, X, i, t)
        end
        for (b, y) in enumerate(sp.states)
            p = 1.0
            for i in 1:sp.N
                p *= Ps[i][x[i], y[i]]
                p == 0 && break
            end
            Pj[a, b] = p
        end
    end
    Pj
end

# The joint observation weight at t. `use(i, t)` says whether individual i's
# observation at t is part of the event being scored; one that is not counts
# as unobserved (weight 1). `alive` lists individuals conditioned to be out of
# the absorbing state at t.
function _joint_obs(model, data::EpidemicData, sp::_JointSpace, X, t::Int,
                    use, alive, absorbing)
    w = ones(Float64, length(sp.states))
    wi = zeros(Float64, sp.K)
    for i in 1:sp.N
        if use(i, t)
            _obs_weights!(wi, model, data, X, i, t)
        else
            fill!(wi, 1.0)
        end
        if i in alive
            wi[absorbing] = 0.0
        end
        for (a, x) in enumerate(sp.states)
            w[a] *= wi[x[i]]
        end
    end
    w
end

# Forward pass over lo:hi with per-step normalisation. Returns the summed log
# normalisers and every filtered distribution (rows by time).
function _joint_forward(model, data::EpidemicData, sp::_JointSpace, lo::Int, hi::Int;
                        use = (i, t) -> true, alive_at = Dict{Int,Vector{Int}}(),
                        alpha0 = nothing)
    absorbing = _absorbing_state(data)
    X = _scaffold(data)
    n = length(sp.states)
    alphas = zeros(Float64, hi - lo + 1, n)
    ll = 0.0
    # `alpha0`, when given, is the filtered distribution at lo - 1; otherwise the
    # pass starts from the starting state at lo.
    if alpha0 === nothing
        alpha = zeros(Float64, n)
        p0s = [data.starting_state(model, data, X, i, lo) for i in 1:sp.N]
        for (a, x) in enumerate(sp.states)
            alpha[a] = prod(p0s[i][x[i]] for i in 1:sp.N)
        end
    else
        alpha = copy(alpha0)
    end
    for t in lo:hi
        if !(alpha0 === nothing && t == lo)
            alpha = vec(alpha' * _joint_transition(model, data, sp, X, t - 1))
        end
        alpha .*= _joint_obs(model, data, sp, X, t, use, get(alive_at, t, Int[]),
                             absorbing)
        z = sum(alpha)
        z > 0 || return (-Inf, alphas)
        alpha ./= z
        ll += log(z)
        alphas[t - lo + 1, :] .= alpha
    end
    reset_aggregates!(data)
    (ll, alphas)
end

"""
    joint_reference(model, data, t_star, M; scored=nothing, alive=Int[])
        -> NamedTuple

Exact quantities for one leave-future-out window of a coupled model, by
enumerating the joint hidden state of every individual. For validation on tiny
populations only (K^N joint states, at most 50,000).

Returns

  * `logp`: `log p(y_S,(t*+1):(t*+M) | y_1:t*)`, the joint predictive density of
    the window's observations of the individuals in `scored` (default: all);
    everyone else's future observations are summed over, not dropped from the
    dynamics;
  * `logp_individual`: that density for each individual on its own, the target
    of an individual-history score under coupling;
  * `log_pA`: `log P(A | y_1:t*)`, where A is "every individual in `alive` is
    out of the absorbing state at t*";
  * `logp_given_A`: `log p(y_S,window | y_1:t*, A)`, so that
    `logp = log_pA + logp_given_A` whenever the window's data imply A;
  * `filtered`, `filtered_A`: the joint state distribution at t* without and
    with the conditioning, over `states`.

Every individual must be observed over the whole series, and a rate at `t` may
read only the aggregates at `t` (the joint state at other times is summed over,
not held in the scaffold the callbacks see).
"""
function joint_reference(model, data::EpidemicData, t_star::Int, M::Int;
                         scored = nothing, alive = Int[])
    sp = _joint_space(data)
    N = sp.N
    S = scored === nothing ? collect(1:N) : collect(scored)
    hi = min(t_star + M, data.n_timepoints)
    absorbing = _absorbing_state(data)

    _, fa = _joint_forward(model, data, sp, 1, t_star)
    alpha = fa[end, :]
    in_A = [all(i -> x[i] != absorbing, alive) for x in sp.states]
    pA = sum(alpha[in_A])
    alphaA = pA > 0 ? (alpha .* in_A) ./ pA : fill(NaN, length(alpha))

    fut(use, a0) = _joint_forward(model, data, sp, t_star + 1, hi; use, alpha0 = a0)[1]
    window(i, t) = t > t_star
    logp = fut((i, t) -> i in S && window(i, t), alpha)
    logpA = pA > 0 ? fut((i, t) -> i in S && window(i, t), alphaA) : -Inf
    logp_i = Dict(i => fut((j, t) -> j == i && window(j, t), alpha) for i in S)
    (; logp, logp_individual = logp_i, log_pA = log(pA), logp_given_A = logpA,
       filtered = alpha, filtered_A = alphaA, states = sp.states)
end

"""
    joint_sample(rng, model, data, t_star; alive=Int[]) -> Matrix{Int}
    joint_sample(rng, model, data, t_star, n; alive=Int[]) -> Vector{Matrix{Int}}

Exact draws of the whole population's hidden trajectory over `1:t_star` given
the observations up to `t_star` and, optionally, the event that every
individual in `alive` is out of the absorbing state at `t_star`: forward
filtering over the joint state, then backward sampling. Cells after `t_star`
are left at 1.

These are what a perfectly mixed latent sampler would return, so estimators
built on them are checked with no MCMC error in the way. For validation on tiny
populations only, as [`joint_reference`](@ref).
"""
joint_sample(rng::AbstractRNG, model, data::EpidemicData, t_star::Int; alive = Int[]) =
    only(joint_sample(rng, model, data, t_star, 1; alive))

function joint_sample(rng::AbstractRNG, model, data::EpidemicData, t_star::Int,
                      n::Int; alive = Int[])
    sp = _joint_space(data)
    _, fa = _joint_forward(model, data, sp, 1, t_star;
                           alive_at = Dict(t_star => collect(alive)))
    Xs = _scaffold(data)
    Ps = [_joint_transition(model, data, sp, Xs, t) for t in 1:(t_star - 1)]
    reset_aggregates!(data)
    out = Vector{Matrix{Int}}(undef, n)
    p = zeros(Float64, length(sp.states))
    for k in 1:n
        X = _scaffold(data)
        b = _sample_categorical(rng, view(fa, t_star, :))
        X[t_star, :] .= sp.states[b]
        for t in (t_star - 1):-1:1
            p .= view(fa, t, :) .* view(Ps[t], :, b)
            p ./= sum(p)
            b = _sample_categorical(rng, p)
            X[t, :] .= sp.states[b]
        end
        out[k] = X
    end
    out
end
