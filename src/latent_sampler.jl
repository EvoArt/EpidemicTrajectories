# Metropolis updates for every latent of a LatentHMMData model.
#
# Shared latents first, one element at a time, scored on every individual at
# once. Then each individual, in parallel: its PerIndividual latents, scored by a
# full forward pass, and its Field elements in time order. A Field element is
# scored only over the timepoints it touches: the running forward vector up to
# the window, the window's steps, and a backward message computed at the start
# of the individual's sweep. Elements are visited in window order, so the
# backward message never includes a step an earlier element changed.

mutable struct LatentSampler{D,SC,MS}
    data::D
    scales::SC                     # per latent: Field n x N, PerIndividual N, Shared n
    mscales::MS                    # per latent: a Field's joint moves (pair n x N, shift N, scale N)
    sweep::Int
    n_adapt::Int
    seed::UInt64
    hist::Vector{Float64}          # current per-individual log likelihood incl. param priors
    accepted::Dict{Symbol,Float64}
    tried::Dict{Symbol,Float64}
    seconds::Dict{Symbol,Float64}
end

"""
    epidemic_latent_sampler(data::LatentHMMData; n_adapt = 1000, seed = 1) -> sampler

`sampler(rng, model, state)` updates every latent in `state` (a
[`LatentState`](@ref)) once and returns it. Proposal scales adapt during the
first `n_adapt` calls. `acceptance_rates(sampler)` gives the last call's rates.

`model` must hold plain numbers: the sampler is never differentiated.
"""
function epidemic_latent_sampler(data::LatentHMMData; n_adapt::Integer = 1000, seed::Integer = 1)
    N = data.n_individuals
    sc = map(data.latent_decls) do k
        k isa Field ? fill(k.scale, k.n, N) :
        k isa PerIndividual ? fill(k.scale, N) : fill(k.scale, k.n)
    end
    ms = map(data.latent_decls) do k
        (pair = fill(k.scale, k isa Field ? k.n : 0, N), shift = fill(k.scale, N), scale = fill(0.1, N),
         neighbour = fill(k.scale, k isa Field ? k.n : 0, N))
    end
    LatentSampler(data, sc, ms, 0, Int(n_adapt), UInt64(seed), zeros(N),
                  Dict{Symbol,Float64}(), Dict{Symbol,Float64}(), Dict{Symbol,Float64}())
end

"""
    acceptance_rates(sampler) -> Dict{Symbol,Float64}

Acceptance rate per latent from the last call of an [`epidemic_latent_sampler`](@ref).
"""
acceptance_rates(s::LatentSampler) = Dict(k => s.accepted[k] / max(s.tried[k], 1) for k in keys(s.tried))

@inline _adapt(s::LatentSampler, x, acc) =
    s.sweep <= s.n_adapt ? clamp(x * exp(s.sweep^-0.6 * (acc - 0.35)), 1e-6, 1e6) : x

_cache_latents(data::LatentHMMData) =
    (data.cache_decl === nothing || !haskey(data.cache_decl, :latents)) ? keys(data.latent_decls) :
    data.cache_decl.latents

function (s::LatentSampler)(rng::AbstractRNG, model0, state::LatentState)
    data = s.data
    s.sweep += 1
    data.current[] = state
    model = _with_per_eval(model0, data)
    for k in keys(s.tried); s.tried[k] = 0.0; s.accepted[k] = 0.0; end
    c = data.cache_decl === nothing ? nothing : _ensure_cache!(data, model, state; weights = true)
    src = c === nothing ? _Fresh() : _FromWeights(c)
    s.seconds[:start] = get(s.seconds, :start, 0.0) + @elapsed begin
        Threads.@threads for i in 1:data.n_individuals
            s.hist[i] = _ind_loglik(src, data, model, i) + _ind_param_priors(data, model, i)
        end
    end
    s.seconds[:shared] = get(s.seconds, :shared, 0.0) + @elapsed _update_shared!(rng, s, model, state, c)
    # per latent: one-element steps, the :pair, :shift, :scale and :neighbour moves, then pgas (changed, tried)
    tallies = [zeros(12 * length(data.latent_decls)) for _ in 1:data.n_individuals]
    s.seconds[:individual] = get(s.seconds, :individual, 0.0) + @elapsed begin
        Threads.@threads for i in 1:data.n_individuals
            r = Random.Xoshiro(hash((s.seed, s.sweep, i)))
            _update_individual!(r, s, model, state, c, i, tallies[i])
        end
    end
    tl = reduce(+, tallies)
    for (n, name) in enumerate(keys(data.latent_decls))
        data.latent_decls[name] isa Shared && continue
        b = 12(n - 1)
        s.accepted[name] = tl[b+1]; s.tried[name] = tl[b+2]
        for (j, mv) in enumerate((_FIELD_MOVES..., :pgas))
            tl[b+2j+2] > 0 || continue
            s.accepted[Symbol(name, :_, mv)] = tl[b+2j+1]; s.tried[Symbol(name, :_, mv)] = tl[b+2j+2]
        end
    end
    restamp!(state)
    c === nothing || (c.stamp = state.stamp)
    state
end

# ---- shared latents ----

function _update_shared!(rng, s::LatentSampler, model, state, c)
    data = s.data
    for (name, k) in pairs(data.latent_decls)
        k isa Shared || continue
        X = state.values[name]
        acc = 0; tried = 0
        for j in 1:k.n
            old = X[:, j]
            new = old .+ s.scales[name][j] .* randn(rng, k.dim)
            lpn = _vec_logprior(k.prior, new)
            a = false
            if isfinite(lpn)
                X[:, j] .= new
                newhist, patch = _shared_trial(s, k, model, c, j, old, new)
                if log(rand(rng)) < sum(newhist) - sum(s.hist) + lpn - _vec_logprior(k.prior, old)
                    a = true; s.hist .= newhist
                    c === nothing || _apply_patch!(c, data, model, patch)
                else
                    X[:, j] .= old
                end
            end
            acc += a; tried += 1
            s.scales[name][j] = _adapt(s, s.scales[name][j], a)
        end
        s.accepted[name] = acc; s.tried[name] = tried
    end
end

# Every individual's log likelihood with element j moved (already written into
# the state). With a cache, only the shared values at touches(j) are redone.
function _shared_trial(s::LatentSampler, k::Shared, model, c, j, old, new)
    data = s.data
    N = data.n_individuals
    newhist = similar(s.hist)
    if c === nothing
        Threads.@threads for i in 1:N
            newhist[i] = _ind_loglik(_Fresh(), data, model, i) + _ind_param_priors(data, model, i)
        end
        return newhist, nothing
    end
    keyset = k.touches === nothing ? (1:data.n_keys) : k.touches(j)
    newsh = similar(c.shared)
    use = falses(data.n_keys, N)
    changed = falses(N)
    Threads.@threads for i in 1:N
        for key in keyset
            oldsh = c.shared[key, i]
            v = k.update === nothing ? _obs_shared(data, model, i, key) :
                k.update(oldsh, model, data, i, key, j, old, new)
            (k.observation_only && v === oldsh) && continue
            newsh[key, i] = v; use[key, i] = true; changed[i] = true
        end
        if changed[i] || !k.observation_only
            newhist[i] = _ind_loglik(_Patched(c, newsh, use), data, model, i) + _ind_param_priors(data, model, i)
        else
            newhist[i] = s.hist[i]
        end
    end
    newhist, (newsh, use)
end

# Cached values, except the shared values flagged in `use`.
struct _Patched{C,SV}
    c::C
    newsh::Matrix{SV}
    use::BitMatrix
end
@inline _shared_at(p::_Patched, data, model, i, key) = @inbounds p.use[key, i] ? p.newsh[key, i] : p.c.shared[key, i]
@inline function _weights_at(p::_Patched, data, model, i, t, sh)
    @inbounds p.use[data.shared_key(t), i] ? _obsw(data.observation(model, data, i, t, sh)) : p.c.weights[t, i]
end

function _apply_patch!(c::LatentCache, data, model, patch)
    newsh, use = patch
    Threads.@threads for i in 1:data.n_individuals
        any(view(use, :, i)) || continue
        lo, hi = data.sampling_period[i]
        for key in 1:data.n_keys
            use[key, i] && (c.shared[key, i] = newsh[key, i])
        end
        for t in lo:hi
            use[data.shared_key(t), i] && (c.weights[t, i] = _store(data, data.observation(model, data, i, t, c.shared[data.shared_key(t), i])))
        end
    end
end

# ---- one individual ----

function _update_individual!(rng, s::LatentSampler, model, state, c, i, tally)
    data = s.data
    n = 0
    for (name, k) in pairs(data.latent_decls)
        n += 1
        b = 12(n - 1)
        if k isa PerIndividual
            _is_free(k, i) || continue
            a = _move_per_individual!(rng, s, name, k, state.values[name], s.scales[name], model, c, i)
            tally[b+1] += a; tally[b+2] += 1
        elseif k isa Field
            a, t = _sweep_field!(rng, s, k, state.values[name], s.scales[name], model, c, i)
            tally[b+1] += a; tally[b+2] += t
            isempty(k.moves) || _field_moves!(rng, s, name, k, state.values[name], s.mscales[name], model, c, i,
                                              view(tally, (b+3):(b+10)))
            if k.pgas !== nothing
                tally[b+11] += _pgas_field!(rng, s, name, k, state.values[name], model, c, i); tally[b+12] += 1
            end
        end
    end
end

# Fresh shared values and weights for individual i, whose latents just moved.
struct _OneFresh{C}
    c::C
end
@inline _shared_at(o::_OneFresh, data, model, i, key) = _obs_shared(data, model, i, key)
@inline _weights_at(o::_OneFresh, data, model, i, t, sh) = _obsw(data.observation(model, data, i, t, sh))

function _move_per_individual!(rng, s::LatentSampler, name, k::PerIndividual, v, scales, model, c, i)
    data = s.data
    old = v[i]
    d = _prior_at(k, model)
    sup = _small_support(d)
    new = if d isa Distributions.ContinuousUnivariateDistribution
        old + scales[i] * randn(rng)
    elseif sup !== nothing
        old == first(sup) ? last(sup) : first(sup)
    else
        old + rand(rng, (-3, -2, -1, 1, 2, 3))
    end
    lpn = Distributions.logpdf(d, new)
    (isfinite(lpn) && (k.allowed === nothing || k.allowed(data, i, new))) || return false
    reads_obs = name in _cache_latents(data)
    v[i] = new
    src = (c === nothing || reads_obs) ? _OneFresh(c) : _FromWeights(c)
    nl = _ind_loglik(src, data, model, i) + _ind_param_priors(data, model, i)
    lp_prior = _param_prior(k) ? 0.0 : lpn - Distributions.logpdf(d, old)
    a = log(rand(rng)) < nl - s.hist[i] + lp_prior
    if a
        s.hist[i] = nl
        if c !== nothing && reads_obs
            lo, hi = data.sampling_period[i]
            for key in 1:data.n_keys; c.shared[key, i] = _obs_shared(data, model, i, key); end
            for t in lo:hi; c.weights[t, i] = _store(data, data.observation(model, data, i, t, c.shared[data.shared_key(t), i])); end
        end
    else
        v[i] = old
    end
    d isa Distributions.ContinuousUnivariateDistribution && (scales[i] = _adapt(s, scales[i], a))
    a
end

# Current observation weights at (i, t).
@inline _cur_weights(c::Nothing, data, model, i, t) =
    _obsw(data.observation(model, data, i, t, _obs_shared(data, model, i, data.shared_key(t))))
@inline _cur_weights(c::LatentCache, data, model, i, t) = @inbounds c.weights[t, i]

function _sweep_field!(rng, s::LatentSampler, k::Field, S, scales, model, c, i)
    data = s.data
    lo, hi = data.sampling_period[i]
    lo <= hi || return 0, 0
    K = _nstates(data)
    # backward messages after each timepoint, normalised
    B = Vector{NTuple{K,Float64}}(undef, hi - lo + 2)
    lb = zeros(hi - lo + 2)
    B[end] = ntuple(_ -> 1.0, Val(K))
    for t in hi:-1:(lo + 1)
        b = _wbwd(_wmul(_cur_weights(c, data, model, i, t), B[t - lo + 2]), data, model, i, t)
        B[t - lo + 1], z = _tnorm(b)
        lb[t - lo + 1] = lb[t - lo + 2] + log(z)
    end
    Bat(t) = B[t - lo + 2]; lbat(t) = lb[t - lo + 2]   # message after timepoint t
    # running forward vector after timepoint p
    f = _start(data, model, i); p = lo - 1; lf = 0.0
    acc = 0; tried = 0
    old = zeros(k.dim)
    for e in 1:k.n
        w = k.touches(e)
        l = max(first(w), lo); h = min(last(w), hi)
        l <= h || continue
        while p < l - 1
            p += 1
            f, z = _obs_step(_wfwd(f, data, model, i, p), _cur_weights(c, data, model, i, p))
            lf += log(z)
        end
        keys_e = k.keys === nothing ? unique(data.shared_key(t) for t in l:h) : k.keys(e)
        cur = _window_log(data, model, i, l, h, f, lo, nothing, c, keys_e, Bat(h)) + lf + lbat(h) +
              _vec_logprior(k.prior, view(S, :, e, i))
        for d in 1:k.dim; old[d] = S[d, e, i]; end
        sc = scales[e, i]
        for d in 1:k.dim; S[d, e, i] = old[d] + sc * randn(rng); end
        lpn = _vec_logprior(k.prior, view(S, :, e, i))
        a = false
        if isfinite(lpn)
            newsh = c === nothing || isempty(keys_e) ? nothing :
                    Dict(key => _obs_shared(data, model, i, key) for key in keys_e)
            new = _window_log(data, model, i, l, h, f, lo, newsh, c, keys_e, Bat(h)) + lf + lbat(h) + lpn
            if isfinite(new) && log(rand(rng)) < new - cur
                a = true
                if c !== nothing && newsh !== nothing
                    for (key, v) in newsh; c.shared[key, i] = v; end
                    for t in l:h
                        data.shared_key(t) in keys_e &&
                            (c.weights[t, i] = _store(data, data.observation(model, data, i, t, c.shared[data.shared_key(t), i])))
                    end
                end
            end
        end
        a || (for d in 1:k.dim; S[d, e, i] = old[d]; end)
        acc += a; tried += 1
        scales[e, i] = _adapt(s, sc, a)
    end
    s.hist[i] = (c === nothing ? _ind_loglik(_Fresh(), data, model, i) : _ind_loglik(_FromWeights(c), data, model, i)) +
                _ind_param_priors(data, model, i)
    acc, tried
end

# ---- joint moves of an individual's Field elements ----
#
# After the one-element steps: moves of several elements of individual i at once,
# each a Metropolis step scored by i's full forward pass and the elements' priors.
#   :pair   each two consecutive elements in use, moved by one common vector
#   :shift  every element in use, moved by one common vector
#   :scale  every element in use, pulled towards or pushed from their mean by a
#           factor exp(u), u ~ N(0, s^2); log Jacobian dim * (m - 1) * u for m elements
#   :neighbour  each element in use with a neighbour in use, proposed close to the
#           previous or the next element (one picked at random): N(neighbour, s^2).
#           It moves a jump between consecutive elements to the next step, or
#           removes it, in one move, which small steps cannot. The proposal does
#           not depend on the element's own value, so the ratio carries
#           q(old) / q(new), q the mixture over its neighbours (unchanged by the move).
# The other proposals are symmetric. Scales adapt in warm-up like the one-element steps.
function _field_moves!(rng, s::LatentSampler, name, k::Field, S, ms, model, c, i, tally)
    data = s.data
    lo, hi = data.sampling_period[i]
    E = [e for e in 1:k.n if (w = k.touches(e); max(first(w), lo) <= min(last(w), hi))]
    isempty(E) && return
    reads_obs = name in _cache_latents(data)
    src = (c === nothing || reads_obs) ? _OneFresh(c) : _FromWeights(c)
    lprior(es) = sum(e -> _vec_logprior(k.prior, view(S, :, e, i)), es; init = 0.0)
    old = zeros(k.dim, k.n)
    # propose by `move!` (writing into S), accept or put back; returns whether accepted
    function step!(es, move!, logjac)
        for e in es, d in 1:k.dim; old[d, e] = S[d, e, i]; end
        before = lprior(es)
        move!()
        after = lprior(es)
        if isfinite(after)
            nl = _ind_loglik(src, data, model, i) + _ind_param_priors(data, model, i)
            if isfinite(nl) && log(rand(rng)) < nl - s.hist[i] + after - before + logjac
                s.hist[i] = nl
                if c !== nothing && reads_obs
                    for key in 1:data.n_keys; c.shared[key, i] = _obs_shared(data, model, i, key); end
                    for t in lo:hi
                        c.weights[t, i] = _store(data, data.observation(model, data, i, t, c.shared[data.shared_key(t), i]))
                    end
                end
                return true
            end
        end
        for e in es, d in 1:k.dim; S[d, e, i] = old[d, e]; end
        false
    end
    if :pair in k.moves
        for j in 1:(length(E) - 1)
            e1, e2 = E[j], E[j+1]
            sc = ms.pair[e1, i]
            v = sc .* randn(rng, k.dim)
            a = step!((e1, e2), () -> (for d in 1:k.dim; S[d, e1, i] += v[d]; S[d, e2, i] += v[d]; end), 0.0)
            tally[1] += a; tally[2] += 1
            ms.pair[e1, i] = _adapt(s, sc, a)
        end
    end
    if :shift in k.moves
        v = ms.shift[i] .* randn(rng, k.dim)
        a = step!(E, () -> (for e in E, d in 1:k.dim; S[d, e, i] += v[d]; end), 0.0)
        tally[3] += a; tally[4] += 1
        ms.shift[i] = _adapt(s, ms.shift[i], a)
    end
    if :scale in k.moves && length(E) > 1
        u = ms.scale[i] * randn(rng)
        m = [sum(S[d, e, i] for e in E) / length(E) for d in 1:k.dim]
        a = step!(E, () -> (for e in E, d in 1:k.dim; S[d, e, i] = m[d] + exp(u) * (S[d, e, i] - m[d]); end),
                  k.dim * (length(E) - 1) * u)
        tally[5] += a; tally[6] += 1
        ms.scale[i] = _adapt(s, ms.scale[i], a)
    end
    if :neighbour in k.moves
        inE = falses(k.n); inE[E] .= true
        for e in E
            nb = [x for x in (e - 1, e + 1) if 1 <= x <= k.n && inE[x]]
            isempty(nb) && continue
            sc = ms.neighbour[e, i]
            # log density of the proposal at x: equal mixture of N(S[:, n, i], sc^2 I) over nb
            lq(x) = logsumexp([-sum(abs2, x .- view(S, :, n, i)) / (2sc^2) for n in nb]) - log(length(nb)) -
                    k.dim * log(sc)
            xo = S[:, e, i]
            xn = S[:, rand(rng, nb), i] .+ sc .* randn(rng, k.dim)
            a = step!((e,), () -> (for d in 1:k.dim; S[d, e, i] = xn[d]; end), lq(xo) - lq(xn))
            tally[7] += a; tally[8] += 1
            ms.neighbour[e, i] = _adapt(s, sc, a)
        end
    end
end

# ---- particle Gibbs with ancestor sampling over an individual's Field run ----
#
# One conditional SMC sweep over individual i's elements in use, E[1..m] (an
# `index` Field). Step j proposes element E[j] and runs the forward pass over
# the timepoints from the first of E[j]'s window up to the one before E[j+1]'s:
# those read only E[j] and E[j-1]. Each particle carries its path and its
# normalised forward vector, so the hidden chain is summed out exactly
# (Rao-Blackwellised). The current path is the reference particle. Its
# ancestor at each step is drawn with weights that include the rest of the
# reference path, through backward messages computed once along it (ancestor
# sampling, Lindsten, Jordan and Schön 2014). Proposal: an equal mixture of
# normals around the ancestor's previous element (one per scale), a normal
# around the anchor (if any), and the prior. Returns whether the path changed.
_prior_draw(rng, prior::Tuple, D) = [rand(rng, prior[d]) for d in 1:D]
_prior_draw(rng, prior, D) = [rand(rng, prior) for _ in 1:D]

function _pgas_field!(rng, s::LatentSampler, name, k::Field, S, model, c, i)
    data = s.data
    lo, hi = data.sampling_period[i]
    lo <= hi || return false
    E = [e for e in 1:k.n if (w = k.touches(e); max(first(w), lo) <= min(last(w), hi))]
    m = length(E)
    m == 0 && return false
    pg = k.pgas; P = pg.particles; D = k.dim; K = _nstates(data)
    cuts = [j == m ? hi : first(k.touches(E[j+1])) - 1 for j in 1:m]
    starts = [j == 1 ? lo : cuts[j-1] + 1 for j in 1:m]
    ref = S[:, E, i]
    # backward messages along the reference: B[t - lo + 2] is the message after t
    B = Vector{NTuple{K,Float64}}(undef, hi - lo + 2)
    B[end] = ntuple(_ -> 1.0, Val(K))
    for t in hi:-1:(lo + 1)
        b = _wbwd(_wmul(_cur_weights(nothing, data, model, i, t), B[t - lo + 2]), data, model, i, t)
        B[t - lo + 1], _ = _tnorm(b)
    end
    # the forward pass over step j with E[j-1] = prev and E[j] = cur, from the
    # normalised vector f; returns the normalised vector and the log increment
    function window(f, j, prev, cur)
        j > 1 && (for d in 1:D; S[d, E[j-1], i] = prev[d]; end)
        for d in 1:D; S[d, E[j], i] = cur[d]; end
        a = j == 1 ? _start(data, model, i) : f
        ll = 0.0
        key = -1; sh = nothing
        for t in starts[j]:cuts[j]
            kt = data.shared_key(t)
            kt == key || (key = kt; sh = _obs_shared(data, model, i, kt))
            a, z = _obs_step(_wfwd(a, data, model, i, t), _obsw(data.observation(model, data, i, t, sh)))
            z > 0 || return a, -Inf
            isone(z) || (ll += log(z))
        end
        sa = sum(a)
        sa > 0 || return a, -Inf
        map(x -> x / sa, a), ll + log(sa)
    end
    lgauss(x, mu, sd) = -sum(abs2, x .- mu) / (2sd^2) - D * log(sd) - D / 2 * log(2pi)
    function comps(j, prev)
        cs = Tuple{Vector{Float64},Float64}[]
        j > 1 && (for sd in pg.scales; push!(cs, (prev, sd)); end)
        if pg.anchors !== nothing
            an = pg.anchors[:, E[j], i]
            all(isfinite, an) && push!(cs, (an, pg.anchor_scale))
        end
        cs
    end
    function propose(j, prev)
        cs = comps(j, prev); r = rand(rng, 1:(length(cs) + 1))
        r > length(cs) ? _prior_draw(rng, k.prior, D) : cs[r][1] .+ cs[r][2] .* randn(rng, D)
    end
    function logq(j, prev, x)
        cs = comps(j, prev)
        logsumexp([[lgauss(x, mu, sd) for (mu, sd) in cs]; _vec_logprior(k.prior, x)]) - log(length(cs) + 1)
    end
    probs(lw) = (w = exp.(lw .- maximum(lw)); w ./ sum(w))
    X = zeros(D, m, P); Xn = similar(X)
    F = Vector{NTuple{K,Float64}}(undef, P); Fn = similar(F)
    lw = zeros(P); lwn = similar(lw)
    ok = true
    for j in 1:m
        if j == 1
            for p in 1:P
                x = p == P ? ref[:, 1] : propose(1, nothing)
                X[:, 1, p] = x
                lp = _vec_logprior(k.prior, x)
                if isfinite(lp)
                    F[p], ll = window(nothing, 1, nothing, x)
                    lw[p] = ll + lp - logq(1, nothing, x)
                else
                    F[p] = ntuple(_ -> 0.0, Val(K)); lw[p] = -Inf
                end
            end
        else
            isfinite(maximum(lw)) || (ok = false; break)
            w = probs(lw)
            anc = [_sample_categorical(rng, w) for _ in 1:P]
            xs = ref[:, j]
            las = fill(-Inf, P)
            for q in 1:P
                isfinite(lw[q]) || continue
                a, ll = window(F[q], j, X[:, j-1, q], xs)
                isfinite(ll) || continue
                d = sum(map(*, a, B[cuts[j] - lo + 2]))
                d > 0 && (las[q] = lw[q] + ll + log(d))
            end
            isfinite(maximum(las)) || (ok = false; break)
            anc[P] = _sample_categorical(rng, probs(las))
            for p in 1:P
                a = anc[p]; prev = X[:, j-1, a]
                x = p == P ? xs : propose(j, prev)
                for jj in 1:(j-1), d in 1:D; Xn[d, jj, p] = X[d, jj, a]; end
                Xn[:, j, p] = x
                lp = _vec_logprior(k.prior, x)
                if isfinite(lp)
                    Fn[p], ll = window(F[a], j, prev, x)
                    lwn[p] = ll + lp - logq(j, prev, x)
                else
                    Fn[p] = F[a]; lwn[p] = -Inf
                end
            end
            X, Xn = Xn, X; F, Fn = Fn, F; lw, lwn = lwn, lw
        end
    end
    if !ok || !isfinite(maximum(lw))
        S[:, E, i] = ref
        return false
    end
    b = _sample_categorical(rng, probs(lw))
    S[:, E, i] = X[:, :, b]
    reads_obs = name in _cache_latents(data)
    src = (c === nothing || reads_obs) ? _OneFresh(c) : _FromWeights(c)
    s.hist[i] = _ind_loglik(src, data, model, i) + _ind_param_priors(data, model, i)
    if c !== nothing && reads_obs
        for key in 1:data.n_keys; c.shared[key, i] = _obs_shared(data, model, i, key); end
        for t in lo:hi
            c.weights[t, i] = _store(data, data.observation(model, data, i, t, c.shared[data.shared_key(t), i]))
        end
    end
    X[:, :, b] != ref
end

# log of f * (steps l..h) * B, f being the forward vector after l-1 (the start
# when l == lo, recomputed in case it reads the latent).
function _window_log(data, model, i, l, h, f, lo, newsh, c, keys_e, Bh)
    a = l == lo ? _start(data, model, i) : f
    ll = 0.0
    for t in l:h
        key = data.shared_key(t)
        w = if newsh !== nothing && key in keys_e
            _obsw(data.observation(model, data, i, t, newsh[key]))
        elseif c === nothing
            _obsw(data.observation(model, data, i, t, _obs_shared(data, model, i, key)))
        else
            c.weights[t, i]
        end
        a, z = _obs_step(_wfwd(a, data, model, i, t), w)
        z > 0 || return -Inf
        ll += log(z)
    end
    dv = sum(map(*, a, Bh))
    dv > 0 ? ll + log(dv) : -Inf
end

"""
    check_local_messages(data::LatentHMMData, model, state; field) -> Float64

Largest difference, over individuals and elements of `field`, between the
windowed score the sampler uses and the full forward pass. Should be rounding
error.
"""
function check_local_messages(data::LatentHMMData, model0, state::LatentState; field::Symbol)
    data.current[] = state
    model = _with_per_eval(model0, data)
    k = data.latent_decls[field]
    worst = 0.0
    for i in 1:data.n_individuals
        lo, hi = data.sampling_period[i]
        lo <= hi || continue
        full = _ind_loglik(_Fresh(), data, model, i)
        K = _nstates(data)
        B = Vector{NTuple{K,Float64}}(undef, hi - lo + 2); lb = zeros(hi - lo + 2)
        B[end] = ntuple(_ -> 1.0, Val(K))
        for t in hi:-1:(lo + 1)
            b = _wbwd(_wmul(_cur_weights(nothing, data, model, i, t), B[t - lo + 2]), data, model, i, t)
            B[t - lo + 1], z = _tnorm(b); lb[t - lo + 1] = lb[t - lo + 2] + log(z)
        end
        f = _start(data, model, i); p = lo - 1; lf = 0.0
        for e in 1:k.n
            w = k.touches(e); l = max(first(w), lo); h = min(last(w), hi)
            l <= h || continue
            while p < l - 1
                p += 1
                f, z = _obs_step(_wfwd(f, data, model, i, p), _cur_weights(nothing, data, model, i, p)); lf += log(z)
            end
            loc = _window_log(data, model, i, l, h, f, lo, nothing, nothing, (), B[h - lo + 2]) + lf + lb[h - lo + 2]
            worst = max(worst, abs(loc - full))
        end
    end
    worst
end

@inline _wmul(w, b) = w === nothing ? b : map(*, w, b)
