# Independent individuals, each with a finite hidden chain whose transitions are
# weights (`@transitions ... :weights`), plus declared latents (latents.jl) that
# the weights and observations may read. The chain is summed out exactly; the
# latents are sampled.
#
# Timing: the chain starts in `starting_state` before the first timepoint. At
# each timepoint t it takes the weighted step INTO t, then the observation at t.
# (EpidemicData's rates move OUT of t; here the step into the first timepoint is
# an ordinary step, which is what entry needs.)

"""
    LatentHMMData

Built by [`latent_hmm_data`](@ref). Extras are read as `data.name`; the current
latent values as `data.latents.name`.
"""
struct LatentHMMData{K,RF,SH,VF,VT,ST,OB,OS,KF,LD,LS,PE,CD,EX<:NamedTuple}
    n_individuals::Int
    n_timepoints::Int
    n_states::Int
    state_space::Vector{Symbol}
    sampling_period::Vector{Tuple{Int,Int}}
    trans_mat::WeightSpec{RF,SH}
    from::Vector{Int}
    to::Vector{Int}
    vfrom::VF                    # the same codes as Vals, so each edge compiles to a fixed slot
    vto::VT
    starting_state::ST
    observation::OB              # (model, data, i, t, shared) -> NTuple{K}
    observation_shared::OS       # (model, data, i, key) -> value, or nothing
    shared_key::KF               # t -> key
    n_keys::Int
    latent_decls::LD
    current::Base.RefValue{LS}
    per_evaluation::PE           # (model, data) -> NamedTuple merged into model, or nothing
    cache_decl::CD               # (; shared = names, weights = names), or nothing
    cache::Base.RefValue{Any}
    extras::EX
end
# `data.name` reads a field, the current latents (`data.latents`) or an extra,
# decided at compile time as for EpidemicData.
@inline Base.getproperty(d::LatentHMMData, s::Symbol) = _lh_field(d, Val(s))
for f in fieldnames(LatentHMMData)
    @eval @inline _lh_field(d::LatentHMMData, ::Val{$(QuoteNode(f))}) = getfield(d, $(QuoteNode(f)))
end
@inline _lh_field(d::LatentHMMData, ::Val{:latents}) = getfield(d, :current)[].values
@inline function _lh_field(d::LatentHMMData, ::Val{s}) where {s}
    extras = getfield(d, :extras)
    hasproperty(extras, s) ||
        throw(ArgumentError("LatentHMMData has no field or extra `$s`; pass it to `latent_hmm_data` as `$s = ...`"))
    getproperty(extras, s)
end
Base.propertynames(d::LatentHMMData) = (fieldnames(LatentHMMData)..., :latents, propertynames(getfield(d, :extras))...)

# A scalar observation weight, seen as the whole tuple.
struct _ScalarObs{K,W}
    w::W
end
@inline (o::_ScalarObs{K})(model, data, i, t, sh) where {K} = ntuple(s -> o.w(model, data, i, t, s, sh), Val(K))

"""
    latent_hmm_data(; n_individuals, n_timepoints, trans_mat, starting_state, latents, init,
                      observation_weights = nothing, observation_weight = nothing,
                      observation_shared = nothing, shared_key = identity, n_keys = n_timepoints,
                      sampling_period = nothing, per_evaluation = nothing, cache = nothing,
                      state_space = trans_mat.states, extras...)

A model of independent individuals whose hidden chain has weighted transitions
and reads latent variables.

- `trans_mat`: a [`WeightSpec`](@ref), from `@transitions ... :weights`. The weight
  at `t` is for the step into `t`. A `@shared` function that returns `nothing` at `t`
  means nothing moves at `t`: every state keeps its weight and the step is skipped.
- `starting_state(model, data, i)`: weights over the states before the first step.
- `observation_weights(model, data, i, t, shared)`: the weight of each state's
  observations at `t`, as a tuple. Return `nothing` for a timepoint with no
  observation. Or `observation_weight(model, data, i, t, s, shared)`
  for one state at a time.
- `observation_shared(model, data, i, key)`: anything the observation weights at
  every `t` with `shared_key(t) == key` need, computed once per key. It arrives
  as `shared`. Keys run from 1 to `n_keys`.
- `latents`: an [`@latents`](@ref) declaration; `init`: starting values as a
  `NamedTuple`; any left out are drawn as [`latent_state`](@ref) describes.
- `per_evaluation(model, data)`: values worth computing once per likelihood call
  rather than once per step, e.g. rates tabulated by age group. Returned as a
  `NamedTuple` and merged into `model`.
- `cache = (; shared = (:a, :b), weights = (:c,), latents = (:x,))`: names the
  parameters `observation_shared` reads (`shared`), the further parameters the
  observation weights read (`weights`), and the latents `observation_shared`
  reads (`latents`; all of them if left out). With it, values that cannot have changed
  are reused: an HMC block that moves none of `shared` reuses the shared values,
  and one that moves none of either reuses the weights.
- `sampling_period`: `(first, last)` timepoint per individual.
- `extras...`: anything else, read as `data.name`.
"""
function latent_hmm_data(; n_individuals, n_timepoints, trans_mat::WeightSpec, starting_state,
                         latents::NamedTuple, init::NamedTuple = (;),
                         observation_weights = nothing, observation_weight = nothing,
                         observation_shared = nothing, shared_key = identity, n_keys = n_timepoints,
                         sampling_period = nothing, per_evaluation = nothing, cache = nothing,
                         state_space = trans_mat.states, extras...)
    states = collect(Symbol, state_space)
    K = length(states)
    idx(s) = (k = findfirst(==(s), states); k === nothing && error("state $s is not in the state space"); k)
    from = [idx(a) for (a, _) in trans_mat.transitions]
    to = [idx(b) for (_, b) in trans_mat.transitions]
    obs = if observation_weights !== nothing
        observation_weights
    elseif observation_weight !== nothing
        _ScalarObs{K,typeof(observation_weight)}(observation_weight)
    else
        (model, data, i, t, sh) -> ntuple(_ -> 1.0, Val(K))
    end
    sp = sampling_period === nothing ? fill((1, n_timepoints), n_individuals) :
         [(Int(a), Int(b)) for (a, b) in sampling_period]
    length(sp) == n_individuals || throw(DimensionMismatch("sampling_period has $(length(sp)) entries"))
    latents = map(k -> _resolve(k, n_timepoints), latents)
    for (name, k) in pairs(latents)
        k isa Field || continue
        w = filter(!isempty, [k.touches(e) for e in 1:k.n])
        all(e -> first(w[e]) <= first(w[e+1]) && last(w[e]) <= last(w[e+1]), 1:(length(w) - 1)) ||
            throw(ArgumentError("Field `$name`: touches(e) must not decrease with e"))
    end
    state = latent_state(latents, n_individuals; init...)
    vfrom = Tuple(Val(a) for a in from); vto = Tuple(Val(b) for b in to)
    LatentHMMData{K,typeof(trans_mat.rate_fns),typeof(trans_mat.shared),typeof(vfrom),typeof(vto),typeof(starting_state),
                  typeof(obs),typeof(observation_shared),typeof(shared_key),typeof(latents),
                  typeof(state),typeof(per_evaluation),typeof(cache),typeof(NamedTuple(extras))}(
        n_individuals, n_timepoints, K, states, sp, trans_mat, from, to, vfrom, vto, starting_state, obs,
        observation_shared, shared_key, n_keys, latents, Ref(state), per_evaluation, cache,
        Ref{Any}(nothing), NamedTuple(extras))
end

"""
    current_state(data::LatentHMMData) -> LatentState

The latent values the data object currently holds: the starting values, or
whatever was last handed to the likelihood or the sampler.
"""
current_state(data::LatentHMMData) = data.current[]

@inline _with_per_eval(model, data::LatentHMMData) =
    data.per_evaluation === nothing ? model : merge(model, data.per_evaluation(model, data))

# ---- one step ----

# alpha' = alpha W_t, along the declared edges.
@inline function _wfwd(alpha::NTuple{K}, data::LatentHMMData, model, i, t) where {K}
    tm = data.trans_mat
    sh = _step_shared(tm.shared, model, data, i, t)
    sh === nothing && return alpha          # nothing moves at t
    z = zero(promote_type(eltype(alpha), _param_eltype(model)))
    _wfwd(alpha, ntuple(_ -> z, Val(K)), tm.rate_fns, data.vfrom, data.vto, sh, model, data, i, t)
end
@inline _wfwd(alpha, acc, ::Tuple{}, ::Tuple{}, ::Tuple{}, sh, model, data, i, t) = acc
@inline function _wfwd(alpha, acc, rates::Tuple, fr::Tuple, to::Tuple, sh, model, data, i, t)
    w = _rate(first(rates), sh, model, data, i, t)
    acc = _addat(acc, first(to), alpha[_val(first(fr))] * w)
    _wfwd(alpha, acc, Base.tail(rates), Base.tail(fr), Base.tail(to), sh, model, data, i, t)
end

@inline _val(::Val{a}) where {a} = a
@inline _addat(acc::NTuple{K}, ::Val{b}, x) where {K,b} = ntuple(j -> j == b ? acc[j] + x : acc[j], Val(K))

# v' = W_t v.
@inline function _wbwd(v::NTuple{K}, data::LatentHMMData, model, i, t) where {K}
    tm = data.trans_mat
    sh = _step_shared(tm.shared, model, data, i, t)
    sh === nothing && return v
    z = zero(promote_type(eltype(v), _param_eltype(model)))
    _wbwd(v, ntuple(_ -> z, Val(K)), tm.rate_fns, data.vfrom, data.vto, sh, model, data, i, t)
end
@inline _wbwd(v, acc, ::Tuple{}, ::Tuple{}, ::Tuple{}, sh, model, data, i, t) = acc
@inline function _wbwd(v, acc, rates::Tuple, fr::Tuple, to::Tuple, sh, model, data, i, t)
    w = _rate(first(rates), sh, model, data, i, t)
    acc = _addat(acc, first(fr), w * v[_val(first(to))])
    _wbwd(v, acc, Base.tail(rates), Base.tail(fr), Base.tail(to), sh, model, data, i, t)
end

@inline function _tnorm(a::NTuple)
    s = sum(a)
    r = inv(s)
    map(x -> x * r, a), s
end

_start(data::LatentHMMData, model, i) = promote(data.starting_state(model, data, i)...)
_conv(::Type{T}, a::Tuple) where {T} = map(x -> convert(T, x), a)

# ---- where the observation weights come from ----
# Fresh: compute everything. Shared: cached shared values, fresh weights.
# Cached: cached weights. The cache holds plain Float64 values only.

mutable struct LatentCache{SV,K}
    stamp::UInt
    skey::Vector{Float64}
    wkey::Vector{Float64}
    shared::Matrix{SV}                   # n_keys x N
    weights::Matrix{NTuple{K,Float64}}   # T x N
    weights_ok::Bool
end

struct _Fresh end
struct _FromShared{C}; c::C; end
struct _FromWeights{C}; c::C; end

@inline _obs_shared(data::LatentHMMData, model, i, key) =
    data.observation_shared === nothing ? nothing : data.observation_shared(model, data, i, key)

@inline _shared_at(::_Fresh, data, model, i, key) = _obs_shared(data, model, i, key)
@inline _shared_at(src::_FromShared, data, model, i, key) = @inbounds src.c.shared[key, i]

@inline _weights_at(src::Union{_Fresh,_FromShared}, data, model, i, t, sh) =
    _obsw(data.observation(model, data, i, t, sh))
@inline _weights_at(src::_FromWeights, data, model, i, t, sh) = @inbounds src.c.weights[t, i]

# ---- one individual's marginal log likelihood ----

@inline _obsw(w) = w === nothing ? nothing : promote(w...)
_store(::LatentHMMData{K}, w) where {K} = w === nothing ? ntuple(_ -> 1.0, Val(K)) : NTuple{K,Float64}(w)

# Weight by the observation (`nothing`: no observation at t), rescaling only when
# the total drifts far from 1. Returns the vector and the factor taken out.
#
# The factor is the total's plain value (`_strip`), not the total itself. That
# is exact: the factor goes into the log likelihood as a constant and the
# derivative stays in the vector. Rescaling by a dual-number total instead
# breaks when the total is below ~1e-154, as a capture likelihood can be:
# the derivative of `inv(s)` is -1/s^2, which overflows, and every entry of
# the gradient becomes NaN while the log likelihood stays finite.
@inline function _obs_step(a, w)
    a = w === nothing ? a : map(*, a, w)
    s = _strip(sum(a))
    (1e-100 < s < 1e100) && return a, one(s)
    s > 0 || return a, s
    r = inv(s)
    map(x -> x * r, a), s
end

function _ind_loglik(src::_FromWeights, data::LatentHMMData, model, i)
    lo, hi = data.sampling_period[i]
    T = _param_eltype(model)
    lo <= hi || return zero(T)
    a = _conv(T, _start(data, model, i))
    ll = zero(T)
    for t in lo:hi
        a, z = _obs_step(_wfwd(a, data, model, i, t), _weights_at(src, data, model, i, t, nothing))
        z > 0 || return T(-Inf)
        isone(z) || (ll += log(z))
    end
    s = sum(a)
    s > 0 ? ll + log(s) : T(-Inf)
end

function _ind_loglik(src, data::LatentHMMData, model, i)
    lo, hi = data.sampling_period[i]
    T = _param_eltype(model)
    lo <= hi || return zero(T)
    a = _conv(T, _start(data, model, i))
    ll = zero(T)
    key = data.shared_key(lo)
    sh = _shared_at(src, data, model, i, key)
    for t in lo:hi
        k = data.shared_key(t)
        k == key || (key = k; sh = _shared_at(src, data, model, i, key))
        a, z = _obs_step(_wfwd(a, data, model, i, t), _weights_at(src, data, model, i, t, sh))
        z > 0 || return T(-Inf)
        isone(z) || (ll += log(z))
    end
    s = sum(a)
    s > 0 ? ll + log(s) : T(-Inf)
end

# Priors that depend on the parameters belong to the likelihood.
@inline function _ind_param_priors(data::LatentHMMData, model, i)
    _sum_pprior(values(data.latent_decls), values(data.latents), model, i, _param_eltype(model))
end
@inline _sum_pprior(ks, vs, model, i, ::Type{T}) where {T} =
    reduce(+, map((k, v) -> _pprior(k, v, model, i, T), ks, vs); init = zero(T))
@inline _pprior(k, v, model, i, ::Type{T}) where {T} = zero(T)
@inline _pprior(k::PerIndividual{<:Function}, v, model, i, ::Type{T}) where {T} =
    convert(T, Distributions.logpdf(k.prior(model), v[i]))

# ---- caches ----

_isplain(x::Float64) = true
_isplain(x::Integer) = true
_isplain(x::AbstractArray) = eltype(x) <: Union{Float64,Integer}
_isplain(x) = false
_allplain(model, names) = all(n -> _isplain(model[n]), names)

# The ForwardDiff extension adds a method for Duals.
_strip(x) = x
_strip(x::AbstractArray) = map(_strip, x)
_strip(x::Tuple) = map(_strip, x)
_strip(x::NamedTuple) = map(_strip, x)
_strip_model(model::NamedTuple) = map(_strip, model)

_cnames(data, f) = haskey(data.cache_decl, f) ? data.cache_decl[f] : ()
_keyvals(model, names) = Float64[v for n in names for v in (model[n] isa AbstractArray ? model[n] : (model[n],))]

function _refresh_shared!(data::LatentHMMData{K}, plain, state) where {K}
    N = data.n_individuals
    s1 = _obs_shared(data, plain, 1, data.shared_key(data.sampling_period[1][1]))
    shared = Matrix{typeof(s1)}(undef, data.n_keys, N)
    Threads.@threads for i in 1:N
        for key in 1:data.n_keys
            shared[key, i] = _obs_shared(data, plain, i, key)
        end
    end
    c = data.cache[]
    weights = (c isa LatentCache && size(c.weights) == (data.n_timepoints, N)) ? c.weights :
              Matrix{NTuple{K,Float64}}(undef, data.n_timepoints, N)
    data.cache[] = LatentCache{typeof(s1),K}(state.stamp, _keyvals(plain, _cnames(data, :shared)),
                                             Float64[], shared, weights, false)
    nothing
end

function _refresh_weights!(c::LatentCache, data::LatentHMMData, plain)
    Threads.@threads for i in 1:data.n_individuals
        lo, hi = data.sampling_period[i]
        for t in lo:hi
            c.weights[t, i] = _store(data, data.observation(plain, data, i, t, c.shared[data.shared_key(t), i]))
        end
    end
    c.wkey = _keyvals(plain, _cnames(data, :weights))
    c.weights_ok = true
    nothing
end

# Bring the cache up to date with `state` and the plain parameter values.
function _ensure_cache!(data::LatentHMMData, plain, state; weights::Bool)
    c = data.cache[]
    if !(c isa LatentCache) || c.stamp != state.stamp || c.skey != _keyvals(plain, _cnames(data, :shared))
        _refresh_shared!(data, plain, state)
        c = data.cache[]
    end
    if weights && (!c.weights_ok || c.wkey != _keyvals(plain, _cnames(data, :weights)))
        _refresh_weights!(c, data, plain)
    end
    c
end

# ---- the marginal likelihood ----

function _chunk_loglik(src, acc, data::LatentHMMData, model, c, nchunk)
    for i in c:nchunk:data.n_individuals
        acc += _ind_loglik(src, data, model, i) + _ind_param_priors(data, model, i)
    end
    acc
end

function _total_loglik(src, data::LatentHMMData, model)
    nchunk = 4 * Threads.nthreads()
    z = zero(_param_eltype(model))
    tasks = [Threads.@spawn(_chunk_loglik(src, z, data, model, c, nchunk)) for c in 1:nchunk]
    sum(fetch.(tasks))
end

function _source(data::LatentHMMData, model, state)
    cd = data.cache_decl
    cd === nothing && return _Fresh()
    _allplain(model, _cnames(data, :shared)) || return _Fresh()
    plain = _strip_model(model)
    if _allplain(model, _cnames(data, :weights))
        return _FromWeights(_ensure_cache!(data, plain, state; weights = true))
    end
    _FromShared(_ensure_cache!(data, plain, state; weights = false))
end

"""
    epidemic_marginal_loglik(data::LatentHMMData) -> mll

`mll(model, data, state)`: the log probability of the observations with every
individual's hidden chain summed out, given the latent values in `state` (a
[`LatentState`](@ref)), plus the latent priors that depend on the parameters.
The HMC target for a [`LatentHMMData`](@ref) model.

Threaded over individuals and differentiable in `model` (forward mode). With
`cache` declared in [`latent_hmm_data`](@ref), observation values that the
differentiated parameters cannot change are reused.
"""
function epidemic_marginal_loglik(data::LatentHMMData)
    function marginal_loglik(model0, data::LatentHMMData, state::LatentState)
        _allfinite(model0) || return _param_eltype(model0)(-Inf)    # a diverged HMC proposal
        data.current[] = state
        model = _with_per_eval(model0, data)
        _total_loglik(_source(data, model, state), data, model)
    end
    marginal_loglik
end

"""
    hmm_logliks(model, data::LatentHMMData, state) -> Vector{Float64}

Each individual's marginal log likelihood, without the latent priors.
"""
function hmm_logliks(model0, data::LatentHMMData, state::LatentState)
    data.current[] = state
    model = _with_per_eval(model0, data)
    [_ind_loglik(_Fresh(), data, model, i) for i in 1:data.n_individuals]
end

_nstates(::LatentHMMData{K}) where {K} = K

"""
    @weights(model; name = value, ...)
    @weights(model, values...)

What a `@shared` step function (the first form) or `observation_weights` (the
second) returns, with every value converted to the parameters' number type.
Plain numbers such as `0` or `1` can then sit beside values computed from the
parameters, and every branch returns the same type, which Julia needs to
compile the function well.

```julia
iseven(t) && return @weights(model; stay = 1, die = 1 - phi, live = phi)
@weights(model, 0, L, L)
```
"""
macro weights(args...)
    T = gensym(:T)
    conv(x) = :(convert($T, $(esc(x))))
    if !isempty(args) && args[1] isa Expr && args[1].head == :parameters
        length(args) == 2 || error("@weights(model; name = value, ...)")
        model = args[2]
        fields = map(args[1].args) do kw
            (kw isa Expr && kw.head == :kw) || error("@weights: expected name = value, got $kw")
            Expr(:kw, kw.args[1], conv(kw.args[2]))
        end
        body = Expr(:tuple, Expr(:parameters, fields...))
    else
        model = args[1]
        body = Expr(:tuple, map(conv, args[2:end])...)
    end
    quote
        $T = $(GlobalRef(@__MODULE__, :_param_eltype))($(esc(model)))
        $body
    end
end

_allfinite(x::Real) = isfinite(x)
_allfinite(x::AbstractArray) = all(_allfinite, x)
_allfinite(x::Union{Tuple,NamedTuple}) = all(_allfinite, x)
_allfinite(x) = true
