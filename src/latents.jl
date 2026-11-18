# Latent variables a weighted HMM reads but does not sum out. The hidden chain is
# summed exactly by `epidemic_marginal_loglik`; these are sampled, one Metropolis
# move at a time, by `epidemic_latent_sampler`.
#
# Three kinds, by who they belong to:
#   Field          a vector per individual at each of n elements (a position per year)
#   PerIndividual  one value per individual (a random effect, a birth year, sex)
#   Shared         vectors common to every individual (an unmapped detector)

abstract type LatentKind end

"""
    Field(dim, n; prior, index = nothing, touches = nothing, keys = nothing, scale = 1.0)

A `dim`-vector for each individual at each of `n` elements, e.g. an activity
centre per year.

- `prior`: a univariate distribution used for every coordinate, or a tuple of
  them, one per coordinate.
- `index(t)`: the element in use at timepoint `t`, e.g. the year. The weights
  and observations at `t` may read elements `index(t)` and `index(t) - 1` (a
  move from last year's position). The sampler works out which timepoints each
  element touches from this.
- `touches(e)`: instead of `index`, the timepoints whose weights or observations read element `e`,
  as a range. The sampler scores a move of element `e` over these alone, so
  every timepoint that reads it must be in the range. Ranges must not decrease
  from one element to the next.
- `keys(e)`: optional, the observation keys (see `shared_key` in
  [`latent_hmm_data`](@ref)) whose `observation_shared` value reads element `e`.
  By default every key met in `touches(e)`.
- `scale`: starting random-walk scale (adapted during warm-up).
"""
struct Field{P,W,KY} <: LatentKind
    dim::Int
    n::Int
    prior::P
    touches::W
    keys::KY
    scale::Float64
end
function Field(dim::Integer, n::Integer; prior, index = nothing, touches = nothing, keys = nothing, scale = 1.0)
    (index === nothing) == (touches === nothing) &&
        throw(ArgumentError("Field: give one of `index` or `touches`"))
    Field(Int(dim), Int(n), prior, touches === nothing ? _ByIndex(index) : touches, keys, Float64(scale))
end

# touches worked out from index once the number of timepoints is known
struct _ByIndex{F}; index::F; end
struct _Windows; w::Vector{UnitRange{Int}}; end
(w::_Windows)(e) = w.w[e]

_resolve(k, T) = k
function _resolve(k::Field{P,<:_ByIndex}, T) where {P}
    idx = [k.touches.index(t) for t in 1:T]
    firsts = [something(findfirst(==(e), idx), 0) for e in 1:k.n]
    w = map(1:k.n) do e
        firsts[e] == 0 && return (e == 1 ? 1 : 0):0
        nxt = findfirst(>(0), view(firsts, (e + 1):k.n))
        firsts[e]:(nxt === nothing ? T : firsts[e + nxt])
    end
    for e in 2:k.n      # elements never in use get an empty window in place
        isempty(w[e]) && (w[e] = last(w[e - 1]):(last(w[e - 1]) - 1))
    end
    Field(k.dim, k.n, k.prior, _Windows(w), k.keys, k.scale)
end

"""
    PerIndividual(prior; observed = nothing, allowed = nothing, scale = 1.0)

One value per individual.

- `prior`: a distribution, or `model -> distribution` when it depends on the
  parameters. A prior that depends on the parameters is part of the marginal
  likelihood, for every individual, observed or not.
- `observed`: a vector with `missing` where the value is unknown. Known values
  are fixed.
- `allowed(data, i, value)`: a constraint; moves that break it are rejected.
- `scale`: random-walk scale for a continuous prior. A discrete value moves by
  1 to 3 steps, or flips when it has two possible values.
"""
struct PerIndividual{D,O,A} <: LatentKind
    prior::D
    observed::O
    allowed::A
    scale::Float64
end
PerIndividual(prior; observed = nothing, allowed = nothing, scale = 1.0) =
    PerIndividual(prior, observed, allowed, Float64(scale))

"""
    Shared(dim, n; prior, touches = nothing, update = nothing, observation_only = false, scale = 1.0)

`n` vectors common to every individual, e.g. the positions of unmapped detectors.

- `prior`: as for [`Field`](@ref).
- `touches(j)`: the observation keys (see `shared_key` in
  [`latent_hmm_data`](@ref)) whose `observation_shared` value reads element `j`.
  `nothing` means all of them.
- `update(old, model, data, i, key, j, oldx, newx)`: optional. Updates an
  `observation_shared` value for a move of element `j` from `oldx` to `newx`
  more cheaply than recomputing it. Return `old` itself when the value is
  unchanged.
- `observation_only`: set when nothing but `observation_shared` reads this
  latent. An individual whose shared values all come back `old` is then skipped.
"""
struct Shared{P,W,U} <: LatentKind
    dim::Int
    n::Int
    prior::P
    touches::W
    update::U
    observation_only::Bool
    scale::Float64
end
Shared(dim::Integer, n::Integer; prior, touches = nothing, update = nothing,
       observation_only = false, scale = 1.0) =
    Shared(Int(dim), Int(n), prior, touches, update, observation_only, Float64(scale))

"""
    @latents begin
        name ~ Kind(...)
        ...
    end

Declare a model's latent variables. Returns a `NamedTuple` of
[`Field`](@ref), [`PerIndividual`](@ref) and [`Shared`](@ref) declarations,
which [`latent_hmm_data`](@ref) takes as `latents`.

```julia
latents = @latents begin
    S   ~ Field(2, n_years; prior = Normal(0, 10.0), touches = y -> year_steps(y))
    u   ~ PerIndividual(Normal(0, 1))
    sex ~ PerIndividual(model -> Bernoulli(model.sex_prop); observed = sex_data)
end
```
"""
macro latents(block)
    block isa Expr && block.head == :block || error("@latents body must be begin...end")
    names = Symbol[]; rhs = Any[]
    for line in block.args
        line isa LineNumberNode && continue
        (line isa Expr && line.head == :call && line.args[1] == :~ && line.args[2] isa Symbol) ||
            error("@latents lines look like `name ~ Field(...)`, got: $line")
        push!(names, line.args[2]); push!(rhs, esc(line.args[3]))
    end
    :(NamedTuple{$(Tuple(names))}(($(rhs...),)))
end

# ---- the sampled values ----

"""
    LatentState

The current value of every declared latent, as `state.values.name`:
a `dim x n x N` array for a Field, a vector for a PerIndividual, a `dim x n`
matrix for a Shared. Build one with [`latent_state`](@ref).

`stamp` changes whenever the values do. The likelihood's caches are keyed on it.
"""
mutable struct LatentState{V<:NamedTuple}
    values::V
    stamp::UInt
end

_new_stamp() = rand(UInt)
restamp!(s::LatentState) = (s.stamp = _new_stamp(); s)
Base.copy(s::LatentState) = LatentState(map(copy, s.values), _new_stamp())

_shape(k::Field, N) = (k.dim, k.n, N)
_shape(k::PerIndividual, N) = (N,)
_shape(k::Shared, N) = (k.dim, k.n)

"""
    latent_state(latents, n_individuals; rng = Xoshiro(1), init...)

Starting values for every latent in `latents`, given as `name = array` keyword
arguments with the shapes [`LatentState`](@ref) describes. Observed
PerIndividual values replace the starting values where they are known.

A latent left out is drawn from its prior. A PerIndividual whose prior depends
on the parameters is filled in from its observed values instead.
"""
function latent_state(latents::NamedTuple, N::Integer; rng = Random.Xoshiro(1), init...)
    vals = map(keys(latents)) do name
        k = latents[name]
        x = haskey(init, name) ? copy(init[name]) : _draw_init(rng, k, N, name)
        size(x) == _shape(k, N) ||
            throw(DimensionMismatch("latent `$name` should be $(_shape(k, N)), got $(size(x))"))
        if k isa PerIndividual && k.observed !== nothing
            for i in 1:N
                ismissing(k.observed[i]) || (x[i] = k.observed[i])
            end
        end
        x
    end
    LatentState(NamedTuple{keys(latents)}(Tuple(vals)), _new_stamp())
end

# ---- priors ----

_coord_prior(p, c) = p
_coord_prior(p::Tuple, c) = p[c]

function _vec_logprior(prior, x)
    lp = 0.0
    @inbounds for c in eachindex(x)
        lp += Distributions.logpdf(_coord_prior(prior, c), x[c])
    end
    lp
end

_param_prior(k::PerIndividual) = k.prior isa Function
_prior_at(k::PerIndividual, model) = k.prior isa Function ? k.prior(model) : k.prior

_is_free(k::PerIndividual, i) = k.observed === nothing || ismissing(k.observed[i])

# The possible values of a discrete prior, if there are few.
function _small_support(d)
    d isa Distributions.DiscreteUnivariateDistribution || return nothing
    lo, hi = Distributions.minimum(d), Distributions.maximum(d)
    (isfinite(lo) && isfinite(hi) && hi - lo <= 1) || return nothing
    Int(lo):Int(hi)
end

# ---- starting values from the priors ----

function _draw_init(rng, k::Union{Field,Shared}, N, name)
    x = zeros(_shape(k, N))
    for c in 1:k.dim
        p = _coord_prior(k.prior, c)
        p isa Distributions.Distribution || throw(ArgumentError("no starting value for `$name`"))
        sel = selectdim(x, 1, c)
        sel .= rand(rng, p, size(sel))
    end
    x
end
function _draw_init(rng, k::PerIndividual, N, name)
    if !(k.prior isa Function)
        return rand(rng, k.prior, N)
    end
    seen = k.observed === nothing ? [] : collect(skipmissing(k.observed))
    isempty(seen) && throw(ArgumentError("no starting value for `$name`: its prior depends on the parameters"))
    [rand(rng, seen) for _ in 1:N]
end
