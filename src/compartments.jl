# State spaces for discrete-time individual-level epidemic models.
#
# A `StateSpace` names the compartments an individual can occupy and fixes their
# integer encoding. The encoding matters because the hidden-state trajectory `X`
# is stored as a `Matrix{Int}` (individual x time), and both the FFBS sampler and
# the likelihood index transition-probability matrices by these codes.
#
# We support arbitrary user encodings (the badger work uses the sparse codes
# S=0, E=3, I=1, Dead=9 to match an external C++/R pipeline), but internally the
# FFBS/likelihood machinery works with a DENSE 1:n_states index — `state_index`
# maps a user code to its dense position. Keeping both lets model authors use
# whatever codes their data already carry while the linear-algebra stays simple.

"""
    StateSpace(codes; names=nothing)

A discrete state space for an individual-level epidemic model. `codes` is the
tuple of integer state codes in the fixed order they index transition-probability
matrices (so `codes[1]` is dense index 1, `codes[2]` is dense index 2, ...).
`names` is an optional tuple of `Symbol`s for pretty-printing / clarity.

The order in `codes` IS the dense ordering used everywhere internally — a
transition matrix `P` produced by the model has `P[a, b] = Prob(dense state a at
t -> dense state b at t+1)`, where dense index `a` corresponds to `codes[a]`.

# Examples
```julia
SI   = StateSpace((0, 1);       names=(:S, :I))         # susceptible / infected
SIS  # same two states, recurrent dynamics (S<->I) — encoding identical to SI
SEID = StateSpace((0, 3, 1, 9); names=(:S, :E, :I, :D)) # badger encoding
```
"""
struct StateSpace{N,C<:NTuple{N,Int},Nm}
    codes::C
    names::Nm  # NTuple{N,Symbol} or Nothing
end

StateSpace(codes::NTuple{N,Int}; names=nothing) where {N} = StateSpace{N,typeof(codes),typeof(names)}(codes, names)
StateSpace(codes::AbstractVector{<:Integer}; names=nothing) = StateSpace(Tuple(Int.(codes)); names=names)

"""
    nstates(ss::StateSpace) -> Int

Number of distinct states (dense dimension of the transition matrices).
"""
nstates(::StateSpace{N}) where {N} = N

"""
    state_index(ss::StateSpace, code::Integer) -> Int

Dense index (1-based) of the user state `code` within `ss.codes`. Errors if
`code` is not one of the state space's codes.
"""
@inline function state_index(ss::StateSpace{N}, code::Integer) where {N}
    @inbounds for a in 1:N
        ss.codes[a] == code && return a
    end
    throw(ArgumentError("state code $code is not in this StateSpace $(ss.codes)"))
end

"""
    SI

The canonical two-state susceptible/infected state space, `StateSpace((0, 1);
names=(:S, :I))`. This is the state space of the iFFBS-paper cattle E. coli
model: an animal is either susceptible (`X=0`) or infected (`X=1`), with
recurrent S<->I dynamics (an infected animal can recover back to susceptible and
be re-infected — there is NO recovered/removed compartment). `SIS` is provided
as an alias to make the recurrent dynamics explicit at the call site; the state
encoding is identical.
"""
const SI = StateSpace((0, 1); names=(:S, :I))

"""
    SIS

Alias for [`SI`](@ref) — the same two-state `{S, I}` encoding, named to make the
recurrent (susceptible-infected-susceptible) dynamics explicit. There is no
separate recovered compartment; recovery returns an individual to `S`.
"""
const SIS = SI

"""
    SEID

The badger bovine-TB state space `StateSpace((0, 3, 1, 9); names=(:S, :E, :I,
:D))`: susceptible, exposed (latently infected, not yet infectious), infectious,
and dead/removed. The sparse codes match the external capture-recapture data
pipeline. Provided for the individual-level SEID milestone; the two-state `SI`
model is the first supported case.
"""
const SEID = StateSpace((0, 3, 1, 9); names=(:S, :E, :I, :D))
