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

# --- The proposal: which chain the filter actually runs ----------------------
#
# `iffbs!` is an exact Gibbs step only when the chain the forward filter runs is
# the chain the likelihood scores. `IFFBSProposal` names that chain explicitly so
# it can be a DIFFERENT one — which is what `iffbs_mh!` (iffbs_mh.jl) corrects for
# with a Metropolis-Hastings accept/reject step.
#
# The default is built from `data` and reproduces the previous behaviour exactly:
# every field is the corresponding field of `data`. Nothing about `iffbs!` changes.
#
# All four fields are parameterised. That is the CLAUDE.md concrete-types rule, and
# specifically the third-instance trap it records: a field annotated with a
# parametric type must bind EVERY one of that type's parameters or the field is
# abstract. Leaving `TM` free (rather than writing `TransitionSpec{RF}`) sidesteps
# that entirely, and has the bonus that a proposal's transition source need not be
# a `TransitionSpec` at all.
struct IFFBSProposal{TM,SS,OP,RC}
    trans_mat::TM              # the chain the filter pushes probability through
    starting_state::SS         # its initial distribution
    observation_process::OP    # the per-state weight vector it multiplies in
    rest_contribution::RC      # its coupling term
end

"""
    iffbs_proposal(data; trans_mat, starting_state, observation_process, rest_contribution)

The chain the iFFBS forward filter runs. Every keyword defaults to the matching
field of `data`, so `iffbs_proposal(data)` is the exact conditional's own chain and
`iffbs!` with it is the ordinary exact Gibbs sweep.

Override a field to make the filter a PROPOSAL for a different target, and correct
the difference with [`iffbs_mh!`](@ref):

  * `rest_contribution = no_rest_contribution` — drop the between-individual term
    from the filter. This is the paper's MHiFFBS (Touloupou, Finkenstadt & Spencer
    2020, section 3.3.2). Note the ratio still needs the TARGET's neighbour terms
    for the two realised paths, so this saves a factor of `n_states / 2` on the
    coupling work, not the whole of it.
  * `trans_mat = <a Markov approximation>` — for a semi-Markov target, whose
    sojourn-dependent hazard no first-order filter can represent. This is the
    paper's SM-iFFBS (section 4.3).

Both at once is the paper's SM-MHiFFBS; one MH step corrects both discrepancies.
"""
iffbs_proposal(data::EpidemicData;
               trans_mat=data.trans_mat,
               starting_state=data.starting_state,
               observation_process=data.observation_process,
               rest_contribution=data.rest_contribution) =
    IFFBSProposal(trans_mat, starting_state, observation_process, rest_contribution)

"""
    uncorrected_proposal(data)

The paper's "uncorrected-iFFBS" proposal: the filter drops the coupling term
entirely (standard FFBS on a single chain). Valid ONLY inside [`iffbs_mh!`](@ref) —
used as a Gibbs step it targets the wrong conditional, which is the failure mode
the paper's Supplementary section A documents.
"""
uncorrected_proposal(data::EpidemicData) =
    iffbs_proposal(data; rest_contribution=no_rest_contribution)

"""
    markov_proposal(data, trans_mat)

A proposal that swaps only the transition spec — the shape needed for a
semi-Markov target, where `trans_mat` is a first-order approximation of it
(e.g. the geometric/kappa = 1 chain of the paper's section 4.3).
"""
markov_proposal(data::EpidemicData, trans_mat) = iffbs_proposal(data; trans_mat=trans_mat)

initialise_forward_filter(model, data::EpidemicData, X, i, t) = data.starting_state(model, data, X, i, t)

# --- Sweep-level scratch for the forward/backward passes ---------------------
#
# `probs` and `trans_cache` are created by the forward pass, consumed by the
# backward pass, and dead by the end of `iffbs_individual!` — nothing outside
# that call ever holds a reference. So they need not be allocated per individual:
# one buffer sized to the LONGEST sampling window serves every individual in the
# sweep, with each one using the leading `1:n_t` slice.
#
# Before this, every individual allocated a fresh `N x N x n_t` `trans_cache`
# (~20 KB at N=4, n_t=78) and an `n_t x N` `probs`, then zeroed both — ~48 MB of
# pointless zeroing per badger sweep, and a fresh cold region of memory each time.
#
# Held in a module-level `Ref` keyed by size rather than threaded through the call
# signature: `iffbs!` is single-threaded by construction (individuals within a
# sweep share mutable aggregate state via the reverse/re-apply invariant, so they
# CANNOT be run in parallel), and this keeps the public `forward_filter` /
# `backward_sample!` signatures unchanged for anyone calling them directly.
#
# CONCRETELY TYPED, and that is not optional. A first version used
# `Ref{Any}(nothing)` holding a NamedTuple: every `_filter_scratch` call then
# returned `Any`, so `s.probs`/`s.cur`/`s.w` were dynamic lookups and the buffers
# reached the inner loops untyped. Measured: the sweep got SLOWER (0.435 ->
# 0.546 s) even though allocation fell 199 -> 139 MB — type instability across
# the whole inner loop cost more than the allocations it removed. Exactly the
# trap CLAUDE.md's performance notes describe, hit while trying to fix a
# different one.
struct FilterScratch
    probs::Matrix{Float64}
    trans::Array{Float64,3}
    rowsum::Vector{Float64}
    cur::Vector{Float64}
    w::Vector{Float64}
    # Two whole-window Int paths, for `iffbs_mh!` only. The MH step must hold the
    # CURRENT path while the candidate is drawn (it scores both against the same
    # filter), so unlike `iffbs!` it cannot sample straight into `X`. Sized with
    # everything else so the reject branch — which copies the current path back —
    # allocates nothing.
    path_cur::Vector{Int}
    path_can::Vector{Int}
end

const _FILTER_SCRATCH = Ref{Union{Nothing,FilterScratch}}(nothing)

function _filter_scratch(data::EpidemicData, n_t::Int)::FilterScratch
    N = data.n_states
    s = _FILTER_SCRATCH[]
    if s === nothing || size(s.probs, 1) < n_t || size(s.probs, 2) != N
        # Size to the longest window in the whole dataset, so this grows at most
        # once per run rather than creeping up individual by individual.
        maxwin = maximum(hi - lo + 1 for (lo, hi) in data.sampling_period; init=n_t)
        cap = max(maxwin, n_t)
        s = FilterScratch(zeros(Float64, cap, N), zeros(Float64, N, N, cap),
                          zeros(Float64, N), zeros(Float64, N), zeros(Float64, N),
                          zeros(Int, cap), zeros(Int, cap))
        _FILTER_SCRATCH[] = s
    end
    return s
end

"""
    forward_filter(xᵢ, start_sampling, end_sampling, model, data, X, i,
                   proposal=iffbs_proposal(data))

The forward pass for individual `i`: at each time, push the previous filtered
distribution through `i`'s transition matrix, then multiply in the observation
likelihood and the coupling term. Returns the filtered distributions and the
cached transition matrices (reused by the backward pass).

The backward pass needs every timepoint's transition matrix still alive when it
runs, so they can't share one buffer the way a single-matrix-at-a-time loop would
— but they need not be `length(xᵢ)` SEPARATE allocations either. `trans_cache` is
one contiguous `N × N × (length(xᵢ)-1)` array, filled a slice at a time via
[`transition_matrix_at!`](@ref); `rowsum` is pure scratch (nothing outside
`transition_matrix_at!` reads it) and is genuinely reused across every timepoint.
Before this, one badger-sized sweep (2384 individuals, ~78-timepoint average
window) allocated close to 185,000 short-lived `N × N` matrices; profiling showed
that allocation, not dispatch, as the dominant cost once the earlier type-stability
fixes had already landed (see the repro log).
"""
function forward_filter(xᵢ, start_sampling, end_sampling, model, data::EpidemicData, X, i,
                        proposal=iffbs_proposal(data))
    n_t = length(xᵢ)
    N = data.n_states
    scratch = _filter_scratch(data, n_t)
    probs = view(scratch.probs, 1:n_t, :)
    trans_cache = view(scratch.trans, :, :, 1:n_t)
    forward_filter!(probs, trans_cache, scratch, xᵢ, start_sampling, end_sampling,
                    model, data, X, i, proposal)
    probs, trans_cache
end

"""
    forward_filter!(probs, trans_cache, scratch, xᵢ, start_sampling, end_sampling,
                    model, data, X, i, proposal=iffbs_proposal(data))

In-place [`forward_filter`](@ref): writes the filtered distributions into `probs`
and the transition matrices into `trans_cache`, allocating nothing per timepoint.

Every intermediate the old version built fresh at each `(i, t)` — the predicted
distribution (`trans' * probs[j-1, :]`, which also materialised a transpose), the
unnormalised product, and the normalised result — is now a write into a reused
`N`-length buffer, and the matrix-vector product is an explicit loop (`N` is 4 in
the reference model; BLAS is not worth its call overhead at that size). A
badger-sized sweep visits ~187k `(i, t)` cells, so each array that was allocated
here cost ~187k allocations per sweep.
"""
function forward_filter!(probs, trans_cache, scratch, xᵢ, start_sampling, end_sampling,
                         model, data::EpidemicData, X, i, proposal=iffbs_proposal(data))
    n_t = length(xᵢ)
    N = data.n_states
    rowsum = scratch.rowsum
    cur = scratch.cur
    # Everything the filter pushes probability through comes off the PROPOSAL, not
    # off `data` — that is the one behavioural seam this refactor opens, and with
    # the default proposal the two are the same objects.
    p_trans = proposal.trans_mat
    p_start = proposal.starting_state
    p_obs = proposal.observation_process
    p_rest = proposal.rest_contribution

    t0 = start_sampling
    base = p_start(model, data, X, i, t0)
    obs = p_obs(model, data, X, i, t0)
    affected = data.affected_individuals === nothing ? nothing : data.affected_individuals[t0, i]
    rest = p_rest(model, data, X, i, t0, N, affected)
    z0 = zero(eltype(probs))
    @inbounds for s in 1:N
        cur[s] = base[s] * obs[s] * rest[s]
        z0 += cur[s]
    end
    @inbounds for s in 1:N
        probs[1, s] = cur[s] / z0
    end
    # trans_cache[:, :, 1] is never read (backward_sample! walks from n_t down to
    # 2, reading trans_cache[j] for j >= 2 only), so it is left untouched — the
    # buffer is reused across individuals and stale values here are harmless.

    @inbounds for j in 2:n_t
        t = start_sampling + j - 1
        trans = view(trans_cache, :, :, j)
        # OFF-BY-ONE, load-bearing: `trans_cache[:, :, j]` is the matrix for the
        # step (j-1 -> j) in WINDOW indices, i.e. absolute (t-1 -> t). The backward
        # pass therefore reads `trans_cache[:, :, j+1]` when moving from window
        # index j to j+1. Slice 1 is never written and never read.
        transition_matrix_at!(trans, rowsum, p_trans, model, data, X, i, t - 1)

        obs_w = p_obs(model, data, X, i, t)
        affected = data.affected_individuals === nothing ? nothing : data.affected_individuals[t, i]
        rest_w = p_rest(model, data, X, i, t, N, affected)

        # pred = trans' * probs[j-1, :], then multiply in the observation and
        # coupling weights — fused into one pass over the N candidate states.
        z = zero(eltype(probs))
        for b in 1:N
            acc = zero(eltype(probs))
            for a in 1:N
                acc += trans[a, b] * probs[j - 1, a]
            end
            w = acc * obs_w[b] * rest_w[b]
            cur[b] = w
            z += w
        end
        if z > 0
            for b in 1:N
                probs[j, b] = cur[b] / z
            end
        else
            uniform = one(eltype(probs)) / N
            for b in 1:N
                probs[j, b] = uniform
            end
        end
    end
    nothing
end

"""
    backward_sample!(probs, trans_cache, xᵢ, start_sampling, end_sampling, model, data, X, i, rng)

The backward pass: draw the final state from its filtered distribution, then walk
backwards drawing each state from its filtered distribution reweighted by the
already-sampled next state. Writes into `xᵢ` (a view into `X`).
"""
function backward_sample!(probs, trans_cache, xᵢ, start_sampling, end_sampling, model, data::EpidemicData, X, i, rng)
    n_t = length(xᵢ)
    N = data.n_states
    # Reuse the same sweep-level scratch the forward pass used — `w` is a buffer
    # kept solely for this loop, so it never collides with `cur`.
    w = _filter_scratch(data, n_t).w

    xᵢ[n_t] = _sample_categorical(rng, view(probs, n_t, :))

    @inbounds for j in (n_t - 1):-1:1
        trans = view(trans_cache, :, :, j + 1)
        bnext = xᵢ[j + 1]
        # cond = probs[j, :] .* trans[:, bnext], normalised — written into `w`
        # rather than allocating two fresh N-vectors per timepoint (~187k cells
        # per sweep, so that was ~374k allocations).
        z = zero(eltype(probs))
        for a in 1:N
            v = probs[j, a] * trans[a, bnext]
            w[a] = v
            z += v
        end
        if z > 0
            for a in 1:N
                w[a] /= z
            end
        else
            uniform = one(eltype(probs)) / N
            for a in 1:N
                w[a] = uniform
            end
        end
        xᵢ[j] = _sample_categorical(rng, w)
    end
    nothing
end

# --- The backward pass's own density ----------------------------------------
#
# `iffbs_mh!` needs the log-density of the backward-sampling kernel, for the drawn
# candidate AND for the path already in `X`. Both are the same walk as
# `backward_sample!`: the candidate's is accumulated as it is drawn, the current
# path's by replacing each draw with a lookup.
#
# WHY THESE ARE SEPARATE FUNCTIONS rather than a keyword on `backward_sample!`:
# `backward_sample!` is on the hot path of every `iffbs!` sweep, and adding a
# `log` per timepoint (plus a branch the compiler has to hoist) to a function that
# does not need it is exactly the kind of change CLAUDE.md's "measure each change
# alone" rule exists to prevent. The duplication is ~20 lines and buys a hot path
# that is byte-for-byte what it was.
#
# The three of them MUST stay in lockstep on two details or the MH ratio is wrong:
#   * the weight is `probs[j, a] * trans_cache[a, x[j+1], j+1]` — note the `j+1`
#     on the cache (see the off-by-one note in `forward_filter!`);
#   * the degenerate `z <= 0` branch falls back to a UNIFORM distribution, so the
#     scorer must return `log(1/N)` there, not `-Inf`.

"""
    backward_logq(probs, trans_cache, path, data, n_t) -> Real

The log-density of `path` under the backward-sampling kernel that
[`backward_sample!`](@ref) draws from — i.e. `log q(path)` for the proposal whose
forward pass produced `probs`/`trans_cache`.

Consumes NO randomness. That matters: with the proposal equal to the target the
acceptance ratio is identically 1, `iffbs_mh!` skips its `rand()` (see
[`iffbs_mh_individual!`](@ref)), and the whole MH sweep then draws exactly the
same random numbers in the same order as the plain `iffbs!` sweep — which is what
makes the RNG-stream equivalence test possible.

`path` is indexed 1:n_t in WINDOW coordinates (window index `j` is absolute time
`start_sampling + j - 1`), matching `xᵢ` in [`backward_sample!`](@ref).
"""
function backward_logq(probs, trans_cache, path, data::EpidemicData, n_t::Int)
    N = data.n_states
    w = _filter_scratch(data, n_t).w

    @inbounds begin
        logq = log(probs[n_t, path[n_t]])

        for j in (n_t - 1):-1:1
            trans = view(trans_cache, :, :, j + 1)
            bnext = path[j + 1]
            z = zero(eltype(probs))
            for a in 1:N
                v = probs[j, a] * trans[a, bnext]
                w[a] = v
                z += v
            end
            if z > 0
                logq += log(w[path[j]] / z)
            else
                logq += log(one(eltype(probs)) / N)
            end
        end
    end
    logq
end

"""
    backward_sample_logq!(dest, probs, trans_cache, data, n_t, rng) -> Real

[`backward_sample!`](@ref), but writing into `dest` (window coordinates, NOT a
view into `X`) and returning the log-density of what it drew.

Writing somewhere other than `X` is required, not stylistic: the MH step has to
score the CURRENT path against the same filter, so `X` must still hold it when the
candidate is drawn.

Draws exactly the same categorical variates, in the same order, as
[`backward_sample!`](@ref) would.
"""
function backward_sample_logq!(dest, probs, trans_cache, data::EpidemicData, n_t::Int, rng)
    N = data.n_states
    w = _filter_scratch(data, n_t).w

    @inbounds begin
        b = _sample_categorical(rng, view(probs, n_t, :))
        dest[n_t] = b
        logq = log(probs[n_t, b])

        for j in (n_t - 1):-1:1
            trans = view(trans_cache, :, :, j + 1)
            bnext = dest[j + 1]
            z = zero(eltype(probs))
            for a in 1:N
                v = probs[j, a] * trans[a, bnext]
                w[a] = v
                z += v
            end
            if z > 0
                for a in 1:N
                    w[a] /= z
                end
            else
                uniform = one(eltype(probs)) / N
                for a in 1:N
                    w[a] = uniform
                end
            end
            a = _sample_categorical(rng, w)
            dest[j] = a
            logq += log(w[a])
        end
    end
    logq
end

"""
    iffbs_individual!(model, data, X, i, rng, proposal=iffbs_proposal(data))

Resample individual `i`'s whole trajectory in place.

Reverses `i`'s contribution out of the aggregates, runs the forward filter and
backward sample, then re-applies `i`'s new contribution — leaving the aggregates
exactly consistent with the updated `X`, and giving `i` leave-one-out statistics
while it is being resampled.
"""
function iffbs_individual!(model, data::EpidemicData, X, i, rng,
                           proposal=iffbs_proposal(data))
    start_sampling, end_sampling = data.sampling_period[i]
    xᵢ = @view X[start_sampling:end_sampling, i]

    # `apply_summaries!` (aggregates.jl), NOT `for ds in data.derived_summaries`:
    # the summaries are a Tuple of distinct closure types, so a plain loop infers
    # their union and dispatches at runtime on every call. These two loops were
    # 42.7% of a sweep's self time, both flagged for GC and dynamic dispatch.
    @inbounds for t in start_sampling:end_sampling
        apply_summaries!(data.derived_summaries, model, data, X, X[t, i], i, t, true)
    end

    data._focal[] = i
    probs, trans_cache = forward_filter(xᵢ, start_sampling, end_sampling, model, data, X, i,
                                        proposal)
    backward_sample!(probs, trans_cache, xᵢ, start_sampling, end_sampling, model, data, X, i, rng)
    data._focal[] = -1

    @inbounds for t in start_sampling:end_sampling
        apply_summaries!(data.derived_summaries, model, data, X, X[t, i], i, t, false)
    end

    nothing
end

"""
    iffbs!(model, data, X, rng; proposal=iffbs_proposal(data)) -> X

One full iFFBS sweep: resample every individual's trajectory in turn, each
conditioning on the others' current trajectories.

Assumes the aggregates already agree with `X` on entry (see
[`apply_derived_summaries!`](@ref)) and preserves that invariant on exit, so a
sweep never rebuilds them.
"""
function iffbs!(model, data::EpidemicData, X, rng; proposal=iffbs_proposal(data))
    # Built ONCE per sweep, not once per individual: `IFFBSProposal` holds a
    # `TransitionSpec` (which has array fields, so it is not isbits), and
    # constructing it inside the loop would be one heap allocation per individual
    # for no reason.
    for i in 1:data.n_individuals
        iffbs_individual!(model, data, X, i, rng, proposal)
    end
    X
end
