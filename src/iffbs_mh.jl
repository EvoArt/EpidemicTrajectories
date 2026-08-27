# iFFBS as a PROPOSAL, corrected by Metropolis-Hastings.
#
# `iffbs!` is an exact Gibbs step ONLY when the chain the forward filter runs is
# the chain the likelihood scores. Three ordinary situations break that:
#
#   * a SEMI-MARKOV target — the `E -> I` hazard depends on time-since-exposure,
#     which no first-order filter can represent (the paper's SM-iFFBS);
#   * a deliberately cheapened proposal — drop the between-individual coupling
#     term from the filter (the paper's MHiFFBS);
#   * a target carrying factors the filter does not — this package's `entry_time`
#     gate, an approximate `coupling_trans_mat`, a hand-written
#     `rest_contribution` that has drifted from `neighbor_logprob`.
#
# All three are the same fix: keep iFFBS, treat its output as a PROPOSAL in an
# independence Metropolis-Hastings step, and accept with
#
#     log alpha = [log pi(x_can | x_rest) - log pi(x_cur | x_rest)]
#               - [log q(x_can)           - log q(x_cur)]
#
# Reference: Touloupou, Finkenstadt & Spencer (2020), "Scalable Bayesian Inference
# for Coupled Hidden Markov and Semi-Markov Models", JCGS 29(2):238-249,
# section 3.3.2 (Algorithm 2) and section 4.3.
#
# The paper is explicit that the UNCORRECTED version is wrong, not merely
# approximate: running a mismatched proposal as though it were a Gibbs step is the
# failure its Supplementary section A documents. Hence `epidemic_latent_sampler`
# refuses `mh=false` together with a `proposal`.

"""
    MHStats(n_individuals)

Per-individual acceptance telemetry for [`iffbs_mh!`](@ref).

Fields:
- `proposed[i]` — candidates that DIFFERED from the current path (the ones an
  accept/reject decision was actually made about)
- `accepted[i]` — how many of those were accepted
- `identical[i]` — proposals that reproduced the current path exactly. Counted
  SEPARATELY, and deliberately: folding them into the denominator makes the
  reported rate depend on how often the chain is frozen rather than on how good
  the proposal is. `acceptance_rate` excludes them; [`identical_rate`](@ref)
  reports them on their own.
- `last_logratio[i]` — the last `log alpha` computed for `i` (`NaN` if none yet)
- `max_abs_logratio`, `worst_individual` — the running worst `|log alpha|` and who
  produced it. These are what makes the sampler its own exactness diagnostic: with
  the proposal equal to the target every ratio must be 0, so a non-zero maximum
  names the individual whose conditional the filter is getting wrong. See
  [`check_iffbs_exact`](@ref).
"""
struct MHStats
    proposed::Vector{Int}
    accepted::Vector{Int}
    identical::Vector{Int}
    last_logratio::Vector{Float64}
    max_abs_logratio::Base.RefValue{Float64}
    worst_individual::Base.RefValue{Int}
end

MHStats(n::Integer) = MHStats(zeros(Int, n), zeros(Int, n), zeros(Int, n),
                              fill(NaN, n), Ref(0.0), Ref(0))

"""
    reset_stats!(stats::MHStats) -> stats

Zero every counter. Useful between warm-up and sampling.
"""
function reset_stats!(stats::MHStats)
    fill!(stats.proposed, 0)
    fill!(stats.accepted, 0)
    fill!(stats.identical, 0)
    fill!(stats.last_logratio, NaN)
    stats.max_abs_logratio[] = 0.0
    stats.worst_individual[] = 0
    stats
end

"""
    acceptance_rate(stats) -> Float64
    acceptance_rate(stats, i) -> Float64

Accepted / proposed, EXCLUDING proposals that reproduced the current path
(see [`MHStats`](@ref)). `NaN` when nothing was proposed.

The paper reports median acceptance above 0.84 for every population size it tried
(up to 1000 individuals per group), declining slowly with population size. A rate
near zero means the proposal chain is too far from the target to be usable — that
is a modelling problem, not a bug in the kernel.
"""
function acceptance_rate(stats::MHStats)
    p = sum(stats.proposed)
    p == 0 ? NaN : sum(stats.accepted) / p
end
acceptance_rate(stats::MHStats, i::Integer) =
    stats.proposed[i] == 0 ? NaN : stats.accepted[i] / stats.proposed[i]

"""
    identical_rate(stats) -> Float64

The fraction of all proposals that reproduced the current path exactly. High
values mean the individual's conditional is sharply peaked (common for
well-observed individuals), and they make [`acceptance_rate`](@ref) rest on few
decisions and [`check_iffbs_exact`](@ref) a weaker check.
"""
function identical_rate(stats::MHStats)
    total = sum(stats.proposed) + sum(stats.identical)
    total == 0 ? NaN : sum(stats.identical) / total
end

@inline function _record!(stats::MHStats, i::Int, logratio::Float64, accepted::Bool)
    stats.proposed[i] += 1
    accepted && (stats.accepted[i] += 1)
    stats.last_logratio[i] = logratio
    a = abs(logratio)
    # `>` not `>=`, and NaN compares false, so a NaN ratio never becomes "the
    # worst" — it is reported through `last_logratio` instead.
    if a > stats.max_abs_logratio[]
        stats.max_abs_logratio[] = a
        stats.worst_individual[] = i
    end
    nothing
end
@inline _record!(::Nothing, i, logratio, accepted) = nothing

@inline _record_identical!(stats::MHStats, i::Int) = (stats.identical[i] += 1; nothing)
@inline _record_identical!(::Nothing, i) = nothing

"""
    iffbs_mh_individual!(model, data, X, i, rng, proposal, target, stats;
                         force=:none, on_nonfinite=:reject) -> log alpha

One Metropolis-Hastings update of individual `i`'s whole trajectory, using the
iFFBS backward-sampling kernel of `proposal` as an independence proposal and
`target` (an [`epidemic_conditional_loglik`](@ref)) as the conditional density.

Returns the log acceptance ratio, or `NaN` when the proposal reproduced the
current path (no decision was made).

## The ordering, and why it is this ordering

The aggregates are the expensive shared state, so the step is arranged to touch
them as little as plain [`iffbs_individual!`](@ref) does:

    1. log pi_cur     -- FREE: on entry the aggregates already agree with X,
                         which still holds the current path
    2. save x_cur; REVERSE i out of the aggregates      (1 reverse)
    3. forward filter under `proposal`, at the leave-one-out aggregate state
    4. score log q_cur (no RNG); draw x_can + log q_can into scratch, NOT into X
    5. identical? re-apply and return                   (1 apply)
    6. write x_can into X; APPLY                        (1 apply); log pi_can
    7. accept -> done. reject -> reverse, restore x_cur, apply  (+1 rev +1 app)

So an ACCEPT costs exactly the same summary passes as a plain iFFBS sweep, and
only a reject pays an extra round trip. Step 1 being free is the whole reason
`log pi_cur` is computed before anything is disturbed rather than after.

## Why no `rand()` when `log alpha >= 0`

Taken from the reference implementation ("when acc >= 1, avoid drawing RNG so
forced-accept runs reproduce classic iFFBS random stream"). A guaranteed-accept
decision then costs no randomness, so a run that always accepts draws exactly the
variates `iffbs!` would, in the same order.

**It does not quite give bit-identical streams for `proposal == target`, and the
reason is worth knowing.** The EXACT ratio there is 0, but the COMPUTED one is a
difference of sums of order 1e-14 whose sign is arbitrary. A ratio of `-1e-16`
takes the `log(rand(rng)) < logratio` branch, consumes a variate, and then accepts
anyway — same trajectory, different stream. The test suite therefore checks
stream equivalence under `force=:accept`, and checks the SIZE of the ratio
separately via [`check_iffbs_exact`](@ref). No tolerance is applied to the
comparison against zero: treating `|log alpha| <= tol` as an automatic accept
would trade a real (if tiny) bias for a cosmetic property.

## Keywords

- `force` — `:none` (decide normally), `:accept`, `:reject`. Test hooks; forcing
  never consumes randomness, so `force=:accept` on an exact spec leaves the RNG
  stream identical to `iffbs!`'s. `force=:reject` is the ONLY way to exercise the
  restore path, which is the branch most likely to harbour a bookkeeping bug.
- `on_nonfinite` — `:reject` (default) or `:error`, for a `NaN` ratio. `+Inf`
  accepts and `-Inf` rejects without complaint: both are meaningful (a current
  path unreachable under the proposal, and a candidate impossible under the
  target, respectively).
"""
function iffbs_mh_individual!(model, data::EpidemicData, X, i::Int, rng,
                              proposal, target, stats;
                              force::Symbol=:none, on_nonfinite::Symbol=:reject)
    first_t, end_raw = data.sampling_period[i]
    last_t = min(end_raw, data.n_timepoints)
    n_t = last_t - first_t + 1
    xᵢ = @view X[first_t:last_t, i]
    ds = data.derived_summaries

    # --- 1. the current path's conditional, at the standing invariant ---------
    # Aggregates agree with X, `data._focal[] == -1`: exactly the state
    # `epidemic_loglik` assumes. Doing this FIRST is what keeps the accept path
    # at the same summary cost as `iffbs_individual!`.
    logpi_cur = Float64(target(model, data, X, i))

    # `_filter_scratch` sizes itself to the longest window in the whole dataset on
    # its first call, so this returns the same object `forward_filter` will get and
    # the views below stay valid across it.
    scratch = _filter_scratch(data, n_t)
    path_cur = view(scratch.path_cur, 1:n_t)
    path_can = view(scratch.path_can, 1:n_t)
    # WINDOW coordinates: path index j is absolute time first_t + j - 1.
    @inbounds for j in 1:n_t
        path_cur[j] = X[first_t + j - 1, i]
    end

    # --- 2. reverse i out: aggregates become leave-one-out --------------------
    @inbounds for t in first_t:last_t
        apply_summaries!(ds, model, data, X, X[t, i], i, t, true)
    end

    # --- 3/4. filter under the proposal; score the current path; draw ---------
    data._focal[] = i
    probs, trans_cache = forward_filter(xᵢ, first_t, last_t, model, data, X, i, proposal)
    logq_cur = backward_logq(probs, trans_cache, path_cur, data, n_t)
    logq_can = backward_sample_logq!(path_can, probs, trans_cache, data, n_t, rng)
    data._focal[] = -1

    # --- 5. identical proposal: nothing to decide ----------------------------
    same = true
    @inbounds for j in 1:n_t
        if path_can[j] != path_cur[j]
            same = false
            break
        end
    end
    if same
        # X was never written, so re-applying `X[t, i]` restores the invariant.
        @inbounds for t in first_t:last_t
            apply_summaries!(ds, model, data, X, X[t, i], i, t, false)
        end
        _record_identical!(stats, i)
        return NaN
    end

    # --- 6. move to the candidate and score it -------------------------------
    @inbounds for j in 1:n_t
        X[first_t + j - 1, i] = path_can[j]
    end
    @inbounds for t in first_t:last_t
        apply_summaries!(ds, model, data, X, X[t, i], i, t, false)
    end
    logpi_can = Float64(target(model, data, X, i))

    logratio = (logpi_can - logpi_cur) - (logq_can - logq_cur)

    # --- 7. decide -----------------------------------------------------------
    local accepted::Bool
    if force === :accept
        accepted = true
    elseif force === :reject
        accepted = false
    elseif force !== :none
        error("iffbs_mh_individual!: `force` must be :none, :accept or :reject, got :$force")
    elseif isnan(logratio)
        if on_nonfinite === :error
            error("iffbs_mh_individual!: NaN acceptance ratio for individual $i " *
                  "(logpi_can=$logpi_can, logpi_cur=$logpi_cur, " *
                  "logq_can=$logq_can, logq_cur=$logq_cur)")
        end
        accepted = false
    elseif logratio >= 0
        # No `rand()` here — see the docstring. Covers +Inf.
        accepted = true
    else
        # Compared in LOG space. `exp(logratio)` would overflow to Inf / underflow
        # to 0 and then need special-casing, which is how the reference ends up
        # with an `isnan(acc)` branch. `log(rand()) < -Inf` is false, so a -Inf
        # ratio rejects, correctly.
        accepted = log(rand(rng)) < logratio
    end

    _record!(stats, i, logratio, accepted)

    if !accepted
        # Restore: reverse the candidate out, put the current path back, re-apply.
        # This is the branch the ratio-1 test never exercises (there everything is
        # accepted), so it has its own aggregate-consistency test.
        @inbounds for t in first_t:last_t
            apply_summaries!(ds, model, data, X, X[t, i], i, t, true)
        end
        @inbounds for j in 1:n_t
            X[first_t + j - 1, i] = path_cur[j]
        end
        @inbounds for t in first_t:last_t
            apply_summaries!(ds, model, data, X, X[t, i], i, t, false)
        end
    end

    logratio
end

"""
    iffbs_mh!(model, data, X, rng; proposal, target, stats=nothing,
              force=:none, on_nonfinite=:reject) -> X

One full MH-corrected iFFBS sweep: propose and accept/reject every individual's
trajectory in turn, each conditioning on the others' current trajectories.

Like [`iffbs!`](@ref), assumes the aggregates agree with `X` on entry and
preserves that on exit — on BOTH branches of every decision.

`proposal` defaults to `iffbs_proposal(data)` and `target` to
`epidemic_conditional_loglik(data)`. With both at their defaults this is the exact
Gibbs sweep with every acceptance ratio equal to 1, which is useful as a
self-check but wasteful as a sampler; use [`iffbs!`](@ref) for that, or
[`check_iffbs_exact`](@ref) to run the check deliberately.

If you pass a `target` built with `entry_time` / `survival` / `step_logprob`, they
must be the SAME ones you gave [`epidemic_loglik`](@ref). See
[`epidemic_conditional_loglik`](@ref).
"""
function iffbs_mh!(model, data::EpidemicData, X, rng;
                   proposal=iffbs_proposal(data),
                   target=epidemic_conditional_loglik(data),
                   stats=nothing, force::Symbol=:none, on_nonfinite::Symbol=:reject)
    _check_mh_supported(data)
    for i in 1:data.n_individuals
        iffbs_mh_individual!(model, data, X, i, rng, proposal, target, stats;
                             force=force, on_nonfinite=on_nonfinite)
    end
    X
end

# The exactness argument in `check_iffbs_exact` rests on the focal-self-contribution
# machinery reproducing "aggregates including i" inside the filter (see
# `_call_rate_with_focal`). With it switched off, the user's rate functions do
# their own accounting against LEAVE-ONE-OUT aggregates inside the filter, while
# `epidemic_loglik` calls those same rates against FULL aggregates — and the
# package has no way to know which of the two the conditional target should use.
# Rather than guess, refuse.
function _check_mh_supported(data::EpidemicData)
    data.focal_self_contribution || error(
        "iffbs_mh!: `focal_self_contribution=false` is not supported. The MH " *
        "acceptance ratio compares the filter's view of a rate against the " *
        "likelihood's, and with the focal re-insertion switched off those two are " *
        "evaluated against different aggregate states with no way to reconcile " *
        "them. Rebuild `data` with the default `focal_self_contribution=true`.")
    nothing
end

"""
    check_iffbs_exact(model, data, X; rng, n_sweeps=1, atol=1e-8, target=nothing)
        -> report

Is plain [`iffbs!`](@ref) an exact Gibbs step for this model?

Runs MH-corrected sweeps with the proposal set equal to the target
(`iffbs_proposal(data)`) and `force=:accept`, so the trajectory evolves exactly as
`iffbs!` would while every acceptance ratio is recorded. When the filter and the
likelihood score the same chain, every ratio is 0; a non-zero one names an
individual whose conditional the filter is getting wrong.

Returns a `NamedTuple`:

- `exact::Bool` — `max_abs_logratio <= atol`
- `max_abs_logratio`, `worst_individual`
- `n_checked` — decisions actually made (proposals that differed from the current
  path). A small number means a weak check, so it is reported.
- `n_identical`
- `offenders` — `(individual, logratio)` pairs exceeding `atol`, worst first

## What it can and cannot tell you

It compares the FILTER against the CONDITIONAL TARGET. It does not compare either
against [`epidemic_loglik`](@ref) — that is the delta-consistency test's job, and
it belongs in your model's own test suite (see `test/iffbs_mh.jl`).

Static detection of "is this spec iFFBS-compatible?" is not possible: rates are
opaque closures, and the interesting mismatches live in keyword arguments to a
different function (`entry_time`), in a second spec (`coupling_trans_mat`), or in
a split observation model. Running the ratio is the check.

## Expected numerical floor

The filter and the likelihood do not share their numerical guards
(`transition_matrix_at!` clamps rates then takes the self-transition as
`1 - rowsum`; `epidemic_loglik` adds `1e-12` inside the `log`;
`make_neighbor_logprob_from_transitions` floors at `1e-12`). On a model whose
probabilities sit well inside those bands the agreement is ~1e-12 relative, so
`atol=1e-8` is comfortable. A model that fails ONLY because it rides the clamp has
a different problem worth knowing about.
"""
function check_iffbs_exact(model, data::EpidemicData, X;
                           rng=Random.default_rng(), n_sweeps::Int=1, atol::Real=1e-8,
                           target=nothing)
    _check_mh_supported(data)
    prop = iffbs_proposal(data)
    tgt = target === nothing ? epidemic_conditional_loglik(data) : target

    worst = 0.0
    worst_i = 0
    n_checked = 0
    n_identical = 0
    offenders = Tuple{Int,Float64}[]

    for _ in 1:n_sweeps, i in 1:data.n_individuals
        lr = iffbs_mh_individual!(model, data, X, i, rng, prop, tgt, nothing;
                                  force=:accept)
        if isnan(lr)
            n_identical += 1
            continue
        end
        n_checked += 1
        a = abs(lr)
        if a > worst
            worst = a
            worst_i = i
        end
        a > atol && push!(offenders, (i, lr))
    end

    sort!(offenders; by=x -> -abs(x[2]))
    (; exact=worst <= atol, max_abs_logratio=worst, worst_individual=worst_i,
       n_checked, n_identical, offenders)
end

"""
    IFFBSMHSampler

What [`epidemic_latent_sampler`](@ref) returns for `mh=true`. Callable as
`(rng, model, X) -> X`, so it drops into a PracticalBayes latent kernel exactly
like the plain sampler does, while still exposing `.stats` for the acceptance
diagnostics:

```julia
latent! = epidemic_latent_sampler(data; mh=true, proposal=uncorrected_proposal(data))
# ... sample ...
acceptance_rate(latent!.stats)
```
"""
struct IFFBSMHSampler{D<:EpidemicData,P,T}
    data::D
    proposal::P
    target::T
    stats::MHStats
end

(s::IFFBSMHSampler)(rng, model, X) =
    iffbs_mh!(model, s.data, X, rng; proposal=s.proposal, target=s.target, stats=s.stats)
