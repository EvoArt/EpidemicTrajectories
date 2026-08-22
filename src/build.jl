# The three things a model spec generates: a simulator, a likelihood, and a latent
# sampler. Each is an ordinary function, closed over the `data`, so they drop into
# any PPL (or none).

"""
    epidemic_simulator(data) -> simulate

Build the simulator: `simulate(rng, model) -> X`, drawing a trajectory forward in
time from the parameters `model`.

Each individual's initial state comes from `data.starting_state`; each subsequent
step samples from that individual's transition matrix. The user's derived
summaries are applied as the simulation advances, so rates that read the
aggregates see values consistent with the trajectory so far.
"""
function epidemic_simulator(data::EpidemicData)
    function simulate(rng, model)
        X = zeros(Int, data.n_timepoints, data.n_individuals)

        # A simulation builds `X` from nothing, so the aggregates must start from
        # nothing too. Without this, a second call would accumulate on top of the
        # first one's counts — inflating whatever the rates read off them, and
        # silently making the same seed give a different trajectory.
        reset_aggregates!(data)

        for i in 1:data.n_individuals
            p0 = data.starting_state(model, data, X, i, 1)
            X[1, i] = _sample_categorical(rng, p0)
        end

        for t in 1:(data.n_timepoints - 1)
            # Fill this time slice's aggregates before the rates read them.
            for i in 1:data.n_individuals
                apply_summaries!(data.derived_summaries, model, data, X, X[t, i], i, t, false)
            end
            for i in 1:data.n_individuals
                P = transition_matrix_at(data.trans_mat, model, data, X, i, t)
                X[t + 1, i] = _sample_categorical(rng, view(P, X[t, i], :))
            end
        end

        # The loop above stops one short, so fill the final time slice too — on
        # exit the aggregates must agree with the whole of `X`, which is the
        # invariant the likelihood and the latent sampler both rely on.
        for i in 1:data.n_individuals
            apply_summaries!(data.derived_summaries, model, data, X,
                             X[data.n_timepoints, i], i, data.n_timepoints, false)
        end

        X
    end

    simulate
end

"""
    epidemic_loglik(data) -> loglik

Build the likelihood: `loglik(model, data, X) -> Real`, the log-probability of the
trajectory `X` under the parameters `model`.

Autodiff-friendly in `model` (`X` and `data` are constants), so it drops straight
into a PPL's log density — e.g. `@addlogprob! loglik(pars, data, X)` in a
PracticalBayes `@model`.

Reads the aggregates rather than rebuilding them: whatever the rate functions read
off `data` must already be consistent with `X`. That invariant is established once
by [`apply_derived_summaries!`](@ref) and preserved by the latent sampler.

Each individual's transitions are only summed over its own `sampling_period`
(defaulting to `1:n_timepoints` when the user doesn't supply one — see
[`epidemic_data`](@ref)), not the full time range. Outside that window there is no
move to explain: nothing observes the individual, so the reference model (which
this package matches) contributes no likelihood term there either. On the badger
dataset the average window is under half the full 161 timepoints, so this roughly
halves the per-gradient cost.

## Conditioning on entry: `entry_time` (+ `survival`)

Many observation designs only START watching an individual partway through its
life — a badger's first capture, a patient's enrolment. Before that entry time we
know it was ALIVE (else we would never have seen it), but we did NOT observe its
disease dynamics. The likelihood should therefore, in the pre-entry window:

  * still score the DISEASE transitions the trajectory makes (infection,
    progression) — the epidemic was happening whether or not we watched;
  * NOT score the SURVIVAL factor — the individual is known alive, so charging
    `P(survive)` there would double-count / bias the survival parameters, and
    charging `P(die)` is simply wrong (it did not die).

This is exactly the reference's `j >= firstCaptureTimes` gate, which multiplies in
`log(survival)` only from entry onward while still scoring the infection/
progression part before it.

Pass BOTH:
  * `entry_time` — a `Vector{Int}`, the per-individual entry time; and
  * `survival` — the SAME survival function the transitions use,
    `survival(model, data, i, t) -> P(survive the t -> t+1 step)`.

For a step `t -> t+1` with `t < entry_time[i]`, the loglik scores
`log(transition_prob) - log(survival)` — i.e. the transition with its survival
factor DIVIDED OUT (in log space, subtracted), leaving just the disease-move part.
From `entry_time[i]` on, the full `log(transition_prob)` is scored. The loop still
runs over the whole `sampling_period` (nothing is skipped).

`survival` MUST be the survival used to build `trans_mat` (so the subtraction
exactly removes what was multiplied in). When `entry_time === nothing` (the
default) neither argument is consulted and the behaviour is unchanged; supplying
`entry_time` without `survival` errors, since the gate cannot be applied without
knowing which factor is survival.

The subtraction `log(transition_prob) - log(survival)` is exact for an
alive->alive move, where `transition_prob = survival * move`. It is NOT exact for
an alive->death move (there `transition_prob = 1 - survival`). This is fine in
practice because a model that conditions on entry also forbids death before the
individual is known alive (e.g. a death-banning observation process makes the
filter never place a pre-entry death), so the pre-entry window contains only
alive->alive moves. If your model can place death before entry, do not use this
gate.

The starting-state term is scored at each individual's OWN window start
(`sampling_period[i][1]`), where the nu mixing is defined and where iFFBS stores
the drawn initial state — NOT at absolute time 1.

## The power-user path: copy this body and hand-optimize it

The whole point of this package is that the likelihood is an ORDINARY function you
can read, copy, and replace. `epidemic_loglik` returns a generic closure that
works for ANY model spec — which means it goes through indirections (`data.trans_mat`,
`data.starting_state`, `transition_prob` walking the rate tuple) that a model
written by hand for ONE fixed spec does not need. When the generic version is not
fast enough, drop to a bespoke `loglik` for your model. The closure this function
returns is, verbatim, the loop below — start from this and specialize it:

```julia
# A hand-written equivalent of the closure `epidemic_loglik(data; entry_time, survival)`
# returns. Copy it, then specialize for YOUR model (see the levers underneath).
function my_loglik(model, data, X)
    ll = zero(eltype(model.some_continuous_param))   # element type follows the params (AD)

    for i in 1:data.n_individuals
        first_t, last_t = data.sampling_period[i]

        # starting state at the individual's own window start
        p0 = data.starting_state(model, data, X, i, first_t)
        ll += log(p0[X[first_t, i]] + 1e-12)

        entry_i = data.first_capture_time[i]         # or first_t if not conditioning on entry
        for t in first_t:min(last_t, data.n_timepoints) - 1
            # the ONE transition this individual actually made (fused survival * move)
            p = transition_prob(data.trans_mat, model, data, X, i, t, X[t, i], X[t + 1, i])
            ll += log(p + 1e-12)

            # pre-entry: divide the survival factor back out (subtract its log)
            if t < entry_i
                s = my_survival(model, data, i, t)   # SAME survival used in trans_mat
                ll -= log(s + 1e-12)
            end
        end
    end
    ll
end
```

Levers a bespoke version can pull that the generic one cannot:

  * **Analytic survival / rates.** `transition_prob` calls YOUR rate functions
    through a tuple walk. If your survival has a closed form (e.g. a Siler or
    Gompertz `exp`), inline it and its derivative here instead of routing through
    the generic `@transitions` dispatch. Hard-code the `(from, to)` branch for your
    state space rather than clamping and summing an arbitrary row.

  * **Condition on entry with FEWER calls.** The gate above evaluates survival
    TWICE per pre-entry step: once fused inside `transition_prob`, once as
    `my_survival` to subtract back out. A bespoke pre-entry branch can compute the
    disease-MOVE directly (`log(infection)` / `log(1-progRate)` / …) in ONE
    transcendental, never forming `survival * move` at all. On a dataset with a
    large pre-entry window (the badgers: ~9452 steps) this roughly halves the
    gate's added cost.

  * **Skip work you know is zero.** The generic loop runs to `last_t` and scores
    post-death `D -> D` steps as `log(1) ≈ 0`; a bespoke loop can stop at the
    imputed death time (as the reference's `lastObsAliveTimes` does) and drop
    those iterations entirely.

  * **Non-allocating starting state.** `data.starting_state` here returns a fresh
    vector per individual; a bespoke term can index the one state it needs without
    allocating (the same trick `observation_weight` uses for the obs term).

A worked, staged version of exactly this — naive spec → `@aggregate` →
`logProbRest` + custom coupling → hand-written likelihood — is in the optimization
example in the docs.
"""
function epidemic_loglik(data::EpidemicData; entry_time=nothing, survival=nothing,
                         step_logprob=nothing)
    et = entry_time
    surv = survival
    if et !== nothing && surv === nothing
        error("epidemic_loglik: `entry_time` requires `survival` too — the entry " *
              "gate removes the survival factor before entry, so it must know which " *
              "factor that is. Pass `survival=<the survival fn used in trans_mat>`.")
    end
    # Dispatch on the step function's CONCRETE type, so `default_step_logprob` (a
    # singleton) devirtualises and inlines and the AD hot path is what it always
    # was. Passing it through an untyped field instead would make one runtime
    # dispatch per (i, t) — the exact `rate_fns` / `derived_summaries` trap
    # CLAUDE.md records twice.
    _build_loglik(step_logprob === nothing ? default_step_logprob : step_logprob, et, surv)
end

"""
    default_step_logprob(model, data, X, i, t)

`log P(X[t, i] -> X[t+1, i])` under `data.trans_mat` — the step term
[`epidemic_loglik`](@ref) and [`epidemic_conditional_loglik`](@ref) use unless a
`step_logprob` is supplied.

## Why this is a seam

The whole point of `step_logprob` is SEMI-MARKOV targets. A sojourn-dependent
hazard (`E -> I` after `s` steps in `E`) is not a function of `(model, data, i, t)`
— it needs the PATH — and rate functions in a [`TransitionSpec`](@ref) are handed
only `(model, data, i, t)`. `step_logprob` receives `X`, so a semi-Markov model
overrides this ONE function and keeps the starting-state, observation, entry-gate
and neighbour machinery unchanged.

Give the SAME `step_logprob` to `epidemic_loglik` and to
`epidemic_conditional_loglik`. If they disagree, the HMC block and the latent Gibbs
block are sampling different posteriors, silently. (The MH kernel cannot catch this
for you: it compares the conditional against the proposal, not against the
population likelihood. `test/iffbs_mh.jl`'s delta-consistency test is what catches
it, and it is worth copying into your own model's tests.)

Note that a semi-Markov `step_logprob` must NOT be used to build the iFFBS
PROPOSAL — the forward filter is a first-order recursion and cannot represent it.
That is precisely what [`iffbs_mh!`](@ref) exists to correct.
"""
@inline default_step_logprob(model, data::EpidemicData, X, i, t) =
    log(transition_prob(data.trans_mat, model, data, X, i, t, X[t, i], X[t + 1, i]) + 1e-12)

function _build_loglik(slp::F, et, surv) where {F}
    function loglik(model, data::EpidemicData, X)
        ll = zero(_param_eltype(model))

        for i in 1:data.n_individuals
            first_t, last_t = data.sampling_period[i]
            # Score the starting state at the individual's OWN window start, NOT at
            # absolute time 1. iFFBS imputes and stores the trajectory over
            # [first_t, last_t] only (iffbs_individual!: xᵢ = X[first_t:last_t, i])
            # and draws the initial state into X[first_t, i]. Reading X[1, i] here
            # for a badger whose window starts later scores a cell iFFBS never
            # touches — a stale X_init value uncoupled from the sampled trajectory.
            # (`badger_starting_state` ignores its t arg and recomputes first_t
            # internally, so the DISTRIBUTION was already correct; only the state
            # index it was scored against was wrong.)
            p0 = data.starting_state(model, data, X, i, first_t)
            ll += log(p0[X[first_t, i]] + 1e-12)

            # The loop covers the WHOLE window; entry conditioning changes WHAT is
            # scored before entry (survival divided out), not WHICH steps.
            entry_i = et === nothing ? first_t : et[i]
            # OFF-BY-ONE: this is `first_t : (min(last_t, n_timepoints) - 1)`, i.e.
            # every step `t -> t+1` that lies wholly inside the window. The last
            # timepoint of the window has no outgoing step to score.
            for t in first_t:(min(last_t, data.n_timepoints) - 1)
                # Only ONE entry of the transition matrix matters here: the move
                # this individual actually made. `transition_prob` computes just
                # that, rather than building the whole matrix per (i, t) — which
                # dominated the gradient (~380k matrix allocations per call).
                ll += slp(model, data, X, i, t)
                # Pre-entry: divide the survival factor back out (subtract its log),
                # leaving just the disease-move part. `transition_prob` returned
                # `survival * move`, so `log(move) = log(p) - log(survival)`.
                if surv !== nothing && t < entry_i
                    s = surv(model, data, i, t)
                    ll -= log(s + 1e-12)
                end
            end
        end

        ll
    end

    loglik
end

"""
    epidemic_obs_loglik(data; observation_process=data.observation_process) -> obs_loglik

Build the OBSERVATION likelihood: `obs_loglik(model, data, X) -> Real`, the
log-probability of the observations given the trajectory `X`.

This is the counterpart to [`epidemic_loglik`](@ref), which covers only the
starting state and the transitions. Neither includes the other, so a model that
wants both writes their sum:

```julia
@addlogprob! loglik(pars, data, X) + obs_loglik(pars, data, X)
```

Without this term the observation parameters get NO likelihood information in the
log density — their gradient entries are prior/Jacobian only, and they are
effectively sampled from the prior. (`observation_process` is otherwise used only
by the iFFBS forward filter, which is not part of the differentiated density.)

## Why `observation_process` is a keyword

The default is whatever `data` already holds, so the common case is
`epidemic_obs_loglik(data)`. Passing a DIFFERENT function is the seam that lets a
user split their observation model between this likelihood and a conjugate Gibbs
block.

The package cannot make that split itself: `observation_process` is one opaque
function returning a weight vector, and nothing in it tells the package which
factors belong to which parameters. A user whose observation process factorises
multiplicatively —

    w(state) = capture_factor(state) * test_factor(state)

— can exploit conjugacy by writing the two factors as separate functions, using
the product as `data.observation_process` (so the latent sampler still sees the
whole thing), and passing only the non-conjugate factor here:

```julia
obs_loglik = epidemic_obs_loglik(data; observation_process = my_test_factor_only)
```

Because the weights enter as a product, the log-likelihood is a SUM of the two
factors' contributions, so dropping one factor here drops exactly its term and
leaves the other's intact. Keeping a factor in BOTH this likelihood and a
conjugate block would double-count it.

## The contract

`observation_process(model, data, X, i, t)` returns a per-state weight vector `w`
where `w[s]` is `P(observation at (i,t) | state s)`. It need not be normalised
over states — it is a likelihood in the observation, not a distribution over
states. This term reads `w[X[t, i]]`: the weight of the state the individual is
actually in.

!!! warning "Take the weight vector's element type from the parameters"
    Allocate `w` as `ones(eltype(model.some_param), data.n_states)`, never
    `ones(Float64, ...)`. A parameter arrives as a plain `Float64` when its Gibbs
    block samples it conjugately, but as a `ForwardDiff.Dual` when it sits in an
    HMC block — so the SAME observation function is called with both, and which
    one you get depends on the BLOCKING, not on the model. Hard-coding `Float64`
    works until someone moves that parameter into an HMC block, then throws on
    the write. (The package does the same thing internally via `_param_eltype`.)

Summed over each individual's own `sampling_period`, matching
[`epidemic_loglik`](@ref) — outside that window nothing observes the individual.

## Performance: supply `observation_weight` for a scalar path

The vector-returning contract above is what the LATENT SAMPLER needs (the forward
filter genuinely reads every state's weight). The LIKELIHOOD needs exactly ONE
entry — `w[X[t,i]]` — so going through the vector allocates one array per `(i,t)`
and throws all but one element away. On the badger model that is ~187k
allocations per call, each an array of `Dual`s under AD.

This is the same trap [`epidemic_loglik`](@ref) already avoids: it uses
[`transition_prob`](@ref) (two scalar accumulators, no allocation) rather than
building the whole transition matrix per `(i, t)`.

So `observation_weight` is the scalar counterpart. Pass a function

    observation_weight(model, data, X, i, t, s) -> P(observation at (i,t) | state s)

and this term calls it with `s = X[t, i]`, never materialising a vector. It must
agree with `observation_process` entry-for-entry; supplying both and letting them
disagree silently changes the posterior, so verify them against each other.

Both arguments **default to whatever `data` carries** (`data.observation_weight`
and `data.observation_process`). So the recommended path is to give the scalar to
[`epidemic_data`](@ref) once — as `observation_weight`, or as the second half of a
both-supplied observation model — and then just call `epidemic_obs_loglik(data)`:
it picks up the stored scalar automatically and takes the fast, allocation-free
path. Pass these keywords here only to override what `data` stores (e.g. the badger
model, which stores the full `capture × tests` vector for the filter but passes
only the `tests` factor here). When `data` carries no scalar and none is passed,
the vector path is used — correct, just slower.
"""
function epidemic_obs_loglik(data::EpidemicData;
                             observation_process=data.observation_process,
                             observation_weight=data.observation_weight)
    obs = observation_process
    obsw = observation_weight

    function obs_loglik(model, data::EpidemicData, X)
        ll = zero(_param_eltype(model))

        for i in 1:data.n_individuals
            first_t, last_t = data.sampling_period[i]
            for t in first_t:min(last_t, data.n_timepoints)
                # Only the weight of the state this individual is actually in
                # matters — the rest of the vector describes states it is not in.
                # With `observation_weight` supplied we compute just that entry;
                # otherwise fall back to the vector-returning process and index it.
                @inbounds s = X[t, i]
                p = obsw === nothing ? obs(model, data, X, i, t)[s] :
                                       obsw(model, data, X, i, t, s)
                ll += log(p + 1e-12)
            end
        end

        ll
    end

    obs_loglik
end

"""
    epidemic_conditional_loglik(data; entry_time, survival, step_logprob,
                                observation_process, observation_weight,
                                neighbor_logprob, coupled_mask, neighbor_window)
        -> conditional

Build the FULL CONDITIONAL of one individual's trajectory:
`conditional(model, data, X, i) -> Real`, equal to `log pi(x_i | x_rest, y, model)`
up to an additive constant that does not depend on `x_i`.

This is the target of [`iffbs_mh!`](@ref)'s accept/reject step. It is also the
generic piece any OTHER per-individual latent kernel needs (particle Gibbs,
blocked Gibbs, MH move-events), which is why it lives here rather than in
`iffbs_mh.jl`.

## What it sums, and why the neighbour term is there

Collecting every factor of `epidemic_loglik + epidemic_obs_loglik` that depends on
`x_i` gives four groups:

  1. `log p0_i(X[first_t, i])` — `data.starting_state`
  2. `sum_t step_logprob(model, data, X, i, t)` — `i`'s own moves, plus the entry
     gate's `- log(survival)` before `entry_time[i]`, exactly as
     [`epidemic_loglik`](@ref) scores them
  3. `sum_t log g_i(X[t, i])` — the observation weights
  4. `sum_t sum_j log P_j(X[t, j] -> X[t+1, j])` over `j` in
     `affected_individuals[t, i]`

Group 4 has no counterpart in `epidemic_loglik`'s per-individual loop: those terms
belong to the NEIGHBOURS' likelihood contributions, and they enter `i`'s
conditional because the neighbours' rates read aggregates that `i` feeds. Leaving
them out is the single most likely way to get this wrong, and it is what
`test/iffbs_mh.jl`'s delta-consistency test exists to catch.

## Evaluation point (do not get this wrong)

`conditional` must be called with the aggregates CONSISTENT WITH `X` — the same
standing invariant [`epidemic_loglik`](@ref) assumes, and the same one
[`iffbs!`](@ref) preserves — and with `data._focal[] == -1`. [`iffbs_mh!`](@ref)
arranges both.

## The two traps

**The observation model must be the FULL one.** `observation_process` /
`observation_weight` default to what `data` carries, which is what the FILTER
sees. Do NOT pass the reduced factor you may have given
[`epidemic_obs_loglik`](@ref): the full conditional of `X` contains every term
that depends on `X`, and a capture factor whose PARAMETERS are drawn conjugately
still depends on `X` (you cannot capture a dead animal). Dropping it here biases
the latent update.

**`entry_time` / `survival` / `step_logprob` must match `epidemic_loglik`'s.**
They define the target; disagreeing makes the HMC block and the latent block
sample different posteriors. Pass the same values to both.

## `neighbor_window`

Which of a neighbour `j`'s steps count in group 4:

  * `:likelihood` (default) — only steps inside `j`'s OWN `sampling_period`, which
    is exactly the set [`epidemic_loglik`](@ref) scores. This makes the conditional
    a true restriction of the joint, which is what group 4 has to be.
  * `:filter` — every `j` in `affected_individuals[t, i]`, matching what
    [`make_rest_contribution`](@ref) does inside the forward filter.

The two coincide when every individual shares a window, and usually coincide
anyway (an out-of-window neighbour typically sits in an absorbing state, whose
self-transition has probability 1 under every candidate and therefore cancels).
When they genuinely differ, `:likelihood` is the correct target and the difference
is a real proposal/target mismatch — [`check_iffbs_exact`](@ref) will report it,
and [`iffbs_mh!`](@ref) will correct it.

## `neighbor_logprob` and `coupled_mask`

Default to `data.neighbor_logprob` / `data.coupled_mask` — the very functions
`epidemic_data` built the default coupling term from. Reusing them is what makes
the acceptance ratio exactly 1 when the proposal equals the target.

If you supplied your OWN `rest_contribution` to [`epidemic_data`](@ref) (the fast
running-total path), `data.neighbor_logprob` is still the DEFAULT one and may not
agree with it. Pass the matching `neighbor_logprob` here, or accept that
`check_iffbs_exact` will flag the difference.

The mask restriction is exact: a neighbour move outside
[`coupled_transition_mask`](@ref) has the same probability under every value of
`X[t, i]`, so it contributes the same constant to both sides of the ratio and
cancels. Same argument as in [`make_rest_contribution`](@ref).
"""
function epidemic_conditional_loglik(data::EpidemicData;
                                     entry_time=nothing, survival=nothing,
                                     step_logprob=nothing,
                                     observation_process=data.observation_process,
                                     observation_weight=data.observation_weight,
                                     neighbor_logprob=data.neighbor_logprob,
                                     coupled_mask=data.coupled_mask,
                                     neighbor_window::Symbol=:likelihood,
                                     min_logprob=-1e12)
    if entry_time !== nothing && survival === nothing
        error("epidemic_conditional_loglik: `entry_time` requires `survival` too, " *
              "for the same reason `epidemic_loglik` does.")
    end
    neighbor_window in (:likelihood, :filter) ||
        throw(ArgumentError("neighbor_window must be :likelihood or :filter, got :$neighbor_window"))

    _build_conditional(step_logprob === nothing ? default_step_logprob : step_logprob,
                       observation_process, observation_weight, neighbor_logprob,
                       entry_time, survival, coupled_mask,
                       neighbor_window === :likelihood, Float64(min_logprob))
end

# Every callable is a separate type parameter, for the reason CLAUDE.md records
# three times over: stored behind an abstract field they would each cost a runtime
# dispatch, here on every (i, t) and every neighbour visit.
function _build_conditional(slp::F, obs::O, obsw::W, nlp::NL,
                            et, surv, mask, window_restrict::Bool,
                            min_logprob::Float64) where {F,O,W,NL}
    function conditional(model, data::EpidemicData, X, i::Int)
        ll = zero(_param_eltype(model))
        first_t, last_t_raw = data.sampling_period[i]
        T = data.n_timepoints
        last_t = min(last_t_raw, T)

        # (1) starting state, at the individual's OWN window start — identical to
        # `epidemic_loglik`, including the 1e-12 guard, because any difference
        # would show up as a spurious acceptance ratio.
        p0 = data.starting_state(model, data, X, i, first_t)
        @inbounds ll += log(p0[X[first_t, i]] + 1e-12)

        # (2) own steps. OFF-BY-ONE: `first_t : (last_t - 1)`, one step per
        # (t -> t+1) wholly inside the window; the window's last timepoint has no
        # outgoing step. Matches `_build_loglik` exactly.
        entry_i = et === nothing ? first_t : et[i]
        for t in first_t:(last_t - 1)
            ll += slp(model, data, X, i, t)
            if surv !== nothing && t < entry_i
                ll -= log(surv(model, data, i, t) + 1e-12)
            end
        end

        # (3) own observations. OFF-BY-ONE: `first_t : last_t` INCLUSIVE — an
        # observation attaches to a timepoint, not to a step. Matches
        # `epidemic_obs_loglik`.
        for t in first_t:last_t
            @inbounds st = X[t, i]
            p = obsw === nothing ? obs(model, data, X, i, t)[st] :
                                   obsw(model, data, X, i, t, st)
            ll += log(p + 1e-12)
        end

        # (4) the neighbours i influences.
        ll += _neighbor_conditional(nlp, mask, min_logprob, window_restrict,
                                    model, data, X, i, first_t, last_t)
        ll
    end

    conditional
end

# Split out so the `mask === nothing` and `window_restrict` branches hoist cleanly
# and `nlp` specialises.
function _neighbor_conditional(nlp::NL, mask, min_logprob::Float64, window_restrict::Bool,
                               model, data::EpidemicData, X, i::Int,
                               first_t::Int, last_t::Int) where {NL}
    acc = zero(_param_eltype(model))
    aff = data.affected_individuals
    aff === nothing && return acc
    T = data.n_timepoints

    # OFF-BY-ONE: a neighbour's STEP is (t -> t+1), so t stops at T-1. This is the
    # same set of timepoints `make_rest_contribution` covers: it is called for
    # every t in the window and returns all-ones (i.e. contributes nothing) at
    # t == n_timepoints.
    t_hi = min(last_t, T - 1)
    @inbounds for t in first_t:t_hi
        for j in aff[t, i]
            j == i && continue
            if window_restrict
                fj, lj = data.sampling_period[j]
                (fj <= t && t <= min(lj, T) - 1) || continue
            end
            if mask !== nothing
                mask[X[t, j], X[t + 1, j]] || continue
            end
            acc += max(nlp(model, data, X, j, t, i), min_logprob)
        end
    end
    acc
end

"""
    epidemic_latent_sampler(data; mh=false, proposal=nothing, target=nothing,
                            stats=nothing) -> latent!

Build the latent-state sampler: `latent!(rng, model, X) -> X`, one sweep
resampling the whole trajectory in place given the parameters.

This is what a PracticalBayes `AbstractLatentKernel`'s `latent_step` calls once per
Gibbs sweep — outside every gradient call, which is the point of the package.

iFFBS is one choice of latent sampler; the role is deliberately just "a function
of `(rng, model, X)` that updates `X`", so other samplers can fill it.

## `mh=false` (default) — the exact Gibbs sweep

Plain [`iffbs!`](@ref). Correct when the chain the forward filter runs IS the
chain the likelihood scores. [`check_iffbs_exact`](@ref) answers that question for
a given model rather than leaving it to inspection.

## `mh=true` — iFFBS as a proposal, corrected

[`iffbs_mh!`](@ref). Returns an [`IFFBSMHSampler`](@ref), still callable as
`(rng, model, X) -> X` but also carrying `.stats` for the acceptance diagnostics.

- `proposal` — an [`iffbs_proposal`](@ref); defaults to the exact one, which makes
  every ratio 1 (a self-check, not a useful sampler).
- `target` — an [`epidemic_conditional_loglik`](@ref); defaults to
  `epidemic_conditional_loglik(data)`. **Pass your own whenever
  `epidemic_loglik` got `entry_time` / `survival` / `step_logprob`**, with the
  same values, or the two blocks target different posteriors.
- `stats` — an [`MHStats`](@ref); one is created if you do not supply it, and is
  reachable as `latent!.stats` either way.

## Why `mh=false` with a `proposal` is an error

Because that combination is precisely the paper's "uncorrected-iFFBS", which
targets the wrong conditional and is the failure mode its Supplementary section A
documents. Reaching it by forgetting a keyword would be silent. Ask for the
correction explicitly, or do not pass a proposal.
"""
function epidemic_latent_sampler(data::EpidemicData; mh::Bool=false,
                                 proposal=nothing, target=nothing, stats=nothing)
    if !mh
        proposal === nothing || error(
            "epidemic_latent_sampler: a `proposal` was given with `mh=false`. An " *
            "iFFBS proposal that differs from the target is only valid inside the " *
            "MH correction — used as a Gibbs step it samples from the wrong " *
            "conditional, silently. Pass `mh=true`, or drop the `proposal`.")
        target === nothing || error(
            "epidemic_latent_sampler: a `target` was given with `mh=false`; the " *
            "plain iFFBS sweep has no accept/reject step to use it in.")
        stats === nothing || error(
            "epidemic_latent_sampler: `stats` were given with `mh=false`; the " *
            "plain iFFBS sweep accepts everything, so there is nothing to record.")
        return (rng, model, X) -> iffbs!(model, data, X, rng)
    end

    _check_mh_supported(data)
    prop = proposal === nothing ? iffbs_proposal(data) : proposal
    tgt = target === nothing ? epidemic_conditional_loglik(data) : target
    st = stats === nothing ? MHStats(data.n_individuals) : stats
    IFFBSMHSampler(data, prop, tgt, st)
end
