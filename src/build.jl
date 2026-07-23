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

The starting-state term is unchanged (it applies at the individual's own start,
where the nu mixing is defined).
"""
function epidemic_loglik(data::EpidemicData; entry_time=nothing, survival=nothing)
    et = entry_time
    surv = survival
    if et !== nothing && surv === nothing
        error("epidemic_loglik: `entry_time` requires `survival` too — the entry " *
              "gate removes the survival factor before entry, so it must know which " *
              "factor that is. Pass `survival=<the survival fn used in trans_mat>`.")
    end
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
            for t in first_t:min(last_t, data.n_timepoints) - 1
                # Only ONE entry of the transition matrix matters here: the move
                # this individual actually made. `transition_prob` computes just
                # that, rather than building the whole matrix per (i, t) — which
                # dominated the gradient (~380k matrix allocations per call).
                p = transition_prob(data.trans_mat, model, data, X, i, t,
                                    X[t, i], X[t + 1, i])
                ll += log(p + 1e-12)
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

When omitted (the default), the vector path is used — correct, just slower.
"""
function epidemic_obs_loglik(data::EpidemicData;
                             observation_process=data.observation_process,
                             observation_weight=nothing)
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
    epidemic_latent_sampler(data) -> latent!

Build the latent-state sampler: `latent!(rng, model, X) -> X`, one iFFBS sweep
resampling the whole trajectory in place given the parameters.

This is what a PracticalBayes `AbstractLatentKernel`'s `latent_step` calls once per
Gibbs sweep — outside every gradient call, which is the point of the package.

iFFBS is one choice of latent sampler; the role is deliberately just "a function
of `(rng, model, X)` that updates `X`", so other samplers can fill it.
"""
function epidemic_latent_sampler(data::EpidemicData)
    (rng, model, X) -> iffbs!(model, data, X, rng)
end
