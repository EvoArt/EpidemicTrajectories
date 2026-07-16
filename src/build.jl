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
                for ds in data.derived_summaries
                    ds(model, data, X, X[t, i], i, t)
                end
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
            for ds in data.derived_summaries
                ds(model, data, X, X[data.n_timepoints, i], i, data.n_timepoints)
            end
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
"""
function epidemic_loglik(data::EpidemicData)
    function loglik(model, data::EpidemicData, X)
        ll = zero(_param_eltype(model))

        for i in 1:data.n_individuals
            p0 = data.starting_state(model, data, X, i, 1)
            ll += log(p0[X[1, i]] + 1e-12)
        end

        for t in 1:(data.n_timepoints - 1)
            for i in 1:data.n_individuals
                P = transition_matrix_at(data.trans_mat, model, data, X, i, t)
                ll += log(P[X[t, i], X[t + 1, i]] + 1e-12)
            end
        end

        ll
    end

    loglik
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
