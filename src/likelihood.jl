# Log-likelihood of the continuous parameters given a FIXED hidden state
# trajectory — the term differentiated by HMC/NUTS.
#
# This is the "single source of truth": it is built from the SAME
# `transition_matrix_at` the FFBS sampler and the simulator use. When you change
# a rate function, the fit, the latent sampler, and (later) the residuals all
# move together, consistently.
#
# `trajectory_loglik(pars, model, data)` sums, over every individual and every
# time step, the log-probability of the observed one-step transition
# `X[i,t] -> X[i,t+1]` under the model's transition matrix. `X` (in `data.states`)
# is treated as FIXED data here — under AD it is a constant, so gradients flow
# only through `pars`. This matches how a PracticalBayes `@addlogprob!` term uses
# a `ValueSlot` latent: the latent state is AD-constant, the parameters are not.

"""
    trajectory_loglik(pars, model, data) -> Real

Log-probability of the hidden state trajectory `data.states` given the
parameters `pars`, under the model's per-step transition matrices. Sums
`log P[a→b]` over every individual `i` and every time step `t → t+1`, where `a`,
`b` are individual `i`'s dense states at `t` and `t+1`.

Pure and autodiff-friendly: differentiates cleanly w.r.t. `pars` (the
transition matrices follow `eltype(pars)`), treats `data.states` as constant.
Drop it straight into a PracticalBayes `@model` via
`@addlogprob! trajectory_loglik(pars, model, data)`, or into any other PPL's
"add to the log density" primitive.

`model.state_space` and `model.rates` are read; `data` must expose `.states`,
`.group`, `.members` (see [`make_data`](@ref)).
"""
function trajectory_loglik(pars, model, data)
    ss = model.state_space::StateSpace
    X = data.states
    n_ind, n_t = size(X)
    T = eltype(pars)
    lp = zero(T)
    for i in 1:n_ind
        for t in 1:(n_t - 1)
            a = state_index(ss, X[i, t])
            b = state_index(ss, X[i, t + 1])
            P = transition_matrix_at(model.rates, pars, model, data, i, t)
            @inbounds lp += log(P[a, b])
        end
    end
    return lp
end

"""
    observation_loglik(pars, model, data, tests, results) -> Real

Log-likelihood of the observed diagnostic-test `results` given the fixed hidden
state trajectory `data.states`. `tests` is a tuple of [`DiagnosticTest`](@ref)s;
`results` is a tuple/vector of `n_individuals x n_times` result matrices (one per
test, entries `-1` missing / `0` negative / `1` positive), aligned with `tests`.

Sums, over every individual and time, the log of the joint per-state observation
likelihood evaluated at the individual's TRUE state (from `data.states`).

Usually the test-sensitivity/specificity parameters have conjugate closed-form
Gibbs updates (see the iFFBS-paper example) and are NOT sampled by HMC — in that
case this term is constant w.r.t. the HMC parameters and can be omitted from the
`@addlogprob!` HMC target. It is provided for completeness and for models that DO
put the test parameters through HMC.
"""
function observation_loglik(pars, model, data, tests, results)
    ss = model.state_space::StateSpace
    X = data.states
    n_ind, n_t = size(X)
    T = promote_type(eltype(pars), Float64)
    lp = zero(T)
    for t in 1:n_t, i in 1:n_ind
        a = state_index(ss, X[i, t])
        rs = ntuple(k -> results[k][i, t], length(results))
        lik = observation_likelihood(tests, pars, ss, rs)
        @inbounds lp += log(lik[a])
    end
    return lp
end
