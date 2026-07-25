# Badger server vs reference fix plan

This document lists the exact code changes I would make to align the server iFFBS/likelihood with the reference. I am not modifying any files; this is a plan only.

## 1. Root cause

`badger_infection` in the consolidated server script uses `data.aggregates.n_alive` as the denominator for the focal individual's `S -> E` force of infection. In `iffbs_individual!` the focal has been reversed out of `data.aggregates`, so `n_alive` is leave-one-out. The reference handles this with `groupLevelInfectionForcePlus1` (`M+1`) in `compute_individual_transition_probs`. The server `badger_infection` does not add the focal back, so the FOI denominator is `M-1` instead of `M`, and infection is over-probable. This propagates into implausible latent trajectories.

The cleanest fix is to keep `data.aggregates` **full** (including the focal) while the focal is being resampled, and to make the coupling/rest-contribution term subtract the focal's *current* state before adding each candidate. This avoids changing the rate-function signature and makes `badger_infection` correct as written.

## 2. Change `iffbs_individual!` to keep aggregates full

File: `src/iffbs.jl`

Currently the function reverses the focal out at the start and re-applies at the end. Instead, save the old trajectory, do not reverse at the start, and at the end replace the old contribution with the new one.

Before:

```julia
function iffbs_individual!(model, data::EpidemicData, X, i, rng)
    start_sampling, end_sampling = data.sampling_period[i]
    xᵢ = @view X[start_sampling:end_sampling, i]

    @inbounds for t in start_sampling:end_sampling
        apply_summaries!(data.derived_summaries, model, data, X, X[t, i], i, t, true)
    end

    probs, trans_cache = forward_filter(xᵢ, start_sampling, end_sampling, model, data, X, i)
    backward_sample!(probs, trans_cache, xᵢ, start_sampling, end_sampling, model, data, X, i, rng)

    @inbounds for t in start_sampling:end_sampling
        apply_summaries!(data.derived_summaries, model, data, X, X[t, i], i, t, false)
    end

    nothing
end
```

After:

```julia
function iffbs_individual!(model, data::EpidemicData, X, i, rng)
    start_sampling, end_sampling = data.sampling_period[i]
    xᵢ = @view X[start_sampling:end_sampling, i]

    # Keep the focal in data.aggregates; the transition rate and coupling term
    # will handle leave-one-out accounting themselves.
    x_old = copy(xᵢ)

    probs, trans_cache = forward_filter(xᵢ, start_sampling, end_sampling, model, data, X, i)
    backward_sample!(probs, trans_cache, xᵢ, start_sampling, end_sampling, model, data, X, i, rng)

    # Replace the old focal contribution with the newly sampled one.
    @inbounds for t in start_sampling:end_sampling
        s_old = x_old[t - start_sampling + 1]
        s_new = X[t, i]
        apply_summaries!(data.derived_summaries, model, data, X, s_old, i, t, true)
        apply_summaries!(data.derived_summaries, model, data, X, s_new, i, t, false)
    end

    nothing
end
```

Why: `forward_filter` then calls `transition_matrix_at!` with `data.aggregates` full. `badger_infection` reads `n_alive` and `n_infectious` that already include the focal, which is exactly the denominator the focal needs for its own `S -> E` probability.

## 3. Make `make_badger_rest_contribution` subtract the current focal

File: `examples/badger_fit_siler_fixed_hmc_5000.jl` (or wherever `make_badger_rest_contribution` lives).

Because `data.aggregates` are now full, `I_minus` and `M_minus` must start from the full group counts and subtract the focal's *current* state before `_grp_foi` adds the candidate.

Before:

```julia
function make_badger_rest_contribution()
    function rest_contribution(model, data, X, i, t, n_states, affected_override=nothing)
        t == data.n_timepoints && return ones(n_states)
        g = data.social_group[i, t]
        g == 0 && return ones(n_states)
        I_minus = data.aggregates.n_infectious[g, t]
        M_minus = data.aggregates.n_alive[g, t]
        nSE = data.aggregates.nSE[g, t]
        nSS = data.aggregates.nSS[g, t]
        logw = zeros(Float64, n_states)
        @inbounds for s in 1:n_states
            c = _foi_case(s)
            foi = _grp_foi(model, data, g, t, c, I_minus, M_minus)
            logw[s] = nSE * log(max(foi, 1e-12)) + nSS * log(max(1.0 - foi, 1e-12))
        end
        logw .-= maximum(logw)
        exp.(logw)
    end
    rest_contribution
end
```

After (only `I_minus`/`M_minus` lines change):

```julia
function make_badger_rest_contribution()
    function rest_contribution(model, data, X, i, t, n_states, affected_override=nothing)
        t == data.n_timepoints && return ones(n_states)
        g = data.social_group[i, t]
        g == 0 && return ones(n_states)
        s_current = X[t, i]
        I_minus = data.aggregates.n_infectious[g, t] - (s_current == _I_CODE ? 1 : 0)
        M_minus = data.aggregates.n_alive[g, t]      - (s_current != _D_CODE ? 1 : 0)
        nSE = data.aggregates.nSE[g, t]
        nSS = data.aggregates.nSS[g, t]
        logw = zeros(Float64, n_states)
        @inbounds for s in 1:n_states
            c = _foi_case(s)
            foi = _grp_foi(model, data, g, t, c, I_minus, M_minus)
            logw[s] = nSE * log(max(foi, 1e-12)) + nSS * log(max(1.0 - foi, 1e-12))
        end
        logw .-= maximum(logw)
        exp.(logw)
    end
    rest_contribution
end
```

Why: `data.aggregates` now include the focal. `_grp_foi` adds the candidate focal state back in; `I_minus`/`M_minus` must therefore exclude the *current* focal state. The `_CASE_I`/`_CASE_SorE`/`_CASE_D` logic inside `_grp_foi` does not change.

## 4. `badger_infection` needs no change

`badger_infection` currently reads `data.aggregates.n_alive` and `data.aggregates.n_infectious`:

```julia
function badger_infection(model, data, i, t)
    g = data.social_group[i, t]
    g == 0 && return 0.0
    I = data.aggregates.n_infectious[g, t]
    M = data.aggregates.n_alive[g, t]
    M == 0 && return 0.0
    lambda_g = model.lambda * model.alpha[g]
    foi = lambda_g + model.beta * I / ((M / data.K)^model.q)
    return -expm1(-foi)
end
```

With the change in section 2, `M` and `I` are now full group counts that already include the focal individual. Since `badger_infection` is only used for `S -> E`, the focal is susceptible, so:

- `M` includes the focal (alive) — correct.
- `I` does not include the focal (not infectious) — correct.

No edit is needed.

## 5. Add post-monitoring survival to the likelihood

File: `examples/badger_fit_siler_fixed_hmc_5000.jl` (the `loglik` definition).

The reference `posterior.jl` adds `log(survival)` for the interval from `lastObsAliveTimes` up to the last capture after monitoring. The server `epidemic_loglik` stops at `endSamplingPeriod` and never scores this tail.

Before:

```julia
const loglik = epidemic_loglik(data; entry_time=raw.first_capture_time,
                               survival=siler_survival_tp1)
```

After:

```julia
const _base_loglik = epidemic_loglik(data; entry_time=raw.first_capture_time,
                                      survival=siler_survival_tp1)

function badger_loglik(model, data, X)
    ll = _base_loglik(model, data, X)
    T = _param_eltype(model)
    for i in 1:data.n_individuals
        end_t  = data.sampling_period[i][2]
        last_t = data.last_capture_time[i]
        t0 = end_t
        t1 = min(last_t - 1, data.n_timepoints - 1)
        for t in t0:t1
            ll += log(siler_survival_tp1(model, data, i, t))
        end
    end
    return ll
end

const loglik = badger_loglik
```

Why: `siler_survival_tp1(model, data, i, t)` already returns the survival from `t` to `t+1` for the age at `t+1`. Summing `log` of that from `endSamplingPeriod` to `last_capture_time - 1` matches the reference `capturesAfterMonit` correction in `posterior.jl`.

## 6. What does NOT need changing

- `siler_survival_tp1` / `siler_surv` are not the site of the death-before-last-capture guard. The guard lives in the iFFBS transition matrix (`compute_individual_transition_probs`) and in the server `badger_obs_capture_deathban`. Those are already equivalent.
- `_grp_foi`'s `+1` for `S/E` and `I` is not a double-count once the aggregate accounting is fixed; it is the counterpart to the reference's `Plus1` FOI.
- The `erlang_cdf_at_1` implementations are the same Horner form.
- The Brock changepoint difference is, as you said, not an issue.

## 7. Order of application

1. Apply section 2 (`src/iffbs.jl`) and section 3 (`make_badger_rest_contribution`). This restores correct FOI denominators during iFFBS.
2. Verify `badger_infection` is unchanged and now receives full aggregates (section 4).
3. Apply section 5 (`badger_loglik`) so the parameter likelihood also scores post-monitoring survival.
4. Re-run the 50-iteration test to confirm the chain still executes, then compare posterior summaries to the reference.
