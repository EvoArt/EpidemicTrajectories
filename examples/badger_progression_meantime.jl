# Corrected E->I progression: the MEAN-TIME convention, matching the reference
# engine and the C++. See progression_bug.md for the full diagnosis.
#
# THE BUG this fixes. `badger_model.jl`'s `badger_progression` uses
# `erlang_cdf_at_1(k, tau/k)` with `erlang_cdf_at_1(k, tau) = 1 - exp(-tau)*Σ...`,
# i.e. at k=1 the E->I probability is `1 - exp(-tau)` and `tau` is a RATE. That
# was copied from `run_base_exp.jl`'s inline top-level `erlang_cdf_at_1`.
#
# But that top-level function is a DECOY. The code path run_base_exp.jl actually
# executes is `semi-markov_src.jl/transitions.jl`, whose `erlang_cdf_at_1` uses
# `x_over_theta = 1/theta` -> at k=1 the probability is `1 - exp(-1/tau)`, and
# `tau` is a MEAN TIME. The C++ agrees: `logProbEtoI = log(1 - exp(-1/tau))`
# (logPost_HMC.cpp:66), `p12 = (1-prDeath)*(1-exp(-1/tau))` (iFFBS_fixedPars.cpp:115).
#
# So our `tau` was the reference's `1/tau`: at the reference's tau=15 we compute
# P(E->I)=1.0 (certain), where they compute 0.0645. Confirmed numerically:
# ours(tau=0.0667) == ref(tau=15) == 0.0645, exact reciprocal. This is why our
# fit inferred tau~0.3 where the references get ~15 — and, because a wrong E->I
# timing corrupts the latent X, why beta and q were dragged off with it (the FOI
# formula itself is identical across all three implementations).
#
# This file leaves badger_model.jl's (rate-convention) version untouched and
# provides the corrected mean-time version for a new fit script to use.

using EpidemicTrajectories
using Distributions
using Random

isdefined(@__MODULE__, :BADGER_STATES) || include(joinpath(@__DIR__, "badger_model.jl"))

# Mean-time Erlang CDF at 1, matching semi-markov_src.jl/transitions.jl:250 and
# the C++. `theta` is a MEAN TIME. At k=1: `1 - exp(-1/theta)`.
function erlang_cdf_at_1_meantime(k, theta)
    x_over_theta = 1.0 / theta
    exp_term = exp(-x_over_theta)
    sum_val = one(x_over_theta)
    term = one(x_over_theta)
    for i in 1:(k - 1)
        term *= x_over_theta / i
        sum_val += term
    end
    return 1.0 - sum_val * exp_term
end

# E->I progression with `tau` as a MEAN TIME (reference/C++ convention).
# `prate(k, tau) = erlang_cdf_at_1_meantime(k, tau/k)`, matching the reference's
# `prate(k,tau) = erlang_cdf_at_1(k, tau/k)`.
badger_progression_meantime(model, data, i, t) =
    erlang_cdf_at_1_meantime(data.k, model.tau / data.k)

# Transition spec using the corrected progression. Survival and infection are
# whatever the caller supplies via the survival choice; this only swaps E->I.
# Two builders — one for Siler survival, one for Gompertz — so either can pair
# with the corrected progression.
function badger_transitions_meantime_siler()
    @transitions BADGER_STATES begin
        @survival siler_survival death=:D
        S -> E = badger_infection
        E -> I = badger_progression_meantime
    end
end

# Gompertz-Makeham survival evaluated at the DESTINATION time (t+1), fixing an
# off-by-one against the reference. See progression_bug.md / the transition-timing
# note below.
#
# THE OFF-BY-ONE. For the transition t -> t+1, the C++ (logPost_HMC.cpp: move
# (j-1)->j uses `TrProbSurvive_(ageMat(i,j))`, i.e. age at the DESTINATION j) and
# the semi-Markov Julia ref (transitions.jl:11, `t_next = t+1`, survival at
# t_next) both evaluate survival at t+1. Our gompertz_makeham_survival /
# siler_survival read `data.age[i, t]` — the SOURCE time — so every survival
# factor is one step of age behind the reference, and the last-capture guard
# compares `t` where the reference compares `t+1`. Systematic across all
# individuals and timepoints; c1/a2/b2 are pulled to compensate.
#
# This version reads age at t+1 and guards on t+1, matching the reference.
function gompertz_makeham_survival_tp1(model, data, i, t)
    tt = t + 1                                  # DESTINATION time of the t->t+1 move
    age = tt <= data.n_timepoints ? data.age[i, tt] : data.age[i, data.n_timepoints] + (tt - data.n_timepoints)
    age < 0 && return 1.0
    a2, b2, c1 = model.a2, model.b2, model.c1
    y1 = b2 * (age - 1); y2 = b2 * age
    late = -exp(y1) * expm1(y2 - y1)
    s = exp(-c1 + (a2 / b2) * late)
    return tt <= data.last_capture_time[i] ? 1.0 : s
end

# Gompertz-Makeham survival (t+1) + corrected mean-time progression — the fully
# C++-matching combination.
function badger_transitions_meantime_gompertz()
    @transitions BADGER_STATES begin
        @survival gompertz_makeham_survival_tp1 death=:D
        S -> E = badger_infection
        E -> I = badger_progression_meantime
    end
end
