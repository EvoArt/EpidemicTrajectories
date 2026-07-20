# The badger model with GOMPERTZ-MAKEHAM survival, matching the C++ reference
# (badger_ref/classic rcpp) rather than the Julia reference (run_base_exp.jl).
#
# WHY THIS EXISTS. The default badger model (badger_model.jl) uses a full 4-param
# SILER survival curve — early-life mortality (a1, b1) + Gompertz late-life
# (a2, b2) + Makeham constant hazard (c1) — copied line-for-line from
# run_base_exp.jl's `probs`. But the C++ code in `classic rcpp` uses a simpler
# 3-param GOMPERTZ-MAKEHAM curve: NO a1/b1 term at all. Confirmed by grep — `a1`
# and `b1` appear in ZERO .cpp files. Its survival probability (TrProbSurvive_.cpp)
# is exactly:
#
#     P(survive | age) = exp( -c1 + (a2/b2) * (exp(b2*(age-1)) - exp(b2*age)) )
#
# i.e. our siler_survival WITHOUT the `(a1/b1) * diffExpsEarlyLife` term. Fitting
# the extra a1/b1 term against data generated (or best-described) by a
# Gompertz-Makeham process lets a1/b1 absorb signal that should sit in c1/a2/b2,
# which is the most likely reason those estimates drift from the C++'s.
#
# This file changes ONLY the survival function and the transition spec that uses
# it. Everything else — infection, progression, starting state, observations,
# aggregates, coupling — is reused unchanged from badger_model.jl /
# badger_model_reststotal.jl / badger_model_obssplit.jl.

using EpidemicTrajectories
using Distributions
using Random

isdefined(@__MODULE__, :BADGER_STATES) || include(joinpath(@__DIR__, "badger_model.jl"))

## ---------------------------------------------------------------------------
## Gompertz-Makeham survival (no a1/b1) — matches TrProbSurvive_.cpp exactly
## ---------------------------------------------------------------------------

# Same shape as siler_survival in badger_model.jl but WITHOUT the early-life
# (a1, b1) term. Reads only c1, a2, b2 — the three parameters the C++ actually
# has. `diffExpsLateLife` is written to match the C++'s
# `-exp(b2*(age-1)) * (exp(b2) - 1)` form (via expm1) exactly.
function gompertz_makeham_survival(model, data, i, t)
    age = t >= 1 ? data.age[i, t] : data.age[i, 1] + (t - 1)
    age < 0 && return 1.0                       # not yet born

    a2, b2, c1 = model.a2, model.b2, model.c1
    y1 = b2 * (age - 1); y2 = b2 * age
    late = -exp(y1) * expm1(y2 - y1)            # exp(b2*(age-1)) - exp(b2*age)
    s = exp(-c1 + (a2 / b2) * late)

    # A badger seen alive later cannot have died yet — same data constraint as
    # siler_survival and as the C++ (which sets survival to 1 up to last capture).
    return t <= data.last_capture_time[i] ? 1.0 : s
end

# Transition spec using the Gompertz-Makeham survival. Infection and progression
# are the same functions the default model uses.
function badger_transitions_gompertz()
    @transitions BADGER_STATES begin
        @survival gompertz_makeham_survival death=:D
        S -> E = badger_infection
        E -> I = badger_progression
    end
end
