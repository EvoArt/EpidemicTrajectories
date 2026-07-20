# The progression bug: tau is inverted (rate vs mean-time)

**Symptom** (user report): our fit gives tau~0.3, beta~24, q~0.7; both references
(C++ classic rcpp AND the semi-Markov Julia ref) give tau~15, beta~0.019, q~0.6.
Low ESS and occasional poor rhat on the HMC params.

**Root cause: our E->I progression uses the RATE convention where the reference
uses the MEAN-TIME convention. Our `tau` is the reference's `1/tau`.**

## The two `erlang_cdf_at_1` are different

Ours (`examples/badger_model.jl:68`), copied from `run_base_exp.jl`'s top-level:
```julia
erlang_cdf_at_1(k, tau) = 1.0 - exp(-tau) * sum((tau^j)/factorial(j) for j in 0:(k-1))
```
At k=1: `1 - exp(-tau)`. `tau` is a RATE.

The reference ENGINE (`badger_ref/semi-markov_src.jl/transitions.jl:250`), which is
what `run_base_exp.jl` actually RUNS:
```julia
function erlang_cdf_at_1(k, theta)
    x_over_theta = 1.0 / theta          # <-- 1/theta
    ...
    return 1.0 - sum_val * exp(-1/theta)
end
prate(k,tau) = erlang_cdf_at_1(k, tau/k)
```
At k=1: `1 - exp(-1/tau)`. `tau` is a MEAN TIME.

The C++ agrees with the engine: `logProbEtoI = log(1 - exp(-1/tau))`
(`logPost_HMC.cpp:66`), `p12 = (1-prDeath)*(1-exp(-1/tau))`
(`iFFBS_fixedPars.cpp:115`).

**We copied the WRONG `erlang_cdf_at_1`.** `run_base_exp.jl` has an inline
top-level definition using `exp(-tau)`, but that function is a decoy — the code
path that actually runs uses `semi-markov_src.jl/transitions.jl`'s version with
`exp(-1/theta)`. We matched the decoy.

## Numerical confirmation

```
k=1, P(E->I):
  tau=0.0667  ours(rate)=0.0645   ref(mean-time)=1.0000
  tau=15.0    ours(rate)=1.0000   ref(mean-time)=0.0645
```
`ours(tau=0.0667) == ref(tau=15) == 0.0645`. Exact reciprocal.

## Why this drags beta and q too

Progression sets how fast E->I happens, which sets how many I animals the latent X
contains. Wrong progression -> wrong X -> the FOI counts (n_infectious) feeding
beta/q are wrong -> beta and q move to compensate. The FOI FORMULA is identical
across all three implementations (verified term-by-term: `a + b*I/(M/K)^q`,
`alpha_g = lambda*alpha[g]`, K=85, I counts I-only, M counts alive); so beta being
off ~1000x is NOT a FOI bug — it is beta absorbing the damage from a corrupted X.

## Also wrong: the prior

`tau ~ Exponential(10)` means "mean RATE 10" under our convention but should mean
"mean TIME 10" under the reference's. So even the prior pulled tau the wrong way.
The Gompertz variant already set tau ~ Exponential(100) to match the C++ Gamma(1,
scale=100) — that is the MEAN-TIME prior, correct ONLY IF the formula is also
mean-time. With our current rate-convention formula, that prior is doubly wrong.

## The fix

Change `erlang_cdf_at_1` to the mean-time convention (`exp(-1/theta)`), matching
the engine and the C++. One-line change to the formula; the prior then also needs
to be the mean-time one (Exp(10) or Exp(100) as intended, now meaning mean time).

Two things to decide:
1. Fix in the SILER model (badger_model.jl) — affects the existing default and
   the reststotal fit. This is a genuine bug fix, not a variant.
2. The Gompertz variant inherits `badger_progression` from badger_model.jl, so
   fixing it there fixes both.

NOT verified yet: whether flipping the convention brings beta/q back into line
too. It SHOULD (via a corrected X), but that needs a run to confirm. Do NOT assume
the fit matches until a converged run shows it.

## What was NOT the cause (ruled out)

- k/K swap: verified data.K=85.0, data.k=1 at runtime. Correct.
- FOI formula / K scaling / I,M counts: identical across all three.
- Randomised L: reverted; current code is fixed L=15. And L only affects mixing.
- thetas/rhos sharing an HMC block: affects mixing (ESS/rhat), never the posterior.
- Low ESS/rhat: SYMPTOMS of the mis-specified target, not causes.
