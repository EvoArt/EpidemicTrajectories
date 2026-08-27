module EpidemicTrajectoriesHypothesisTestsExt

# The uniformity checks. In an extension rather than the core so that
# `HypothesisTests` (and its own dependency tree) never enters the package's
# dependency list — a user who only wants residual VALUES pays nothing for these.

using HypothesisTests: OneSampleADTest, pvalue
using Distributions: Uniform

using EpidemicTrajectories
using EpidemicTrajectories: SummaryResult, _pit_values, draw_values

import EpidemicTrajectories: uniformity_test, pvalue_distribution, pi_05

# `mean` and `median` written out rather than pulled from `Statistics`. An
# extension may only use packages the parent declares, and adding Statistics to the
# core for two one-line functions is a worse trade than these six lines.
_mean(x) = sum(x) / length(x)   # over a concrete vector only, never a generator
function _median(x)
    y = sort(collect(x))
    n = length(y)
    isodd(n) ? y[(n + 1) ÷ 2] : (y[n ÷ 2] + y[n ÷ 2 + 1]) / 2
end

# Anderson–Darling against Uniform(0,1). AD rather than KS because it weights the
# TAILS, which is where a miscalibrated waiting-time residual shows up first: a
# censoring or clock-origin error piles residuals against 0 or 1 while leaving the
# middle of the distribution looking perfectly reasonable.
#
# Returns `(NaN, NaN)` rather than throwing when there is nothing to test — a draw
# with two usable residuals is not a failure, it is a draw with no information,
# and the caller filters those out.
function _ad_uniform(values)
    clean = filter(x -> isfinite(x) && 0.0 < x < 1.0, values)
    length(clean) < 3 && return (NaN, NaN)
    try
        ad = OneSampleADTest(clean, Uniform(0, 1))
        return (ad.A², pvalue(ad))
    catch
        # AD can fail on degenerate input (every value identical, say). That is a
        # property of the sample, not an error worth propagating.
        return (NaN, NaN)
    end
end

function uniformity_test(r::SummaryResult, name::Symbol; per_draw::Bool=true)
    _pit_values(r, name)          # the :pit guard, and the "does it exist" check

    if !per_draw
        # Pooled: correct only if you know what you are giving up. Residuals from
        # one individual across draws are correlated, so this understates the
        # p-value and over-rejects.
        v = _pit_values(r, name)
        stat, p = _ad_uniform(v)
        return (; statistic=stat, pvalue=p, n=length(v), n_draws=r.n_draws)
    end

    ps = pvalue_distribution(r, name)
    stats = Float64[]
    for k in 1:r.n_draws
        s, _ = _ad_uniform(draw_values(r, name, k))
        isnan(s) || push!(stats, s)
    end
    isempty(ps) && return (; statistic=NaN, pvalue=NaN, n=0, n_draws=0)

    # The MEDIAN across draws, not the mean: a single pathological draw should not
    # move the verdict, and the p-value distribution is skewed by construction.
    return (; statistic=_median(stats), pvalue=_median(ps),
              n=length(draw_values(r, name, 1)), n_draws=length(ps))
end

function pvalue_distribution(r::SummaryResult, name::Symbol)
    _pit_values(r, name)
    ps = Float64[]
    for k in 1:r.n_draws
        _, p = _ad_uniform(draw_values(r, name, k))
        isnan(p) || push!(ps, p)
    end
    return ps
end

function pi_05(r::SummaryResult, name::Symbol)
    ps = pvalue_distribution(r, name)
    isempty(ps) && return NaN
    return count(p -> p < 0.05, ps) / length(ps)
end

end # module
