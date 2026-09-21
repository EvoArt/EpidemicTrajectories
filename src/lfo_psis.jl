# Pareto-smoothed importance sampling for leave-future-out reuse.
#
# Refitting at every cutoff is expensive, so Buerkner, Gabry & Vehtari (2020)
# reuse a fit from an earlier cutoff `l < t` and correct the mismatch by
# importance sampling, with ratio proportional to p(y_{l+1:t} | x_l, theta).
# Ratios are Pareto-smoothed, giving stabilised weights and a k-hat; when
# k-hat > tau the approximation is unreliable and the model is refit.
#
# Two things worth knowing before using this to compare granularities. The
# ratio is a joint density over all individuals, so it reweights draws toward
# those that predict the cohort well jointly, which is exactly what joint
# scoring rewards: verdicts flipped in ~21% of cells moving PSIS -> exact, so
# exact refitting is the comparison of record whenever granularities are being
# compared. And a "pointwise ratio" is not an available fix, because the score
# is a modelling choice while the ratio is pinned by a density identity.

"""
    psis_smooth(logr; tail_frac=0.2) -> (logw, khat)

Pareto-smoothed importance weights from log ratios `logr`.

Returns log weights (unnormalised) and the Pareto shape `khat`. Rules of thumb:
`khat < 0.5` is fine, `0.5-0.7` is usable, `> 0.7` means refit.

# The degenerate case, which once looked like a finding

When most ratios are `-Inf`: common when forward simulation kills an individual
the data prove was alive — a naive implementation takes the tail cutoff over all
ratios, gets `-Inf`, and every exceedance becomes `-Inf - (-Inf) = NaN`. Every
weight is then `NaN`, so the window scores `-Inf` regardless of granularity.

That did not look like a bug. It looked like the substantive result "the naive
proposal collapses every window", and was written up as one before being caught.
The tell was a total of exactly `0.00` with every window non-finite, a fully
collapsed run summing to zero, which then *beat* every finite competitor.

Here: only the finite ratios are smoothed, and `khat = Inf` is returned when
there are too few of them, which correctly forces a refit.
"""
function psis_smooth(logr::AbstractVector{<:Real}; tail_frac::Real = 0.2)
    S = length(logr)
    S == 0 && throw(ArgumentError("no ratios"))
    finite_idx = findall(isfinite, logr)
    # Too few usable draws to estimate a tail: flag rather than fabricate.
    if length(finite_idx) < 5
        w = fill(-Inf, S)
        for i in finite_idx; w[i] = logr[i]; end
        return (w, Inf)
    end

    lr = Float64.(logr[finite_idx])
    n = length(lr)
    M = max(3, min(n - 1, ceil(Int, tail_frac * n)))
    ord = sortperm(lr)
    tail_pos = ord[(n - M + 1):n]
    cutoff = lr[ord[n - M]]

    exceed = [lr[i] - cutoff for i in tail_pos]
    k, sigma = _gpd_fit(exp.(exceed) .- 1.0)

    smoothed = copy(lr)
    if isfinite(k) && sigma > 0
        for (j, i) in enumerate(tail_pos)
            p = (j - 0.5) / M
            q = k == 0 ? -sigma * log1p(-p) : sigma * expm1(-k * log1p(-p)) / k
            smoothed[i] = min(cutoff + log1p(q), maximum(lr))
        end
    end

    w = fill(-Inf, S)
    for (j, i) in enumerate(finite_idx); w[i] = smoothed[j]; end
    (w, isfinite(k) ? k : Inf)
end

"""
    _gpd_fit(x) -> (k, sigma)

Generalised-Pareto shape and scale by the Zhang & Stephens (2009) profile method.

Returns `(Inf, 0.0)` for a degenerate sample rather than a NaN: a NaN here
propagates into every weight and silently voids the window.
"""
function _gpd_fit(x::AbstractVector{<:Real})
    n = length(x)
    n < 3 && return (Inf, 0.0)
    xs = sort(x)
    (xs[end] <= 0 || !all(isfinite, xs)) && return (Inf, 0.0)
    prior = 3
    m = 30 + floor(Int, sqrt(n))
    b = [1 - sqrt(m / (j - 0.5)) for j in 1:m] ./ (prior * xs[max(1, n ÷ 4)]) .+ 1 / xs[end]
    ls = similar(b)
    @inbounds for j in eachindex(b)
        k = -sum(log1p.(-b[j] .* xs)) / n
        ls[j] = n * (log(b[j] / k) + k - 1)
    end
    w = exp.(ls .- maximum(ls))
    w ./= sum(w)
    bhat = sum(w .* b)
    # Sign matters here. Zhang & Stephens profile over b = -k/sigma, so the mean of
    # log1p(-b*x) estimates -k, and khat is its negative. Getting this backwards
    # is silent: sigma still comes out right (it is k/b, and both flip together),
    # weights are still finite and plausible, and only the diagnostic is
    # inverted -- so a heavy tail reports as healthy and no refit is triggered.
    # Caught by fitting samples of known shape; asserted in the tests.
    khat = sum(log1p.(-bhat .* xs)) / n
    sigmahat = -khat / bhat
    (isfinite(khat) ? khat : Inf, isfinite(sigmahat) ? max(sigmahat, 0.0) : 0.0)
end

"""
    psis_ess(logw) -> Float64

Effective sample size of self-normalised weights. Reported per window because
the `-Inf` collapse is the endpoint of weight concentration, not a separate
pathology: on one study median ESS fell 600 -> 250 -> 22 from pointwise to
group to joint scoring, and joint was also the arm that lost windows.
"""
function psis_ess(logw::AbstractVector{<:Real})
    any(isfinite, logw) || return 0.0
    m = maximum(filter(isfinite, logw))
    w = [isfinite(x) ? exp(x - m) : 0.0 for x in logw]
    s = sum(w)
    s <= 0 && return 0.0
    sum(w)^2 / sum(abs2, w)
end
