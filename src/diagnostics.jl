# The checks and plots that turn a `SummaryResult` into a verdict.
#
# These are STUBS. The implementations live in weak-dependency extensions, so that
# neither `HypothesisTests` nor `Makie` enters the package's core dependency list —
# the core stays at five (four plus the `Serialization` stdlib), and a user who
# only wants residual VALUES pays for nothing else.
#
# Load the extension by loading its package:
#
#     using HypothesisTests   # enables uniformity_test, pvalue_distribution, pi_05
#     using CairoMakie        # enables residual_plot, pvalue_plot, residual_qq
#
# Calling one of these without its package loaded gives a clear error saying which
# one to load, rather than a `MethodError` about an undefined function.

"""
    uniformity_test(result, name; per_draw=true) -> NamedTuple

Test whether a `:pit` summary's residuals are Uniform(0,1) — the question a PIT
residual exists to answer. Requires `using HypothesisTests`.

Returns `(; statistic, pvalue, n, n_draws)`.

`per_draw=true` (the default) runs the Anderson–Darling test **within each draw**
and reports the MEDIAN p-value across draws. That is the correct treatment:
residuals from the same individual at different draws are correlated, so pooling
them inflates the effective sample size and makes the test anticonservative — it
will reject a model that is fine. Pass `per_draw=false` to pool anyway, knowing
that.

For the fuller picture use [`pvalue_distribution`](@ref) and [`pi_05`](@ref),
which report the whole distribution rather than collapsing it to one number.

A `:raw` summary is not a PIT residual and this will refuse to test it.
"""
function uniformity_test end

"""
    pvalue_distribution(result, name) -> Vector{Float64}

The Anderson–Darling p-value for each draw separately. Requires
`using HypothesisTests`.

**Under a correctly specified model this vector is itself Uniform(0,1)**, which is
a far stronger check than any single p-value: a residual that is subtly
miscalibrated passes one test by luck but cannot produce a uniform spread of
p-values across draws. It is also what [`pi_05`](@ref) summarises, and what
[`pvalue_plot`](@ref) draws.

Draws with fewer than three usable residuals are skipped, so the result may be
shorter than `result.n_draws`.
"""
function pvalue_distribution end

"""
    pi_05(result, name) -> Float64

π₀.₀₅: the posterior probability that a draw's uniformity test rejects at the 5%
level — i.e. the fraction of draws whose p-value falls below 0.05. Requires
`using HypothesisTests`.

**Read it against 0.05, not against 0.** A correctly specified model gives ≈0.05
by construction, because that is what a 5% test does. Values materially above 0.05
indicate misfit; values near 0 suggest the residual is overdispersed or that the
test has no power here, which is its own kind of problem.

This is the reference implementation's headline calibration statistic
(`compute_pi_05`), and the single number worth quoting for a fitted model.
"""
function pi_05 end

"""
    residual_plot(result, name; kwargs...) -> Figure

A histogram of a summary's residuals, with the Uniform(0,1) reference line a
correctly specified model should follow. Requires a Makie backend
(`using CairoMakie`).

Also available as `residual_plot!(ax, result, name)` to draw into an existing
axis, and as `residual_plot(values)` for a bare vector.

Attributes: `bins` (default 20), `color`, `reference` (draw the uniform line,
default `true`), `normalization` (default `:pdf`, so the reference line is at 1).
"""
function residual_plot end

"""
    residual_plot!(ax, result, name; kwargs...)

In-place [`residual_plot`](@ref), drawing into an existing axis.
"""
function residual_plot! end

"""
    pvalue_plot(result, name; kwargs...) -> Figure

A histogram of the per-draw uniformity p-values (see [`pvalue_distribution`](@ref)),
with the Uniform(0,1) reference and the 0.05 threshold marked. Requires both
`using HypothesisTests` and a Makie backend.

This is the plot that says whether the model fits: under a correct model the bars
are flat, and about 5% of the mass sits left of the marked threshold. A spike
against zero is misfit.
"""
function pvalue_plot end

"""
    pvalue_plot!(ax, result, name; kwargs...)

In-place [`pvalue_plot`](@ref).
"""
function pvalue_plot! end

"""
    residual_qq(result, name; kwargs...) -> Figure

A QQ plot of a summary's residuals against Uniform(0,1). Requires a Makie backend.

Complements [`residual_plot`](@ref): a histogram shows WHERE the mass is, a QQ plot
shows how far the whole distribution departs and in which direction — residuals
bowing above the diagonal mean the modelled waiting times are too long, below means
too short. That directional reading is what localises a misspecification to a
component rather than merely announcing one.

Also available as `residual_qq!(ax, ...)`.
"""
function residual_qq end

"""
    residual_qq!(ax, result, name; kwargs...)

In-place [`residual_qq`](@ref).
"""
function residual_qq! end

"""
    residual_panel(result; kwargs...) -> Figure

One figure summarising every `:pit` summary in a result: a residual histogram and
a p-value histogram per summary, one row each. Requires both `using HypothesisTests`
and a Makie backend.

This is the "did my model fit?" view — the convenience layer over the individual
plots, for when you want the answer rather than a specific panel. `:raw` summaries
are skipped, since uniformity is not a meaningful question to ask of them.
"""
function residual_panel end

# A helper the extensions share: the values a uniformity test should see, with the
# `:raw` guard applied. Kept in the core so both extensions agree on it, and so the
# refusal to test a `:raw` summary is stated once.
function _pit_values(r::SummaryResult, name::Symbol)
    haskey(r, name) || error("no summary named :$name (have $(keys(r)))")
    r.kinds[name] === :pit || error(
        "summary :$name has kind :$(r.kinds[name]), not :pit — uniformity is not a " *
        "meaningful question to ask of a non-PIT quantity. Declare it as :pit only " *
        "if it really is a probability-integral-transform residual.")
    return residual_values(r, name)
end
