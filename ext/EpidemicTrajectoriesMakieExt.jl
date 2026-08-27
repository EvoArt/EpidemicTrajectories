module EpidemicTrajectoriesMakieExt

# The residual plots. In an extension so Makie never enters the core dependency
# list; load any Makie backend (CairoMakie, GLMakie) to enable them.
#
# Three plots, and they answer different questions:
#
#   residual_plot — a histogram of the residuals. Says WHERE the mass is: a pile
#     against 0 or 1 is the signature of a wrong waiting-time distribution.
#   residual_qq   — the same residuals against the Uniform(0,1) diagonal. Says how
#     far the whole distribution departs and IN WHICH DIRECTION, which is what
#     localises a misspecification to a component.
#   pvalue_plot   — the per-draw uniformity p-values. Says whether the departure is
#     big enough to matter, accounting for how many individuals there are.
#
# All three take a `SummaryResult` and a summary name, or a bare vector.

using Makie

using EpidemicTrajectories
using EpidemicTrajectories: SummaryResult, residual_values, _pit_values

import EpidemicTrajectories: residual_plot, residual_plot!, pvalue_plot, pvalue_plot!,
                             residual_qq, residual_qq!, residual_panel,
                             pvalue_distribution, pi_05

# =============================================================================
# The recipes
# =============================================================================

# A histogram of PIT residuals with the Uniform(0,1) density as a reference. The
# reference is at height 1 because the histogram is a :pdf — that is the whole
# reason for the normalization default, and changing it without moving the
# reference line would make a correct model look wrong.
Makie.@recipe(ResidualHist, values) do scene
    Makie.Theme(;
        bins = 20,
        color = Makie.RGBAf(0.25, 0.45, 0.75, 0.85),
        strokecolor = :white,
        strokewidth = 1,
        reference = true,
        referencecolor = Makie.RGBAf(0.8, 0.2, 0.2, 0.9),
        referencewidth = 2,
        referencestyle = :dash,
    )
end

function Makie.plot!(p::ResidualHist)
    v = p[:values]

    # Bin edges pinned to [0, 1] rather than to the data's own range. Both
    # quantities plotted here — PIT residuals and p-values — live on [0,1] by
    # construction, so the axis is meaningful independently of the sample, and
    # comparing two panels side by side requires the same bins in each.
    #
    # It also fixes a failure that hits exactly when the plot matters most: when a
    # model is badly misspecified every p-value collapses to ~1e-6, data-derived
    # edges put them all in one hair-thin bin, and `:pdf` sends that bin's density
    # to ~1e8 — a panel with an unreadable axis and no visible bars, precisely at
    # rejection. Fixed edges make that case a single bar hard against zero, which
    # is what it should look like.
    edges = Makie.lift(p.bins) do nb
        range(0, 1; length = nb + 1)
    end

    Makie.hist!(p, v;
                bins = edges,
                normalization = :pdf,
                color = p.color,
                strokecolor = p.strokecolor,
                strokewidth = p.strokewidth)

    if p.reference[]
        # Uniform(0,1) has density 1 everywhere on [0,1]. Flat bars at this line
        # is what a correctly specified model looks like.
        Makie.hlines!(p, [1.0];
                      color = p.referencecolor,
                      linewidth = p.referencewidth,
                      linestyle = p.referencestyle)
    end
    return p
end

# A QQ plot of residuals against Uniform(0,1). The diagonal is the reference; the
# direction of departure from it is the diagnostic content.
Makie.@recipe(ResidualQQ, values) do scene
    Makie.Theme(;
        color = Makie.RGBAf(0.25, 0.45, 0.75, 0.7),
        markersize = 5,
        referencecolor = Makie.RGBAf(0.8, 0.2, 0.2, 0.9),
        referencewidth = 2,
    )
end

function Makie.plot!(p::ResidualQQ)
    pts = Makie.lift(p[:values]) do v
        y = sort(collect(v))
        n = length(y)
        # Plotting positions (k - 1/2)/n: the standard choice, and unlike k/n it
        # does not force the largest observation onto the boundary, which would put
        # a spurious point at exactly 1 in every plot.
        x = [(k - 0.5) / n for k in 1:n]
        Makie.Point2f.(x, y)
    end

    Makie.lines!(p, [Makie.Point2f(0, 0), Makie.Point2f(1, 1)];
                 color = p.referencecolor, linewidth = p.referencewidth)
    Makie.scatter!(p, pts; color = p.color, markersize = p.markersize)
    return p
end

# =============================================================================
# The user-facing entry points
# =============================================================================

# Accept either a bare vector or a (result, name) pair everywhere, so a user with
# residuals from somewhere else is not locked out of the plots.
_values(v::AbstractVector) = collect(float.(v))
_values(r::SummaryResult, name::Symbol) = _pit_values(r, name)

function residual_plot!(ax, v::AbstractVector; kwargs...)
    residualhist!(ax, _values(v); kwargs...)
end
residual_plot!(ax, r::SummaryResult, name::Symbol; kwargs...) =
    residual_plot!(ax, _values(r, name); kwargs...)

function residual_plot(v::AbstractVector; figure=(;), axis=(;), kwargs...)
    fig = Makie.Figure(; figure...)
    ax = Makie.Axis(fig[1, 1];
                    xlabel = "PIT residual", ylabel = "density",
                    title = "Residuals vs Uniform(0,1)",
                    axis...)
    residual_plot!(ax, v; kwargs...)
    Makie.xlims!(ax, 0, 1)
    return fig
end

function residual_plot(r::SummaryResult, name::Symbol; figure=(;), axis=(;), kwargs...)
    v = _values(r, name)
    cov = round(100 * r.coverage[name]; digits=1)
    fig = residual_plot(v;
        figure,
        axis = (; title = "$name — $(length(v)) residuals, $(cov)% coverage", axis...),
        kwargs...)
    return fig
end

function residual_qq!(ax, v::AbstractVector; kwargs...)
    residualqq!(ax, _values(v); kwargs...)
end
residual_qq!(ax, r::SummaryResult, name::Symbol; kwargs...) =
    residual_qq!(ax, _values(r, name); kwargs...)

function residual_qq(v::AbstractVector; figure=(;), axis=(;), kwargs...)
    fig = Makie.Figure(; figure...)
    ax = Makie.Axis(fig[1, 1];
                    xlabel = "theoretical quantile", ylabel = "observed quantile",
                    title = "QQ vs Uniform(0,1)", aspect = 1,
                    axis...)
    residual_qq!(ax, v; kwargs...)
    Makie.xlims!(ax, 0, 1); Makie.ylims!(ax, 0, 1)
    return fig
end

residual_qq(r::SummaryResult, name::Symbol; figure=(;), axis=(;), kwargs...) =
    residual_qq(_values(r, name); figure,
                axis = (; title = "$name — QQ vs Uniform(0,1)", axis...), kwargs...)

function pvalue_plot!(ax, ps::AbstractVector; threshold=0.05, kwargs...)
    residualhist!(ax, _values(ps); kwargs...)
    # The 5% line: under a correct model about 5% of the mass sits left of it, so
    # this is what π₀.₀₅ is reading off. Marking it makes the plot self-explaining.
    Makie.vlines!(ax, [threshold]; color = (:black, 0.6), linewidth = 1.5, linestyle = :dot)
    return ax
end

function pvalue_plot!(ax, r::SummaryResult, name::Symbol; kwargs...)
    ps = _require_pvalues(r, name)
    pvalue_plot!(ax, ps; kwargs...)
end

# `pvalue_distribution` lives in the HYPOTHESISTESTS extension, not this one, so a
# user with only Makie loaded would otherwise hit a bare MethodError on a stub and
# have no idea which package to add. Say it plainly instead. (The plots that need
# only residual VALUES — residual_plot, residual_qq — work with Makie alone; it is
# specifically the p-value ones that need both.)
function _require_pvalues(r::SummaryResult, name::Symbol)
    ps = try
        pvalue_distribution(r, name)
    catch err
        err isa MethodError && error(
            "this plot needs the per-draw uniformity p-values, which come from the " *
            "HypothesisTests extension. Run `using HypothesisTests` as well as your " *
            "Makie backend. (residual_plot and residual_qq need Makie only.)")
        rethrow()
    end
    isempty(ps) && error(
        "no draw of :$name had at least 3 usable residuals, so there is nothing to test")
    return ps
end

function pvalue_plot(ps::AbstractVector; figure=(;), axis=(;), kwargs...)
    fig = Makie.Figure(; figure...)
    ax = Makie.Axis(fig[1, 1];
                    xlabel = "uniformity p-value", ylabel = "density",
                    title = "p-values across draws", axis...)
    pvalue_plot!(ax, ps; kwargs...)
    Makie.xlims!(ax, 0, 1)
    return fig
end

function pvalue_plot(r::SummaryResult, name::Symbol; figure=(;), axis=(;), kwargs...)
    ps = _require_pvalues(r, name)
    p05 = round(pi_05(r, name); digits=3)
    return pvalue_plot(ps; figure,
        axis = (; title = "$name — p-values, π₀.₀₅ = $p05 (target 0.05)", axis...),
        kwargs...)
end

# =============================================================================
# The panel — the "did my model fit?" view
# =============================================================================

function residual_panel(r::SummaryResult; figure=(;), kwargs...)
    # Only :pit summaries: uniformity is not a meaningful question to ask of a
    # :raw quantity, so plotting one against a Uniform reference would be
    # actively misleading rather than merely unhelpful.
    pit_names = [n for n in keys(r) if r.kinds[n] === :pit]
    isempty(pit_names) && error(
        "residual_panel: no :pit summaries in this result (only $(keys(r))). " *
        "A :raw summary is not a PIT residual, so there is no uniformity to check.")

    fig = Makie.Figure(; size = (1000, 320 * length(pit_names)), figure...)

    for (row, name) in enumerate(pit_names)
        v = residual_values(r, name)
        cov = round(100 * r.coverage[name]; digits=1)

        ax1 = Makie.Axis(fig[row, 1];
                         xlabel = "PIT residual", ylabel = "density",
                         title = "$name — residuals ($(length(v)), $(cov)% coverage)")
        residual_plot!(ax1, v; kwargs...)
        Makie.xlims!(ax1, 0, 1)

        ax2 = Makie.Axis(fig[row, 2];
                         xlabel = "theoretical quantile", ylabel = "observed",
                         title = "$name — QQ", aspect = 1)
        residual_qq!(ax2, v)
        Makie.xlims!(ax2, 0, 1); Makie.ylims!(ax2, 0, 1)

        ps = _require_pvalues(r, name)
        ax3 = Makie.Axis(fig[row, 3];
                         xlabel = "uniformity p-value", ylabel = "density",
                         title = "$name — π₀.₀₅ = $(round(pi_05(r, name); digits=3))")
        pvalue_plot!(ax3, ps)
        Makie.xlims!(ax3, 0, 1)
    end

    return fig
end

end # module
