# Adaptation diagnostics for samplers running inside a truncated LFO window.
#
# WHY THIS EXISTS. Truncation clamps the fields that window a likelihood, so any
# parameter scored over `sampling_period` (or over any other clamped field) has a
# FLAT likelihood beyond the cutoff: proposal and current state score identically
# and the move is accepted on the prior alone.
#
# A random-walk adapter targeting ~0.234 then sees ~100% acceptance and widens
# forever. Measured on a real fit, a changepoint bounded in 1:160 reached a
# proposal sd of 939 over 5000 sweeps while never dropping below 0.77
# acceptance. At that width almost every proposal is out of range and discarded
# as a no-op -- and because out-of-range proposals were not counted as
# REJECTIONS, the adapter never learned. The parameter was frozen while
# reporting healthy acceptance.
#
# That reads as healthy in every summary. It needs an explicit check, which is
# what this file provides.

"""
    AdaptTrace(name; target_lo=0.39, target_hi=0.49)

Records a random-walk kernel's acceptance and proposal scale so
[`check_adaptation`](@ref) can say whether it converged or ran away.

Push one observation per adaptation interval with [`record!`](@ref). The default
target window is the one used by the badger changepoint kernel; pass your own if
your adapter targets something else (0.234 is the usual random-walk optimum).
"""
mutable struct AdaptTrace
    name::Symbol
    target_lo::Float64
    target_hi::Float64
    sweeps::Vector{Int}
    acceptance::Vector{Float64}
    scale::Vector{Float64}
    n_out_of_range::Int
    n_proposed::Int
end

AdaptTrace(name::Symbol; target_lo = 0.39, target_hi = 0.49) =
    AdaptTrace(name, target_lo, target_hi, Int[], Float64[], Float64[], 0, 0)

"""
    record!(tr, sweep, acceptance, scale; n_out_of_range=0, n_proposed=0)

Add one adaptation-interval observation.

`n_out_of_range` matters more than it looks: a proposal rejected for being
outside the parameter's support is a REJECTION, not a no-op. Kernels that skip
it without counting it hide exactly the signal an adapter needs.
"""
function record!(tr::AdaptTrace, sweep::Int, acceptance::Real, scale::Real;
                 n_out_of_range::Int = 0, n_proposed::Int = 0)
    push!(tr.sweeps, sweep)
    push!(tr.acceptance, Float64(acceptance))
    push!(tr.scale, Float64(scale))
    tr.n_out_of_range += n_out_of_range
    tr.n_proposed += n_proposed
    tr
end

"""
    check_adaptation(tr; scale_growth=10.0, support=nothing) -> NamedTuple

Did this kernel's adaptation converge, or run away?

Returns `(; ok, warnings, final_acceptance, final_scale, scale_ratio)`. Each
warning names a specific pathology rather than saying "looks odd":

- **acceptance never fell to target** — the signature of a flat likelihood. Under
  LFO this usually means the parameter is being scored over a window that
  truncation emptied, so proposals beyond the cutoff are decided by the prior.
- **proposal scale grew by more than `scale_growth`x** — an adapter that never
  found a rejection to push back against.
- **proposal scale exceeds the support** — at that width nearly every proposal is
  illegal, so the chain is effectively frozen wherever it happens to sit.
- **out-of-range proposals not counted as rejections** — reported whenever they
  are a large share of proposals, because that is what lets the previous two
  pathologies persist unnoticed.

This is a heuristic, not a proof: a genuinely well-identified parameter can sit
near the top of its target window. It is meant to make a silent failure loud.
"""
function check_adaptation(tr::AdaptTrace; scale_growth::Real = 10.0,
                          support::Union{Nothing,Tuple{Real,Real}} = nothing)
    warnings = String[]
    isempty(tr.acceptance) && return (; ok = true, warnings, final_acceptance = NaN,
                                      final_scale = NaN, scale_ratio = NaN)

    fa, fs = last(tr.acceptance), last(tr.scale)
    ratio = first(tr.scale) > 0 ? fs / first(tr.scale) : Inf

    if minimum(tr.acceptance) > tr.target_hi
        push!(warnings, "$(tr.name): acceptance never fell to the target window " *
            "(min $(round(minimum(tr.acceptance), digits=3)) > $(tr.target_hi)). " *
            "Under LFO this usually means a FLAT likelihood: the parameter is " *
            "scored over a window that truncation emptied, so moves beyond the " *
            "cutoff are accepted on the prior alone.")
    end
    if ratio > scale_growth
        push!(warnings, "$(tr.name): proposal scale grew $(round(ratio, digits=1))x " *
            "($(round(first(tr.scale), digits=2)) -> $(round(fs, digits=2))) — " *
            "the adapter never met a rejection to push back against.")
    end
    if support !== nothing
        width = support[2] - support[1]
        if fs > width
            push!(warnings, "$(tr.name): proposal scale $(round(fs, digits=1)) " *
                "exceeds the support width $(width) — nearly every proposal is " *
                "illegal, so the chain is effectively frozen.")
        end
    end
    if tr.n_proposed > 0 && tr.n_out_of_range / tr.n_proposed > 0.5
        push!(warnings, "$(tr.name): $(tr.n_out_of_range) of $(tr.n_proposed) " *
            "proposals were out of range. Count these as REJECTIONS, not no-ops, " *
            "or the adapter cannot see them.")
    end

    (; ok = isempty(warnings), warnings, final_acceptance = fa,
       final_scale = fs, scale_ratio = ratio)
end

"""
    flat_likelihood_range(data, plan, cutoff) -> Vector{Symbol}

Which declared extras become UNINFORMATIVE beyond `cutoff` once truncated.

A `clamp`ed or `filter`ed extra carries no information past the cutoff by
construction — that is the point of truncating it — so any likelihood windowed by
one of them is flat out there. Listing them makes the consequence visible before
a sweep runs, rather than after a 2-hour fit produces a diverged adapter.
"""
function flat_likelihood_range(plan::TruncationPlan)
    [n for n in propertynames(plan.rules)
       if getproperty(plan.rules, n) isa Union{Clamp,Filter}]
end
