using EpidemicTrajectories
using Test
using Random
using StableRNGs: StableRNG
using Statistics: mean, std

# A test file NOT listed here silently does not run. Adding one is part of
# adding the feature, not an afterthought.
@testset "EpidemicTrajectories.jl" begin
    include("spec.jl")
    include("aggregates.jl")
    include("iffbs.jl")
    include("iffbs_mh.jl")
    include("residuals.jl")

    # Leave-future-out cross-validation.
    include("truncate.jl")
    include("lfo.jl")
    include("lfo_score.jl")
    include("lfo_run.jl")
    include("lfo_psis.jl")
    include("lfo_backend.jl")
    include("lfo_adapt.jl")

    # The example-level window scorer. A different thing from `lfo.jl` above,
    # which tests the package's LFO core; the two arrived on separate branches
    # under the same name.
    #
    # It reads `examples/lfo_common.jl`, which is NOT tracked, so it cannot run
    # from a clean checkout. Gated rather than deleted: the test is real, its
    # fixture simply has not been committed.
    if isfile(joinpath(@__DIR__, "..", "examples", "lfo_common.jl"))
        include("lfo_window_scoring.jl")
    else
        @info "skipping lfo_window_scoring.jl: examples/lfo_common.jl is not present"
    end
end
