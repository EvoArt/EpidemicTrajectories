using EpidemicTrajectories
using Test
using Random
using StableRNGs: StableRNG
using Statistics: mean

@testset "EpidemicTrajectories.jl" begin
    include("spec.jl")
    include("aggregates.jl")
    include("iffbs.jl")
    include("truncate.jl")
    include("lfo.jl")
    include("lfo_score.jl")
    include("lfo_run.jl")
    include("lfo_psis.jl")
end
