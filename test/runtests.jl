using EpidemicTrajectories
using Test
using Random
using StableRNGs: StableRNG
using Statistics: mean, std

@testset "EpidemicTrajectories.jl" begin
    include("spec.jl")
    include("aggregates.jl")
    include("iffbs.jl")
    include("iffbs_mh.jl")
    include("lfo.jl")
    include("residuals.jl")
end
