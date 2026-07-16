using EpidemicTrajectories
using Test
using Random
using StableRNGs: StableRNG
using Statistics: mean

@testset "EpidemicTrajectories.jl" begin
    include("spec.jl")
    include("aggregates.jl")
    include("iffbs.jl")
end
