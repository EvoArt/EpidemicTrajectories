using EpidemicTrajectories
using Test
using Random
using StableRNGs: StableRNG
using Statistics: mean

@testset "EpidemicTrajectories.jl" begin
    include("compartments.jl")
    include("functional_style.jl")
    include("transition_matrix_style.jl")
    include("consistency.jl")
end
