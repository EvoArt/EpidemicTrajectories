using EpidemicTrajectories
using Test
using Random
using StableRNGs: StableRNG
using Statistics: mean, std, var
using Serialization: Serialization

# A test file NOT listed here silently does not run. Adding one is part of
# adding the feature, not an afterthought.
@testset "EpidemicTrajectories.jl" begin
    include("spec.jl")
    include("aggregates.jl")
    include("iffbs.jl")
    include("iffbs_mh.jl")
    include("residuals.jl")

    # Exact marginalisation for independent individuals.
    include("hmm.jl")

    # Weighted chains with sampled latents.
    include("latent_hmm.jl")

end
