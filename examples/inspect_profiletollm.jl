using ProfileToLLM

open(joinpath(@__DIR__, "profile_llm_names.txt"), "w") do io
    for n in names(ProfileToLLM)
        println(io, n)
    end
end

open(joinpath(@__DIR__, "profile_llm_methods.txt"), "w") do io
    for m in methods(ProfileToLLM.profile_to_llm)
        println(io, m)
    end
end
