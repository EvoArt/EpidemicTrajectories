function inspect(path, outpath)
    open(outpath, "w") do io
        for (i, l) in enumerate(readlines(path))
            any(c -> codepoint(c) > 127, l) || continue
            println(io, "line ", i, ":")
            println(io, l)
            println(io, "codes: ", [Int(codepoint(c)) for c in l])
            println(io)
        end
    end
end
inspect("examples/badger_fit_siler_fixed_hmc_5000.jl", "examples/nonascii.txt")
