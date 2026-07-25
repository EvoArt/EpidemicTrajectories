function inspect(path)
    for (i, l) in enumerate(readlines(path))
        any(c -> c == Char(0x3f) || codepoint(c) > 127, l) || continue
        codes = [Int(codepoint(c)) for c in l]
        println("line $i: $codes")
    end
end
inspect("examples/badger_fit_siler_fixed_hmc_5000.jl")
