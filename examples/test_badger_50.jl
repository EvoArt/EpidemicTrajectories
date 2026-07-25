s = replace(read("examples/badger_fit_siler_fixed_hmc_5000.jl", String), "run_badger_fit(5000" => "run_badger_fit(50")
Base.include_string(Main, s)
