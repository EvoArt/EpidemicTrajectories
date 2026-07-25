# Drive the staged benchmark: run each stage script in a FRESH Julia process (so
# their module-level `const data` etc. don't collide), collect the STAGE_TIMING
# line each prints, and tabulate how iFFBS and full-sweep times move stage to stage.
#
#   julia --project=. bench_stages.jl
#
# Each stage is timed by `time_stage` in badger_common.jl (one iFFBS sweep at fixed
# params; one full Gibbs sweep averaged over a few reps after warm-up).

const STAGES = ["naive", "coupled", "obsweight", "reststotal"]
const HERE = @__DIR__

results = Tuple{String,Float64,Float64}[]
for stage in STAGES
    script = joinpath(HERE, "badger_$(stage).jl")
    println("\n===== running stage: $stage =====")
    # Inherit the caller's env (so BADGER_NFULL propagates) and add BADGER_BENCH.
    env = copy(ENV); env["BADGER_BENCH"] = "1"
    out = read(setenv(`$(Base.julia_cmd()) --project=$(HERE) $script`, env), String)
    line = only(filter(l -> startswith(l, "STAGE_TIMING"), split(out, '\n')))
    fields = Dict(split(kv, '=')[1] => split(kv, '=')[2] for kv in split(line)[2:end])
    push!(results, (stage, parse(Float64, fields["ffbs"]), parse(Float64, fields["full"])))
    println("  iFFBS=$(results[end][2]) s/sweep   full=$(results[end][3]) s/sweep")
end

println("\n\n########## staged optimization: s/sweep ##########")
println(rpad("stage", 14), rpad("iFFBS", 12), rpad("full", 12),
        rpad("iFFBS speedup", 15), "full speedup   (vs naive)")
base_f, base_full = results[1][2], results[1][3]
for (stage, tf, tfull) in results
    println(rpad(stage, 14), rpad(round(tf, digits=3), 12), rpad(round(tfull, digits=3), 12),
            rpad(string(round(base_f / tf, digits=2), "x"), 15),
            round(base_full / tfull, digits=2), "x")
end
