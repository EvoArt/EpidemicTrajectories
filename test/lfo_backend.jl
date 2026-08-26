# Execution backends. No scheduler is available in CI, so these test the parts
# that are pure: the generated script, the array splitting, and the status logic.
#
# The gate that matters most is that completion is counted from RESULT FILES, not
# from a progress log -- a progress row is only written when a task reaches the
# END of its script, so startup failures leave no row and are invisible.

@testset "write_sbatch: the directives that matter are present" begin
    be = SlurmArray(partition = "hmq", cpus = 8, mem = "64G", time = "8:00:00",
                    exclude = ["node-bad", "node-worse"], project = "/proj")
    io = IOBuffer()
    write_sbatch(io, be, "/proj/my_lfo.jl", "/out", 0, 30)
    s = String(take!(io))

    @test occursin("#SBATCH --partition=hmq", s)
    @test occursin("#SBATCH --cpus-per-task=8", s)
    @test occursin("#SBATCH --mem=64G", s)
    @test occursin("#SBATCH --time=8:00:00", s)
    @test occursin("#SBATCH --exclude=node-bad,node-worse", s)   # the bad-node lever
    @test occursin("#SBATCH --array=0-29", s)                    # 30 tasks, 0-indexed
    @test occursin("my_lfo.jl", s)
    @test occursin("--project=/proj", s)
    # the worker learns which item it is from the environment, not from a closure
    @test occursin("LFO_WORK_ITEM", s)
    @test occursin("JULIA_NUM_THREADS", s)
end

@testset "write_sbatch: the offset shifts the item, not the array index" begin
    # Clusters cap the array INDEX, not the count: --array=99-197 is rejected
    # outright. So a second slice must still start at 0 and add an offset.
    be = SlurmArray(partition = "p")
    io = IOBuffer()
    write_sbatch(io, be, "s.jl", "/out", 99, 50)
    s = String(take!(io))
    @test occursin("#SBATCH --array=0-49", s)     # index starts at 0 ...
    @test occursin("99 + \${SLURM_ARRAY_TASK_ID}", s)   # ... offset does the rest
end

@testset "work_item reads the environment" begin
    haskey(ENV, "LFO_WORK_ITEM") && delete!(ENV, "LFO_WORK_ITEM")
    @test work_item() === nothing          # a launcher
    ENV["LFO_WORK_ITEM"] = "37"
    @test work_item() == 37                # a worker
    delete!(ENV, "LFO_WORK_ITEM")
end

@testset "sweep_status counts result FILES, not progress rows" begin
    dir = mktempdir()
    h = SweepHandle(String[], joinpath(dir, "manifest"), dir, 5, SlurmArray(partition = "p"))

    st = sweep_status(h)
    @test st.n_done == 0
    @test st.missing == collect(0:4)

    # three items finished (the content is irrelevant; existence is the signal)
    for i in (0, 2, 4)
        write(EpidemicTrajectories._item_file(dir, i), "result")
    end
    st2 = sweep_status(h)
    @test st2.n_done == 3
    @test st2.missing == [1, 3]            # exactly the ones with no output

    # A progress log claiming everything finished must NOT change the answer:
    # tasks that die at startup never write a row, so a log is evidence of
    # success only, never of failure.
    open(joinpath(dir, "progress.csv"), "w") do io
        for i in 0:4; println(io, "$i,0,done"); end
    end
    @test sweep_status(h).missing == [1, 3]
end

@testset "sweep_status: no scheduler is not an error" begin
    # On a laptop `sacct` does not exist. That must report "nothing known", not
    # crash, and must not be read as "no failures".
    dir = mktempdir()
    h = SweepHandle(["999999"], joinpath(dir, "m"), dir, 2, SlurmArray(partition = "p"))
    st = sweep_status(h)
    @test st.by_node isa Dict
    @test isempty(st.by_node)
end
