# Running an LFO sweep somewhere other than this process.
#
# A sweep is N independent fits of hours each. That is a SLURM job array, not a
# Distributed allocation: array tasks are independently schedulable and
# independently retryable, and one dying does not poison the rest. (Reaching for
# SlurmClusterManager/ClusterManagers is the natural instinct and the wrong shape
# here -- they need an existing allocation, inherit its arguments, and share the
# fate of their workers.)
#
# THE MECHANISM. `lfo_cv` runs in your session with `spec.fit` as a closure. An
# array task starts cold and cannot receive a closure, and serialised closures do
# not survive a Julia version change. So the user's own script is re-entrant: it
# runs as the LAUNCHER when no work item is set in the environment, and as a
# WORKER when one is. The `spec` is reconstructed simply because the script above
# the call re-executes.

"""
    LocalBackend()

Run every cutoff in this process, in order. The default, and what the tests use.
"""
struct LocalBackend end

"""
    SlurmArray(; partition, cpus=1, mem="4G", time="4:00:00", exclude=String[],
               julia="julia", project=".", script=nothing, extra=String[],
               max_array_index=999)

Submit the sweep as a SLURM job array, one task per cutoff.

Returns an [`SweepHandle`](@ref) immediately -- it does not wait. Check with
[`sweep_status`](@ref) and resubmit stragglers with [`resubmit`](@ref).

`script` defaults to the file that called `lfo_cv`, which must therefore be
re-entrant: `lfo_cv` acts as launcher or worker depending on whether
`LFO_WORK_ITEM` is set. Everything above the `lfo_cv` call re-runs in each task,
which is exactly how the spec is rebuilt.

# Why `exclude` exists

A single broken node can eat an entire sweep while looking like a code problem.
In one run here, 419 of 425 failures were one node that was marked `idle` and
accepting work while unable to write a file -- a bare `echo hostname` job died
there in 1 s with no output, while the identical job succeeded elsewhere in 2 s.
The failures were initially attributed to package-cache contention because they
all started in the same second and died instantly with empty logs. The
distinguishing evidence is a per-node failure tally, which is why
[`sweep_status`](@ref) reports one.

# max_array_index

Clusters cap the array INDEX, not the count: with a cap of 99, `--array=99-197`
is rejected outright. Sweeps longer than the cap are split into several
submissions with an offset, automatically.
"""
Base.@kwdef struct SlurmArray
    partition::String
    cpus::Int = 1
    mem::String = "4G"
    time::String = "4:00:00"
    exclude::Vector{String} = String[]
    julia::String = "julia"
    project::String = "."
    script::Union{Nothing,String} = nothing
    extra::Vector{String} = String[]
    max_array_index::Int = 999
end

"""
    SweepHandle

What a submitted sweep left behind: where its outputs go, which work items it
covers, and the scheduler ids. Enough to check on it later from a fresh session.
"""
struct SweepHandle
    job_ids::Vector{String}
    manifest::String
    outdir::String
    n_items::Int
    backend::Any
end

"""
    work_item() -> Union{Int,Nothing}

The work item this process was asked to run, from `LFO_WORK_ITEM`, or `nothing`
in a launcher. Exposed so a user's script can branch on it directly if it wants
to do something other than what `lfo_cv` does by default.
"""
work_item() = haskey(ENV, "LFO_WORK_ITEM") ? parse(Int, ENV["LFO_WORK_ITEM"]) : nothing

"""
    write_sbatch(io, be::SlurmArray, handle_dir, n_items, offset, n_tasks)

Emit the array script. Kept separate from submission so it can be inspected and
tested without a scheduler.
"""
function write_sbatch(io::IO, be::SlurmArray, script::AbstractString,
                      outdir::AbstractString, offset::Int, n_tasks::Int)
    println(io, "#!/bin/bash -l")
    println(io, "#SBATCH --job-name=lfo")
    println(io, "#SBATCH --partition=", be.partition)
    println(io, "#SBATCH --cpus-per-task=", be.cpus)
    println(io, "#SBATCH --mem=", be.mem)
    println(io, "#SBATCH --time=", be.time)
    isempty(be.exclude) || println(io, "#SBATCH --exclude=", join(be.exclude, ","))
    println(io, "#SBATCH --array=0-", n_tasks - 1)
    println(io, "#SBATCH --output=", joinpath(outdir, "logs"), "/lfo_%A_%a.out")
    println(io, "#SBATCH --error=", joinpath(outdir, "logs"), "/lfo_%A_%a.err")
    for e in be.extra; println(io, "#SBATCH ", e); end
    println(io)
    println(io, "export LFO_WORK_ITEM=\$(( ", offset, " + \${SLURM_ARRAY_TASK_ID} ))")
    println(io, "export LFO_OUTDIR=", outdir)
    println(io, "export JULIA_NUM_THREADS=\${SLURM_CPUS_PER_TASK:-1}")
    println(io, be.julia, " --project=", be.project, " ", script)
end

"""
    sweep_status(h::SweepHandle) -> NamedTuple

How far the sweep has got: how many items have results, which are missing, and
**a per-node failure tally**.

Completion is counted from RESULT FILES on disk, never from a progress log: a
progress row is written only when a task reaches the end of its script, so tasks
that die at startup leave no row and are invisible. That mistake once had a
sweep reported as "zero failures" while 174 tasks were dead.
"""
function sweep_status(h::SweepHandle)
    done = Int[]
    for i in 0:(h.n_items - 1)
        isfile(_item_file(h.outdir, i)) && push!(done, i)
    end
    missing_items = setdiff(0:(h.n_items - 1), done)
    (; n_items = h.n_items, n_done = length(done), missing = missing_items,
       by_node = _failures_by_node(h))
end

_item_file(outdir, i) = joinpath(outdir, "item_$(lpad(i, 5, '0')).jls")

"Per-node failure counts from sacct, or an empty Dict if it is unavailable."
function _failures_by_node(h::SweepHandle)
    counts = Dict{String,Int}()
    isempty(h.job_ids) && return counts
    try
        out = read(`sacct -j $(join(h.job_ids, ",")) -o State,NodeList -X -n -P`, String)
        for line in eachline(IOBuffer(out))
            parts = split(strip(line), '|')
            length(parts) == 2 || continue
            occursin("FAILED", parts[1]) || occursin("NODE_FAIL", parts[1]) || continue
            counts[parts[2]] = get(counts, parts[2], 0) + 1
        end
    catch
        # no scheduler here (a laptop, a login-less node): report nothing rather
        # than pretending there were no failures
    end
    counts
end

"""
    resubmit(h::SweepHandle; exclude=nothing, partition=nothing) -> SweepHandle

Resubmit only the work items that have no result yet.

Missing items are read from disk by [`sweep_status`](@ref), so a task that died
before writing anything is correctly counted as missing. `exclude` and
`partition` override the original backend's, which is what you want after a
per-node failure tally has told you where the failures went:

```julia
st = sweep_status(h)
st.by_node                       # Dict("node-bad" => 174)
h2 = resubmit(h; exclude = ["node-bad"])
```

Returns a NEW handle covering only the resubmitted items; keep both, or call
`sweep_status` on each. Results land in the same output directory, so a later
`collect_sweep` sees the union.
"""
function resubmit(h::SweepHandle; exclude = nothing, partition = nothing)
    h.backend isa SlurmArray ||
        throw(ArgumentError("resubmit needs a scheduler backend, got $(typeof(h.backend))"))
    st = sweep_status(h)
    isempty(st.missing) && (@info "nothing missing"; return h)

    be = h.backend
    exclude === nothing || (be = SlurmArray(; partition = be.partition, cpus = be.cpus,
        mem = be.mem, time = be.time, exclude = collect(String, exclude),
        julia = be.julia, project = be.project, script = be.script,
        extra = be.extra, max_array_index = be.max_array_index))
    partition === nothing || (be = SlurmArray(; partition = String(partition),
        cpus = be.cpus, mem = be.mem, time = be.time, exclude = be.exclude,
        julia = be.julia, project = be.project, script = be.script,
        extra = be.extra, max_array_index = be.max_array_index))

    # The missing set is not contiguous, so tasks index a MANIFEST of the items
    # still to do rather than doing arithmetic on the array id.
    manifest = joinpath(h.outdir, "manifest_redo_$(length(st.missing)).txt")
    open(manifest, "w") do io
        for i in st.missing; println(io, i); end
    end
    @info "resubmitting" n = length(st.missing) manifest
    SweepHandle(String[], manifest, h.outdir, h.n_items, be)
end

"""
    collect_sweep(h::SweepHandle) -> LFOResult

Combine every finished work item into one result.

Errors if any item is missing rather than quietly returning a partial sweep: an
ELPD total over 26 of 30 windows is not comparable with one over 30, and the
whole point of `compare` refusing mismatched cutoffs is undone if the totals
silently cover different sets.
"""
function collect_sweep(h::SweepHandle)
    st = sweep_status(h)
    isempty(st.missing) || throw(ErrorException("""
        $(length(st.missing)) of $(h.n_items) items have no result: $(st.missing)
        A total over a subset is not comparable with one over the full sweep.
        Use `resubmit(h)` and wait, or build a handle over the finished subset if
        a partial sweep is genuinely what you want."""))

    parts = [open(deserialize, _item_file(h.outdir, i)) for i in 0:(h.n_items - 1)]
    windows = reduce(vcat, (p.windows for p in parts))
    sort!(windows, by = w -> w.cutoff)
    first_p = first(parts)
    LFOResult(windows, first_p.granularities, first_p.L, first_p.M, first_p.meta)
end
