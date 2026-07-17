# Set up the badger iFFBS sweep for interactive profiling, then get out of the way.
#
# Run from the repo root:
#     julia --project=examples
#     julia> include("examples/profile_iffbs_repl.jl")
#     julia> @profview sweep()          # flamegraph
#
# Leaves `data`, `X0`, `sweep_pars`, `latent!` and `sweep()` in Main to poke at.
#
# Bar WIDTH in the flamegraph is TOTAL time (self + callees), which is the number
# that matters here: a self-time table makes this sweep look flat and hides that
# rest_contribution owns ~70% of it through what it calls.

ENV["BADGER_RUN"] = "0"          # badger_fit.jl auto-runs the fit unless told not to
include(joinpath(@__DIR__, "badger_fit.jl"))

using ProfileView                # exports @profview
using StableRNGs: StableRNG
using ProfileToLLM: profile_table, print_profile   # text table, for copy/paste

init0 = badger_initial_params(raw; rng=StableRNG(13))
X0 = copy(raw.X_init)

# The rate functions read `model.nu` (the combined S/E/I mixing matrix), not the
# separate nuE/nuI badger_initial_params returns — same merge badger_fit.jl's own
# `_pars(c)` does.
sweep_pars = merge(Base.structdiff(init0, (; nuE=0, nuI=0)), (; nu=hcat(init0.nuE, init0.nuI)))

# The aggregates must agree with X before the sampler runs; it maintains the
# invariant from there.
reset_aggregates!(data)
apply_derived_summaries!(sweep_pars, data, X0)

latent! = epidemic_latent_sampler(data)
rng = StableRNG(1)

"""
    sweep(n=1)

`n` iFFBS sweeps over all 2384 badgers. A function rather than a bare loop so
`@profview sweep()` profiles compiled code, not the REPL's global-scope wrapper.
"""
sweep(n::Int=1) = (for _ in 1:n; latent!(rng, sweep_pars, X0); end; nothing)

println("warming up (compiles the whole iFFBS call graph — never profile this one)...")
sweep()

t = @elapsed sweep()
println("one warm iFFBS sweep: ", round(t, digits=3), " s   (reference: ~1.47 s)")

println("""

Ready.

  @profview sweep(2)        # flamegraph — click a bar for its stack frame,
                            # right-click to zoom, ctrl-click to descend
  @profview sweep(2)        # run it twice; the first opens a cold window

Text tables (same data, no window):

  print_profile(profile_table(Profile.fetch(); sortby=:total); max_rows=25)   # who OWNS the time
  print_profile(profile_table(Profile.fetch(); sortby=:self);  max_rows=25)   # where cycles land

What to look for — the open question is why our sweep is ~2.6-2.9 s against the
reference's ~1.47 s. Current evidence (total%): iffbs_individual! 98 ->
forward_filter 84 -> rest_contribution 69 -> transition_prob 66. So the coupling
term dominates: for each (i, t) it tries all 4 candidate states and, for each,
loops over every affected neighbour calling transition_prob. The reference does
not do that loop at all — it keeps logProbRest[s, j, t] / logProbRestTotal[s, t]
and patches them incrementally per individual (updaters.jl:290), so its coupling
cost is an O(1) array read.
""")
