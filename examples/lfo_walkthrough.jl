# Leave-future-out cross-validation, end to end, on a small SID model.
#
# Runnable as-is:  julia --project=. examples/lfo_walkthrough.jl
#
# It is also the template for a cluster sweep: the SAME file works unchanged as a
# SLURM array job, because `lfo_cv` acts as launcher or worker depending on
# whether a work item is set in the environment. Nothing below needs to change to
# scale from a laptop to 240 fits -- only the `backend` argument at the end.

using EpidemicTrajectories
using Random

# ─────────────────────────────────────────────────────────────────────────
# 1. A model. Three states, one absorbing, so mortality is in play -- which is
#    where the granularity axis earns its keep.
# ─────────────────────────────────────────────────────────────────────────
const N_IND, N_T, N_GROUPS = 60, 40, 4
state_space = [:S, :I, :D]

aggs = @aggregate state_space begin
    @array n_infected Int (N_GROUPS, N_T)
    n_infected[data.group[i], t] += (state == :I)
end

spec = @transitions state_space begin
    S -> I = (m, d, i, t) ->
        -expm1(-(m.α + m.β * d.aggregates[:n_infected][d.group[i], t]))
    S -> D = (m, d, i, t) -> m.μ
    I -> S = (m, d, i, t) -> 1 / m.recover
    I -> D = (m, d, i, t) -> m.μ_I
end

group = repeat(1:N_GROUPS; inner = N_IND ÷ N_GROUPS)
truth = (; α = 0.01, β = 0.02, recover = 5.0, μ = 0.005, μ_I = 0.02)

# `last_seen` is the field that makes this interesting: it records the last time
# each individual was KNOWN present, and a model's observation weight typically
# forbids the absorbing state at or before it. Left untruncated it leaks the
# future into every training fit.
# `last_seen` is filled in AFTER simulating, from the trajectory itself: an
# individual is known present up to the step before it dies. Setting it to N_T
# for everyone would declare every individual present to the end, so every
# simulated death would be a contradiction and every window would be -Inf --
# a broken fixture rather than a broken estimator, and worth knowing the
# difference before reading anything into an -Inf.
data0 = epidemic_data(;
    n_individuals = N_IND, n_timepoints = N_T, group = group,
    trans_mat = spec, aggregates = aggs,
    starting_state = (m, d, X, i, t) -> [0.9, 0.1, 0.0],
    last_seen = fill(N_T, N_IND),
    sex = repeat([1, 2], N_IND ÷ 2))

X_true = epidemic_simulator(data0)(MersenneTwister(1), truth)
last_seen = [something(findfirst(t -> X_true[t, i] == 3, 1:N_T), N_T + 1) - 1
             for i in 1:N_IND]

data = epidemic_data(;
    n_individuals = N_IND, n_timepoints = N_T, group = group,
    trans_mat = spec, aggregates = aggs,
    starting_state = (m, d, X, i, t) -> [0.9, 0.1, 0.0],
    last_seen = last_seen,
    sex = repeat([1, 2], N_IND ÷ 2))

println("simulated: ", count(==(3), X_true), " dead cells of ", length(X_true),
        "; ", count(<(N_T), last_seen), " of ", N_IND, " individuals die")

# ─────────────────────────────────────────────────────────────────────────
# 2. DECLARE how each extra behaves under truncation.
#
#    This is the step that prevents the failure that is hardest to detect.
#    Shortening `sampling_period` is NOT enough: any extra carrying post-cutoff
#    information leaks it into the training fit, inflating the score of whichever
#    model exploits it, and nothing about the result looks wrong.
#
#    An undeclared time-shaped extra is an ERROR. Try deleting the `keep` line
#    below to see what that looks like.
# ─────────────────────────────────────────────────────────────────────────
plan = truncation(
    clamp = (:last_seen,),   # min.(v, cutoff): no "known present" past the cutoff
    keep  = (:sex,),         # a covariate; an explicit "I checked, it is timeless"
)

println("extras that go uninformative past the cutoff: ",
        flat_likelihood_range(plan))

# ─────────────────────────────────────────────────────────────────────────
# 3. What LFO needs from you: how to FIT, and how to score ONE cell.
# ─────────────────────────────────────────────────────────────────────────

# A stand-in for MCMC. A real one returns (draws, X_draws) where draws[s] is
# whatever `cell_logdensity` wants as its model, and X_draws[s] is that draw's
# latent trajectory over the whole series.
function my_fit(train_data, cutoff)
    n_draws = 20
    rng = MersenneTwister(100 + cutoff)
    draws = [(; α = truth.α * exp(0.1randn(rng)), β = truth.β * exp(0.1randn(rng)),
               recover = truth.recover, μ = truth.μ, μ_I = truth.μ_I)
             for _ in 1:n_draws]
    (draws, [copy(X_true) for _ in 1:n_draws])
end

# The log density of ONE individual's observation at ONE time, given the
# simulated state. Return -Inf for an inadmissible trajectory: it is charged to
# that individual's own cell, never to the whole window.
function my_density(model, d, X, i, t)
    state = X[t, i]
    state == 3 && t <= d.last_seen[i] && return -Inf   # dead when known present
    state == 2 ? log(0.7) : log(0.9)
end

# Which cells actually depend on the latent state. STRONGLY worth supplying: on
# one real dataset only ~2% of cells carried an observation at all, the rest
# contributing a constant identical under every granularity -- so the granularity
# comparison was near-null and nobody noticed until this was measured.
my_informative(d, i, t) = true

# ─────────────────────────────────────────────────────────────────────────
# 4. Run it.
# ─────────────────────────────────────────────────────────────────────────
lfo = LFOSpec(fit = my_fit, cell_logdensity = my_density,
              is_informative = my_informative, plan = plan, n_sim = 4)

res = lfo_cv(lfo, data; L = 20, M = 2,
             granularity = (Pointwise(), ByGroup(group), Joint()),
             cache = mktempdir(), verbose = false)

println("\n", res)
println("informative cells: ", n_informative(res),
        "  (if this is a small fraction of the total, no granularity can")
println("                     discriminate much and the comparison is near-null)")

# ─────────────────────────────────────────────────────────────────────────
# 5. Compare two models.
#
#    `compare` refuses the comparisons that are never valid: across
#    granularities (a pointwise total sums N*M densities per window, a joint
#    total sums one), across different M, and across unshared cutoffs.
# ─────────────────────────────────────────────────────────────────────────
worse = LFOSpec(fit = my_fit, plan = plan, n_sim = 4,
                cell_logdensity = (m, d, X, i, t) ->
                    (X[t, i] == 3 && t <= d.last_seen[i]) ? -Inf : log(0.5))
res_worse = lfo_cv(worse, data; L = 20, M = 2,
                   granularity = (Pointwise(), ByGroup(group), Joint()),
                   verbose = false)

for g in (:pointwise, :by_group, :joint)
    c = compare(res, res_worse; granularity = g)
    println("  $(rpad(g, 10)) diff $(round(c.diff, digits = 2))  " *
            "se $(round(c.se, digits = 2))  " *
            "se_indep $(round(c.se_indep, digits = 2))  (n=$(c.n_windows))")
end
println("""
  `se` treats windows as independent; with M > 1 they overlap, so it is
  optimistic. `se_indep` uses every M-th window and is the conservative read.""")

# ─────────────────────────────────────────────────────────────────────────
# 6. The same script on a cluster.
#
#    Uncomment. On the first run there is no work item in the environment, so
#    `lfo_cv` submits the array and returns a handle. Each task then re-runs THIS
#    FILE with LFO_WORK_ITEM set and does exactly one cutoff -- which is why the
#    spec never has to be serialised: everything above rebuilds it.
# ─────────────────────────────────────────────────────────────────────────
#
#   handle = lfo_cv(lfo, data; L = 20, M = 2,
#                   granularity = (Pointwise(), Joint()),
#                   backend = SlurmArray(partition = "hmq", cpus = 8,
#                                        mem = "64G", time = "8:00:00",
#                                        exclude = ["a-known-bad-node"]))
#
#   st = sweep_status(handle)     # counts RESULT FILES, never a progress log:
#   st.by_node                    # a task that dies at startup writes no row.
#   handle2 = resubmit(handle; exclude = ["whatever by_node blamed"])
#   final  = collect_sweep(handle)   # errors if anything is still missing
