# Is the badger iFFBS sweep an EXACT Gibbs step?
#
# `iffbs!` is exact only when the chain the forward filter runs is the chain the
# likelihood scores. `iffbs_mh_design.md` §1.3 predicted two ways the badger model
# might break that. This script measures both on the real data.
#
#   (a) the `entry_time` gate -- `epidemic_loglik(data; entry_time, survival)`
#       divides the survival factor out of every pre-entry step while the FILTER
#       still multiplies it in.
#
#   (b) the window-end coupling term -- `forward_filter!` calls
#       `rest_contribution` at the focal's LAST timepoint, where it scores each
#       neighbour's `t -> t+1` move. When that neighbour's own window ends at `t`,
#       the joint contains no such factor and `X[t+1, j]` is a cell iFFBS never
#       writes.
#
# ANSWERS (2026-08-22, 2384 badgers x 161 timepoints):
#
#   (a) DOES NOT FIRE. Identical ratios with the gate on and off, always. The
#       prediction in the design doc holds: `@survival` applies one
#       non-state-dependent factor, so it cancels on normalisation for every path
#       that stays alive, and the badger observation model gives the dead state
#       zero weight before last capture so no other path is reachable.
#
#   (b) FIRES, on 11 of 1974 badgers, worst |log alpha| = 0.586. All of them are
#       animals BORN MID-STUDY, whose group contains other mid-study animals whose
#       windows end early. Switching the conditional to `neighbor_window=:filter`
#       (i.e. adopting the filter's own convention) collapses the discrepancy to
#       3.3e-8, which localises it exactly. For the worst offender the filter
#       scores precisely 2 extra `(j, t)` pairs, both at `t == last_t(j)`, both
#       reading an `X[t+1, j]` outside `j`'s window.
#
#       `:likelihood` is the CORRECT rule: it reproduces the joint's delta to
#       1e-11 (see the delta-consistency section below). So the FILTER is what
#       needs fixing, not the target. Left unfixed here on purpose -- it changes
#       what `iffbs!` computes and so needs its own measured commit.
#
# Run:  julia --project=examples examples/badger_check_iffbs_exact.jl

ENV["BADGER_NO_RUN"] = "1"          # build the model, do not fit it
include(joinpath(@__DIR__, "badger_naive.jl"))

using Statistics: median, quantile
using EpidemicTrajectories: transition_prob

println()
println("="^74)
println("badger check_iffbs_exact — ", data.n_individuals, " badgers x ",
        data.n_timepoints, " timepoints")
println("="^74)

# ---------------------------------------------------------------------------
# PARAMETERS. `time_stage`'s benchmark placeholders (a2=3.0, b2=1.7, c1=0.45) are
# NOT usable here: they make the Siler survival underflow to ~1e-20 by age 5, so
# `survival * infection` lands on `transition_matrix_at!`'s 1e-12 clamp for 15.7%
# of all scored transitions. A clamped probability is CONSTANT, so those steps
# stop responding to the focal at all and every comparison below is measuring the
# clamp rather than the model. (That is the same survival-annihilation failure the
# survival-free `coupling` view of `@survival` exists to prevent -- and note which
# way round it falls: the coupling view stays responsive while the full
# `trans_mat` goes flat.)
#
# Use a survival curve that stays inside the representable range instead.
# ---------------------------------------------------------------------------
let G = raw.n_groups, NT = raw.n_tests, NS = raw.n_seasons, NNU = raw.n_nu_times
    global pars = (; tau=5.0, alpha=fill(0.05, G), lambda=0.5, beta=0.3, q=0.2,
                     a1=0.1, b1=0.5, a2=1e-4, b2=0.03, c1=0.01,
                     thetas=fill(0.3, NT), rhos=fill(0.5, NT), phis=fill(0.9, NT),
                     etas=fill(0.3, NS), nu=fill(0.05, NNU, 2))
end
const ET = raw.first_capture_time

X = Matrix{Int}(raw.X_init)
reset_aggregates!(data); apply_derived_summaries!(pars, data, X)
println("loglik = ", loglik(pars, data, X),
        "   obs_loglik = ", obs_loglik(pars, data, X))

let n = Ref(0), tot = Ref(0)
    for i in 1:data.n_individuals
        ft, lt = data.sampling_period[i]
        for t in ft:(min(lt, data.n_timepoints) - 1)
            p = transition_prob(data.trans_mat, pars, data, X, i, t, X[t, i], X[t+1, i])
            tot[] += 1
            p <= 1.0000001e-12 && (n[] += 1)
        end
    end
    println("transitions pinned on the 1e-12 clamp: ", n[], "/", tot[],
            "  (", round(100 * n[] / tot[]; digits=3), "%)  <- must be ~0")
end

# ---------------------------------------------------------------------------
# BURN IN FIRST -- a prerequisite of the diagnostic, not of the model.
#
# Run against `raw.X_init` directly and EVERY ratio is +Inf, correctly.
# `badger_obs_capture` writes a hard structural zero into the dead state's weight
# while a badger is known alive, so many cells of the reference's `Xinit` are
# unreachable under the filter: `log q(current) = -Inf`, for which accepting
# unconditionally is the right MH response. Measured: 2377/2384 initial paths are
# unreachable; after ONE `iffbs!` sweep, 0/2384 are.
# ---------------------------------------------------------------------------
for _ in 1:3
    iffbs!(pars, data, X, StableRNG(1))
end
println("burned in with 3 iffbs! sweeps")

# ---------------------------------------------------------------------------
# The four combinations. `entry_time` on/off isolates (a); `neighbor_window`
# isolates (b).
# ---------------------------------------------------------------------------
variants = [
    ("entry gate ON,  :likelihood  (the true target)",
     (; entry_time=ET, survival=siler_survival)),
    ("entry gate OFF, :likelihood",
     (;)),
    ("entry gate ON,  :filter",
     (; entry_time=ET, survival=siler_survival, neighbor_window=:filter)),
    ("entry gate OFF, :filter",
     (; neighbor_window=:filter)),
]

reports = Dict{String,Any}()
for (label, kw) in variants
    Xc = copy(X)
    reset_aggregates!(data); apply_derived_summaries!(pars, data, Xc)
    rep = check_iffbs_exact(pars, data, Xc; rng=StableRNG(20260822), n_sweeps=1,
                            target=epidemic_conditional_loglik(data; kw...))
    reports[label] = rep
    println()
    println(label)
    println("  max |log alpha|   : ", rep.max_abs_logratio)
    println("  decisions checked : ", rep.n_checked, "  (identical: ", rep.n_identical, ")")
    println("  offenders > 1e-8  : ", length(rep.offenders))
end

println()
println("Who are the offenders under the TRUE target?")
let rep = reports["entry gate ON,  :likelihood  (the true target)"]
    for (i, lr) in rep.offenders
        ft, lt = data.sampling_period[i]
        println("  i=", lpad(i, 4), "  log alpha=", lpad(round(lr; digits=6), 10),
                "  window=(", ft, ",", lt, ")",
                "  birth=", data.birth_time[i],
                "  born mid-study: ", data.birth_time[i] == ft)
    end
end

# ---------------------------------------------------------------------------
# Localise (b): which (j, t) does the FILTER score that the JOINT does not?
# ---------------------------------------------------------------------------
let rep = reports["entry gate ON,  :likelihood  (the true target)"]
    isempty(rep.offenders) && return
    i = rep.worst_individual
    ft, lt = data.sampling_period[i]
    extra = Tuple{Int,Int,Int,Int}[]
    for t in ft:min(lt, data.n_timepoints - 1)
        for j in data.affected_individuals[t, i]
            j == i && continue
            fj, lj = data.sampling_period[j]
            (fj <= t && t <= min(lj, data.n_timepoints) - 1) ||
                push!(extra, (j, t, fj, lj))
        end
    end
    println()
    println("worst offender i=", i, " window=(", ft, ",", lt, ")")
    println("  (j,t) the FILTER scores but epidemic_loglik does not: ", length(extra))
    for (j, t, fj, lj) in extra
        println("     j=", j, " t=", t, "  j's window=(", fj, ",", lj, ")",
                t == min(lj, data.n_timepoints) ? "   <- t == last_t(j)" : "",
                t + 1 > min(lj, data.n_timepoints) ?
                    "   X[t+1,j] is never written by iFFBS" : "")
    end
end

# ---------------------------------------------------------------------------
# The arbiter: the conditional must be the JOINT's restriction. This decides
# which neighbour-window rule is right, and it is the check worth copying into
# any model's own tests.
#
# Note which observation term to use. The badger `obs_loglik` is the TEST factor
# only, because the capture factor's parameters (`etas`) are drawn conjugately.
# But the full conditional of X contains every term that depends on X, and the
# capture factor does. So the comparison uses `epidemic_obs_loglik(data)`, the
# FULL observation likelihood -- which is also what the conditional defaults to.
# ---------------------------------------------------------------------------
println()
println("delta-consistency: conditional vs (loglik + FULL obs_loglik)")
let full_obs = epidemic_obs_loglik(data),
    t_lik = epidemic_conditional_loglik(data; entry_time=ET, survival=siler_survival),
    t_flt = epidemic_conditional_loglik(data; entry_time=ET, survival=siler_survival,
                                        neighbor_window=:filter)

    rng = StableRNG(7)
    Xd = copy(X)
    w_lik = Ref(0.0); w_flt = Ref(0.0)
    for _ in 1:10
        i = rand(rng, 1:data.n_individuals)
        ft, lt = data.sampling_period[i]
        reset_aggregates!(data); apply_derived_summaries!(pars, data, Xd)
        j0 = loglik(pars, data, Xd) + full_obs(pars, data, Xd)
        a0, b0 = t_lik(pars, data, Xd, i), t_flt(pars, data, Xd, i)
        saved = copy(Xd[ft:lt, i])
        for t in ft:lt
            Xd[t, i] = rand(rng, 1:data.n_states)
        end
        reset_aggregates!(data); apply_derived_summaries!(pars, data, Xd)
        j1 = loglik(pars, data, Xd) + full_obs(pars, data, Xd)
        a1, b1 = t_lik(pars, data, Xd, i), t_flt(pars, data, Xd, i)
        if isfinite(j0) && isfinite(j1)
            dj = j1 - j0
            w_lik[] = max(w_lik[], abs((a1 - a0) - dj))
            w_flt[] = max(w_flt[], abs((b1 - b0) - dj))
        end
        Xd[ft:lt, i] = saved
    end
    println("  neighbor_window = :likelihood  worst error ", w_lik[], "   <- correct")
    println("  neighbor_window = :filter      worst error ", w_flt[])
end

println()
println("="^74)
println("""
CONCLUSION (2026-08-22)

  (a) The `entry_time` gate does NOT break exactness. Identical ratios with the
      gate on and off, at every parameter set tried.

  (b) The window-end coupling term DOES, on 11 of 1974 badgers (worst
      |log alpha| = 0.586, all of them born mid-study). The filter scores a
      neighbour's `t -> t+1` move at `t == last_t(j)`, which the joint does not
      contain and which reads an `X[t+1, j]` iFFBS never writes.

      `neighbor_window=:filter` reproduces the filter exactly (3.3e-8), so the
      diagnosis is certain; and `:likelihood` reproduces the JOINT exactly
      (~1e-11), so the FILTER is the side that is wrong. The fix belongs in
      `make_rest_contribution` and changes what `iffbs!` computes, so it needs its
      own measured commit. Until then `iffbs_mh!` corrects it.

  Also worth carrying forward: `time_stage`'s benchmark parameters make the Siler
  survival underflow so badly that 15.7% of scored transitions sit on the 1e-12
  clamp. Every comparison run at those values measures the clamp, not the model.
""")
println("="^74)
