# Stage 1 — NAIVE. The least code: let the package do everything.
#
#   * aggregates via `@aggregate` (the package allocates + maintains them),
#   * coupling via the package DEFAULT `make_rest_contribution` — the brute-force
#     O(n_states × |affected|) counterfactual recompute (no `rest_contribution`,
#     no `coupled_transitions` given),
#   * observation likelihood via the vector-returning `observation_process` only
#     (no scalar `observation_weight`),
#   * one HMC block + conjugate etas/nu + iFFBS(X).
#
# This is the "just describe the model" baseline. Every later stage removes one of
# these package conveniences in favour of a hand-written shortcut and we measure
# what each buys. Model definitions are shared in badger_common.jl.

include(joinpath(@__DIR__, "badger_common.jl"))

function build(dir)
    d = load_badger_data(dir)
    data = epidemic_data(;
        n_individuals=d.n_individuals, n_timepoints=d.n_timepoints,
        trans_mat=badger_transitions(),
        starting_state=badger_starting_state,
        observation_process=badger_observations,
        aggregates=badger_aggregates(d.n_groups, d.n_timepoints),
        sampling_period=d.sampling_period,
        affected_individuals=badger_affected_lists(d),
        # No `coupled_transitions` and no custom `rest_contribution`: the default
        # coupling assumes every transition is coupled and recomputes it per (i,t).
        state_space=BADGER_STATES,
        social_group=d.social_group, age=d.age, capture=d.capture,
        capt_effort=d.capt_effort, tests=d.tests, season=d.season,
        birth_time=d.birth_time, last_capture_time=d.last_capture_time,
        nu_times=d.nu_times, K=Float64(d.K), k=d.k, n_groups=d.n_groups)
    (; data, raw=d)
end

const STAGE = "naive"
const (data, raw) = build(DATA_DIR())

# Both halves of the likelihood; the obs half uses the vector process (slow path).
const loglik = epidemic_loglik(data; entry_time=raw.first_capture_time,
                               survival=siler_survival)
const obs_loglik = epidemic_obs_loglik(data; observation_process=badger_obs_tests)

if get(ENV, "BADGER_BENCH", "0") == "1"
    time_stage(STAGE, data, raw, loglik, obs_loglik)
elseif get(ENV, "BADGER_NO_RUN", "0") != "1"
    n = parse(Int, get(ENV, "BADGER_SWEEPS", "50"))
    println("[$STAGE] ", data.n_individuals, " badgers x ", data.n_timepoints, " t")
    chn, elapsed = run_badger_fit(data, raw, loglik, obs_loglik; n_sweeps=n)
    report(chn)
end
