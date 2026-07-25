# Stage 3 — + scalar observation_weight. Targets the GRADIENT, not iFFBS.
#
# The observation likelihood only ever needs one entry of the weight vector —
# `w[X[t,i]]`, the weight of the state the badger is actually in. The vector-
# returning `observation_process` (stages 1-2) allocates a whole array per (i,t)
# and throws all but one element away: on badgers ~187k arrays of Duals per
# gradient call. `observation_weight` returns just that one entry. CLAUDE.md
# measures the obs-term gradient going 0.226 → 0.075 s (3.0x), 18 MB → 16 bytes.
#
# `epidemic_data` and the coupling are unchanged from badger_coupled.jl; only the
# `obs_loglik` gains the scalar path.

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
        coupled_transitions=[(:S, :E)],
        state_space=BADGER_STATES,
        social_group=d.social_group, age=d.age, capture=d.capture,
        capt_effort=d.capt_effort, tests=d.tests, season=d.season,
        birth_time=d.birth_time, last_capture_time=d.last_capture_time,
        nu_times=d.nu_times, K=Float64(d.K), k=d.k, n_groups=d.n_groups)
    (; data, raw=d)
end

const STAGE = "obsweight"
const (data, raw) = build(DATA_DIR())

const loglik = epidemic_loglik(data; entry_time=raw.first_capture_time,
                               survival=siler_survival)
# The scalar `observation_weight` is the only change vs badger_coupled.jl.
const obs_loglik = epidemic_obs_loglik(data;
                                 observation_process=badger_obs_tests,
                                 observation_weight=badger_obs_tests_weight)

if get(ENV, "BADGER_BENCH", "0") == "1"
    time_stage(STAGE, data, raw, loglik, obs_loglik)
elseif get(ENV, "BADGER_NO_RUN", "0") != "1"
    n = parse(Int, get(ENV, "BADGER_SWEEPS", "50"))
    println("[$STAGE] ", data.n_individuals, " badgers x ", data.n_timepoints, " t")
    chn, elapsed = run_badger_fit(data, raw, loglik, obs_loglik; n_sweeps=n)
    report(chn)
end
