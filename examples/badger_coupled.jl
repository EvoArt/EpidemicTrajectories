# Stage 2 — + coupled_transitions. One line more than the naive stage.
#
# The only effect one badger has on another is contributing to its force of
# infection, i.e. only a neighbour's S→E move is influenced by the focal. Declaring
# `coupled_transitions=[(:S, :E)]` lets the sampler skip any neighbour whose realised
# move the focal cannot affect — exact, not an approximation (a skipped move has the
# same probability under every candidate focal state and cancels on normalisation).
# CLAUDE.md measures this at ~10x on the badger coupling.
#
# Everything else is identical to badger_naive.jl.

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
        coupled_transitions=[(:S, :E)],          # <-- the only change vs naive
        state_space=BADGER_STATES,
        social_group=d.social_group, age=d.age, capture=d.capture,
        capt_effort=d.capt_effort, tests=d.tests, season=d.season,
        birth_time=d.birth_time, last_capture_time=d.last_capture_time,
        nu_times=d.nu_times, K=Float64(d.K), k=d.k, n_groups=d.n_groups)
    (; data, raw=d)
end

const STAGE = "coupled"
const (data, raw) = build(DATA_DIR())

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
