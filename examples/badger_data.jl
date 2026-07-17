# Loading and reshaping the badger data for EpidemicTrajectories.
#
# All of this is user code: reading the CSVs, filtering, and converting the
# reference's conventions into ours. The package has no idea any of it exists.
#
# Two conventions differ from the reference and are converted here:
#   * the reference is individual-major (`X[i, t]`); we are time-major (`X[t, i]`)
#   * the reference encodes states sparsely (S=0, E=3, I=1, D=9); we use the
#     position in `state_space`, so S=1, E=2, I=3, D=4
#
# The Brock changepoint is applied statically (see BROCK_CHANGEPOINT below).

using CSV
using DataFrames

# Our state space. The order fixes the encoding: S=1, E=2, I=3, D=4.
const BADGER_STATES = [:S, :E, :I, :D]

# The reference's sparse state codes, mapped to our positions.
const REF_STATE_CODE = Dict(0 => 1, 3 => 2, 1 => 3, 9 => 4)

# When the Brock test switched from version 1 to version 2.
#
# The reference treats this as unknown, starting at `xi = 80` and updating it by
# RWMH. Crucially, the raw `TestMat` columns are already correct AT `xi = 80` —
# the reference reads them as-is and only ever *swaps* the two Brock columns for
# tests lying between the current and proposed changepoint
# (`TestMatAsFieldProposal`). So fixing the changepoint at 101 is not a matter of
# choosing which column applies at each time: it means taking the raw data and
# applying that same swap once, for the tests in `[80, 101)`.
const BROCK_CHANGEPOINT_RAW = 80        # what the raw TestMat columns encode
const BROCK_CHANGEPOINT = 101           # what we fix it to

"""
    load_badger_data(dir; brock_changepoint=BROCK_CHANGEPOINT, known_sex_only=true)

Read the badger CSVs from `dir` and reshape them into this package's conventions.

Returns a NamedTuple with, among others:
- `n_individuals`, `n_timepoints`, `n_groups`, `n_tests`
- `X_init` — the reference's starting trajectory, `[t, i]`, in our state encoding
- `social_group` — `[i, t]`, the group each individual is in at each time (it
  moves between groups, so this genuinely varies with `t`; 0 means "not present")
- `age` — `[i, t]`, age in timesteps (`-10` before birth)
- `capture` — `[t, i]`, 1 if captured
- `tests` — `[t, i, test]`, the six test results (`-1` = not tested)
- `sampling_period` — `(first, last)` per individual
- `birth_time`, `last_capture_time`, `season`, `nu_times`, `sex`, `K`, `k`

`known_sex_only` filters to badgers of known sex, as the reference does — the base
model has no sex effects, but it keeps the dataset comparable with the sex models.
"""
function load_badger_data(dir; brock_changepoint=BROCK_CHANGEPOINT, known_sex_only=true)
    rd(f) = CSV.read(joinpath(dir, f), DataFrame)

    dims = rd("dimensions.csv")
    m, maxt = dims[1, :m], dims[1, :maxt]
    n_groups, n_tests = dims[1, :G], dims[1, :numTests]
    n_seasons, n_nu_times = dims[1, :numSeasons], dims[1, :numNuTimes]

    Xinit_raw = Matrix(rd("Xinit.csv"))                 # [i, t], sparse codes
    test_mat = Matrix{Float64}(rd("TestMat.csv"))       # long: time, id, group, 6 tests
    capt_hist = Matrix{Int}(rd("CaptHist.csv"))         # [i, t]
    birth_time = vec(Matrix{Int}(rd("birthTimes.csv")))
    start_period = vec(Matrix{Int}(rd("startSamplingPeriod.csv")))
    end_period = vec(Matrix{Int}(rd("endSamplingPeriod.csv")))
    nu_times = vec(Matrix{Int}(rd("nuTimes.csv")))
    sex_raw = vec(Matrix{Int}(rd("sex.csv")))           # 0=unknown, 1=F, 2=M
    K = rd("Kay.csv")[1, :K]
    k = rd("k.csv")[1, :k]

    # ── filter to known-sex badgers, and renumber ────────────────────────────
    keep = known_sex_only ? findall(!=(0), sex_raw) : collect(1:m)
    old_to_new = zeros(Int, m)
    for (new_i, old_i) in enumerate(keep)
        old_to_new[old_i] = new_i
    end

    Xinit_raw = Xinit_raw[keep, :]
    capt_hist = capt_hist[keep, :]
    birth_time = birth_time[keep]
    start_period = start_period[keep]
    end_period = end_period[keep]
    sex = Int[sex_raw[i] == 1 ? 1 : 0 for i in keep]    # 1=F, 0=M

    kept_rows = [old_to_new[Int(test_mat[r, 2])] != 0 for r in 1:size(test_mat, 1)]
    test_mat = test_mat[kept_rows, :]
    for r in 1:size(test_mat, 1)
        test_mat[r, 2] = old_to_new[Int(test_mat[r, 2])]
    end
    m = length(keep)

    # ── states: sparse codes -> our positions, and [i,t] -> [t,i] ────────────
    X_init = Matrix{Int}(undef, maxt, m)
    for i in 1:m, t in 1:maxt
        code = Xinit_raw[i, t]
        X_init[t, i] = code == -10 ? 1 : REF_STATE_CODE[Int(code)]   # unknown -> S
    end

    # ── group membership over time (LocateIndiv): the group at the most recent
    #    capture, carried forward. Genuinely varies with t — badgers move.
    social_group = zeros(Int, m, maxt)
    for i in 1:m
        rows = findall(==(Float64(i)), @view test_mat[:, 2])
        isempty(rows) && continue
        times_i = test_mat[rows, 1]
        groups_i = test_mat[rows, 3]
        g = Int(groups_i[argmin(times_i)])              # group at first capture
        for t in max(1, birth_time[i]):maxt
            at_t = findfirst(==(Float64(t)), times_i)
            at_t === nothing || (g = Int(groups_i[at_t]))
            social_group[i, t] = g
        end
    end

    # ── age: t - birth, from birth onward ────────────────────────────────────
    age = fill(-10, m, maxt)
    for i in 1:m, t in max(1, birth_time[i]):maxt
        age[i, t] = t - birth_time[i]
    end

    # ── tests: long -> [t, i, test], with the static Brock assignment ────────
    # -10 means missing in the reference; we use -1 throughout.
    tests = fill(-1, maxt, m, n_tests)
    for r in 1:size(test_mat, 1)
        t = Int(test_mat[r, 1])
        i = Int(test_mat[r, 2])
        (1 <= t <= maxt && 1 <= i <= m) || continue
        for j in 1:n_tests
            v = test_mat[r, 3 + j]
            (isnan(v) || v == -10) && continue
            tests[t, i, j] = Int(v)
        end
    end
    tests = apply_brock_changepoint!(tests, brock_changepoint)

    # ── capture history: [i,t] -> [t,i] ──────────────────────────────────────
    capture = permutedims(capt_hist, (2, 1))

    # last time each individual was seen alive — before this, death is impossible
    last_capture_time = [findlast(==(1), @view capture[:, i]) for i in 1:m]
    last_capture_time = [lc === nothing ? 0 : lc for lc in last_capture_time]

    season = make_season_vec(n_seasons, 1, maxt)
    sampling_period = [(start_period[i], end_period[i]) for i in 1:m]

    return (; n_individuals=m, n_timepoints=maxt, n_groups, n_tests, n_seasons,
            n_nu_times, X_init, social_group, age, capture, tests, sampling_period,
            birth_time, last_capture_time, season, nu_times, sex, K, k)
end

"""
    apply_brock_changepoint!(tests, changepoint; raw=BROCK_CHANGEPOINT_RAW,
                             brock1=1, brock2=2)

Move the Brock changepoint from `raw` (what the raw data encodes) to
`changepoint`, by swapping the two Brock columns for every test in between —
exactly the reference's `TestMatAsFieldProposal`, applied once.

Moving the changepoint *later* means tests in `[raw, changepoint)` were recorded
as Brock2 but now belong to Brock1 (and vice versa); moving it *earlier* swaps
`[changepoint, raw)` the other way. Each test occasion fills only one of the two
columns, so a swap is lossless — it relabels which version was used, and never
discards a result.
"""
function apply_brock_changepoint!(tests, changepoint;
                                  raw=BROCK_CHANGEPOINT_RAW, brock1=1, brock2=2)
    changepoint == raw && return tests
    window = changepoint > raw ? (raw:(changepoint - 1)) : (changepoint:(raw - 1))
    for t in window
        1 <= t <= size(tests, 1) || continue
        for i in axes(tests, 2)
            tests[t, i, brock1], tests[t, i, brock2] = tests[t, i, brock2], tests[t, i, brock1]
        end
    end
    tests
end

"""
    make_season_vec(n_seasons, season_start, maxt)

Seasons cycling `1..n_seasons` from `season_start`.
"""
function make_season_vec(n_seasons, season_start, maxt)
    v = ones(Int, maxt)
    v[1] = season_start
    for t in 2:maxt
        v[t] = v[t-1] < n_seasons ? v[t-1] + 1 : 1
    end
    v
end
