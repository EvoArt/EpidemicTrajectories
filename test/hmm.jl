# Exact marginalisation for independent individuals.
#
# Every value here is pinned against brute-force enumeration of hidden paths,
# written independently of the forward recursion, not against another estimator:
#  * the marginal likelihood, per individual and in total, and its gradient;
#  * the filtered state and backward-sampled paths;
#  * coupling, declared or hidden in a callback, is refused.

isdefined(@__MODULE__, :ForwardDiff) || @eval import ForwardDiff
using EpidemicTrajectories: logsumexp

# Capture-recapture with serology: N(egative), P(ositive), D(ead). `y` is 0 not
# caught, 1 caught negative, 2 caught positive, -1 no survey (weight 1). The
# first capture is conditioned on, so it carries the status but no capture
# probability. Time- and individual-varying rates, irregular intervals.
function _cjs_weight(model, data, i, t, s)
    y = data.y[t, i]
    y == -1 && return one(model.p)
    s == 3 && return y == 0 ? one(model.p) : zero(model.p)
    caught = y > 0
    w = t == data.first[i] ? one(model.p) : (caught ? model.p : 1 - model.p)
    caught || return w
    y == s ? w : zero(model.p)
end

function _cjs(y, first, sp; group = collect(1:size(y, 2)), coupled = nothing,
              infection = nothing, start = nothing)
    n_t, n = size(y)
    ss = [:N, :P, :D]
    aggs = @aggregate ss begin
        @array n_pos Int (1, n_t)
        n_pos[1, t] += (state == :P)
    end
    inf = infection === nothing ?
        ((model, data, i, t) -> model.a * data.dt[t] * (1 + data.z[i])) : infection
    spec = @transitions ss begin
        N -> P = inf
        N -> D = (model, data, i, t) -> model.mu * data.dt[t]
        P -> D = (model, data, i, t) -> model.mu2 * data.dt[t]
    end
    st = start === nothing ?
        ((model, data, X, i, t) -> [1 - model.nu, model.nu, zero(model.nu)]) : start
    epidemic_data(; n_individuals = n, n_timepoints = n_t, trans_mat = spec,
                  aggregates = aggs, starting_state = st,
                  observation_weight = (model, data, X, i, t, s) ->
                      _cjs_weight(model, data, i, t, s),
                  sampling_period = sp, group = group,
                  coupled_transitions = coupled,
                  y = y, first = first, dt = [1.0, 0.5, 2.0, 1.0, 1.5, 0.0],
                  z = [0.0, 1.0, 0.5, 0.0, 1.0][1:n])
end

# Columns are individuals, rows times. Individual 4 enters at t = 5.
const _Y = [ 1  0  0  0;
             0  2  0  0;
             1  0  1  0;
             2 -1  1  0;
             0  2  0  1;
             0  0  0  2]
const _FIRST = [1, 2, 3, 5]
const _SP = [(1, 6), (2, 6), (3, 5), (5, 6)]
const _THETA = (; a = 0.15, mu = 0.1, mu2 = 0.2, nu = 0.3, p = 0.6)

_data() = _cjs(copy(_Y), copy(_FIRST), _SP)

# Sum of p0 * w * prod(P * w) over every path of individual `i` on `lo:hi`,
# generic in the number type so ForwardDiff can run through it too.
function _enum_paths(model, data, i, lo, hi)
    K = data.n_states
    X = ones(Int, data.n_timepoints, data.n_individuals)
    n = hi - lo + 1
    Ps = [transition_matrix_at(data.trans_mat, model, data, X, i, t - 1) for t in (lo + 1):hi]
    p0 = data.starting_state(model, data, X, i, lo)
    w(t, s) = _cjs_weight(model, data, i, t, s)
    out = Dict{Any,Any}()
    for path in Iterators.product(ntuple(_ -> 1:K, n)...)
        pr = p0[path[1]] * w(lo, path[1])
        for j in 2:n
            pr *= Ps[j - 1][path[j - 1], path[j]] * w(lo + j - 1, path[j])
        end
        out[path] = pr
    end
    out
end
_enum_loglik(model, data, i, lo, hi) = log(sum(values(_enum_paths(model, data, i, lo, hi))))
_enum_total(model, data) =
    sum(_enum_loglik(model, data, i, data.sampling_period[i]...) for i in 1:data.n_individuals)

@testset "marginal likelihood equals enumeration over paths" begin
    data = _data()
    mll = epidemic_marginal_loglik(data)
    per = hmm_logliks(_THETA, data)
    for i in 1:4
        @test per[i] ≈ _enum_loglik(_THETA, data, i, _SP[i]...) atol = 1e-12
    end
    @test mll(_THETA, data) ≈ sum(per) atol = 1e-12
    @test mll(_THETA, data) ≈ _enum_total(_THETA, data) atol = 1e-12

    # the same factors as the complete-path likelihood, with X summed out: at a
    # model with no zero-probability paths the two agree up to its 1e-12 guards
    data2 = _cjs(fill(-1, 6, 4), copy(_FIRST), _SP)   # no surveys: every path possible
    loglik = epidemic_loglik(data2); obsll = epidemic_obs_loglik(data2)
    for i in 1:4
        lo, hi = _SP[i]
        lps = Float64[]
        for path in Iterators.product(ntuple(_ -> 1:3, hi - lo + 1)...)
            X = ones(Int, 6, 4)
            X[lo:hi, i] .= collect(path)
            one_i = _cjs(fill(-1, 6, 4), copy(_FIRST),
                         [k == i ? _SP[k] : (1, 0) for k in 1:4])
            push!(lps, epidemic_loglik(one_i)(_THETA, one_i, X) +
                       epidemic_obs_loglik(one_i)(_THETA, one_i, X))
        end
        # impossible paths (alive after D) score log(1e-12) there, not -Inf
        @test hmm_logliks(_THETA, data2)[i] ≈ logsumexp(lps) atol = 1e-8
    end
end

@testset "impossible histories are -Inf, and say where" begin
    y = copy(_Y); y[3, 1] = 1; y[4, 1] = 2; y[5, 1] = 1     # seroreversion at t = 5
    data = _cjs(y, copy(_FIRST), _SP)
    @test epidemic_marginal_loglik(data)(_THETA, data) == -Inf
    f = hmm_filter(_THETA, data, 1)
    @test f.loglik == -Inf && f.zero_at == 5
    # the zero is charged to the individual whose data it is
    @test hmm_filter(_THETA, data, 2).zero_at == 0
    @test isfinite(hmm_logliks(_THETA, data)[3])
end

@testset "gradient: enumeration and finite differences" begin
    data = _data()
    mll = epidemic_marginal_loglik(data)
    nt(v) = (; a = v[1], mu = v[2], mu2 = v[3], nu = v[4], p = v[5])
    v0 = collect(values(_THETA))
    g = ForwardDiff.gradient(v -> mll(nt(v), data), v0)
    ge = ForwardDiff.gradient(v -> _enum_total(nt(v), data), v0)
    @test g ≈ ge rtol = 1e-10
    h = 1e-6
    fd = [(mll(nt(v0 .+ h .* (1:5 .== k)), data) -
           mll(nt(v0 .- h .* (1:5 .== k)), data)) / 2h for k in 1:5]
    @test g ≈ fd rtol = 1e-6
    @test abs(g[5]) > 1e-3             # the observation parameter is informed
    @test all(isfinite, g)
end

@testset "reentrant: repeated, interleaved and concurrent calls agree" begin
    data = _data()
    mll = epidemic_marginal_loglik(data)
    ref = mll(_THETA, data)
    ForwardDiff.gradient(v -> mll((; _THETA..., p = v[1]), data), [0.6])
    @test mll(_THETA, data) == ref
    thetas = [(; _THETA..., p = p) for p in 0.3:0.1:0.8]
    seq = [mll(th, data) for th in thetas]
    par = fetch.([Threads.@spawn mll(th, data) for th in thetas])
    @test par == seq
end

# The same capture-recapture model with every step's shared quantities computed
# once: the infection hazard and the interval for the rates, the capture weight
# for the observations.
_cjs_step(model, data, i, t) =
    (; dt = data.dt[t], inf = model.a * data.dt[t] * (1 + data.z[i]))
function _cjs_obs_step(model, data, i, t)
    y = data.y[t, i]
    (; y, w = t == data.first[i] ? one(model.p) : (y > 0 ? model.p : 1 - model.p))
end
function _cjs_shared_weight(model, data, X, i, t, s, sh)
    sh.y == -1 && return one(model.p)
    s == 3 && return sh.y == 0 ? one(model.p) : zero(model.p)
    sh.y > 0 || return sh.w
    sh.y == s ? sh.w : zero(model.p)
end
function _cjs_shared(y, first, sp)
    n_t, n = size(y)
    ss = [:N, :P, :D]
    aggs = @aggregate ss begin
        @array n_pos Int (1, n_t)
        n_pos[1, t] += (state == :P)
    end
    spec = @transitions ss begin
        @shared _cjs_step
        N -> P = (model, data, i, t, shared) -> shared.inf
        N -> D = (model, data, i, t, shared) -> model.mu * shared.dt
        P -> D = (model, data, i, t, shared) -> model.mu2 * shared.dt
    end
    epidemic_data(; n_individuals = n, n_timepoints = n_t, trans_mat = spec,
                  aggregates = aggs,
                  starting_state = (model, data, X, i, t) -> [1 - model.nu, model.nu, zero(model.nu)],
                  observation_weight = _cjs_shared_weight, observation_shared = _cjs_obs_step,
                  sampling_period = sp, group = collect(1:n),
                  y = y, first = first, dt = [1.0, 0.5, 2.0, 1.0, 1.5, 0.0],
                  z = [0.0, 1.0, 0.5, 0.0, 1.0][1:n])
end

@testset "shared step values and threads: the same likelihood" begin
    plain = _data()
    shared = _cjs_shared(copy(_Y), copy(_FIRST), _SP)
    X = ones(Int, 6, 4)
    # every entry the samplers and the complete-path likelihood read
    for i in 1:4, t in 1:5
        P = transition_matrix_at(plain.trans_mat, _THETA, plain, X, i, t)
        @test transition_matrix_at(shared.trans_mat, _THETA, shared, X, i, t) == P
        for a in 1:3, b in 1:3
            @test transition_prob(shared.trans_mat, _THETA, shared, X, i, t, a, b) == P[a, b]
        end
        for s in 1:3
            @test shared.observation_weight(_THETA, shared, X, i, t, s) ==
                  _cjs_weight(_THETA, plain, i, t, s)
        end
    end
    # the value is pinned to enumeration, not only to the plain spec
    ref = _enum_total(_THETA, plain)
    for data in (plain, shared), threads in (1, 3)
        @test epidemic_marginal_loglik(data; threads)(_THETA, data) ≈ ref atol = 1e-12
    end
    nt(v) = (; a = v[1], mu = v[2], mu2 = v[3], nu = v[4], p = v[5])
    v0 = collect(values(_THETA))
    ge = ForwardDiff.gradient(v -> _enum_total(nt(v), plain), v0)
    for data in (plain, shared), threads in (1, 3)
        mll = epidemic_marginal_loglik(data; threads)
        @test ForwardDiff.gradient(v -> mll(nt(v), data), v0) ≈ ge rtol = 1e-10
    end
    # blocks partition the individuals, in order, and an impossible history
    # still gives -Inf through the threaded sum
    for n in 1:6
        @test reduce(vcat, collect.(EpidemicTrajectories._work_blocks(plain, n))) == 1:4
    end
    y = copy(_Y); y[3, 1] = 1; y[4, 1] = 2; y[5, 1] = 1
    bad = _cjs_shared(y, copy(_FIRST), _SP)
    @test epidemic_marginal_loglik(bad; threads = 2)(_THETA, bad) == -Inf
    @test_throws ArgumentError epidemic_marginal_loglik(plain; threads = 0)
end

@testset "the callbacks' X is a scaffold: perturbing it changes nothing" begin
    data = _data()
    X1 = ones(Int, 6, 4)
    X2 = [mod1(t * i, 3) for t in 1:6, i in 1:4]
    @test hmm_logliks(_THETA, data; X = X1) == hmm_logliks(_THETA, data; X = X2)
    @test check_independent(_THETA, data) === data
end

@testset "filtered state equals enumeration over paths" begin
    data = _data()
    upto = 4
    paths = _enum_paths(_THETA, data, 1, 1, upto)
    Z = sum(values(paths))
    for s in 1:3
        ps = sum(v for (k, v) in paths if k[end] == s) / Z
        @test hmm_filter(_THETA, data, 1; upto).filtered[s] ≈ ps atol = 1e-12
    end
end

@testset "backward sampling draws the exact smoothed states" begin
    data = _data()
    paths = _enum_paths(_THETA, data, 1, 1, 6)
    Z = sum(values(paths))
    # Not caught at t = 5 after a positive capture at 4: alive-positive or dead,
    # a genuinely uncertain state (an observed one would pass vacuously).
    p3 = sum(v for (k, v) in paths if k[5] == 3) / Z      # P(x_5 = D | y)
    @test 0.05 < p3 < 0.95
    rng = StableRNG(7)
    X = ones(Int, 6, 4)
    n = 20_000
    hits = 0
    allowed = true
    for _ in 1:n
        hmm_sample!(rng, _THETA, data, X; individuals = 1:1)
        hits += X[5, 1] == 3
        # every draw is a path the data allow
        allowed &= paths[Tuple(X[1:6, 1])] > 0
    end
    @test allowed
    @test abs(hits / n - p3) < 4 * sqrt(p3 * (1 - p3) / n)
end

@testset "coupling is refused, declared or hidden" begin
    # one shared group, no declaration: refused
    @test_throws ArgumentError epidemic_marginal_loglik(
        _cjs(copy(_Y), copy(_FIRST), _SP; group = ones(Int, 4)))
    # declared coupling: refused
    @test_throws ArgumentError epidemic_marginal_loglik(
        _cjs(copy(_Y), copy(_FIRST), _SP; group = ones(Int, 4), coupled = [(:N, :P)]))
    # declared uncoupled: accepted
    d = _cjs(copy(_Y), copy(_FIRST), _SP; group = ones(Int, 4),
             coupled = Tuple{Symbol,Symbol}[])
    @test epidemic_marginal_loglik(d)(_THETA, d) ≈ epidemic_marginal_loglik(_data())(_THETA, _data())

    # a rate reading a latent aggregate passes the structural check (it cannot
    # see inside the function) but not the numeric one
    leaky = _cjs(copy(_Y), copy(_FIRST), _SP;
                 infection = (model, data, i, t) ->
                     0.05 + 0.1 * data.aggregates.n_pos[1, t] / 4)
    @test require_independent(leaky) === leaky
    @test_throws ArgumentError check_independent(_THETA, leaky)
    # nor a starting state that reads X
    peeky = _cjs(copy(_Y), copy(_FIRST), _SP;
                 start = (model, data, X, i, t) ->
                     X[t, i] == 1 ? [0.7, 0.3, 0.0] : [0.5, 0.5, 0.0])
    @test_throws ArgumentError check_independent(_THETA, peeky)
end
