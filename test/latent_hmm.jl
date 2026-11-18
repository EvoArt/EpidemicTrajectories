using Distributions: Normal, Bernoulli, Poisson, Uniform, DiscreteUniform, logpdf, pdf
using LinearAlgebra: Diagonal, SymTridiagonal, diag
import ForwardDiff

@testset "weighted chains with latents" begin

    @testset ":weights tag" begin
        spec = @transitions [:a, :b] :weights begin
            a -> a = 0.5
            a -> b = 2.0
            b -> b = 1
        end
        @test spec isa WeightSpec
        @test spec.transitions == [(:a, :a), (:a, :b), (:b, :b)]
        # tag before the state space too
        @test (@transitions :weights [:a] begin a -> a = 1 end) isa WeightSpec
        @test_throws Exception @eval @transitions [:a, :d] :weights begin
            @survival 0.9 death = :d
            a -> a = 1
        end
        @test_throws Exception epidemic_data(; n_individuals = 1, n_timepoints = 2, trans_mat = spec,
                                             starting_state = (m, d, X, i, t) -> [1.0, 0.0],
                                             aggregates = Dict{Symbol,Any}())
    end

    @testset "reduces to the probability chain" begin
        # S -> I -> D with time-varying infection, noisy observations of the state
        N, T = 4, 8
        rng = StableRNG(1)
        y = rand(rng, 1:3, T, N)
        obsw(model, data, X, i, t, s) = s == data.y[t, i] ? 0.8 : 0.1
        inf(model, data, i, t) = model.b * (1 + 0.1t)
        ps = @transitions [:S, :I, :D] begin
            S -> I = inf
            I -> D = (model, data, i, t) -> model.d
        end
        pd = epidemic_data(; n_individuals = N, n_timepoints = T, trans_mat = ps,
                           starting_state = (m, d, X, i, t) -> [0.7, 0.3, 0.0],
                           aggregates = Dict{Symbol,Any}(), derived_summaries = (), observation_weight = obsw, y = y,
                           group = collect(1:N))
        model = (; b = 0.1, d = 0.2)
        ref = epidemic_marginal_loglik(pd)(model, pd)

        # same chain: the step into t uses the rates at t - 1, nothing moves into t = 1
        r(model, data, i, t) = t == 1 ? 0.0 : inf(model, data, i, t - 1)
        dd(model, data, i, t) = t == 1 ? 0.0 : model.d
        ws = @transitions [:S, :I, :D] :weights begin
            S -> S = 1 - r
            S -> I = r
            I -> I = 1 - dd
            I -> D = dd
            D -> D = 1
        end
        wd = latent_hmm_data(; n_individuals = N, n_timepoints = T, trans_mat = ws,
                             starting_state = (m, d, i) -> (0.7, 0.3, 0.0),
                             observation_weight = (m, d, i, t, s, sh) -> obsw(m, d, nothing, i, t, s),
                             latents = (;), init = (;), y = y)
        @test epidemic_marginal_loglik(wd)(model, wd, current_state(wd)) ≈ ref atol = 1e-10
    end

    @testset "nothing: no move, no observation" begin
        N4, T4 = 3, 10
        y4 = rand(StableRNG(6), 1:2, T4, N4)
        full(model, data, i, t) = iseven(t) ? (; stay = 1.0, go = 0.0) : (; stay = 1 - model.r, go = model.r)
        lazy(model, data, i, t) = iseven(t) ? nothing : full(model, data, i, t)
        ob_full(m, d, i, t, sh) = t % 3 == 0 ? (1.0, 1.0) : (d.y[t, i] == 1 ? 0.8 : 0.2, d.y[t, i] == 2 ? 0.7 : 0.3)
        ob_lazy(m, d, i, t, sh) = t % 3 == 0 ? nothing : ob_full(m, d, i, t, sh)
        sf = @transitions [:a, :b] :weights begin
            @shared full
            a -> a = shared.stay
            a -> b = shared.go
            b -> b = 1
        end
        sl = @transitions [:a, :b] :weights begin
            @shared lazy
            a -> a = shared.stay
            a -> b = shared.go
            b -> b = 1
        end
        mk(spec, ob) = latent_hmm_data(; n_individuals = N4, n_timepoints = T4, trans_mat = spec,
                                       starting_state = (m, d, i) -> (0.9, 0.1), observation_weights = ob,
                                       latents = (;), init = (;), y = y4)
        df = mk(sf, ob_full); dl = mk(sl, ob_lazy)
        model = (; r = 0.15)
        @test hmm_logliks(model, dl, current_state(dl)) ≈ hmm_logliks(model, df, current_state(df)) atol = 1e-12
        f = x -> epidemic_marginal_loglik(dl)((; r = x), dl, current_state(dl))
        @test ForwardDiff.derivative(f, 0.15) ≈ (f(0.15 + 1e-6) - f(0.15 - 1e-6)) / 2e-6 rtol = 1e-5
    end

    # A random walk observed with noise: one hidden state, a 1-d Field per
    # timepoint. The posterior of the field is Gaussian and known exactly.
    N, T, σ, τ, s0 = 3, 12, 0.7, 0.5, 10.0
    rng = StableRNG(2)
    truth = cumsum(σ .* randn(rng, T, N); dims = 1)
    yobs = truth .+ τ .* randn(rng, T, N)
    step(model, data, i, t) = t == 1 ? 1.0 :
        pdf(Normal(data.latents.S[1, t-1, i], model.σ), data.latents.S[1, t, i])
    rw_spec = @transitions [:a] :weights begin
        a -> a = step
    end
    rw_lat = @latents begin
        S ~ Field(1, T; prior = Normal(0, s0), touches = e -> e:min(e + 1, T))
    end
    rw_obs(model, data, i, t, sh) = (pdf(Normal(data.latents.S[1, t, i], model.τ), data.y[t, i]),)
    rw = latent_hmm_data(; n_individuals = N, n_timepoints = T, trans_mat = rw_spec,
                         starting_state = (m, d, i) -> (1.0,), observation_weights = rw_obs,
                         latents = rw_lat, init = (; S = reshape(copy(yobs), 1, T, N)), y = yobs)
    rw_model = (; σ = σ, τ = τ)

    @testset "local messages" begin
        @test check_local_messages(rw, rw_model, current_state(rw); field = :S) < 1e-10
    end

    @testset "field sampler against the exact posterior" begin
        Q = SymTridiagonal(fill(1 / s0^2 + 1 / τ^2, T) .+ [1; fill(2, T - 2); 1] ./ σ^2, fill(-1 / σ^2, T - 1))
        Σ = inv(Matrix(Q))
        sampler = epidemic_latent_sampler(rw; n_adapt = 500)
        state = current_state(rw)
        r = StableRNG(3)
        for _ in 1:500; sampler(r, rw_model, state); end
        draws = zeros(T, N, 6000)
        for k in 1:6000
            sampler(r, rw_model, state)
            draws[:, :, k] .= state.values.S[1, :, :]
        end
        for i in 1:N
            m = Σ * (yobs[:, i] ./ τ^2)
            @test maximum(abs.(vec(mean(draws[:, i, :]; dims = 2)) .- m) ./ sqrt.(diag(Σ))) < 0.3
            v = vec(var(draws[:, i, :]; dims = 2))
            @test all(0.7 .< v ./ diag(Σ) .< 1.4)
        end
    end

    @testset "per-individual latent against enumeration" begin
        # z ~ Bernoulli(p) per individual changes the a -> b rate
        N2, T2 = 4, 6
        r = StableRNG(4)
        y2 = rand(r, 1:2, T2, N2)
        rate(model, data, i, t) = data.latents.z[i] == 1 ? model.hi : model.lo
        sp = @transitions [:a, :b] :weights begin
            a -> a = 1 - rate
            a -> b = rate
            b -> b = 1
        end
        lat = @latents begin
            z ~ PerIndividual(model -> Bernoulli(model.p))
        end
        d2 = latent_hmm_data(; n_individuals = N2, n_timepoints = T2, trans_mat = sp,
                             starting_state = (m, d, i) -> (1.0, 0.0),
                             observation_weight = (m, d, i, t, s, sh) -> s == d.y[t, i] ? 0.85 : 0.15,
                             latents = lat, init = (; z = zeros(Int, N2)), y = y2)
        model = (; p = 0.3, hi = 0.4, lo = 0.05)
        st = current_state(d2)
        exact = map(1:N2) do i
            lz = map((0, 1)) do zv
                st.values.z[i] = zv
                hmm_logliks(model, d2, st)[i] + logpdf(Bernoulli(model.p), zv)
            end
            1 / (1 + exp(lz[1] - lz[2]))
        end
        st.values.z .= 0
        sampler = epidemic_latent_sampler(d2; n_adapt = 0)
        hits = zeros(N2)
        for _ in 1:20000
            sampler(r, model, st)
            hits .+= st.values.z
        end
        @test maximum(abs.(hits ./ 20000 .- exact)) < 0.02
        # the parameter-dependent prior is part of the likelihood
        mll = epidemic_marginal_loglik(d2)
        @test mll(model, d2, st) ≈ sum(hmm_logliks(model, d2, st)) +
                                   sum(logpdf.(Bernoulli(model.p), st.values.z)) atol = 1e-10
    end

    @testset "shared latent" begin
        # a common offset mu, observed through every individual
        N3, T3 = 5, 4
        r = StableRNG(5)
        y3 = 1.3 .+ randn(r, T3, N3)
        sp = @transitions [:a] :weights begin
            a -> a = 1
        end
        lat = @latents begin
            mu ~ Shared(1, 1; prior = Normal(0, 10), observation_only = true)
        end
        d3 = latent_hmm_data(; n_individuals = N3, n_timepoints = T3, trans_mat = sp,
                             starting_state = (m, d, i) -> (1.0,),
                             observation_shared = (m, d, i, key) -> d.latents.mu[1, 1],
                             observation_weights = (m, d, i, t, sh) -> (pdf(Normal(sh, 1.0), d.y[t, i]),),
                             cache = (; shared = (), weights = (), latents = (:mu,)),
                             latents = lat, init = (; mu = zeros(1, 1)), y = y3)
        prec = 1 / 100 + N3 * T3
        m = sum(y3) / prec
        sampler = epidemic_latent_sampler(d3; n_adapt = 500)
        st = current_state(d3)
        for _ in 1:500; sampler(r, (;), st); end
        mus = [(sampler(r, (;), st); st.values.mu[1, 1]) for _ in 1:8000]
        @test abs(mean(mus) - m) < 0.3 / sqrt(prec)
        @test 0.7 < var(mus) * prec < 1.4
    end

    @testset "Field index, default starting values, @weights" begin
        f = Field(1, 3; prior = Normal(0, 1), index = t -> cld(t, 4))
        d = latent_hmm_data(; n_individuals = 2, n_timepoints = 12, trans_mat = rw_spec,
                            starting_state = (m, d, i) -> (1.0,), latents = (; S = f), y = zeros(12, 2))
        k = d.latent_decls.S
        @test [k.touches(e) for e in 1:3] == [1:5, 5:9, 9:12]
        @test size(current_state(d).values.S) == (1, 3, 2)
        lat = @latents begin
            b ~ PerIndividual(DiscreteUniform(1, 5); observed = [3, missing, missing])
            s ~ PerIndividual(model -> Bernoulli(model.p); observed = [1, missing, 1])
            X ~ Shared(2, 4; prior = (Uniform(0, 1), Uniform(5, 6)))
        end
        st = EpidemicTrajectories.latent_state(lat, 3)
        @test st.values.b[1] == 3 && all(in(1:5), st.values.b)
        @test all(==(1), st.values.s)
        @test all(0 .<= st.values.X[1, :] .<= 1) && all(5 .<= st.values.X[2, :] .<= 6)
        g(model, z) = z ? @weights(model; a = 0, b = model.x) : @weights(model; a = model.x, b = 1)
        D = ForwardDiff.Dual{Nothing,Float64,1}
        @test g((; x = D(0.5)), true) isa NamedTuple{(:a, :b),Tuple{D,D}}
        @test g((; x = 0.5), false) === (; a = 0.5, b = 1.0)
        @test @weights((; x = 0.5), 0, 1) === (0.0, 1.0)
    end

    @testset "cache and gradients" begin
        # observation_shared reads the field and one parameter; the weights
        # read another. Cached and fresh evaluations must agree, and so must
        # AD and finite differences, whichever block carries the Duals.
        sh(model, data, i, key) = exp(-model.c * data.latents.S[1, key, i]^2)
        ob(model, data, i, t, s) = (s * model.w + (1 - s) * (1 - model.w) + model.a * data.y[t, i]^2,)
        mk(cache) = latent_hmm_data(; n_individuals = N, n_timepoints = T, trans_mat = rw_spec,
                                    starting_state = (m, d, i) -> (1.0,), observation_shared = sh,
                                    observation_weights = ob, cache = cache,
                                    latents = rw_lat, init = (; S = reshape(copy(yobs), 1, T, N)),
                                    y = yobs ./ 10)
        dc = mk((; shared = (:c,), weights = (:w,), latents = (:S,)))
        dn = mk(nothing)
        model = (; σ = σ, τ = τ, c = 0.3, w = 0.6, a = 0.1)
        fc = epidemic_marginal_loglik(dc); fn = epidemic_marginal_loglik(dn)
        sc = current_state(dc); sn = current_state(dn)
        @test fc(model, dc, sc) ≈ fn(model, dn, sn) atol = 1e-10
        @test fc(model, dc, sc) ≈ fn(model, dn, sn) atol = 1e-10    # second call uses the cache
        for p in (:c, :w, :σ)
            f = x -> fc(merge(model, NamedTuple{(p,)}((x,))), dc, sc)
            g = ForwardDiff.derivative(f, model[p])
            fd = (f(model[p] + 1e-6) - f(model[p] - 1e-6)) / 2e-6
            @test g ≈ fd rtol = 1e-5
        end
    end
end
