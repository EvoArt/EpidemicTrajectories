# Transition-matrix (gemlib-esque) style: SimpleEpiTransitionMatrix (chain
# binomial), EpiTransitionMatrix (general per-individual), and the @transitions
# modelling-language macro.

using Distributions: Binomial, logpdf
using ForwardDiff

@testset "transition-matrix style: incidence matrix" begin
    m = SimpleEpiTransitionMatrix(
        state_space = SI,
        transitions = [(0, 1), (1, 0)],           # S->I, I->S
        rates = [
            (pars, counts, t) -> -expm1(-(pars.α + pars.β * counts[2])),  # infection
            (pars, counts, t) -> 1 / pars.m,                              # recovery
        ],
    )
    A = incidence_matrix(m)
    @test size(A) == (2, 2)
    @test A[:, 1] == [-1, 1]   # S->I
    @test A[:, 2] == [1, -1]   # I->S
end

@testset "transition-matrix style: simulate + loglik shapes" begin
    m = SimpleEpiTransitionMatrix(
        state_space = SI,
        transitions = [(0, 1), (1, 0)],
        rates = [
            (pars, counts, t) -> -expm1(-(pars.α + pars.β * counts[2] / sum(counts))),
            (pars, counts, t) -> 1 / pars.m,
        ],
    )
    pars = (; α=0.05, β=0.4, m=5.0)
    counts0 = [50, 5]
    rng = StableRNG(3)
    counts, events = simulate_chain_binomial(rng, m, pars, counts0; n_times=40)
    @test size(counts) == (2, 40)
    @test size(events) == (2, 39)
    @test all(counts .>= 0)
    @test all(sum(counts; dims=1) .== sum(counts0))  # closed population

    lp = chain_binomial_loglik(pars, m, counts, events)
    @test isfinite(lp)
    @test lp ≤ 0
end

@testset "transition-matrix style: loglik matches hand-written binomial sum" begin
    m = SimpleEpiTransitionMatrix(
        state_space = SI,
        transitions = [(0, 1), (1, 0)],
        rates = [
            (pars, counts, t) -> -expm1(-(pars.α + pars.β * counts[2])),
            (pars, counts, t) -> 1 / pars.m,
        ],
    )
    pars = (; α=0.02, β=0.03, m=4.0)
    counts0 = [20, 3]
    rng = StableRNG(9)
    counts, events = simulate_chain_binomial(rng, m, pars, counts0; n_times=15)

    ref = 0.0
    for t in 1:14
        nS = counts[1, t]; nI = counts[2, t]
        pInf = -expm1(-(pars.α + pars.β * nI))
        pRec = 1 / pars.m
        ref += logpdf(Binomial(nS, pInf), events[1, t])
        ref += logpdf(Binomial(nI, pRec), events[2, t])
    end
    @test chain_binomial_loglik(pars, m, counts, events) ≈ ref rtol=1e-10
end

@testset "transition-matrix style: @transitions macro (simple)" begin
    si = @transitions SI begin
        S -> I = (pars, counts, t) -> -expm1(-(pars.α + pars.β * counts[2]))
        I -> S = (pars, counts, t) -> 1 / pars.m
    end
    @test si isa SimpleEpiTransitionMatrix
    @test si.transitions == [(0, 1), (1, 0)]

    # equivalence with a hand-built model
    pars = (; α=0.02, β=0.03, m=4.0)
    rng = StableRNG(9)
    counts, events = simulate_chain_binomial(rng, si, pars, [20, 3]; n_times=15)
    lp = chain_binomial_loglik(pars, si, counts, events)
    @test isfinite(lp)
end

@testset "transition-matrix style: EpiTransitionMatrix drives functional machinery" begin
    # General per-individual rates: f(pars, model, data, i, t). Here we make the
    # infection rate depend on a per-individual covariate `data.susceptibility[i]`
    # — something the count-based SimpleEpiTransitionMatrix CANNOT express — to
    # demonstrate the strictly-more-general seam.
    m = @transitions :individual SI begin
        S -> I = (pars, model, data, i, t) -> begin
            g = data.group[i]
            I_minus = count(j -> j != i && data.states[j, t] == 1, data.members(data, g))
            frailty = data.susceptibility[i]
            -expm1(-frailty * (pars.α + pars.β * I_minus))
        end
        I -> S = (pars, model, data, i, t) -> 1 / pars.m
    end
    @test m isa EpiTransitionMatrix

    # Build data with a per-individual covariate and use it with trajectory_loglik.
    pars = (; α=0.1, β=0.1, m=4.0)
    states = [0 0 1; 1 1 1; 0 1 1]  # 3 individuals x 3 times
    base = EpidemicTrajectories.make_data(states, [1, 1, 1])
    data = merge(base, (; susceptibility=[1.0, 1.5, 0.7]))
    model = (; state_space=SI, rates=m, pars=pars)

    # transition_matrix_at works for the general type -> rows sum to 1
    P = EpidemicTrajectories.transition_matrix_at(m, pars, model, data, 1, 1)
    @test all(≈(1.0), sum(P; dims=2))

    lp = trajectory_loglik(pars, model, data)
    @test isfinite(lp)

    # And it's autodiff-friendly through the general seam.
    g = ForwardDiff.gradient(v -> trajectory_loglik((; α=v[1], β=v[2], m=v[3]),
                                                    (; state_space=SI, rates=m, pars=(; α=v[1], β=v[2], m=v[3])),
                                                    data),
                             [0.1, 0.1, 4.0])
    @test all(isfinite, g)
end
