# The model spec and its sugar: @transitions, rate forms, grouped sources,
# state_space handling.

foi(model, data, i, t) = 0.1
rec(model, data, i, t) = 1 / model.m
surv(model, data, i, t) = 0.9

@testset "@transitions: states inferred from the transitions" begin
    spec = @transitions begin
        S -> I = foi
        I -> S = rec
    end
    @test spec isa TransitionSpec
    @test spec.states == [:S, :I]                      # order of first appearance
    @test spec.transitions == [(:S, :I), (:I, :S)]
    @test spec.auto_self == true                       # the default
end

@testset "@transitions: explicit state_space fixes the numbering" begin
    ss = [:I, :S]                                      # deliberately reversed
    spec = @transitions ss begin
        S -> I = foi
        I -> S = rec
    end
    @test spec.states == [:I, :S]                      # user's order wins

    # a state_space missing a state used in the transitions is an error
    @test_throws Exception @eval @transitions [:S] begin
        S -> I = $foi
    end
end

@testset "@transitions: rate sugar" begin
    model = (; m=5.0, α=0.01)
    data = nothing

    # bare function name is called for us
    s1 = @transitions begin
        S -> I = foi
    end
    @test s1.rate_fns[1](model, data, 1, 1) ≈ 0.1

    # arithmetic composition of bare names
    s2 = @transitions begin
        S -> I = foi * surv
    end
    @test s2.rate_fns[1](model, data, 1, 1) ≈ 0.1 * 0.9

    # bare expression referring to the parameters
    s3 = @transitions begin
        I -> S = 1 / model.m
    end
    @test s3.rate_fns[1](model, data, 1, 1) ≈ 1 / 5.0

    # explicit call still works
    s4 = @transitions begin
        S -> I = foi(model, data, i, t)
    end
    @test s4.rate_fns[1](model, data, 1, 1) ≈ 0.1

    # explicit lambda: the power-user fallback
    s5 = @transitions begin
        S -> I = (model, data, i, t) -> 0.25
    end
    @test s5.rate_fns[1](model, data, 1, 1) ≈ 0.25
end

@testset "@transitions: grouped sources expand to one transition each" begin
    spec = @transitions begin
        S -> I = foi
        (S, I) -> D = 0.01
    end
    @test (:S, :D) in spec.transitions
    @test (:I, :D) in spec.transitions
    @test length(spec.transitions) == 3
end

@testset "@transitions: auto_self opt-out" begin
    spec = @transitions :no_auto_self begin
        S -> I = foi
    end
    @test spec.auto_self == false
end

@testset "@survival: scales the live transitions and adds the deaths" begin
    # The badger structure: S -> E -> I -> D, every step conditional on surviving.
    spec = @transitions [:S, :E, :I, :D] begin
        @survival surv death=:D
        S -> E = foi
        E -> I = 0.2
    end
    model = (; m=5.0)

    # every live state can die — including I, which only ever appears as a
    # DESTINATION above and so has no declared transition out of it
    @test (:S, :D) in spec.transitions
    @test (:E, :D) in spec.transitions
    @test (:I, :D) in spec.transitions

    rate_of(from, to) = spec.rate_fns[findfirst(==((from, to)), spec.transitions)](model, nothing, 1, 1)

    # the live transitions are scaled by survival...
    @test rate_of(:S, :E) ≈ 0.9 * 0.1
    @test rate_of(:E, :I) ≈ 0.9 * 0.2
    # ...and death takes the leftover
    @test rate_of(:S, :D) ≈ 1 - 0.9
    @test rate_of(:I, :D) ≈ 1 - 0.9
end

@testset "@survival: needs a death state" begin
    @test_throws Exception @eval @transitions [:S, :D] begin
        @survival $surv
        S -> D = 0.1
    end
end

@testset "@transitions: bad tag errors" begin
    @test_throws Exception @eval @transitions :nonsense begin
        S -> I = $foi
    end
end
