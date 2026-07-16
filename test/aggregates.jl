# User-declared aggregates: storage + reversible update declared together, bare
# array names, states by name or number, and the reversibility the sampler needs.

@testset "@aggregate: declares storage and update together" begin
    state_space = [:S, :I]
    n_groups, n_t = 3, 5

    aggs = @aggregate state_space begin
        @array n_infected Int (n_groups, n_t)
        n_infected[data.group[i], t] += (state == :I)
    end

    @test aggs isa AggregateDeclaration
    @test length(aggs.specs) == 1
    @test aggs.specs[1].name == :n_infected
    @test aggs.specs[1].eltype == Int
    @test aggs.specs[1].dims == (n_groups, n_t)
    @test length(aggs.summaries) == 1

    store = allocate_aggregates(aggs)
    @test store[:n_infected] isa Matrix{Int}
    @test size(store[:n_infected]) == (n_groups, n_t)
    @test all(iszero, store[:n_infected])
end

@testset "@aggregate: the update is reversible (the property iFFBS relies on)" begin
    state_space = [:S, :I]
    aggs = @aggregate state_space begin
        @array n_infected Int (2, 3)
        n_infected[data.group[i], t] += (state == :I)
    end
    summary = aggs.summaries[1]
    data = (; aggregates=allocate_aggregates(aggs), group=[1, 2])
    X = nothing
    model = nothing

    # applying the infected state increments; reversing it puts the count back
    summary(model, data, X, 2, 1, 1)                      # individual 1, group 1, t=1, state I
    @test data.aggregates[:n_infected][1, 1] == 1
    summary(model, data, X, 2, 1, 1; reverse=true)
    @test data.aggregates[:n_infected][1, 1] == 0

    # the susceptible state contributes nothing either way
    summary(model, data, X, 1, 1, 1)
    @test data.aggregates[:n_infected][1, 1] == 0

    # apply/reverse round-trips exactly, whatever the order
    for s in (1, 2), i in (1, 2), t in 1:3
        summary(model, data, X, s, i, t)
    end
    for s in (1, 2), i in (1, 2), t in 1:3
        summary(model, data, X, s, i, t; reverse=true)
    end
    @test all(iszero, data.aggregates[:n_infected])
end

@testset "@aggregate: states by name and by number agree" begin
    state_space = [:S, :I]

    by_name = @aggregate state_space begin
        @array a Int (1, 1)
        a[1, 1] += (state == :I)
    end
    by_number = @aggregate state_space begin
        @array a Int (1, 1)
        a[1, 1] += (state == 2)
    end

    d1 = (; aggregates=allocate_aggregates(by_name))
    d2 = (; aggregates=allocate_aggregates(by_number))
    by_name.summaries[1](nothing, d1, nothing, 2, 1, 1)
    by_number.summaries[1](nothing, d2, nothing, 2, 1, 1)
    @test d1.aggregates[:a][1, 1] == 1
    @test d2.aggregates[:a][1, 1] == 1
end

@testset "@aggregate: an unknown state name errors" begin
    state_space = [:S, :I]
    aggs = @aggregate state_space begin
        @array a Int (1, 1)
        a[1, 1] += (state == :NotAState)
    end
    d = (; aggregates=allocate_aggregates(aggs))
    @test_throws ErrorException aggs.summaries[1](nothing, d, nothing, 1, 1, 1)
end

@testset "@aggregate: guarded update and several arrays" begin
    state_space = [:S, :I, :D]
    aggs = @aggregate state_space begin
        @array n_infected Int (2, 2)
        @array n_alive Int (2, 2)
        n_infected[data.group[i], t] += (state == :I)      # condition folded in
        if state != :D                                     # guarded form
            n_alive[data.group[i], t] += 1
        end
    end
    @test length(aggs.specs) == 2
    @test length(aggs.summaries) == 2

    data = (; aggregates=allocate_aggregates(aggs), group=[1, 2])
    for s in aggs.summaries
        s(nothing, data, nothing, 2, 1, 1)     # individual 1 is infected (and alive)
    end
    @test data.aggregates[:n_infected][1, 1] == 1
    @test data.aggregates[:n_alive][1, 1] == 1

    for s in aggs.summaries
        s(nothing, data, nothing, 3, 2, 1)     # individual 2 is dead
    end
    @test data.aggregates[:n_infected][2, 1] == 0
    @test data.aggregates[:n_alive][2, 1] == 0
end

@testset "@aggregate: needs an @array declaration and a reversible operator" begin
    @test_throws Exception @eval @aggregate begin
        n_infected[1, 1] += 1
    end
    @test_throws Exception @eval @aggregate begin
        @array a Int (1, 1)
        a[1, 1] = 1                      # `=` has no inverse
    end
end
