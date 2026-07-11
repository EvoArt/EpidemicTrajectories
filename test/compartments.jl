@testset "compartments / state space" begin
    @test nstates(SI) == 2
    @test SI.codes == (0, 1)
    @test state_index(SI, 0) == 1
    @test state_index(SI, 1) == 2
    @test_throws ArgumentError state_index(SI, 5)

    @test SIS === SI  # alias, same encoding

    @test nstates(SEID) == 4
    @test SEID.codes == (0, 3, 1, 9)
    @test state_index(SEID, 3) == 2   # E is dense index 2
    @test state_index(SEID, 9) == 4   # D is dense index 4

    ss = StateSpace([2, 5, 7]; names=(:a, :b, :c))
    @test nstates(ss) == 3
    @test state_index(ss, 5) == 2
end
