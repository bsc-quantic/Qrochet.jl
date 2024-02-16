@testset "Quac" begin
    using Quac

    @testset "QFT" begin
        n = 3
        qftcirc = Quac.Algorithms.QFT(n)
        qftqtn = Quantum(qftcirc)

        # correct number of inputs and outputs
        @test ninputs(qftqtn) == n
        @test noutputs(qftqtn) == n
        @test socket(qftqtn) == Operator()

        # all open indices are sites
        siteinds = getindex.((qftqtn,), sites(qftqtn))
        @test issetequal(inds(TensorNetwork(qftqtn), :open), siteinds)

        # all inner indices are not sites
        notsiteinds = setdiff(inds(TensorNetwork(qftqtn)), siteinds)
        @test_skip issetequal(inds(TensorNetwork(qftqtn), :inner), notsiteinds)
    end
end