@testset "Quac" begin
    using Quac

    @testset "QFT" begin
        n = 3
        qftcirc = Quac.Algorithms.QFT(n)
        qftqtn = Quantum(qftcirc)

        siteinds = getindex.((qftqtn,), sites(qftqtn))
        notsiteinds = filter(idx -> idx ∉ getindex.((qftqtn,), sites(qftqtn)), keys(TensorNetwork(qftqtn).indexmap))

        # correct number of inputs and outputs
        @test ninputs(qftqtn) == n
        @test noutputs(qftqtn) == n
        @test socket(qftqtn) == Operator()
        # all open indices are sites
        @test issetequal(inds(TensorNetwork(qftqtn), :open), siteinds)
        # all inner indices are not sites
        @test issetequal(inds(TensorNetwork(qftqtn), :inner), notsiteinds)
    end
end