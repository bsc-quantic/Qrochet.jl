@testset "ChainRulesCore" begin
    using Qrochet
    using Tenet
    using ChainRulesTestUtils

    @testset "Quantum" begin
        test_frule(Quantum, TensorNetwork([Tensor(fill(1.0, 2), [:i])]), Dict{Site,Symbol}(site"1" => :i))
        test_rrule(Quantum, TensorNetwork([Tensor(fill(1.0, 2), [:i])]), Dict{Site,Symbol}(site"1" => :i))
    end
end