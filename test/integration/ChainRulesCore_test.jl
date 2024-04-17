@testset "ChainRulesCore" begin
    using Qrochet
    using Tenet
    using ChainRulesTestUtils

    @testset "Quantum" begin
        test_frule(Quantum, TensorNetwork([Tensor(ones(2), [:i])]), Dict{Site,Symbol}(site"1" => :i))
        test_rrule(Quantum, TensorNetwork([Tensor(ones(2), [:i])]), Dict{Site,Symbol}(site"1" => :i))
    end

    @testset "Ansatz" begin
        @testset "Product" begin
            tn = TensorNetwork([Tensor(ones(2), [:i]), Tensor(ones(2), [:j]), Tensor(ones(2), [:k])])
            qtn = Quantum(tn, Dict([site"1" => :i, site"2" => :j, site"3" => :k]))

            test_frule(Product, qtn)
            test_rrule(Product, qtn)
        end

        @testset "Chain" begin
            tn = Chain(State(), Open(), [ones(2, 2), ones(2, 2, 2), ones(2, 2)])
            # test_frule(Chain, Quantum(tn), Open())
            test_rrule(Chain, Quantum(tn), Open())

            tn = Chain(State(), Periodic(), [ones(2, 2, 2), ones(2, 2, 2), ones(2, 2, 2)])
            # test_frule(Chain, Quantum(tn), Periodic())
            test_rrule(Chain, Quantum(tn), Periodic())

            tn = Chain(Operator(), Open(), [ones(2, 2, 2), ones(2, 2, 2, 2), ones(2, 2, 2)])
            # test_frule(Chain, Quantum(tn), Open())
            test_rrule(Chain, Quantum(tn), Open())

            tn = Chain(Operator(), Periodic(), [ones(2, 2, 2, 2), ones(2, 2, 2, 2), ones(2, 2, 2, 2)])
            # test_frule(Chain, Quantum(tn), Periodic())
            test_rrule(Chain, Quantum(tn), Periodic())
        end
    end
end
