@testset "QrochetQuacExt" begin
    using Quac

    @testset "QFT_3qubits" begin
        qft3circ = Quac.Algorithms.QFT(3)
        qft3qrochet = Quantum(qft3circ)

        # test sites (inputs and outputs) of the quantum circuit
        for site in values(qft3qrochet.sites)
            @test length(qft3qrochet.tn.indexmap[site]) == 1
        end

        # test inner tensors
        for notsite in filter(idx -> idx ∉ values(qrqft3.sites), keys(qrqft3.tn.indexmap))
            @test length(qft3qrochet.tn.indexmap[notsite]) > 1
        end
    end
end