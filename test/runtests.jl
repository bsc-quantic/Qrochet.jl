using Test
using Qrochet

@testset "Unit tests" verbose = true begin
    include("Site_test.jl")
    include("Quantum_test.jl")
    include("Ansatz/Product_test.jl")
    include("Ansatz/Chain_test.jl")
end

@testset "Integration tests" verbose = true begin end

if haskey(ENV, "ENABLE_AQUA_TESTS")
    @testset "Aqua" verbose = true begin
        using Aqua
        Aqua.test_all(Qrochet, stale_deps = false)
    end
end
