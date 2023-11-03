using Test
using Qrochet
using OMEinsum

@testset "Unit tests" verbose = true begin end

@testset "Integration tests" verbose = true begin end

if haskey(ENV, "ENABLE_AQUA_TESTS")
    @testset "Aqua" verbose = true begin
        using Aqua
        Aqua.test_all(Qrochet, stale_deps = false)
    end
end
