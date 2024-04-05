module QrochetChainRulesTestUtilsExt

using Qrochet
using ChainRulesCore
using ChainRulesTestUtils
using Random

function ChainRulesTestUtils.rand_tangent(rng::AbstractRNG, x::Quantum)
    return Tangent{Quantum}(; tn = rand_tangent(rng, x.tn), sites = NoTangent())
end

# WARN type-piracy
# NOTE used in `Quantum` constructor
ChainRulesTestUtils.rand_tangent(::AbstractRNG, x::Dict{<:Site,Symbol}) = NoTangent()

end
