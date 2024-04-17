module QrochetChainRulesCoreExt

using Qrochet
using ChainRulesCore
using ChainRulesCore: AbstractTangent
using Tenet

@non_differentiable Qrochet.currindex()
@non_differentiable Qrochet.nextindex()

# WARN type-piracy
@non_differentiable Base.setdiff(::Vector{Symbol}, ::Base.ValueIterator)

ChainRulesCore.ProjectTo(x::Quantum) = ProjectTo{Quantum}(; tn = ProjectTo(TensorNetwork(x)))
(projector::ProjectTo{Quantum})(Δ) = Quantum(projector.tn(Δ.tn), Δ.sites)

function ChainRulesCore.frule((_, ẋ, _), ::Type{Quantum}, x::TensorNetwork, sites)
    y = Quantum(x, sites)
    ẏ = Tangent{Quantum}(; tn = ẋ)
    y, ẏ
end

Quantum_pullback(ȳ) = (NoTangent(), ȳ.tn, NoTangent())
Quantum_pullback(ȳ::AbstractThunk) = Quantum_pullback(unthunk(ȳ))
ChainRulesCore.rrule(::Type{Quantum}, x::TensorNetwork, sites) = Quantum(x, sites), Quantum_pullback

end
