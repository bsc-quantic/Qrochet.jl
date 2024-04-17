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

Base.zero(x::Dict{Site,Symbol}) = x

ChainRulesCore.ProjectTo(x::T) where {T<:Ansatz} = ProjectTo{T}(; super = ProjectTo(Quantum(x)))
(projector::ProjectTo{T})(Δ::Union{T,Tangent{T}}) where {T<:Ansatz} = T(projector.super(Δ.super), Δ.boundary)

# NOTE edge case: `Product` has no `boundary`. should it?
(projector::ProjectTo{T})(Δ::Union{T,Tangent{T}}) where {T<:Product} = T(projector.super(Δ.super))

ChainRulesCore.frule((_, ẋ), ::Type{T}, x::Quantum) where {T<:Ansatz} = T(x), Tangent{T}(; super = ẋ)

Ansatz_pullback(ȳ) = (NoTangent(), ȳ.super)
Ansatz_pullback(ȳ::AbstractThunk) = Ansatz_pullback(unthunk(ȳ))
function ChainRulesCore.rrule(::Type{T}, x::Quantum) where {T<:Ansatz}
    y = T(x)
    y, Ansatz_pullback
end

function ChainRulesCore.frule((_, ẋ, _), ::Type{T}, x::Quantum, boundary) where {T<:Ansatz}
    T(x, boundary), Tangent{T}(; super = ẋ, boundary = NoTangent())
end

Ansatz_boundary_pullback(ȳ) = (NoTangent(), ȳ.super, NoTangent())
Ansatz_boundary_pullback(ȳ::AbstractThunk) = Ansatz_boundary_pullback(unthunk(ȳ))
function ChainRulesCore.rrule(::Type{T}, x::Quantum, boundary) where {T<:Ansatz}
    T(x, boundary), Ansatz_boundary_pullback
end

# Ansatz_from_arrays_pullback(ȳ) = (NoTangent(), NoTangent(), NoTangent(), parent.(tensors(ȳ.super.tn)))
# Ansatz_from_arrays_pullback(ȳ::AbstractThunk) = Ansatz_from_arrays_pullback(unthunk(ȳ))
# function ChainRulesCore.rrule(
#     ::Type{T},
#     socket::Qrochet.Socket,
#     boundary::Qrochet.Boundary,
#     arrays;
#     kwargs...,
# ) where {T<:Ansatz}
#     y = T(socket, boundary, arrays; kwargs...)
#     y, Ansatz_from_arrays_pullback
# end

copy_pullback(ȳ) = (NoTangent(), ȳ)
copy_pullback(ȳ::AbstractThunk) = unthunk(ȳ)
function ChainRulesCore.rrule(::typeof(copy), x::Quantum)
    y = copy(x)
    y, copy_pullback
end

end
