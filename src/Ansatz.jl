using Tenet
using ValSplit

"""
    Ansatz

[`Quantum`](@ref) Tensor Network with a predefined structure.

# Notes

  - Any subtype must define `super::Quantum` field or specialize the `Quantum` method.
"""
abstract type Ansatz end

Quantum(@nospecialize tn::Ansatz) = tn.super

# TODO forward `Quantum` methods
for f in [:(Tenet.TensorNetwork), :ninputs, :noutputs, :sites, :nsites, :socket, :(Tenet.tensors)]
    @eval $f(@nospecialize tn::Ansatz) = $f(Quantum(tn))
end

abstract type Boundary end
struct Open <: Boundary end
struct Periodic <: Boundary end

function boundary end

alias(::A) where {A} = string(A)
function Base.summary(io::IO, tn::A) where {A<:Ansatz}
    print(io, "$(alias(tn)) (inputs=$(ninputs(tn)), outputs=$(noutputs(tn)))")
end
Base.show(io::IO, tn::A) where {A<:Ansatz} = Base.summary(io, tn)

@valsplit 2 Tenet.select(tn::Ansatz, query::Symbol, args...) = select(Quantum(tn), query, args...)

function Tenet.select(tn::Ansatz, ::Val{:between}, site1::Site, site2::Site)
    @assert site1 ∈ sites(tn) "Site $site1 not found"
    @assert site2 ∈ sites(tn) "Site $site2 not found"
    @assert site1 != site2 "Sites must be different"

    tensor1 = select(Quantum(tn), :tensor, site1)
    tensor2 = select(Quantum(tn), :tensor, site2)

    !isdisjoint(inds(tensor1), inds(tensor2)) && return nothing

    Iterators.filter(Iterators.filter(==(2) ∘ ndims, neighbors(TensorNetwork(tn), tensor1))) do tensor
        tensor === tensor2
    end |> only
end

struct MissingSchmidtCoefficientsException <: Base.Exception
    bond::NTuple{2,Site}
end

function Base.showerror(io::IO, e::MissingSchmidtCoefficientsException)
    print(io, "Can't access the spectrum on bond $(e.bond)")
end
