using Tenet

"""
    Ansatz

[`Quantum`](@ref) Tensor Network with a predefined structure.

# Notes

  - Any subtype must define `super::Quantum` field or specialize the `Quantum` method.
"""
abstract type Ansatz end

Quantum(@nospecialize tn::Ansatz) = tn.super

# TODO forward `Quantum` methods
for f in [:(Tenet.TensorNetwork), :ninputs, :noutputs, :sites, :socket, :(Tenet.tensors)]
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
