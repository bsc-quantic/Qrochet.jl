using Tenet
using LinearAlgebra

struct Product{S<:Socket} <: Ansatz
    wrap::TensorNetwork

    function Product{S}(arrays; order = defaultorder(Product{S})) where {S}
        if S <: State
            all(==(1) ∘ ndims, arrays) || throw(DimensionMismatch("Product{State} is constructed with vectors"))
        elseif S <: Operator
            all(==(2) ∘ ndims, arrays) || throw(DimensionMismatch("Product{Operator} is constructed with matrices"))
        end

        n = length(arrays)

        oinds = map(genoutsym, 1:n)
        iinds = map(geninsym, 1:n)

        tensors::Vector{Tensor} = map(enumerate(arrays)) do (i, array)
            inds = map(order) do dir
                if dir === :o
                    oinds[i]
                elseif dir === :i
                    iinds[i]
                end
            end
            Tensor(array, inds)
        end

        return new{S}(TensorNetwork(tensors))
    end
end

defaultorder(::Type{Product{Scalar}}) = ()
defaultorder(::Type{Product{State}}) = (:o,)
defaultorder(::Type{Product{Operator}}) = (:i, :o)

socket(::Type{<:Product{S}}) where {S} = S()

# AbstractTensorNetwork interface
for f in [:(Tenet.inds), :(Tenet.tensors), :(Base.size)]
    @eval $f(ψ::Product; kwargs...) = $f(ψ.wrap; kwargs...)
end

function LinearAlgebra.norm(ψ::Product{State}, p::Real = 2)
    mapreduce(*, tensors(ψ)) do tensor
        mapreduce(Base.Fix2(^, p), +, parent(tensor))
    end^(1 // p)
end

function LinearAlgebra.normalize!(ψ::Product{State}, p::Real = 2; insert::Union{Nothing,Int} = nothing)
    norm = LinearAlgebra.norm(ψ, p)

    n = length(tensors(ψ))
    norm ^= 1 / n
    for tensor in tensors(ψ)
        tensor ./= norm
    end

    ψ
end
