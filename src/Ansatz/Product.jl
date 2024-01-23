using Tenet
using Tenet: letter
using LinearAlgebra

struct Product <: Ansatz
    super::Quantum

    function Product(tn::TensorNetwork, sites)
        @assert isempty(inds(tn, set = :inner)) "Product ansatz must not have inner indices"
        new(Quantum(tn, sites))
    end
end

function Product(arrays::Vector{<:Vector})
    _tensors = map(enumerate(arrays)) do (i, array)
        Tensor(array, [letter(i)])
    end

    sitemap = Dict(Site(i) => letter(i) for i in 1:length(arrays))

    Product(TensorNetwork(_tensors), sitemap)
end

LinearAlgebra.norm(tn::Product; kwargs...) = LinearAlgebra.norm(socket(tn), tn; kwargs...)
function LinearAlgebra.norm(::State, tn::Product, p::Real = 2)
    mapreduce(*, tensors(tn)) do tensor
        mapreduce(Base.Fix2(^, p), +, parent(tensor))
    end^(1 // p)
end

LinearAlgebra.normalize!(tn::Product; kwargs...) = LinearAlgebra.normalize!(socket(tn), tn; kwargs...)
function LinearAlgebra.normalize!(::State, tn::Product, p::Real = 2; insert::Union{Nothing,Int} = nothing)
    norm = LinearAlgebra.norm(tn, p)

    n = length(tensors(tn))
    norm ^= 1 / n
    for tensor in tensors(tn)
        tensor ./= norm
    end

    tn
end
