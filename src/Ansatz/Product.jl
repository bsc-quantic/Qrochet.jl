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

function Product(arrays::Vector{<:Matrix})
    n = length(arrays)
    _tensors = map(enumerate(arrays)) do (i, array)
        Tensor(array, [letter(i + n), letter(i)])
    end

    sitemap = merge!(Dict(Site(i, dual = true) => letter(i) for i in 1:n), Dict(Site(i) => letter(i + n) for i in 1:n))

    Product(TensorNetwork(_tensors), sitemap)
end

LinearAlgebra.norm(tn::Product, p::Real = 2) = LinearAlgebra.norm(socket(tn), tn, p)
function LinearAlgebra.norm(::Union{State,Operator}, tn::Product, p::Real)
    mapreduce(*, tensors(tn)) do tensor
        norm(tensor, p)
    end^(1 // p)
end

LinearAlgebra.opnorm(tn::Product, p::Real = 2) = LinearAlgebra.opnorm(socket(tn), tn, p)
function LinearAlgebra.opnorm(::Operator, tn::Product, p::Real)
    mapreduce(*, tensors(tn)) do tensor
        opnorm(tensor, p)
    end^(1 // p)
end

LinearAlgebra.normalize!(tn::Product, p::Real = 2) = LinearAlgebra.normalize!(socket(tn), tn, p)
function LinearAlgebra.normalize!(::Union{State,Operator}, tn::Product, p::Real)
    for tensor in tensors(tn)
        normalize!(tensor, p)
    end
    tn
end
