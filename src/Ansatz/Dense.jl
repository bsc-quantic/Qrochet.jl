struct Dense <: Ansatz
    super::Quantum
end

function Dense(::State, array::AbstractArray; sites::Vector{Site} = Site.(1:ndims(array)))
    @assert ndims(array) > 0
    @assert all(>(1), size(array))

    symbols = [nextindex() for _ in 1:ndims(array)]
    sitemap = map(sites, 1:ndims(array)) do site, i
        site => symbols[i]
    end |> Dict{Site,Symbol}

    tensor = Tensor(array, symbols)

    tn = TensorNetwork([tensor])
    qtn = Quantum(tn, sitemap)
    Dense(qtn)
end

function Dense(::Operator, array::AbstractArray; sites::Vector{Site})
    @assert ndims(array) > 0
    @assert all(>(1), size(array))
    @assert length(sites) == ndims(array)

    tensor_inds = [nextindex() for _ in 1:ndims(array)]
    tensor = Tensor(array, tensor_inds)
    tn = TensorNetwork([tensor])

    sitemap = map(splat(Pair), zip(sites, tensor_inds)) |> Dict{Site,Symbol}
    qtn = Quantum(tn, sitemap)

    Dense(qtn)
end

Base.copy(qtn::Dense) = Dense(copy(Quantum(qtn)))
