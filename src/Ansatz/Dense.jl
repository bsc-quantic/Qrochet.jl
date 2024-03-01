struct Dense <: Ansatz
    super::Quantum
end

function Dense(::State, array::AbstractArray)
    @assert ndims(array) > 0
    @assert all(>(1), size(array))

    sitemap = map(1:ndims(array)) do i
        Site(i) => letter(i)
    end |> Dict{Int,Symbol}

    tensor = Tensor(array, [letter(i) for i in 1:ndims(array)])

    tn = TensorNetwork([tensor])
    qtn = Quantum(tn, sitemap)
    Dense(qtn)
end

function Dense(::Operator, array::AbstractArray; sitemap::Vector{Site})
    @assert ndims(array) > 0
    @assert all(>(1), size(array))
    @assert length(sitemap) == ndims(array)

    tensor_inds = [letter(i) for i in 1:ndims(array)]
    tensor = Tensor(array, tensor_inds)
    tn = TensorNetwork([tensor])

    sitemap = map(splat(Pair), zip(sitemap, tensor_inds)) |> Dict{Site,Symbol}
    qtn = Quantum(tn, sitemap)

    Dense(qtn)
end
