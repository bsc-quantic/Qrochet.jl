using Tenet
using Tenet: letter
using LinearAlgebra

struct Chain <: Ansatz
    super::Quantum
    boundary::Boundary
end

boundary(tn::Chain) = tn.boundary

MPS(arrays) = Chain(State(), Open(), arrays)
pMPS(arrays) = Chain(State(), Periodic(), arrays)
MPO(arrays) = Chain(Operator(), Open(), arrays)
pMPO(arrays) = Chain(Operator(), Periodic(), arrays)

alias(tn::Chain) = alias(socket(tn), boundary(tn), tn)
alias(::State, ::Open, ::Chain) = "MPS"
alias(::State, ::Periodic, ::Chain) = "pMPS"
alias(::Operator, ::Open, ::Chain) = "MPO"
alias(::Operator, ::Periodic, ::Chain) = "pMPO"

function Chain(tn::TensorNetwork, sites, args...; kwargs...)
    Chain(Quantum(tn, sites), args...; kwargs...)
end

function Chain(::State, boundary::Periodic, arrays::Vector{<:AbstractArray})
    @assert all(==(3) ∘ ndims, arrays) "All arrays must have 3 dimensions"

    n = length(arrays)

    _tensors = map(enumerate(arrays)) do (i, array)
        Tensor(array, [letter(i), letter(n + mod1(i - 1, length(arrays))), letter(n + mod1(i, length(arrays)))])
    end

    sitemap = Dict(Site(i) => letter(i) for i in 1:length(arrays))

    Chain(Quantum(TensorNetwork(_tensors), sitemap), boundary)
end

function Chain(::State, boundary::Open, arrays::Vector{<:AbstractArray})
    @assert ndims(arrays[1]) == 2 "First array must have 2 dimensions"
    @assert all(==(3) ∘ ndims, arrays[2:end-1]) "All arrays must have 3 dimensions"
    @assert ndims(arrays[end]) == 2 "Last array must have 2 dimensions"

    n = length(arrays)
    _tensors = map(enumerate(arrays)) do (i, array)
        if i == 1
            Tensor(array, [letter(1), letter(1 + n)])
        elseif i == n
            Tensor(array, [letter(n), letter(n + mod1(n - 1, length(arrays)))])
        else
            Tensor(array, [letter(i), letter(n + mod1(i - 1, length(arrays))), letter(n + mod1(i, length(arrays)))])
        end
    end

    sitemap = Dict(Site(i) => letter(i) for i in 1:length(arrays))

    Chain(Quantum(TensorNetwork(_tensors), sitemap), boundary)
end

function Chain(::Operator, boundary::Periodic, arrays::Vector{<:AbstractArray})
    @assert all(==(4) ∘ ndims, arrays) "All arrays must have 3 dimensions"

    n = length(arrays)

    _tensors = map(enumerate(arrays)) do (i, array)
        Tensor(
            array,
            [
                letter(i),
                letter(i + n),
                letter(2 * n + mod1(i - 1, length(arrays))),
                letter(2 * n + mod1(i, length(arrays))),
            ],
        )
    end

    sitemap = Dict(Site(i) => letter(i) for i in 1:length(arrays))
    merge!(sitemap, Dict(Site(i; dual = true) => letter(i + n) for i in 1:length(arrays)))

    Chain(Quantum(TensorNetwork(_tensors), sitemap), boundary)
end

function Chain(::Operator, boundary::Open, arrays::Vector{<:AbstractArray})
    @assert ndims(arrays[1]) == 3 "First array must have 3 dimensions"
    @assert all(==(4) ∘ ndims, arrays[2:end-1]) "All arrays must have 4 dimensions"
    @assert ndims(arrays[end]) == 3 "Last array must have 3 dimensions"

    n = length(arrays)
    _tensors = map(enumerate(arrays)) do (i, array)
        if i == 1
            Tensor(array, [letter(1), letter(n + 1), letter(1 + 2 * n)])
        elseif i == n
            Tensor(array, [letter(n), letter(2 * n), letter(2 * n + mod1(n - 1, length(arrays)))])
        else
            Tensor(
                array,
                [
                    letter(i),
                    letter(i + n),
                    letter(2 * n + mod1(i - 1, length(arrays))),
                    letter(2 * n + mod1(i, length(arrays))),
                ],
            )
        end
    end

    sitemap = Dict(Site(i) => letter(i) for i in 1:length(arrays))
    merge!(sitemap, Dict(Site(i; dual = true) => letter(i + n) for i in 1:length(arrays)))

    Chain(Quantum(TensorNetwork(_tensors), sitemap), boundary)
end

rightsite(tn::Chain, site::Site) = rightsite(boundary(tn), tn, site)
rightsite(::Open, tn::Chain, site::Site) = Site(site.id + 1)

leftsite(tn::Chain, site::Site) = leftsite(boundary(tn), tn, site)
leftsite(::Open, tn::Chain, site::Site) = Site(site.id - 1)

leftindex(tn::Chain, site::Site) = leftindex(boundary(tn), tn, site)
leftindex(::Periodic, tn::Chain, site::Site) = (select(tn, :tensor, site)|>inds)[end-1]
function leftindex(::Open, tn::Chain, site::Site)
    if site == site"1"
        nothing
    else
        (select(tn, :tensor, site)|>inds) ∩ (select(tn, :tensor, leftsite(tn, site))|>inds) |> only
    end
end

rightindex(tn::Chain, site::Site) = rightindex(boundary(tn), tn, site)
rightindex(::Periodic, tn::Chain, site::Site) = (select(tn, :tensor, site)|>inds)[end]
function rightindex(::Open, tn::Chain, site::Site)
    if site == Site(nsites(tn)) # TODO review
        nothing
    else
        (select(tn, :tensor, site)|>inds) ∩ (select(tn, :tensor, rightsite(tn, site))|>inds) |> only
    end
end

canonize(tn::Chain, args...; kwargs...) = canonize!(deepcopy(tn), args...; kwargs...)
canonize!(tn::Chain, args...; kwargs...) = canonize!(boundary(tn), tn, args...; kwargs...)

# NOTE: in mode == :svd the spectral weights are stored in a vector connected to the now virtual hyperindex!
function canonize!(::Open, tn::Chain, site::Site; direction::Symbol, mode::Symbol = :qr)
    left_inds = Symbol[]
    right_inds = Symbol[]

    virtualind = if direction === :left
        site == Site(nsites(tn)) && throw(ArgumentError("Cannot right-canonize right-most tensor"))
        push!(right_inds, rightindex(tn, site))

        site == Site(1) || push!(left_inds, leftindex(tn, site))
        push!(left_inds, Quantum(tn)[site])

        only(right_inds)
    elseif direction === :right
        site == Site(1) && throw(ArgumentError("Cannot left-canonize left-most tensor"))
        push!(right_inds, leftindex(tn, site))

        site == Site(nsites(tn)) || push!(left_inds, rightindex(tn, site))
        push!(left_inds, Quantum(tn)[site])

        only(right_inds)
    else
        throw(ArgumentError("Unknown direction=:$direction"))
    end

    tmpind = gensym(:tmp)
    if mode == :qr
        qr!(TensorNetwork(tn); left_inds, right_inds, virtualind = tmpind)
    elseif mode == :svd
        svd!(TensorNetwork(tn); left_inds, right_inds, virtualind = tmpind)
    else
        throw(ArgumentError("Unknown mode=:$mode"))
    end

    contract!(TensorNetwork(tn), virtualind)
    replace!(TensorNetwork(tn), tmpind => virtualind)

    return tn
end
