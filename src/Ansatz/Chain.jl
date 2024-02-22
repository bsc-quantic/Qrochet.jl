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

leftsite(tn::Chain, site::Site) = leftsite(boundary(tn), tn, site)
function leftsite(::Open, tn::Chain, site::Site)
    site.id ∉ range(2, length(sites(tn))) && throw(ArgumentError("Invalid site $site"))
    Site(site.id - 1)
end
leftsite(::Periodic, tn::Chain, site::Site) = Site(mod1(site.id - 1, length(sites(tn))))

rightsite(tn::Chain, site::Site) = rightsite(boundary(tn), tn, site)
function rightsite(::Open, tn::Chain, site::Site)
    site.id ∉ range(1, length(sites(tn)) - 1) && throw(ArgumentError("Invalid site $site"))
    Site(site.id + 1)
end
rightsite(::Periodic, tn::Chain, site::Site) = Site(mod1(site.id + 1, length(sites(tn))))

leftindex(tn::Chain, site::Site) = leftindex(boundary(tn), tn, site)
function leftindex(::Union{Open,Periodic}, tn::Chain, site::Site)
    if site == site"1"
        nothing
    else
        (select(tn, :tensor, site) |> inds) ∩ (select(tn, :tensor, leftsite(tn, site)) |> inds) |> only
    end
end

rightindex(tn::Chain, site::Site) = rightindex(boundary(tn), tn, site)
function rightindex(::Union{Open,Periodic}, tn::Chain, site::Site)
    if site == Site(nsites(tn)) # TODO review
        nothing
    else
        (select(tn, :tensor, site) |> inds) ∩ (select(tn, :tensor, rightsite(tn, site)) |> inds) |> only
    end
end

canonize_site(tn::Chain, args...; kwargs...) = canonize_site!(deepcopy(tn), args...; kwargs...)
canonize_site!(tn::Chain, args...; kwargs...) = canonize_site!(boundary(tn), tn, args...; kwargs...)

# NOTE: in method == :svd the spectral weights are stored in a vector connected to the now virtual hyperindex!
function canonize_site!(::Open, tn::Chain, site::Site; direction::Symbol, method = :qr)
    left_inds = Symbol[]
    right_inds = Symbol[]

    virtualind = if direction === :left
        site == Site(1) && throw(ArgumentError("Cannot right-canonize left-most tensor"))
        push!(right_inds, leftindex(tn, site))

        site == Site(nsites(tn)) || push!(left_inds, rightindex(tn, site))
        push!(left_inds, Quantum(tn)[site])

        only(right_inds)
    elseif direction === :right
        site == Site(nsites(tn)) && throw(ArgumentError("Cannot left-canonize right-most tensor"))
        push!(right_inds, rightindex(tn, site))

        site == Site(1) || push!(left_inds, leftindex(tn, site))
        push!(left_inds, Quantum(tn)[site])

        only(right_inds)
    else
        throw(ArgumentError("Unknown direction=:$direction"))
    end

    tmpind = gensym(:tmp)
    if method === :svd
        svd!(TensorNetwork(tn); left_inds, right_inds, virtualind = tmpind)
    elseif method === :qr
        qr!(TensorNetwork(tn); left_inds, right_inds, virtualind = tmpind)
    else
        throw(ArgumentError("Unknown factorization method=:$method"))
    end

    contract!(TensorNetwork(tn), virtualind)
    replace!(TensorNetwork(tn), tmpind => virtualind)

    return tn
end

function isleftcanonical(qtn::Chain, site; atol::Real = 1e-12)
    right_ind = rightindex(qtn, site)

    # we are at right-most site, which cannot be left-canonical
    if isnothing(right_ind)
        return false
    end

    # TODO is replace(conj(A)...) copying too much?
    tensor = select(qtn, :tensor, site)
    contracted = contract(tensor, replace(conj(tensor), right_ind => :new_ind_name))

    return isapprox(contracted, Matrix{Float64}(I, size(A, right_ind), size(A, right_ind)); atol)
end

function isrightcanonical(qtn::Chain, site; atol::Real = 1e-12)
    left_ind = leftindex(qtn, site)

    # we are at left-most site, which cannot be right-canonical
    if isnothing(left_ind)
        return false
    end

    #TODO is replace(conj(A)...) copying too much?
    tensor = select(qtn, :tensor, site)
    contracted = contract(tensor, replace(conj(tensor), left_ind => :new_ind_name))

    return isapprox(contracted, Matrix{Float64}(I, size(A, left_ind), size(A, left_ind)); atol)
end

mixed_canonize(tn::Chain, args...; kwargs...) = mixed_canonize!(deepcopy(tn), args...; kwargs...)
mixed_canonize!(tn::Chain, args...; kwargs...) = mixed_canonize!(boundary(tn), tn, args...; kwargs...)

"""
    mixed_canonize!(boundary::Boundary, tn::Chain, center::Site)

Transform a `Chain` tensor network into the mixed-canonical form, that is,
for i < center the tensors are left-canonical and for i > center the tensors are right-canonical,
and in the center there is a matrix with singular values.
"""
function mixed_canonize!(::Open, tn::Chain, center::Site) # TODO: center could be a range of sites
    # left-to-right QR sweep (left-canonical tensors)
    for i in 1:center.id-1
        canonize_site!(tn, Site(i); direction = :right, method = :qr)
    end

    # right-to-left QR sweep (right-canonical tensors)
    for i in nsites(tn):-1:center.id+1
        canonize_site!(tn, Site(i); direction = :left, method = :qr)
    end

    return tn
end
