using Tenet
using Tenet: letter
using LinearAlgebra
using Random
using Muscle: gramschmidt!

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

function Base.adjoint(chain::Chain)
    chain = deepcopy(chain)
    # TODO: Make this work for `Chain`s with singular values between sites
    length(tensors(chain)) != length(sites(chain)) && throw(ArgumentError("Cannot adjoint a Chain with singular values"))
    q = chain.super

    # Update the virtual indices
    for site in sites(chain)
        left = leftindex(chain, site)
        left !== nothing && replace!(q.tn, left => Symbol(string(left, "'")))
    end
    right = rightindex(chain, sites(chain)[end])
    right !== nothing && replace!(q.tn, right => Symbol(string(right, "'")))

    foreach(conj!, tensors(q.tn))
    return chain
end

function LinearAlgebra.norm(ψ::Chain, p::Real = 2; kwargs...)
    p != 2 && throw(ArgumentError("p=$p is not implemented yet"))

    # TODO: Replace with contract(hcat(ψ, ψ')...) when implemented
    return contract(merge(TensorNetwork(ψ), TensorNetwork(ψ')); kwargs...) |> only |> sqrt |> abs
end

struct ChainSampler{B<:Qrochet.Boundary, S<:Qrochet.Socket, NT<:NamedTuple} <: Random.Sampler{Chain}
    parameters::NT

    ChainSampler{B, S}(; kwargs...) where {B, S} = new{B, S,typeof(values(kwargs))}(values(kwargs))
end

Base.rand(A::Type{<:Chain}, B::Type{<:Qrochet.Boundary}, S::Type{<:Qrochet.Socket}; kwargs...) =
    rand(Random.default_rng(), A, B, S; kwargs...)

Base.rand(rng::AbstractRNG, ::Type{A}, ::Type{B}, ::Type{S}; kwargs...) where {A<:Chain, B<:Qrochet.Boundary, S<:Qrochet.Socket} =
    rand(rng, ChainSampler{B, S}(; kwargs...), B, S)

function Base.rand(rng::Random.AbstractRNG, sampler::ChainSampler, ::Type{Open}, ::Type{State})
    n = sampler.parameters.n
    χ = sampler.parameters.χ
    p = get(sampler.parameters, :p, 2)
    T = get(sampler.parameters, :eltype, Float64)

    arrays::Vector{AbstractArray{T,N} where {N}} = map(1:n) do i
        χl, χr = let after_mid = i > n ÷ 2, i = (n + 1 - abs(2i - n - 1)) ÷ 2
            χl = min(χ, p^(i - 1))
            χr = min(χ, p^i)

            # swap bond dims after mid and handle midpoint for odd-length MPS
            (isodd(n) && i == n ÷ 2 + 1) ? (χl, χl) : (after_mid ? (χr, χl) : (χl, χr))
        end

        # fix for first site
        i == 1 && ((χl, χr) = (χr, 1))

        # orthogonalize by Gram-Schmidt algorithm
        A = gramschmidt!(rand(rng, T, χl, χr * p))

        A = reshape(A, χl, χr, p)
        permutedims(A, (3, 1, 2))
    end

    # reshape boundary sites
    arrays[1] = reshape(arrays[1], p, p)
    arrays[n] = reshape(arrays[n], p, p)

    # normalize state
    arrays[1] ./= sqrt(p)

    Chain(State(), Open(), arrays)
end

# TODO let choose the orthogonality center
# TODO different input/output physical dims
function Base.rand(rng::Random.AbstractRNG, sampler::ChainSampler, ::Type{Open}, ::Type{Operator})
    n = sampler.parameters.n
    χ = sampler.parameters.χ
    p = get(sampler.parameters, :p, 2)
    T = get(sampler.parameters, :eltype, Float64)

    ip = op = p

    arrays::Vector{AbstractArray{T,N} where {N}} = map(1:n) do i
        χl, χr = let after_mid = i > n ÷ 2, i = (n + 1 - abs(2i - n - 1)) ÷ 2
            χl = min(χ, ip^(i - 1) * op^(i - 1))
            χr = min(χ, ip^i * op^i)

            # swap bond dims after mid and handle midpoint for odd-length MPS
            (isodd(n) && i == n ÷ 2 + 1) ? (χl, χl) : (after_mid ? (χr, χl) : (χl, χr))
        end

        shape = if i == 1
            (χr, ip, op)
        elseif i == n
            (χl, ip, op)
        else
            (χl, χr, ip, op)
        end

        # orthogonalize by Gram-Schmidt algorithm
        A = gramschmidt!(rand(rng, T, shape[1], prod(shape[2:end])))
        A = reshape(A, shape)

        (i == 1 || i == n) ? permutedims(A, (2, 3, 1)) : permutedims(A, (3, 4, 1, 2))
    end

    # normalize
    ζ = min(χ, ip * op)
    arrays[1] ./= sqrt(ζ)

    Chain(Operator(), Open(), arrays)
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

truncate(tn::Chain, args...; kwargs...) = truncate!(deepcopy(tn), args...; kwargs...)

"""
    truncate!(qtn::Chain, bond; threshold::Union{Nothing,Real} = nothing, maxdim::Union{Nothing,Int} = nothing)

Truncate the dimension of the virtual `bond`` of the [`Chain`](@ref) Tensor Network by keeping only the `maxdim` largest Schmidt coefficients or those larger than`threshold`.

# Notes

  - Either `threshold` or `maxdim` must be provided. If both are provided, `maxdim` is used.
  - The bond must contain the Schmidt coefficients, i.e. a site canonization must be performed before calling `truncate!`.
"""
function truncate!(qtn::Chain, bond; threshold::Union{Nothing,Real} = nothing, maxdim::Union{Nothing,Int} = nothing)
    # TODO replace for select(:between)
    vind = rightindex(qtn, bond[1])
    if vind != leftindex(qtn, bond[2])
        throw(ArgumentError("Invalid bond $bond"))
    end

    if vind ∉ inds(TensorNetwork(qtn), :hyper)
        throw(MissingSchmidtCoefficientsException(bond))
    end

    tensor = TensorNetwork(qtn)[vind]
    spectrum = parent(tensor)

    extent = if !isnothing(maxdim)
        1:maxdim
    elseif !isnothing(threshold)
        findall(>(threshold) ∘ abs, spectrum)
    else
        throw(ArgumentError("Either `threshold` or `maxdim` must be provided"))
    end

    slice!(TensorNetwork(qtn), vind, extent)

    return qtn
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
    n = size(tensor, right_ind)
    identity_matrix = Matrix(I, n, n)

    return isapprox(contracted, identity_matrix; atol)
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
    n = size(tensor, left_ind)
    identity_matrix = Matrix(I, n, n)

    return isapprox(contracted, identity_matrix; atol)
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
