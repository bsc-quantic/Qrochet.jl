using Tenet

# TODO Should we store here some information about quantum numbers?
"""
    Site(id[, dual = false])
    site"i,j,..."

Represents a physical index.
"""
struct Site{N}
    id::NTuple{N,Int}
    dual::Bool

    Site(id::NTuple{N,Int}; dual = false) where {N} = new{N}(id, dual)
end

Site(id::Int; kwargs...) = Site((id,); kwargs...)
Site(id::Vararg{Int,N}; kwargs...) where {N} = Site(id; kwargs...)

id(site::Site{1}) = only(site.id)
id(site::Site) = site.id

Base.CartesianIndex(site::Site) = CartesianIndex(id(site))

isdual(site::Site) = site.dual
Base.show(io::IO, site::Site) = print(io, "$(id(site))$(site.dual ? "'" : "")")
Base.adjoint(site::Site) = Site(id(site); dual = !site.dual)
Base.isless(a::Site, b::Site) = id(a) < id(b)

macro site_str(str)
    m = match(r"^(\d+,)*\d+('?)$", str)
    if isnothing(m)
        error("Invalid site string: $str")
    end

    id = tuple(map(eachmatch(r"(\d+)", str)) do match
        parse(Int, only(match.captures))
    end...)

    dual = endswith(str, "'")

    return :(Site($id; dual = $dual))
end

"""
    Quantum

Tensor Network with a notion of "causality". This leads to the notion of sites and directionality (input/output).

# Notes

  - Indices are referenced by `Site`s.
"""
struct Quantum
    tn::TensorNetwork

    # WARN keep them synchronized
    sites::Dict{Site,Symbol}
    # sitetensors::Dict{Site,Tensor}

    function Quantum(tn::TensorNetwork, sites)
        for (_, index) in sites
            if !haskey(tn.indexmap, index)
                error("Index $index not found in TensorNetwork")
            elseif index ∉ inds(tn, set = :open)
                error("Index $index must be open")
            end
        end

        # sitetensors = map(sites) do (site, index)
        #     site => tn[index]
        # end |> Dict{Site,Tensor}

        new(tn, sites)
    end
end

Quantum(qtn::Quantum) = qtn

"""
    TensorNetwork(q::Quantum)

Returns the underlying `TensorNetwork` of a [`Quantum`](@ref) Tensor Network.
"""
Tenet.TensorNetwork(q::Quantum) = q.tn

Base.copy(q::Quantum) = Quantum(copy(TensorNetwork(q)), copy(q.sites))

Base.similar(q::Quantum) = Quantum(similar(TensorNetwork(q)), copy(q.sites))
Base.zero(q::Quantum) = Quantum(zero(TensorNetwork(q)), copy(q.sites))

Base.:(==)(a::Quantum, b::Quantum) = a.tn == b.tn && a.sites == b.sites
Base.isapprox(a::Quantum, b::Quantum; kwargs...) = isapprox(a.tn, b.tn; kwargs...) && a.sites == b.sites

"""
    adjoint(q::Quantum)

Returns the adjoint of a [`Quantum`](@ref) Tensor Network; i.e. the conjugate Tensor Network with the inputs and outputs swapped.
"""
function Base.adjoint(qtn::Quantum)
    sites = Iterators.map(qtn.sites) do (site, index)
        site' => index
    end |> Dict{Site,Symbol}

    tn = conj(TensorNetwork(qtn))

    # rename inner indices
    physical_inds = values(sites)
    virtual_inds = setdiff(inds(tn), physical_inds)
    replace!(tn, map(virtual_inds) do i
        i => Symbol(i, "'")
    end...)

    Quantum(tn, sites)
end

"""
    ninputs(q::Quantum)

Returns the number of input sites of a [`Quantum`](@ref) Tensor Network.
"""
ninputs(q::Quantum) = count(isdual, keys(q.sites))

"""
    noutputs(q::Quantum)

Returns the number of output sites of a [`Quantum`](@ref) Tensor Network.
"""
noutputs(q::Quantum) = count(!isdual, keys(q.sites))

"""
    inputs(q::Quantum)

Returns the input sites of a [`Quantum`](@ref) Tensor Network.
"""
inputs(q::Quantum) = sort!(filter(isdual, keys(q.sites)) |> collect)

"""
    outputs(q::Quantum)

Returns the output sites of a [`Quantum`](@ref) Tensor Network.
"""
outputs(q::Quantum) = sort!(filter(!isdual, keys(q.sites)) |> collect)

Base.summary(io::IO, q::Quantum) = print(io, "$(length(q.tn.tensormap))-tensors Quantum")
Base.show(io::IO, q::Quantum) = print(io, "Quantum (inputs=$(ninputs(q)), outputs=$(noutputs(q)))")

"""
    sites(q::Quantum)

Returns the sites of a [`Quantum`](@ref) Tensor Network.
"""
function sites(tn::Quantum; kwargs...)
    if isempty(kwargs)
        collect(keys(tn.sites))
    elseif keys(kwargs) === (:at,)
        findfirst(i -> i === kwargs[:at], tn.sites)
    else
        throw(MethodError(sites, (Quantum,), kwargs))
    end
end

"""
    nsites(q::Quantum)

Returns the number of sites of a [`Quantum`](@ref) Tensor Network.
"""
nsites(tn::Quantum) = length(tn.sites)

"""
    lanes(q::Quantum)

Returns the lanes of a [`Quantum`](@ref) Tensor Network.
"""
lanes(tn::Quantum) = unique(Iterators.map(Iterators.flatten([inputs(tn), outputs(tn)])) do site
    isdual(site) ? site' : site
end)

"""
    nlanes(q::Quantum)

Returns the number of lanes of a [`Quantum`](@ref) Tensor Network.
"""
nlanes(tn::Quantum) = length(lanes(tn))

"""
    getindex(q::Quantum, site::Site)

Returns the index associated with a site in a [`Quantum`](@ref) Tensor Network.
"""
Base.getindex(q::Quantum, site::Site) = inds(q; at = site)

"""
    Socket

Abstract type representing the socket of a [`Quantum`](@ref) Tensor Network.
"""
abstract type Socket end

"""
    Scalar <: Socket

Socket representing a scalar; i.e. a Tensor Network with no open sites.
"""
struct Scalar <: Socket end

"""
    State <: Socket

Socket representing a state; i.e. a Tensor Network with only input sites (or only output sites if `dual = true`).
"""
Base.@kwdef struct State <: Socket
    dual::Bool = false
end

"""
    Operator <: Socket

Socket representing an operator; i.e. a Tensor Network with both input and output sites.
"""
struct Operator <: Socket end

"""
    socket(q::Quantum)

Returns the socket of a [`Quantum`](@ref) Tensor Network; i.e. whether it is a [`Scalar`](@ref), [`State`](@ref) or [`Operator`](@ref).
"""
function socket(q::Quantum)
    _sites = sites(q)
    if isempty(_sites)
        Scalar()
    elseif all(!isdual, _sites)
        State()
    elseif all(isdual, _sites)
        State(dual = true)
    else
        Operator()
    end
end

# forward `TensorNetwork` methods
for f in [:(Tenet.arrays), :(Base.collect)]
    @eval $f(@nospecialize tn::Quantum) = $f(TensorNetwork(tn))
end

"""
    inds(tn::Quantum, set::Symbol = :all, args...; kwargs...)

Options:

  - `:at`: index at a site
"""
function Tenet.inds(tn::Quantum; kwargs...)
    if keys(kwargs) === (:at,)
        inds(tn, Val(:at), kwargs[:at])
    else
        inds(TensorNetwork(tn); kwargs...)
    end
end

Tenet.inds(tn::Quantum, ::Val{:at}, site::Site) = tn.sites[site]

"""
    tensors(tn::Quantum, query::Symbol, args...; kwargs...)

Options:

  - `:at`: tensor at a site
"""
function Tenet.tensors(tn::Quantum; kwargs...)
    if keys(kwargs) === (:at,)
        tensors(tn, Val(:at), kwargs[:at])
    else
        tensors(TensorNetwork(tn); kwargs...)
    end
end

Tenet.tensors(tn::Quantum, ::Val{:at}, site::Site) = only(tensors(tn; intersects = inds(tn; at = site)))

function reindex!(a::Quantum, ioa, b::Quantum, iob)
    ioa ∈ [:inputs, :outputs] || error("Invalid argument: :$ioa")

    sitesb = if iob === :inputs
        inputs(b)
    elseif iob === :outputs
        outputs(b)
    else
        error("Invalid argument: :$iob")
    end

    replacements = map(sitesb) do site
        inds(b; at = site) => inds(a; at = ioa != iob ? site' : site)
    end

    if issetequal(first.(replacements), last.(replacements))
        return b
    end

    replace!(TensorNetwork(b), replacements...)

    for site in sitesb
        b.sites[site] = inds(a; at = ioa != iob ? site' : site)
    end

    b
end

"""
    @reindex! a => b

Reindexes the input/output sites of a [`Quantum`](@ref) Tensor Network `b` to match the input/output sites of another [`Quantum`](@ref) Tensor Network `a`.
"""
macro reindex!(expr)
    @assert Meta.isexpr(expr, :call) && expr.args[1] == :(=>)
    Base.remove_linenums!(expr)
    a, b = expr.args[2:end]

    @assert Meta.isexpr(a, :call)
    @assert Meta.isexpr(b, :call)
    ioa, ida = a.args
    iob, idb = b.args
    return :((reindex!(Quantum($(esc(ida))), $(Meta.quot(ioa)), Quantum($(esc(idb))), $(Meta.quot(iob)))); $(esc(idb)))
end

"""
    merge(a::Quantum, b::Quantum...)

Merges multiple [`Quantum`](@ref) Tensor Networks into a single one by connecting input/output sites.
"""
Base.merge(a::Quantum, others::Quantum...) = foldl(merge, others, init = a)
function Base.merge(a::Quantum, b::Quantum)
    @assert issetequal(outputs(a), map(adjoint, inputs(b))) "Outputs of $a must match inputs of $b"

    @reindex! outputs(a) => inputs(b)
    tn = merge(TensorNetwork(a), TensorNetwork(b))

    sites = Dict{Site,Symbol}()

    for site in inputs(a)
        sites[site] = inds(a; at = site)
    end

    for site in outputs(b)
        sites[site] = inds(b; at = site)
    end

    Quantum(tn, sites)
end
