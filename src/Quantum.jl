using Tenet
using ValSplit

"""
    Site(id[, dual = false])
    site"id"

Represents a physical index.

# Unresolved questions

  - Should we store here some information about quantum numbers?
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

Tenet.TensorNetwork(q::Quantum) = q.tn

Base.copy(q::Quantum) = Quantum(copy(TensorNetwork(q)), copy(q.sites))

Base.:(==)(a::Quantum, b::Quantum) = a.tn == b.tn && a.sites == b.sites
Base.isapprox(a::Quantum, b::Quantum; kwargs...) = isapprox(a.tn, b.tn; kwargs...) && a.sites == b.sites

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

ninputs(q::Quantum) = count(isdual, keys(q.sites))
noutputs(q::Quantum) = count(!isdual, keys(q.sites))

inputs(q::Quantum) = sort!(filter(isdual, keys(q.sites)) |> collect)
outputs(q::Quantum) = sort!(filter(!isdual, keys(q.sites)) |> collect)

Base.summary(io::IO, q::Quantum) = print(io, "$(length(q.tn.tensormap))-tensors Quantum")
Base.show(io::IO, q::Quantum) = print(io, "Quantum (inputs=$(ninputs(q)), outputs=$(noutputs(q)))")

sites(tn::Quantum) = collect(keys(tn.sites))
nsites(tn::Quantum) = length(tn.sites)
lanes(tn::Quantum) = unique(Iterators.map(Iterators.flatten([inputs(tn), outputs(tn)])) do site
    isdual(site) ? site' : site
end)
nlanes(tn::Quantum) = length(lanes(tn))

Base.getindex(q::Quantum, site::Site) = q.sites[site]

abstract type Socket end
struct Scalar <: Socket end
Base.@kwdef struct State <: Socket
    dual::Bool = false
end
struct Operator <: Socket end

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
for f in [:(Tenet.tensors), :(Base.collect)]
    @eval $f(@nospecialize tn::Quantum) = $f(TensorNetwork(tn))
end

@valsplit 2 Tenet.select(tn::Quantum, query::Symbol, args...) = error("Query ':$query' not defined")
Tenet.select(tn::Quantum, ::Val{:index}, site::Site) = tn[site]
Tenet.select(tn::Quantum, ::Val{:tensor}, site::Site) = select(TensorNetwork(tn), :any, tn[site]) |> only

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
        select(b, :index, site) => select(a, :index, ioa != iob ? site' : site)
    end
    replace!(TensorNetwork(b), replacements...)

    for site in sitesb
        b.sites[site] = select(a, :index, ioa != iob ? site' : site)
    end

    b
end

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
