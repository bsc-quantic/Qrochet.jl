using Tenet
using ValSplit

"""
    Site(id[, dual = false])
    site"id"

Represents a physical index.

# Unresolved questions

  - Should we store here some information about quantum numbers?
"""
struct Site
    id::Int
    dual::Bool

    Site(id; dual = false) = new(id, dual)
end

isdual(site::Site) = site.dual
Base.show(io::IO, site::Site) = print(io, "$(site.id)$(site.dual ? "'" : "")")
Base.adjoint(site::Site) = Site(site.id; dual = !site.dual)
Base.isless(a::Site, b::Site) = a.id < b.id

macro site_str(str)
    m = match(r"^(\d+)('?)$", str)
    if isnothing(m)
        error("Invalid site string: $str")
    end

    id = parse(Int, m.captures[1])
    dual = m.captures[2] == "'"

    quote
        Site($id; dual = $dual)
    end
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

Tenet.TensorNetwork(q::Quantum) = q.tn

Base.copy(q::Quantum) = Quantum(copy(TensorNetwork(q)), copy(q.sites))

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
for f in [:(Tenet.tensors)]
    @eval $f(@nospecialize tn::Quantum) = $f(TensorNetwork(tn))
end

@valsplit 2 Tenet.select(tn::Quantum, query::Symbol, args...) = error("Query ':$query' not defined")
Tenet.select(tn::Quantum, ::Val{:index}, site::Site) = tn[site]
Tenet.select(tn::Quantum, ::Val{:tensor}, site::Site) = select(TensorNetwork(tn), :any, tn[site]) |> only
