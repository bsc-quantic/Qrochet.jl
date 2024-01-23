using Tenet

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

    Site(id, dual = false) = new(id, dual)
end

isdual(site::Site) = site.dual
Base.show(io::IO, site::Site) = print(io, "$(site.id)$(site.dual ? "†" : "")")
Base.adjoint(site::Site) = Site(site.id, !site.dual)

macro site_str(str)
    m = match(r"^(\d+)(')$", str)
    if isnothing(m)
        error("Invalid site string: $str")
    end

    id = parse(Int, m.captures[1])
    dual = m.captures[2] == "'"

    quote
        Site($id, $dual)
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

    function Quantum(tn::TensorNetwork, sites::Dict{Site,Symbol})
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

# TODO (@mofeing) Return `copy`?
Tenet.TensorNetwork(q::Quantum) = q.tn

ninputs(q::Quantum) = count(isdual, keys(q.sites))
noutputs(q::Quantum) = count(!isdual, keys(q.sites))

Base.summary(io::IO, q::Quantum) = print(io, "$(length(q.tn.tensormap))-tensors Quantum")
Base.show(io::IO, q::Quantum) = print(io, "Quantum (inputs=$(ninputs(q)), outputs=$(noutputs(q)))")

# forward `TensorNetwork` methods
for f in [:(Tenet.tensors)]
    @eval $f(@nospecialize tn::Quantum) = $f(TensorNetwork(tn))
end

sites(tn::Quantum) = collect(keys(tn.sites))

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
