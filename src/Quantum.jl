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
    siteinds::Dict{Site,Symbol}
    sitetensors::Dict{Site,Tensor}

    function Quantum(tn::TensorNetwork, sites::Dict{Site,Symbol})
        for (site, index) in sites
            if !haskey(tn.indexmap, index)
                error("Index $index not found in TensorNetwork")
            elseif index ∉ inds(tn, set = :open)
                error("Index $index must be open")
            end
        end

        sitetensors = map(sites) do (site, index)
            site => tn[index]
        end |> Dict{Site,Tensor}

        new(tn, sites, sitetensors)
    end
end

# TODO (@mofeing) Return `copy`?
Tenet.TensorNetwork(q::Quantum) = q.tn

ninputs(q::Quantum) = count(isdual, keys(q.sites))
noutputs(q::Quantum) = count(!isdual, keys(q.sites))

Base.summary(io::IO, q::Quantum) = print(io, "$(length(q.tn.tensormap))-tensors Quantum")
Base.show(io::IO, q::Quantum) = print(io, "Quantum (inputs=$(ninputs(q)), outputs=$(noutputs(q)))")

sites(tn::Quantum) = collect(keys(tn.siteinds))

Base.getindex(q::Quantum, site::Site) = q.siteinds[site]