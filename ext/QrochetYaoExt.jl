module QrochetYaoExt

using Qrochet
using Tenet
using Yao

function flatten_circuit(x)
    if any(i -> i isa ChainBlock, subblocks(x))
        flatten_circuit(Yao.Optimise.eliminate_nested(x))
    else
        x
    end
end

function Qrochet.Quantum(circuit::AbstractBlock)
    @assert nlevel(circuit) == 2 "Only support 2-level qubits"

    n = nqubits(circuit)
    wire = [[Qrochet.nextindex()] for _ in 1:n]
    tensors = Tensor[]

    for gate in flatten_circuit(circuit)
        if gate isa Swap
            (a, b) = occupied_locs(gate)
            wire[a], wire[b] = wire[b], wire[a]
            continue
        end

        operator = content(gate)
        array = reshape(mat(operator), fill(2, 2 * nqubits(operator))...)

        inds = map(occupied_locs(gate)) do l
            from, to = last(wire[l]), Qrochet.nextindex()
            push!(wire[l], to)
            (from, to)
        end |> x -> zip(x...) |> Iterators.flatten |> collect

        tensor = Tensor(array, tuple(inds...))
        push!(tensors, tensor)
    end

    sites = merge(
        Dict([Site(site; dual = true) => first(index) for (site, index) in enumerate(wire)]),
        Dict([Site(site; dual = false) => last(index) for (site, index) in enumerate(wire)]),
    )

    Quantum(Tenet.TensorNetwork(tensors), sites)
end

end
