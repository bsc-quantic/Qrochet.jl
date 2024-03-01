module QrochetQuacExt

using Qrochet
using Tenet
using Quac: Gate, Circuit, lanes, arraytype, Swap

function Qrochet.Dense(gate::Gate)
    Qrochet.Dense(arraytype(gate)(gate); sitemap = [Site.(lanes(gate))..., Site.(lanes(gate); dual = true)])
end

Qrochet.evolve!(qtn::Ansatz, gate::Gate) = evolve!(qtn, Qrochet.Dense(gate))

function Qrochet.Quantum(circuit::Circuit)
    n = lanes(circuit)
    wire = [[Tenet.letter(i)] for i in 1:n]
    tensors = Tensor[]

    i = n + 1
    for gate in circuit
        G = arraytype(gate)
        array = G(gate)

        if gate isa Swap
            (a, b) = lanes(gate)
            wire[a], wire[b] = wire[b], wire[a]
            continue
        end

        inds = map(lanes(gate)) do l
            from, to = last(wire[l]), Tenet.letter(i)
            i += 1
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
