# `Quantum` Tensor Networks

```@docs
Quantum
Tenet.TensorNetwork(::Quantum)
Base.adjoint(::Quantum)
sites
nsites
```

## Queries

```@docs
Tenet.select(::Quantum, ::Val{:index}, ::Site)
Tenet.select(::Quantum, ::Val{:tensor}, ::Site)
```

## Connecting `Quantum` Tensor Networks

```@docs
inputs
outputs
lanes
ninputs
noutputs
nlanes
```

```@docs
Socket
socket(::Quantum)
Scalar
State
Operator
```

```@docs
Base.merge(::Quantum, ::Quantum...)
```
