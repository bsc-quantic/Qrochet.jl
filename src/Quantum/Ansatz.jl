using Tenet

abstract type Ansatz <: Tenet.AbstractTensorNetwork end

abstract type Socket end
struct Scalar <: Socket end
struct State <: Socket end
struct Operator <: Socket end

function socket end
socket(::A) where {A<:Ansatz} = socket(A)

abstract type Boundary end
struct Open <: Boundary end
struct Periodic <: Boundary end

function boundary end
boundary(::A) where {A<:Ansatz} = boundary(A)

Base.summary(io::IO, tn::A) where {A<:Ansatz} = print(io, "$A(n=$(length(tensors(tn))))")
Base.show(io::IO, tn::A) where {A<:Ansatz} = Base.summary(io, tn)

# helpers
genoutsym(i) = Symbol(:p, i, :o)
geninsym(i) = Symbol(:p, i, :i)
