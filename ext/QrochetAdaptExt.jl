module QrochetAdaptExt

using Qrochet
using Tenet
using Adapt

Adapt.adapt_structure(to, x::Quantum) = Quantum(adapt(to, TensorNetwork(x)), x.sites)
Adapt.adapt_structure(to, x::Product) = Product(adapt(to, Quantum(x)))
Adapt.adapt_structure(to, x::Chain) = Chain(adapt(to, Quantum(x)), boundary(x))

end
