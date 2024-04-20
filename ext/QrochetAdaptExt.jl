module QrochetAdaptExt

using Qrochet
using Tenet
using Adapt

Adapt.adapt_structure(to, x::Quantum) = Quantum(adapt(to, TensorNetwork(x)), x.sites)

end
