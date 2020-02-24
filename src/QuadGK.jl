# Adaptive Gauss-Kronrod quadrature routines (arbitrary precision),
# originally written and contributed to Julia by Steven G. Johnson, 2013.
#
# This file was formerly a part of Julia. License is MIT: http://julialang.org/license

"""
The `QuadGK` module implements 1d numerical integration by an adaptive Gauss–Kronrod algorithm.

The package provides three functions: `quadgk`, `gauss`, and `kronrod`.

* `quadgk` performs adaptive integration over arbitrary intervals
* `gauss` computes Gaussian quadrature points and weights for integrating over the interval [-1, 1]
* `kronrod` computes Kronrod points, weights, and embedded Gaussian quadrature weights for integrating over [-1, 1].

Typical usage looks like:
```
using QuadGK
integral, err = quadgk(x -> exp(-x^2), 0, 1, rtol=1e-8)
```
which computes the integral of exp(–x²) from x=0 to x=1 to a relative tolerance of 10⁻⁸,
and returns the approximate `integral = 0.746824132812427` and error estimate
`err = 7.887024366937112e-13`.
"""
module QuadGK

export quadgk, gauss, kronrod, cauchy

using DataStructures, LinearAlgebra
import Base.Order.Reverse

include("gausskronrod.jl")
include("evalrule.jl")
include("adapt.jl")
include("weightedgauss.jl")
include("cauchy.jl")

end # module QuadGK
