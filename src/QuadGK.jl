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

export quadgk, quadgk!, gauss, kronrod, alloc_segbuf, quadgk_count, quadgk_print

using DataStructures, LinearAlgebra
import Base.Order.Reverse

# an in-place integrand function f!(result, x) and
# temporary mutable data (e.g. arrays) of type R for evalrule
struct InplaceIntegrand{F,R,RI}
    f!::F

    # temporary arrays for evalrule
    fg::R
    fk::R
    Ig::R
    Ik::R
    fx::R
    Idiff::RI

    # final result array
    I::RI
end

InplaceIntegrand(f!::F, I::RI, fx::R) where {F,RI,R} =
    InplaceIntegrand{F,R,RI}(f!, similar(fx), similar(fx), similar(fx), similar(fx), fx, similar(I), I)

struct Sequential end
struct Parallel{T,S}
    f::Vector{T} # array to store function evaluations
    old_segs::Vector{S} # array to store segments popped off of heap
    new_segs::Vector{S} # array to store segments to add to heap
end

include("gausskronrod.jl")
include("evalrule.jl")
include("adapt.jl")
include("weightedgauss.jl")

"""
    Parallel(domain_type=Float64, range_type=Float64, error_type=Float64; order=7)

This helper will allocate a buffer to parallelize `quadgk` calls across function evaluations
with a given `domain_type`, i.e. the type of the integration limits, `range_type`, i.e. the
type of the range of the integrand, and `error_type`, the type returned by the `norm` given
to `quadgk`. The keyword `order` allocates enough memory so that the Gauss-Kronrod rule of
that order can initially be evaluated without additional allocations. By passing this buffer
to multiple compatible `quadgk` calls, they can all be parallelized without repeated
allocation.
"""
function Parallel(TX=Float64, TI=Float64, TE=Float64; order=7)
    Parallel(Vector{TI}(undef, 2*order+1), alloc_segbuf(TX,TI,TE), alloc_segbuf(TX,TI,TE, size=2))
end

end # module QuadGK
