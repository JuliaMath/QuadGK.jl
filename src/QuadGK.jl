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

struct BatchIntegrand{F,Y,X}
    # in-place function f!(y, x) that takes an array of x values and outputs an array of results in-place
    f!::F
    y::Vector{Y}
    x::Vector{X}
    max_batch::Int # maximum number of x to supply in parallel (defaults to typemax(Int))
    function BatchIntegrand{F,Y,X}(f!::F, y::Vector{Y}, x::Vector{X}, max_batch::Int) where {F,Y,X}
        max_batch > 0 || throw(ArgumentError("maximum batch size must be positive"))
        return new{F,Y,X}(f!, y, x, max_batch)
    end
end

BatchIntegrand(f!::F, y::Vector{Y}, x::Vector{X}; max_batch::Integer=typemax(Int)) where {F,Y,X} =
    BatchIntegrand{F,Y,X}(f!, y, x, max_batch)
BatchIntegrand(f!::F, ::Type{Y}, ::Type{X}=Nothing; kwargs...) where {F,Y,X} =
    BatchIntegrand(f!, Y[], X[]; kwargs...)

include("gausskronrod.jl")
include("evalrule.jl")
include("evalsegs.jl")
include("adapt.jl")
include("weightedgauss.jl")

end # module QuadGK
