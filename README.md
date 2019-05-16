# QuadGK.jl

[![Build Status](https://travis-ci.org/JuliaMath/QuadGK.jl.svg?branch=master)](https://travis-ci.org/JuliaMath/QuadGK.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaMath/QuadGK.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaMath/QuadGK.jl?branch=master)

Documentation:
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaMath.github.io/QuadGK.jl/stable)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaMath.github.io/QuadGK.jl/latest)

This package provides support for one-dimensional numerical integration in Julia using adaptive
Gauss-Kronrod quadrature.
The code was originally part of Base Julia.

The package provides three functions: `quadgk`, `gauss`, and `kronrod`.
`quadgk` performs the integration, `gauss` computes Gaussian quadrature points and weights for integrating
over the interval [-1, 1], and `kronrod` computes Kronrod points, weights, and embedded Gaussian quadrature
weights for integrating over [-1, 1].   Typical usage looks like:
```jl
using QuadGK
integral, err = quadgk(x -> exp(-x^2), 0, 1, rtol=1e-8)
```
which computes the integral of exp(–x²) from x=0 to x=1 to a relative tolerance of 10⁻⁸, and returns the approximate `integral = 0.746824132812427` and error estimate `err = 7.887024366937112e-13` (which is actually smaller than the requested tolerance: convergence was very rapid because the integrand is smooth).

For more information, see the [documentation](https://JuliaMath.github.io/QuadGK.jl/stable).

## Similar packages

The [FastGaussQuadrature.jl](https://github.com/ajt60gaibb/FastGaussQuadrature.jl) package provides
non-adaptive Gaussian quadrature with a wider variety of weight functions — it is a good choice you need to go to very high orders N, e.g. to integrate rapidly oscillating functions, or use weight functions that incorporate some known singularity in your integrand.  QuadGK, on the other hand, keeps the order N of the quadrature rule fixed and improves accuracy by subdividing the integration domain, which can be better if fine resolution is required only in a part of your domain (e.g if your integrand has a sharp peak or singularity somewhere that is not known in advance).

For multidimensional integration, see the [HCubature.jl](https://github.com/stevengj/HCubature.jl), [Cubature.jl](https://github.com/stevengj/Cubature.jl), and
[Cuba.jl](https://github.com/giordano/Cuba.jl) packages.
