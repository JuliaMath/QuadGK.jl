# QuadGK.jl

[![Coverage Status](https://coveralls.io/repos/github/JuliaMath/QuadGK.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaMath/QuadGK.jl?branch=master)

Documentation:
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaMath.github.io/QuadGK.jl/stable)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaMath.github.io/QuadGK.jl/dev)

This package provides support for one-dimensional numerical integration in Julia using adaptive
Gauss-Kronrod quadrature.
The code was originally part of Base Julia.  It supports integration of arbitrary numeric types,
including arbitrary precision (`BigFloat`), and even integration of arbitrary normed vector spaces
(e.g. matrix-valued integrands).

The package provides three basic functions: `quadgk`, `gauss`, and `kronrod`.
`quadgk` performs the integration, `gauss` computes Gaussian quadrature points and weights for integrating
over the interval [a, b], and `kronrod` computes Kronrod points, weights, and embedded Gaussian quadrature
weights for integrating over [-1, 1].   Typical usage looks like:
```jl
using QuadGK
integral, err = quadgk(x -> exp(-x^2), 0, 1, rtol=1e-8)
```
which computes the integral of exp(–x²) from x=0 to x=1 to a relative tolerance of 10⁻⁸, and returns the approximate `integral = 0.746824132812427` and error estimate `err = 7.887024366937112e-13` (which is actually smaller than the requested tolerance: convergence was very rapid because the integrand is smooth).

For more information, see the [documentation](https://JuliaMath.github.io/QuadGK.jl/stable).
