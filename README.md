# QuadGK.jl

[![Build Status](https://travis-ci.org/JuliaMath/QuadGK.jl.svg?branch=master)](https://travis-ci.org/JuliaMath/QuadGK.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaMath/QuadGK.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaMath/QuadGK.jl?branch=master)

Latest release:
[![QuadGK](http://pkg.julialang.org/badges/QuadGK_0.5.svg)](http://pkg.julialang.org/?pkg=QuadGK)
[![QuadGK](http://pkg.julialang.org/badges/QuadGK_0.6.svg)](http://pkg.julialang.org/?pkg=QuadGK)

Documentation:
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaMath.github.io/QuadGK.jl/stable)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaMath.github.io/QuadGK.jl/latest)

This package provides support for one-dimensional numerical integration in Julia using adaptive
Gauss-Kronrod quadrature.
The code was originally part of Base Julia.

The package provides three functions: `quadgk`, `gauss`, and `kronrod`.
`quadgk` performs the integration, `gauss` computes Gaussian quadrature points and weights for integrating
over the interval [-1, 1], and `kronrod` computes Kronrod points, weights, and embedded Gaussian quadrature
weights for integrating over [-1, 1].

For more information, see the documentation.

## Similar packages

The [FastGaussQuadrature.jl](https://github.com/ajt60gaibb/FastGaussQuadrature.jl) package provides
non-adaptive Gaussian quadrature with a wider variety of weight functions.
It should be preferred to this package for higher orders *N*, since the algorithms here are
*O*(*N*<sup>2</sup>) whereas the FastGaussQuadrature algorithms are *O*(*N*).

For multidimensional integration, see the [HCubature.jl](https://github.com/stevengj/HCubature.jl), [Cubature.jl](https://github.com/stevengj/Cubature.jl), and
[Cuba.jl](https://github.com/giordano/Cuba.jl) packages.
