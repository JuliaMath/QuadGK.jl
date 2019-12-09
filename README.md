# QuadGK.jl

[![Build Status](https://travis-ci.org/JuliaMath/QuadGK.jl.svg?branch=master)](https://travis-ci.org/JuliaMath/QuadGK.jl)
[![Coverage Status](https://coveralls.io/repos/github/JuliaMath/QuadGK.jl/badge.svg?branch=master)](https://coveralls.io/github/JuliaMath/QuadGK.jl?branch=master)

Documentation:
[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://JuliaMath.github.io/QuadGK.jl/stable)
[![](https://img.shields.io/badge/docs-latest-blue.svg)](https://JuliaMath.github.io/QuadGK.jl/latest)

This package provides support for one-dimensional numerical integration in Julia using adaptive
Gauss-Kronrod quadrature.
The code was originally part of Base Julia.  It supports integration of arbitrary numeric types,
including arbitrary precision (`BigFloat`), and even integration of arbitrary normed vector spaces
(e.g. matrix-valued integrands).

The package provides three functions: `quadgk`, `gauss`, and `kronrod`.
`quadgk` performs the integration, `gauss` computes Gaussian quadrature points and weights for integrating
over the interval [a, b], and `kronrod` computes Kronrod points, weights, and embedded Gaussian quadrature
weights for integrating over [-1, 1].   Typical usage looks like:
```jl
using QuadGK
integral, err = quadgk(x -> exp(-x^2), 0, 1, rtol=1e-8)
```
which computes the integral of exp(–x²) from x=0 to x=1 to a relative tolerance of 10⁻⁸, and returns the approximate `integral = 0.746824132812427` and error estimate `err = 7.887024366937112e-13` (which is actually smaller than the requested tolerance: convergence was very rapid because the integrand is smooth).

For more information, see the [documentation](https://JuliaMath.github.io/QuadGK.jl/stable).

## Gaussian quadrature and arbitrary weight functions

If you are computing many similar integrals of smooth functions, you may not need an adaptive
integration — with a little experimentation, you may be able to decide on an appropriate number
`N` of integration points in advance, and re-use this for all of your integrals.    In this case
you can use `x, w = gauss(N, a, b)` to find the quadrature points `x` and weights `w`, so that
`sum(f.(x) .* w)` is an `N`-point approximation to `∫f(x)dx` from `a` to `b`.

For computing many integrands of similar functions with *singularities*,
`x, w = gauss(W, N, a, b)` function allows you to pass a *weight function* `W(x)` as the first argument,
so that `sum(f.(x) .* w)` is an `N`-point approximation to `∫W(x)f(x)dx` from `a` to `b`.   In this way,
you can put all of the singularities etcetera into `W` and precompute an accurate quadrature rule as
long as the remaining `f(x)` terms are smooth.   For example,
```jl
using QuadGK
x, w = gauss(x -> exp(-x) / sqrt(x), 10, 0, 20)
```
computes the points and weights for performing `∫exp(-x)f(x)/√x dx` integrals from `0` to `20`, so that
there is a `1/√x` singularity in the integrand at `x=0` and a rapid decay for increasing `x`.    For example, with `f(x) = sin(x)`, the exact answer is `0.57037055568959477…`.  Using the points and weights above with `sum(sin.(x) .* w)`, we obtain `0.5703706546318231`, which is correct to 6–7 digits using only 10 `f(x)` evaluations.  Obtaining similar
accuracy for the same integral from `quadgk` requires nearly 300 function evaluations.   However, the
`gauss` function itself computes many (`2N`) numerical integrals of your weight function (multiplied
by polynomials), so this is only more efficient if your `f(x)` is very expensive or if you need
to compute a large number of integrals with the same `W`.

See the [`gauss` documentation](https://juliamath.github.io/QuadGK.jl/stable/#QuadGK.gauss) for more information.

## Similar packages

The [FastGaussQuadrature.jl](https://github.com/ajt60gaibb/FastGaussQuadrature.jl) package provides
non-adaptive Gaussian quadrature variety of built-in weight functions — it is a good choice you need to go to very high orders N, e.g. to integrate rapidly oscillating functions, or use weight functions that incorporate some standard singularity in your integrand.  QuadGK, on the other hand, keeps the order N of the quadrature rule fixed and improves accuracy by subdividing the integration domain, which can be better if fine resolution is required only in a part of your domain (e.g if your integrand has a sharp peak or singularity somewhere that is not known in advance).

For multidimensional integration, see the [HCubature.jl](https://github.com/stevengj/HCubature.jl), [Cubature.jl](https://github.com/stevengj/Cubature.jl), and
[Cuba.jl](https://github.com/giordano/Cuba.jl) packages.
