# QuadGK.jl

This package implements one-dimensional numerical integration
("quadrature") in Julia using adaptive [Gauss–Kronrod quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Kronrod_quadrature_formula).

That is, it computes integrals $\int_a^b f(x) dx$ numerically,
given the endpoints $(a,b)$ and an arbitrary function $f$, to
any desired accuracy, using the function [`quadgk`](@ref).

(The code was originally [part of Base Julia](https://github.com/JuliaLang/julia/pull/3140).)

## Features

Features of the QuadGK package include:

* h-adaptive integration: automatically subdivides the integration integral into smaller segments until a desired accuracy is reached, allowing it to evaluate the integrand more densely in regions where it is badly behaved (e.g. oscillating rapidly).
* arbitrary integrand types: the integrand `f(x)` can return real numbers, complex numbers, vectors, matrices, or any Julia type supporting `±`, multiplication by scalars, and `norm` (i.e. implementing any [Banach space](https://en.wikipedia.org/wiki/Banach_space)).
* Arbitrary precision: arbitrary-precision arithmetic types such as `BigFloat` can be integrated to arbitrary accuracy
* ["Improper" integrals](https://en.wikipedia.org/wiki/Improper_integral): Integral endpoints can be $\pm \infty$ (`±Inf` in Julia).
* [Contour integrals](https://en.wikipedia.org/wiki/Contour_integration): You can specify a sequence of points in the complex plane to perform a contour integral along a piecewise-linear contour.
* Arbitrary-order and custom quadrature rules: Any polynomial `order` of the Gauss–Kronrod quadrature rule can be specified, as well as generating quadrature rules and weights directly; see [Gauss and Gauss–Kronrod quadrature rules](@ref).  Custom Gaussian-quadrature rules can also be constructed for arbitrary weight functions; see [Gaussian quadrature and arbitrary weight functions](@ref).
* In-place integration: For memory efficiency, integrand functions that write in-place into a pre-allocated buffer (e.g. for vector-valued integrands) can be used with the [`quadgk!`](@ref) function, along with pre-allocated buffers using [`alloc_segbuf`](@ref).
* Batched integrand evaluation: Providing an integrand that can evaluate multiple points simultaneously allows for user-controlled parallelization (e.g. using threads, the GPU, or distributed memory).

## Quick start

The following code computes $\int_0^1 \cos(200x) dx$ numerically, to the default accuracy (a [relative error](https://en.wikipedia.org/wiki/Approximation_error) $\lesssim 10^{-8}$), using [`quadgk`](@ref):
```
julia> using QuadGK

julia> integral, error = quadgk(x -> cos(200x), 0, 1)
(-0.004366486486069925, 2.552917865170437e-13)
```
Notice that the result is a [tuple](https://docs.julialang.org/en/v1/manual/functions/#Tuples) of two values: the estimated `integral`
of `-0.004366486486069925`, an estimated upper bound `error ≈ 2.55e-13`
on the [truncation error](https://en.wikipedia.org/wiki/Truncation_error) in the computed integral (due to the finite number of points at which `quadgk` evaluates the integrand).

By default, `quadgk` evaluates the integrand at more and more points ("adaptive quadrature") until
the relative error estimate is less than `sqrt(eps())`, corresponding to about 8 significant digits.  Often, however, you should change this by passing a relative tolerance (`rtol`) and/or an absolute tolerance (`atol`), e.g.:
```
julia> quadgk(x -> cos(200x), 0, 1, rtol=1e-3)
(-0.004366486486069085, 2.569238200052031e-6)
```

For extremely [smooth functions](https://en.wikipedia.org/wiki/Smoothness) like $\cos(200x)$, even though it is highly oscillatory, `quadgk` often gives a very accurate result, even more accurate than the minimum accuracy you requested (defaulting to about 8 digits).  In this particular case, we know that the exact integral is $\sin(200)/200 \approx -0.004366486486069972908665092105754\ldots$, and `integral` matches this to about 14 [significant digits](https://en.wikipedia.org/wiki/Significant_figures) with the default tolerance and to about 13 digits even for `rtol=1e-3`.

## Tutorial examples

The [`quadgk` examples](@ref) chapter of this manual presents several other examples, including improper integrals, vector-valued integrands, batched integrand evaluation, improper integrals, singular or near-singular integrands, and Cauchy principal values.

## In-place operations for array-valued integrands

For integrands whose values are *small* arrays whose length is known at compile-time,
it is usually most efficient to modify your integrand to return
an `SVector` from the [StaticArrays.jl package](https://github.com/JuliaArrays/StaticArrays.jl).

However, for integrands that return large or runtime-length arrays, we also provide a function
`quadgk!(f!, result, a,b...)` in order to exploit in-place operations where possible.   The
`result` argument is used to store the estimated integral `I` in-place, and the integrand function
is now of the form `f!(r, x)` and should write `f(x)` in-place into the result array `r`.  See the [`quadgk!`](@ref) documentation for more detail.

## API Reference

See the [API reference](@ref) chapter for a detailed description of the
QuadGK programming interface.

## Other Julia quadrature packages

The [FastGaussQuadrature.jl](https://github.com/ajt60gaibb/FastGaussQuadrature.jl) package provides
non-adaptive Gaussian quadrature variety of built-in weight functions — it is a good choice you need to go to very high orders $N$, e.g. to integrate rapidly oscillating functions, or use weight functions that incorporate some standard singularity in your integrand.  QuadGK, on the other hand, keeps the order $N$ of the quadrature rule fixed and improves accuracy by subdividing the integration domain, which can be better if fine resolution is required only in a part of your domain (e.g if your integrand has a sharp peak or singularity somewhere that is not known in advance).

For multidimensional integration, see also the [HCubature.jl](https://github.com/stevengj/HCubature.jl), [Cubature.jl](https://github.com/stevengj/Cubature.jl), and
[Cuba.jl](https://github.com/giordano/Cuba.jl) packages.

Note that all of the above quadrature routines assume that you supply you integrand
as a *function* $f(x)$ that can be evaluated at *arbitrary points* inside the
integration domain.  This is ideal, because then the integration algorithm can
choose points so that the accuracy improves rapidly (often exponentially rapidly)
with the number of points.   However if you only have function values supplied
at pre-determined points, such as on a regular grid, then you should use
another (probably slower-converging) algorithm in a package such as
[Trapz.jl](https://github.com/francescoalemanno/Trapz.jl), [Romberg.jl](https://github.com/fgasdia/Romberg.jl), or [NumericalIntegration.jl](https://github.com/dextorious/NumericalIntegration.jl).
