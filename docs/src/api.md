# API reference

## `quadgk`

The most commonly used function from the QuadGK package is the `quadgk` function, used to compute numerical integrals (by h-adaptive Gauss–Kronrod quadrature):

```@docs
QuadGK.quadgk
```

The `quadgk` function also has variants `quadgk_count` (which also returns a count of the integrand evaluations), `quadgk_print` (which also prints each integrand evaluation), `quadgk!` (which implements an in-place API for array-valued functions), as well as an `alloc_segbuf` function to pre-allocate
internal buffers used by `quadgk`:

```@docs
QuadGK.quadgk_count
QuadGK.quadgk_print
QuadGK.quadgk!
QuadGK.alloc_segbuf
```

## Gauss and Gauss–Kronrod rules

For more specialized applications, you may wish to construct your own Gauss or Gauss–Kronrod quadrature rules, as described in [Gauss and Gauss–Kronrod quadrature rules](@ref).   To compute rules for $\int_{-1}^{+1} f(x) dx$ and $\int_a^b f(x) dx$ (unweighted integrals), use:

```@docs
QuadGK.gauss(::Type, ::Integer)
QuadGK.kronrod(::Type, ::Integer)
```

More generally, to compute rules for $\int_a^b w(x) f(x) dx$ (weighted integrals, as described in [Gaussian quadrature and arbitrary weight functions](@ref)), use the following methods if you know the [Jacobi matrix](https://en.wikipedia.org/wiki/Jacobi_operator) for the orthogonal
polynomials associated with your weight function:

```@docs
QuadGK.gauss(::AbstractMatrix, ::Real)
QuadGK.kronrod(::AbstractMatrix, ::Integer, ::Real)
QuadGK.HollowSymTridiagonal
```

Most generally, if you know only the weight function $w(x)$ and the interval $(a,b)$, you
can construct Gauss and Gauss–Kronrod rules completely numerically using:

```@docs
QuadGK.gauss(::Any, ::Integer, ::Real, ::Real)
QuadGK.kronrod(::Any, ::Integer, ::Real, ::Real)
```
