# Gaussian quadrature and arbitrary weight functions

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
x, w = gauss(x -> exp(-x) / sqrt(x), 10, 0, -log(1e-10), rtol=1e-9)
```
computes the points and weights for performing `∫exp(-x)f(x)/√x dx` integrals from `0` to `-log(1e-10) ≈ 23`, so that there is a `1/√x` singularity in the integrand at `x=0` and a rapid decay for increasing `x`.  (The `gauss` function currently does not support infinite integration intervals, but for a rapidly decaying weight function you can approximate an infinite interval to any desired accuracy by a sufficiently broad interval, with a tradeoff in computational expense.)  For example, with `f(x) = sin(x)`, the exact answer is `0.570370556005742…`.  Using the points and weights above with `sum(sin.(x) .* w)`, we obtain `0.5703706212868831`, which is correct to 6–7 digits using only 10 `f(x)` evaluations.  Obtaining similar
accuracy for the same integral from `quadgk` requires nearly 300 function evaluations.   However, the
`gauss` function itself computes many (`2N`) numerical integrals of your weight function (multiplied
by polynomials), so this is only more efficient if your `f(x)` is very expensive or if you need
to compute a large number of integrals with the same `W`.

See the [`gauss`](@ref) documentation for more information.  See also our example using a [weight function interpolated from tabulated data](https://nbviewer.jupyter.org/urls/math.mit.edu/~stevenj/Solar-Quadrature.ipynb).