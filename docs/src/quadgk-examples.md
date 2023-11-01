# `quadgk` examples

The following are several examples illustrating the usage of the main [`quadgk`](@ref) numerical-integration function of QuadGK, focusing on more complicated circumstances than the smooth scalar integral of the [Quick start](@ref) section.

## Improper integrals: Infinite limits

`quadgk` supports "improper" integrals over infinite and semi-infinite intervals,
simply by passing `±Inf` for the endpoints.

For example, $\int_0^\infty e^{-x} dx = 1$ is computed by:
```
julia> quadgk(x -> exp(-x), 0, Inf)
(1.0, 4.507383379289404e-11)
```
which give gives the correct answer (1) exactly in this case.  Note that
the error estimate `≈ 4.5e-11` is pessimistic, as is often the case.

The [Gaussian integral](https://en.wikipedia.org/wiki/Gaussian_integral)
$\int_{-\infty}^{+\infty} e^{-x^2} dx = \sqrt{\pi} = 1.772453850905516027298167483341\ldots$ is computed by:
```
julia> quadgk(x -> exp(-x^2), -Inf, Inf)
(1.7724538509055137, 6.4296367126505234e-9)
```
which is the correct answer to nearly [machine precision](https://en.wikipedia.org/wiki/Machine_epsilon), despite the pessimistic error estimate `≈ 6.4e-9`.

Internally, `quadgk` handles infinite limits by the [changes of variables](https://en.wikipedia.org/wiki/Integration_by_substitution)
```math
\int_a^\infty f(x)dx = \int_0^1 f\left(a + \frac{t}{1-t}\right) \frac{1}{(1-t)^2} dt
```
and
```math
\int_{-\infty}^\infty f(x)dx = \int_{-1}^1 f\left(\frac{t}{1-t^2}\right) \frac{1+t^2}{(1-t^2)^2} dt
```
respectively.   Although the transformed integrands are singular at the endpoints
$t = 1$ and $t = \pm 1$, respectively, because the singularities are integrable
and `quadgk` *never evaluates the integrand exactly at the endpoints*, it is able
to perform the numerical integration successfully.

Important tip: This change-of-variables trick works best if your function decays with $|x|$ over a lengthscale of order $\sim 1$.
If your decay length is much larger or shorter than that, it will perform poorly.  For example, with $f(x) = e^{-x/10^6}$ the
decay is over a lengthscale $\sim 10^6$ and `quadgk` requires many more evaluations (705) than for $e^{-x}$ (135, as measured by [`quadgk_count`](@ref),
described below):
```
julia> quadgk_count(x -> exp(-x), 0, Inf)
(1.0, 4.507382674563286e-11, 135)

julia> quadgk_count(x -> exp(-x/1e6), 0, Inf)
(1.000000000001407e6, 0.0014578728207218505, 705)
```
If your function decays over a lengthscale $\sim L$, it is a good idea to compute improper integrals using a change of variables
$\int_a^b f(x)dx = \int_{a/L}^{b/L} f(uL) L \, du$, for example:
```
julia> L = 1e6 # decay lengthscale
1.0e6

julia> f(x) = exp(-x/L)
f (generic function with 1 method)

julia> quadgk_count(u -> f(u*L) * L, 0, Inf) # rescaled integration over u = x/L
(1.0e6, 4.507388807828416e-5, 135)
```

## Counting and printing integrand evaluations

Often, it is a good idea to count the number of times the integrand is evaluated,
in order to have a sense of how efficiently `quadgk` is performing the integral;
this is especially useful with badly behaved integrand (e.g. with singularities,
discontinuities, sharp spikes, and/or rapid oscillations) to see whether some
transformation of the problem might be helpful (see below).

This is easy enough by simply incrementing a global counter in your integrand
function and printing progress as desired.    But it is such a common desire
in pedagogy and debugging that we provide convenience functions [`quadgk_count`](@ref)
and [`quadgk_print`](@ref) to automate this task.

For example, in our $\int_0^\infty e^{-x} dx$ example from above, we could do:
```
julia> quadgk_count(x -> exp(-x), 0, Inf)
(1.0, 4.507383379289404e-11, 135)
```
to return the number of function evaluations (`135`) along with the integral (`1.0`)
and error estimate (`≈ 4.5e-11`).   A relatively large number of function evaluations
are required, even though the function $e^{-x}$ is very smooth, because the infinite
endpoint implicitly introduces a singularity (via the change of variables discussed
above).

We can also print the evaluation points, setting a lower requested relative accuracy of `rtol=1e-2` so that we don't get so much output, by:
```
julia> quadgk_print(x -> exp(-x), 0, Inf, rtol=1e-2)
f(1.0) = 0.36787944117144233
f(0.655923922306948) = 0.5189623601162878
f(1.52456705113438) = 0.21771529605384055
f(0.026110451522445993) = 0.9742274787577301
f(38.29883980138536) = 2.3282264081806294e-17
f(0.00429064542600238) = 0.9957185462423211
f(233.06516868994527) = 6.04064500678147e-102
f(0.14841469193194298) = 0.8620735458624529
f(6.737877409458626) = 0.0011851600983136456
f(0.07246402202084404) = 0.9300992091482783
f(13.799951646519888) = 1.015680581505947e-6
f(0.42263178703605514) = 0.6553198859704107
f(2.3661258586654506) = 0.09384358625528809
f(0.26096469051417465) = 0.7703081183184751
f(3.831936029467112) = 0.021667625834125973
(0.9999887201849575, 0.0009180738585039538, 15)
```
to see that (for a relatively low requested relative accuracy of `rtol=1e-2`) it
evaluates the integrand only 15 times at points from $x \approx 0.00429$ to $x \approx 233.1$, and still managed to get about 5 significant digits correct.  (Note that
`quadgk_print` again returns a 3-element tuple, like `quadgk_count`, where the
third element is the number of integrand evaluations.)

## Integrands with singularities and discontinuities

The integral $\int_0^1 x^{-1/2} dx = \left. 2 \sqrt{x} \right|_0^1 = 2$ is perfectly finite even though the integrand $1/\sqrt{x}$ blows up at $x=0$.  This is an example
of an *integrable singularity*, and `quadgk` can compute this integral:
```
julia> quadgk_count(x -> 1/sqrt(x), 0, 1)
(1.9999999845983916, 2.3762511924588765e-8, 1305)
```
Notice the large number (`1305`) of integrand evaluations returned by
[`quadgk_count`](@ref): this is an indication of how much more work it
is to evaluate an integral with a singularity or any other form of non-smoothness.
At its heart, the Gauss–Kronrod algorithm employed by `quadgk` works by
interpolating the integrand with polynomials over segments of the domain,
and polynomials are bad at representing [non-analytic functions](https://en.wikipedia.org/wiki/Analytic_function) like $1/\sqrt{x}$.

The good news is that `quadgk` **never evaluates functions exactly at the endpoints**,
so it is okay if your function blows up or errors at those points.   (However, you may have
to relax your error tolerance because of the slow convergence, and floating-point limitations
may prevent `quadgk` from reaching very low error tolerances for singular integrands.)   Of
course, it is always better to remove the singularity by some analytical transformation if you
can.  For example, if you need $\int_0^a f(x) x^{-1/2} dx$, you can do a change of variables $x = y^2$
to obtain an equivalent integral $\int_0^\sqrt{a} f(y^2) 2 dy$ that has no singularity and will
therefore converge *much* more quickly.

If your integrand blows up (or has *any* singularity or discontinuity) in the *interior* of the integration domain,
you should *add an extra "endpoint"* at that point to make sure we never evaluate it.
(Also, `quadgk` can often converge more quickly if you tell it where your singularities
are via the endpoints.)  For example, suppose we are integrating
$\int_0^2 |x-1|^{-1/2} dx = 4$, which has an (integrable) singularity at $x=1$.
If we don't tell `quadgk` about the singularity, it gets "unlucky" and evaluates
the integrand exactly at $x=1$, which ends up throwing an error:
```
julia> quadgk_count(x -> 1/sqrt(abs(x-1)), 0, 2)
ERROR: DomainError with 1.0:
integrand produced NaN in the interval (0, 2)
...
```
Instead, if we *tell* it to subdivide the integral at $x=1$, we get the correct answer(`≈ 4`):
```
julia> quadgk_count(x -> 1/sqrt(abs(x-1)), 0, 1, 2)
(3.9999999643041515, 5.8392038954259235e-8, 2580)
```

In general, the syntax `quadgk(f, a, b, c, ...)` denotes the integral
$\int_a^b f(x)dx + \int_b^c f(x)dx  + \cdots$, and `quadgk` never evaluates
the integrand $f(x)$ exactly at the endpoints $a, b, c, \ldots$.

As another example, consider an integral $\int_0^3 H(x-1) = 2$ of the discontinuous
[Heaviside step function](https://en.wikipedia.org/wiki/Heaviside_step_function)
$H(x)$, which $=1$ when $x > 0$ and $=0$ when $x \le 0$:
```
julia> quadgk_count(x -> x > 1, 0, 3)
(2.0000000043200235, 1.7916158219741817e-8, 705)
```
Even though $H(x-1)$ is nearly constant, `quadgk` struggles to integrate it (`705`
function evaluations to get about 8 digits), thanks to the discontinuity at $x=1$.
(Note that `true` and `false` in Julia are equal to numeric `0` and `1`, which is
why we could implement $H(x-1)$ as simply `x > 1`.)
On the other hand, if we *tell it* the location of the discontinuity:
```
julia> quadgk_count(x -> x > 1, 0, 1, 3)
(2.0, 0.0, 30)
```
then it gives the *exact* answer in only 30 evaluations.  The reason it takes
30 evaluations is because `quadgk` defaults to 7th-order Gauss–Kronrod integration
rule, which uses 15 points to interpolate with a high-degree polynomial.  Once
we subdivide the integral, we could actually get away with a lower-order rule
by setting the `order` parameter, e.g.:
```
julia> quadgk_count(x -> x > 1, 0, 1, 3, order=1)
(2.0, 0.0, 6)
```

## Complex and vector-valued integrands

The integrand `f(x)` can return not just real numbers, but also complex numbers, vectors, matrices, or any Julia type supporting `±`, multiplication by scalars, and `norm` (i.e. implementing any [Banach space](https://en.wikipedia.org/wiki/Banach_space)).

For example, we can integrate $1/\sqrt{x}$ from $x=-1$ to $x=1$, where we
[tell the `sqrt` function to return a complex result](https://docs.julialang.org/en/v1/manual/faq/#faq-domain-errors) for negative arguments:
```
julia> quadgk(x -> 1/sqrt(complex(x)), -1, 0, 1)
(1.9999999891094182 - 1.9999999845983916im, 4.056765398346683e-8)
```
which correctly gives $\approx 2 - 2i$.  Note that we explicitly put an
endpoint at $x=0$ to tell `quadgk` about the singularity at that point,
as described above.

Or let's integrate the vector-valued function $f(x) = [1, x, x^2, x^3]$ for $x \in (0,1)$:
```
julia> quadgk(x -> [1,x,x^2,x^3], 0, 1)
([1.0, 0.5, 0.3333333333333333, 0.25], 6.206335383118183e-17)
```
which correctly returns $\approx [1, \frac{1}{2}, \frac{1}{3}, \frac{1}{4}]$.  Note that the error estimate
in this case is an approximate bound on the [norm](https://en.wikipedia.org/wiki/Norm_(mathematics)) of the error, as computed by the [`LinearAlgebra.norm`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.norm) function in Julia.  It defaults to the Euclidean (L2) norm, but you can change this with the `norm`
argument:
```
julia> quadgk(x -> [1,x,x^2,x^3], 0, 1, norm=v->maximum(abs, v))
([1.0, 0.5, 0.3333333333333333, 0.25], 5.551115123125783e-17)
```
e.g. to use the [maximum norm](https://en.wikipedia.org/wiki/Chebyshev_distance)
or some other norm (e.g. a weighted norm if the different components have different
units or have unequal error tolerances).

For integrands whose values are *small* arrays whose length is known at compile time,
it is [usually most efficient](https://docs.julialang.org/en/v1/manual/performance-tips/#Consider-StaticArrays.jl-for-small-fixed-size-vector/matrix-operations) to modify your integrand to return
an `SVector` from the [StaticArrays.jl package](https://github.com/JuliaArrays/StaticArrays.jl).  For the example above:
```
julia> using StaticArrays

julia> integral, error = quadgk(x -> @SVector[1,x,x^2,x^3], 0, 1)
([1.0, 0.5, 0.3333333333333333, 0.25], 6.206335383118183e-17)

julia> typeof(integral)
SVector{4, Float64} (alias for SArray{Tuple{4}, Float64, 1, 4})
```
Note that the return value also gives the `integral` as an `SVector` (a [statically](https://en.wikipedia.org/wiki/Static_program_analysis) sized array).

The QuadGK package did not need any code specific to StaticArrays, and was written long before that package even existed.  The
fact that unrelated packages like this can be [composed](https://en.wikipedia.org/wiki/Composability) is part of the [beauty of multiple dispatch](https://www.youtube.com/watch?v=kc9HwsxE1OY) and [duck typing](https://en.wikipedia.org/wiki/Duck_typing) for [generic programming](https://en.wikipedia.org/wiki/Generic_programming).

## Batched integrand evaluation

User-side parallelization of integrand evaluations is also possible by providing
an in-place function of the form `f!(y,x) = y .= f.(x)`, which evaluates the
integrand at multiple points simultaneously. To use this API, `quadgk`
dispatches on a [`BatchIntegrand`](@ref) type containing `f!` and buffers for
`y` and `x`. These buffers may be pre-allocated and reused for multiple
`BatchIntegrand`s with the same domain and range types.

For example, we can perform multi-threaded integration of a highly oscillatory
function that needs to be refined globally:
```
julia> f(x) = sin(100x)
f (generic function with 1 method)

julia> function f!(y, x)
           n = Threads.nthreads()
           Threads.@threads for i in 1:n
                y[i:n:end] .= f.(@view(x[i:n:end]))
           end
       end
f! (generic function with 1 method)

julia> quadgk(BatchIntegrand{Float64}(f!), 0, 1)
(0.0013768112771231598, 8.493080824940099e-12)
```

Batching also changes how the adaptive refinement is done, which typically leads
to slightly different results and sometimes more integrand evaluations. You
can limit the maximum batch size by setting the `max_batch` parameter
of the [`BatchIntegrand`](@ref), which can be useful in order to set an
upper bound on the size of the buffers allocated by `quadgk`.


## Arbitrary-precision integrals

`quadgk` also supports [arbitrary-precision arithmetic](https://en.wikipedia.org/wiki/Arbitrary-precision_arithmetic) using Julia's [`BigFloat` type](https://docs.julialang.org/en/v1/base/numbers/#BigFloats-and-BigInts) to compute integrals to arbitrary accuracy (albeit at increased computational cost).

For example, we can compute the [error function](https://en.wikipedia.org/wiki/Error_function) $\frac{\sqrt{\pi}}{2} \text{erf}(1) = \int_0^1 e^{-x^2} dx$ to 50 digits by:
```
julia> setprecision(60, base=10) # use 60-digit arithmetic
60

julia> quadgk_count(x -> exp(-x^2), big"0.0", big"1.0", rtol=1e-50)
(0.74682413281242702539946743613185300535449968681260632902766195, 6.8956257635323481758755998484087241330474674891762053644928492e-51, 15345)
```
The correct answer is `≈ 0.746824132812427025399467436131853005354499686812606329027654498958605…`, and
we are matching that to nearly the full precision (≈ 60 digits).  (As usual,
the error estimate of `quadgk` is very conservative for smooth functions.)

Unfortunately, it took 15345 function evaluations to obtain such an accurate
answer.  Since this a smooth integrand, for high-accuracy calculations it is
often advisable to increase the "order" of the quadrature algorithm (which
is related to the degree of polynomials used for interpolation).  The default
is `order=7`, but let's try tripling it to `order=21`:
```
julia> quadgk_count(x -> exp(-x^2), big"0.0", big"1.0", rtol=1e-50, order=21)
(0.74682413281242702539946743613185300535449968681260632902765324, 2.1873898701681913100611385149037136705674736373054902472850425e-58, 129)
```
It got the same accuracy (≈ 60 digits) with only 129 integrand evaluations!

## Contour integration

You can specify a sequence of points in the complex plane to perform a [contour integrals](https://en.wikipedia.org/wiki/Contour_integration) with `quadgk` along a piecewise-linear contour.

For example, consider the function $f(z) = \cos(z)/z$.  By the
[residue theorem](https://en.wikipedia.org/wiki/Residue_theorem)
of complex analysis, if we integrate counter-clockwise in a "loop"
around the [pole](https://en.wikipedia.org/wiki/Zeros_and_poles) at
$z=0$, we should get exactly $2\pi i \cos(0) = 2\pi i$.

One way to do this integral is to [parameterize a contour](https://en.wikipedia.org/wiki/Parametric_equation), say a
circle $|z|=1$ parameterized by $z= e^{i\phi}$ (`= cis(ϕ)` in Julia), which gives $dz = i z\, d\phi$, to
obtain an ordinary integral $\int_0^{2\pi} \frac{\cos(e^{i\phi})}{e^{i\phi}} ie^{i\phi} d\phi$ over the *real* parameter $\phi \in (0,2\pi)$:
```
julia> quadgk(ϕ -> cos(cis(ϕ)) * im, 0, 2π)
(0.0 + 6.283185307179586im, 1.8649646913725044e-8)
```
which indeed gives us $\approx 2\pi i$ (to machine precision).

As an alternative, however, you can directly supply a sequence of
*complex "endpoints"* to `quadgk` and it will perform the contour
integral along a sequence of line segments connecting these points.  For example, instead of integrating around a circular contour, we can integrate
around the diamond (rotated square) connecting the corners $\pm 1$ and $\pm i$:
```
julia> quadgk(z -> cos(z)/z, 1, im, -1, -im, 1)
(0.0 + 6.283185307179587im, 5.369976662961913e-9)
```
which again gives $\approx 2\pi i$ (to machine precision).  Note that it
is critically important to have a *closed* contour (a loop): the final
endpoint must be the same as the starting point ($z=1$).

## Cauchy principal values

Integrands $f(x) = g(x)/x$ that diverge $\sim 1/x$ cannot be integrated
through $x=0$ in the usual way (the singularity is not integrable).
However, if you integrate *around* $x=0$, for both signs of $x$, then you
can define a kind of integral that is the "difference" of the divergence
on the two sides.  This definition is called a [Cauchy principal value](https://en.wikipedia.org/wiki/Cauchy_principal_value), and is usually presented
as a limit:
```math
\text{p.v.}\int_a^b \frac{g(x)}{x} dx =
\lim_{\varepsilon\to 0^+} \left[
    \int_a^{-\varepsilon} \frac{g(x)}{x} dx + \int_{+\varepsilon}^b \frac{g(x)}{x} dx
\right] \, .
```
That is, you subtract a "ball" of radius $\varepsilon$ from the integration
domain $a < 0 < b$ to eliminate the singularity at $x=0$, and take the limit of the resulting integral as the $\varepsilon$ goes to zero.

In principle, you might imagine taking this limit numerically by
extrapolation of numerical integrals for a sequence of $\varepsilon > 0$
values, perhaps using Richardson extrapolation via the [Richardson.jl package](https://github.com/JuliaMath/Richardson.jl).  However, it is
mathematically equivalent and *much* more efficient to use a simple
singularity-subtraction procedure:
```math
\frac{g(x)}{x}  =
\frac{g(x)-g(0)}{x} + \frac{g(0)}{x}
```
where the first term is *not singular* if $g(x)$ is differentiable at $x=0$,
and the latter term can be integrated analytically, giving:
```math
\text{p.v.}\int_a^b \frac{g(x)}{x} dx = \int_a^b \frac{g(x)-g(0)}{x} dx +
g(0) \log|b/a|
```
Since the remaining integral has no singularity, we can do it numerically
directly.  There are two tricks to help us a bit further:

* As in [Integrands with singularities and discontinuities](@ref) above, we'll want to put an extra endpoint at $x=0$ to make sure `quadgk` doesn't evaluate the integrand exactly at that point (which would give `NaN` from $0/0$).
* We should be careful with the integration tolerances, to make sure that any relative tolerance `rtol` is applied with respect to the whole principal part and not just to the $g(x)-g(0)$ integral.  An easy way to do this is to add $g(0) \frac{\log|b/a|}{b-a}$ to the *integrand*, so that we no longer compute the two pieces separately.

Putting it all together, here is a function `cauchy_quadgk(g, a, b)` that
computes our Cauchy principal part $\text{p.v.}\int_a^b g(x)/x$:
```
function cauchy_quadgk(g, a, b; kws...)
    a < 0 < b || throw(ArgumentError("domain must include 0"))
    g₀ = g(0)
    g₀int = b == -a ? zero(g₀) : g₀ * log(abs(b/a)) / (b - a)
    return quadgk_count(x -> (g(x)-g₀)/x + g₀int, a, 0, b; kws...)
end
```
For example, [Mathematica tells us](https://www.wolframalpha.com/input?i=Integrate%5BCos%5Bx%5E2-1%5D%2Fx%2C+%7Bx%2C+-1%2C+2%7D%2C+PrincipalValue+-%3E+True%5D) that
```math
\text{p.v.} \int_{-1}^2 \frac{\cos(x^2-1)}{x} dx \approx 0.212451309942989788929352736695\ldots ,
```
and we can reproduce this with `cauchy_quadgk`:
```
julia> cauchy_quadgk(x -> cos(x^2-1), -1, 2)
(0.21245130994298977, 1.8366794196644776e-11, 60)
```
which is correct to about 16 digits.

This approach and other approaches to computing Cauchy principal
values are discussed in [Keller and Wróbel (2016)](https://doi.org/10.1016/j.cam.2015.08.021).
This kind of "singularity subtraction" is a powerful approach to efficient
computation of integrals with singularities or near singularities.
A huge variety of related techniques have been developed for
[boundary element methods](https://en.wikipedia.org/wiki/Boundary_element_method), where a vast number of singular integrals
must be computed and efficiency is at a premium.  See, for example,
[Reid *et al.* (2014)](http://doi.org/10.1109/TAP.2014.2367492) and
references therein.

## Nearly singular integrands

Even if the integrand is only *nearly* singular, so that there is a
sharp but *finite* peak within the integration domain, it can greatly
increase the efficiency of numerical integration if you can separate
the sharp peak analytically.

For example, suppose that you are integrating:
```math
I = \int_a^b \frac{g(x)}{x - i\alpha} dx
```
for a small $0 < \alpha \ll 1$.  For $\alpha \to 0^+$, it approaches $i\pi g(0)$ plus a Cauchy principal part (the latter being zero if $a = -b$ and $g(x)=g(-x)$), but for small $\alpha > 0$ you have to numerically integrate (for a general function $g(x)$) a function with a sharp spike at $x=0$, which will require a large number of quadrature points.  But you can subtract out the singularity analytically:
```math
I = \int_a^b \left[ \frac{g(x)-g(0)}{x - i\alpha} + \frac{g(0)}{x - i\alpha} \right] dx \\
= \int_a^b \frac{g(x)-g(0)}{x - i\alpha}dx + \underbrace{g(0) \left[\frac{1}{2}\log(x^2 + \alpha^2) + i\tan^{-1}(x/\alpha) \right]_a^b}_{I_0}
```
and then you only need to numerically integrate $I - I_0$, which has the spike subtracted.

As for [Cauchy principal values](@ref) above, we want to include
a $I_0 / (b-a)$ term directly in the integrand so that the error
tolerances are computed correctly, and include $x=0$ as an explicit
endpoint to let `quadgk` know that the integral is badly behaved there.

In code:
```
using QuadGK

function int_slow(g, α, a, b; kws...)
    if a < 0 < b
        # put an explicit endpoint at x=0 since we know it is badly behaved there
        return quadgk_count(x -> g(x) / (x - im*α), a, 0, b; kws...)
    else
        return quadgk_count(x -> g(x) / (x - im*α), a, b; kws...)
    end
end

function int_fast(g, α, a, b; kws...)
    g₀ = g(0)
    denom_int(x) = log(x^2 + α^2)/2 + im * atan(x/α)
    I₀ = g₀ * (denom_int(b) - denom_int(a))
    if a < 0 < b
        # put an explicit endpoint at x=0 since we know it is badly behaved there
        (I,E,c) = quadgk_count(x -> I₀/(b-a) + (g(x) - g₀) / (x - im*α), a, 0, b; kws...)
    else
        (I,E,c) = quadgk_count(x -> I₀/(b-a) + (g(x) - g₀) / (x - im*α), a, b; kws...)
    end
    return (I,E,c+1) # add 1 for g(0) evaluation
end
```
This gives:
```
julia> int_slow(cos, 1e-6, -1, 1)
(1.1102230246251565e-16 + 3.1415896808206125im, 1.4895091715264936e-9, 1230)

julia> int_fast(cos, 1e-6, -1, 1)
(3.3306690738754696e-16 + 3.1415896808190418im, 1.8459683038047577e-12, 31)
```
which agree to about 13 digits, but the slow brute-force method requires 1230 function evaluations while the fast singularity-subtracted method requires only 31 function evaluations.

As an added bonus, `int_fast` works even for `α = 0`, where it gives you $i\pi g(0)$ (for $0 \in (a,b)$) plus the Cauchy principal part as above.
