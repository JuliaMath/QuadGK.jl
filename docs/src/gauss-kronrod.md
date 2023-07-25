# Gauss and Gauss–Kronrod quadrature rules

The foundational algorithm of the QuadGK package is a
[Gauss–Kronrod quadrature rule](https://en.wikipedia.org/wiki/Gauss%E2%80%93Kronrod_quadrature_formula), an extension of
[Gaussian quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature).
In this chapter of
the QuadGK manual, we briefly explain what these are, and describe how
you can use QuadGK to generate your own Gauss and Gauss–Kronrod rules,
including for more complicated weighted integrals.

## Quadrature rules and Gaussian quadrature

A **quadrature rule** is simply a way to approximate an integral by
a sum:
```math
\int_a^b f(x) dx \approx \sum_{i=1}^n w_i f(x_i)
```
where the $n$ evaluation points $x_i$ are known as the **quadrature points**
and the coefficients $w_i$ are the **quadrature weights**.   We typically
want to design quadrature rules that are as accurate as possible for
as small an `n` as possible, for a wide range of functions $f(x)$ (for
example, for [smooth functions](https://en.wikipedia.org/wiki/Smoothness)).
The underlying assumption is that evaluating the integrand $f(x)$ is
computationally expensive, so you want to do this as few times as possible
for a given error tolerance.

There are [many numerical-integration techniques](https://en.wikipedia.org/wiki/Numerical_integration) for designing quadrature rules.  For example,
one could simply pick the points $x_i$ uniformly at random in $(a,b)$ and
use a weight $w_i = 1/n$ to take the average — this is [Monte-Carlo integration](https://en.wikipedia.org/wiki/Monte_Carlo_method), which is
simple but converges rather slowly (its error scales as $\sim 1/\sqrt{n}$).

A particularly efficient class of quadrature rules is known as
[Gaussian quadrature](https://en.wikipedia.org/wiki/Gaussian_quadrature),
which exploits the remarkable theory of [orthogonal polynomials](https://en.wikipedia.org/wiki/Orthogonal_polynomials) in order to design $n$-point
rules that *exactly* integrate all polynomial functions $f(x)$ up to degree
$2n-1$.  More importantly, the error goes to zero extremely rapidly
even for non-polynomial $f(x)$, as long as $f(x)$ is sufficiently smooth.
(They converge *exponentially* rapidly for [analytic functions](https://en.wikipedia.org/wiki/Analytic_function).)  There are many variants of
Gaussian quadrature, as we will discuss further below, but the specific
case of computing $\int_{-1}^{1} f(x) dx$ is known as [Gauss–Legendre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Legendre_quadrature), and $\int_a^b f(x) dx$ over other
intervals $(a,b)$ is equivalent to Gauss–Legendre under a simple
change of variables (given explicitly below).

The QuadGK package can compute the points $x_i$ and weights $w_i$ of a Gauss–Legendre quadrature rule (optionally rescaled to an arbitrary interval ``(a,b)``) for you via the [`gauss`](@ref) function.
For example, the $n=5$ point rule for integrating from $a=1$ to $b=3$
is computed by:
```
julia> a = 1; b = 3; n = 5;

julia> x, w = gauss(n, a, b);

julia> [x w] # show points and weights as a 2-column matrix
5×2 Matrix{Float64}:
 1.09382  0.236927
 1.46153  0.478629
 2.0      0.568889
 2.53847  0.478629
 2.90618  0.236927
```
We can see that there are 5 points $a < x_i < b$.  They are *not* equally spaced or equally weighted, nor do they quite reach the endpoints.  We can now approximate integrals by evaluating the integrand $f(x)$ at these points, multiplying by the weights, and summing.  For example, $f(x)=\cos(x)$ can be integrated via:
```
julia> sum(w .* cos.(x)) # evaluate ∑ᵢ wᵢ f(xᵢ)
-0.7003509770773674

julia> sin(3) - sin(1)   # the exact integral
-0.7003509767480293
```
Even with just $n = 5$ points, Gaussian quadrature can integrate such
a smooth function as this to 8–9 significant digits!

The `gauss` function allows you to compute Gaussian quadrature
rules to any desired precision, even supporting [arbitrary-precision arithmetic](https://en.wikipedia.org/wiki/Arbitrary-precision_arithmetic) types such as `BigFloat`.  For example, we can compute the same rule as above to about 30 digits:
```
julia> setprecision(30, base=10);

julia> x, w = gauss(BigFloat, n, a, b); @show x; @show w;
x = BigFloat[1.0938201540613360072023731217019, 1.4615306898943169089636855793001, 2.0, 2.5384693101056830910363144207015, 2.9061798459386639927976268782981]
w = BigFloat[0.23692688505618908751426404072106, 0.47862867049936646804129151483584, 0.56888888888888888888888888888975, 0.47862867049936646804129151483584, 0.23692688505618908751426404072106]
```
This allows you to compute numerical integrals to very high accuracy if you want.  (The [`quadgk`](@ref) function also supports arbitrary-precision arithmetic types.)

## Gauss–Kronrod: Error estimation and embedded rules

A good quadrature rule is often not enough: you also want
to have an **estimate of the error** for a given $f(x)$, in order to
decide whether you are happy with your approximate integral or if you
want to get a more accurate estimate by increasing $n$.

The most basic way to do this is to evaluate *two* quadrature rules, one with
fewer points $n' < n$, and use their *difference* as an error
estimate.  (If the error is rapidly converging with $n$, this is usually
a conservative upper bound on the error.)
```math
\text{error estimate} = \Big|
\underbrace{\sum_{i=1}^n w_i f(x_i)}_{\text{first rule}} -
\underbrace{\sum_{j=1}^{n'} w_j' f(x_j')}_{\text{second rule}}
\Big|
```
Naively, this requires us to evaluate our integrand $f(x)$ an extra
$n'$ times to get the error estimate from the second rule.  However,
we can do better: if the points $\{ x_j' \}$ of the second ($n'$-point) rule
are a *subset* of the points $\{ x_i \}$ of the points from the first
($n$-point) rule, then we only need $n$ function evaluations for the
first rule and can *re-use* them when evaluating the second rule.
This is called an **embedded** (or **nested**) quadrature rule.

There are many ways of designing embedded quadrature rules.  Unfortunately,
the nice Gaussian quadrature rules cannot be directly nested: the $n'$-point
Gaussian quadrature points are *not* a subset of the $n$-point Gaussian
quadrature points for *any* $1 < n' < n$.   Fortunately, there is a slightly
modified scheme that works, called [Gauss–Kronrod quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Kronrod_quadrature_formula): if you start with an $n'$-point Gaussian-quadrature scheme, you can extend it with
$n'+1$ additional points to obtain a quadrature scheme with $n=2n'+1$
points that exactly integrates polynomials up to degree $3n'+1$.
Although this is slightly worse than an $n$-point Gaussian quadrature
scheme, it is still quite accurate, still converges very fast
for smooth functions, and gives you a built-in error estimate that
requires no additional function evaluations.   (In QuadGK, we refer
to the size $n'$ of the embedded Gauss rule as the "order", although
other authors use that term to refer to the degree of polynomials
that are integrated exactly.)

The [`quadgk`](@ref) function uses Gauss–Kronrod quadrature internally,
defaulting to order $n'=7$ (i.e. $n=15$ points), though you can change
this with the `order` parameter.   This gives it both an estimated
integral and an estimated error.  If the error is larger than your requested
tolerance, `quadgk` splits the integration interval into two halves and
applies the same Gauss–Kronrod rule to each half, and continues to
subdivide the intervals until the desired tolerance is achieved, a
process called $h$-[adaptive quadrature](https://en.wikipedia.org/wiki/Adaptive_quadrature).  (An alternative called $p$-adaptive quadrature
would increase the order $n'$ on the same interval.  $h$-adaptive
quadrature is more robust if your function has localized bad behaviors
like sharp peaks or discontinuities, because it will progressively
add more points mostly in these "bad" regions.)

You can use the [`kronrod`](@ref) function to compute a Gauss–Kronrod
rule to any desired order (and to any precision).  For example, we can extend our 5-point Gaussian-quadrature rule for $\int_1^3$ from the previous section to an 11-point (`2n+1`) Gauss-Kronrod rule:
```
julia> x, w, gw = kronrod(n, a, b); [ x w ] # points and weights
11×2 Matrix{Float64}:
 1.01591  0.042582
 1.09382  0.115233
 1.24583  0.186801
 1.46153  0.24104
 1.72037  0.27285
 2.0      0.282987
 2.27963  0.27285
 2.53847  0.24104
 2.75417  0.186801
 2.90618  0.115233
 2.98409  0.042582
```
Similar to Gaussian quadrature, notice that all of the Gauss–Kronrod points
$a < x_i < b$ lie in the interior $(a,b)$ of our integration interval,
and that they are unequally spaced (clustered more near the edges).
The third return value, `gw`, gives the weights of the embedded 5-point
Gaussian-quadrature rule, which corresponds to the *even-indexed* points
`x[2:2:end]` of the 11-point Gauss–Kronrod rule:
```
julia> [ x[2:2:end] gw ] # embedded Gauss points and weights
5×2 Matrix{Float64}:
 1.09382  0.236927
 1.46153  0.478629
 2.0      0.568889
 2.53847  0.478629
 2.90618  0.236927
```
So, we can evaluate our integrand $f(x)$ at the 11 Gauss–Kronrod points, and then re-use 5 of these values to obtain an error estimate.  For example, with $f(x) = \cos(x)$, we obtain:
```
julia> fx = cos.(x); # evaluate f(xᵢ)

julia> integral = sum(w .* fx) # ∑ᵢ wᵢ f(xᵢ)
-0.7003509767480292

julia> error = abs(integral - sum(gw .* fx[2:2:end])) # |integral - ∑ⱼ wⱼ′ f(xⱼ′)|
3.2933822335934337e-10

julia> abs(integral - (sin(3) - sin(1))) # true error ≈ machine precision
1.1102230246251565e-16
```
As noted above, the error estimate tends to actually be quite a conservative
upper bound on the error, because it is effectively a measure of the error of the lower-order *embedded* 5-point Gauss rule rather than that of the higher-order 11-point Gauss–Kronrod rule.  For smooth functions like $\cos(x)$, an 11-point rule can have an error orders of magnitude smaller than that of the 5-point rule.  (Here, the 11-point rule's accuracy
is so good that it is actually limited by [floating-point roundoff error](https://en.wikipedia.org/wiki/Machine_epsilon); in infinite precision the error would have been `≈ 6e-23`.)

You may notice that both the Gauss–Kronrod and the Gaussian quadrature
rules are *symmetric* around the center $(a+b)/2$ of the integration interval.   In fact, we provide a lower-level function `kronrod(n)` that only computes roughly the first half of the points and weights for $\int_{-1}^{1}$ ($b = -a = 1$), corresponding to $x_i \le 0$.
```
julia> x, w, gw = kronrod(5); [x w] # points xᵢ ≤ 0 and weights
6×2 Matrix{Float64}:
 -0.984085  0.042582
 -0.90618   0.115233
 -0.754167  0.186801
 -0.538469  0.24104
 -0.27963   0.27285
  0.0       0.282987

julia> [x[2:2:end] gw] # embedded Gauss points ≤ 0 and weights
3×2 Matrix{Float64}:
 -0.90618   0.236927
 -0.538469  0.478629
  0.0       0.568889
```
Of course, you still have to evaluate $f(x)$ at all $2n+1$ points,
but summing the results requires a bit less arithmetic
and storing the rule takes less memory.  Note also that the $(-1,1)$ rule can be applied to any desired interval $(a,b)$ by a change of variables
```math
\int_a^b f(x) dx = \frac{b-a}{2} \int_{-1}^{+1} f\left( (u+1)\frac{b-a}{2} + a  \right) du \, ,
```
so the $(-1,1)$ rule can be computed once (for a given order and precision) and re-used.  In consequence, `kronrod(n)` is `quadgk` uses internally.  The higher-level `kronrod(n, a, b)` function is more convenient for casual use, however.

As with `gauss`, the `kronrod` function works with arbitrary precision,
such as `BigFloat` numbers.  `kronrod(n, a, b)` uses the precision of
the endpoints `(a,b)` (converted to floating point), while for
`kronrod(n)` you can explicitly pass a floating-point type `T` as
the first argument, e.g. for 50-digit precision:
```
julia> setprecision(50, base=10); x, w, gw = kronrod(BigFloat, 5); x
6-element Vector{BigFloat}:
 -0.9840853600948424644961729346361394995805528241884714
 -0.9061798459386639927976268782993929651256519107625304
 -0.7541667265708492204408171669461158663862998043714845
 -0.5384693101056830910363144207002088049672866069055604
 -0.2796304131617831934134665227489774362421188153561727
  0.0
```

## Quadrature rules for weighted integrals

More generally, one can compute quadrature rules for a **weighted** integral:

```math
\int_a^b W(x) f(x) dx \approx \sum_{i=1}^n w_i f(x_i)
```
where the effect of **weight function** $W(x)$ (usually required to be $≥ 0$ in ``(a,b)``) is
included in the quadrature weights $w_i$ and points $x_i$.   The main motivation
for weighted quadrature rules is to handle *poorly behaved* integrands — singular, discontinuous, highly oscillatory, and so on — where the "bad" behavior is *known*
and can be *factored out* into $W(x)$.  By designing a quadrature rule with $W(x)$
taken into account, one can obtain fast convergence as long as the remaining
factor $f(x)$ is smooth, regardless of how "bad" $W(x)$ is.  Moreover, the rule
can be re-used for many different $f(x)$ as long as $W(x)$ remains the same.

The QuadGK package can compute both Gauss and Gauss–Kronrod quadrature rules
for arbitrary weight functions $W(x)$, to arbitrary precision, as described
in the section: [Gaussian quadrature and arbitrary weight functions](@ref).
