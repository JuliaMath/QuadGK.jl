# Gaussian quadrature and arbitrary weight functions

The manual chapter on [Gauss and Gauss–Kronrod quadrature rules](@ref) explains the fundamentals
of numerical integration ("quadrature") to approximate $\int_a^b f(x) dx$ by a weighted sum of
$f(x_i)$ values at quadrature points $x_i$.  Make sure you understand that chapter before reading
this one!

More generally, one can compute quadrature rules for
a **weighted** integral:
```math
\int_a^b w(x) f(x) dx \approx \sum_{i=1}^n w_i f(x_i) \, ,
```
where the effect of **weight function** $w(x)$ (usually required to be $≥ 0$ in ``(a,b)``) is
included in the quadrature weights $w_i$ and points $x_i$.  The main motivation
for weighted quadrature rules is to handle *poorly behaved* integrands — singular, discontinuous, highly oscillatory, and so on — where the "bad" behavior is *known*
and can be *factored out* into $w(x)$.  By designing a quadrature rule with $w(x)$
taken into account, one can obtain fast convergence as long as the remaining
factor $f(x)$ is smooth, regardless of how "bad" $w(x)$ is.  Moreover, the rule
can be re-used for many different $f(x)$ as long as $w(x)$ remains the same.

Gaussian quadrature is ideally suited to designing weighted quadrature rules, and QuadGK
includes functions to construct the points $x_i$ and weights $w_i$ for nearly any desired weight
function $w(x) \ge 0$, in principle, to any desired precision.   The case of Gauss–Kronrod rules (if you
want an error estimate) is a bit trickier: it turns out that Gauss–Kronrod rules may not exist
for arbitrary weight functions \[see the review in [Notaris (2016)](https://etna.ricam.oeaw.ac.at/vol.45.2016/pp371-404.dir/pp371-404.pdf)\], but if a (real-valued) rule *does* exist then QuadGK can compute it for you (to arbitrary precision) using an algorithm by [Laurie (1997)](https://www.ams.org/journals/mcom/1997-66-219/S0025-5718-97-00861-2/S0025-5718-97-00861-2.pdf).  You can specify the weight function $w(x)$ and the interval $(a,b)$ in one of two
ways:

* Via the [Jacobi matrix](https://en.wikipedia.org/wiki/Jacobi_operator) of the [orthogonal polynomials](https://en.wikipedia.org/wiki/Orthogonal_polynomials) associated with this weighted integral.  That may sound complicated, but it turns out that these are tabulated for many important weighted integrals.  For example, all of the weighted integrals in the [FastGaussQuadrature.jl](https://github.com/JuliaApproximation/FastGaussQuadrature.jl) package are based on well-known recurrences that you can look up easily.
* By explicitly providing the weight function $w(x)$, in which case QuadGK can perform a sequence of numerical integrals of $w(x)$ against polynomials (using `quadgk`) to numerically construct the Jacobi matrix and hence the Gauss or Gauss–Kronrod quadrature rule.  (This can be computationally expensive, especially to attain high accuracy, but it can still be worthwhile if you re-use the quadrature rule for many different $f(x)$ and/or $f(x)$ is extremely computationally expensive.)

## Weight functions and Jacobi matrices

For any weighted integral $I[f] = \int_a^b w(x) f(x)$ with non-negative $w(x)$, there is an associated set of [orthogonal polynomials](https://en.wikipedia.org/wiki/Orthogonal_polynomials) $p_k(x)$ of degrees $k = 0,1,\ldots$, such that $I[p_j p_k] = 0$ for $j \ne k$.   Amazingly,
the $n$-point Gaussian quadrature points $x_i$ are simply the roots of $p_n(x)$, and in general there is a
deep relationship between quadrature and the theory of orthogonal polynomials.   A key part of this theory
ends up being the [Jacobi matrix](https://en.wikipedia.org/wiki/Jacobi_operator) describing the three-term
recurrence of these polynomials for a given weighted integral.

It turns out that orthogonal polynomials always obey a three-term recurrence relationship
```math
p_{k+1}(x) = (a_k x + b_k)p_k(x) - c_k q_{k-1}(x)
```
for some coefficients $a_k > 0$, $b_k$, and $c_k>0$ that depend on the integral $I$. By a rescaling
$p_k = q_k \prod_{j<k} a_k$, this simplifies to:
```math
q_{k+1}(x) = (x - \alpha_k)q_k(x) - \beta_k q_{k-1}(x)
```
for  coefficients $\alpha_k = -b_k/a_k$ and $\beta_k = c_k/a_k a_{k-1} > 0$.  (Once you know these coefficients,
in fact, you can obtain all of the orthogonal polynomials by $q_{-1}=0, q_0=1, q_1=(x-\alpha_0), q_2=(x-\alpha_1)(x-\alpha_0) - \beta_1,\ldots$.)
The coefficients are also associated with an infinite real-symmetric tridiagonal ["Jacobi" matrix](https://en.wikipedia.org/wiki/Jacobi_operator):
```math
J = \begin{pmatrix}
\alpha_0 & \sqrt{\beta_1} & & & \\
\sqrt{\beta_1} & \alpha_1 & \sqrt{\beta_2} & & \\
& \sqrt{\beta_2} & \alpha_2 & \sqrt{\beta_3} & \\
& & \ddots & \ddots & \ddots
\end{pmatrix} .
```
Let $J_n$ be the $n \times n$ upper-left corner of $J$.   Astonishingly, the
quadrature points $x_i$ turn out to be exactly the eigenvalues of $J$, and the quadrature weights $w_i$
are the first components² of the corresponding normalized eigenvectors, scaled by $I[1]$ [(Golub & Welch, 1968)](https://www.ams.org/journals/mcom/1969-23-106/S0025-5718-69-99647-1/S0025-5718-69-99647-1.pdf)!

Given the $n \times n$ matrix $J_n$ (represented by a [`LinearAlgebra.SymTridiagonal`](https://docs.julialang.org/en/v1/stdlib/LinearAlgebra/#LinearAlgebra.SymTridiagonal) object, which only stores the $\alpha_k$ and $\sqrt{\beta_k}$ coefficients) and the integral `unitintegral` $= I[1]$, you can construct the points $x_i$ and
weights $w_i$ of the $n$-point Gaussian quadrature rule in QuadGK via `x, w = gauss(Jₙ, unitintegral)`.  To construct
the $(2n+1)$-point Kronrod rule, then you need the $m \times m$ matrix $J_m$ where `m ≥ div(3n+3,2)` ($m \ge \lfloor (3n+3)/2 \rfloor$), and then obtain the points `x` and weights `w` (along with embedded Gauss weights `gw`) via `x, w, gw = kronrod(Jₘ, n, unitintegral)`.  Much of the time, you can simply look up formulas for the recurrence relations
for weight functions of common interest.   Hopefully, this will be clearer with some examples below.

### Gauss–Legendre quadrature via the Jacobi matrix

The common case of integrals $I[f] = \int_{-1}^{+1} f(x) dx$, corresponding to the weight function $w(x) = 1$ over
the interval $(-1,1)$, leads to the [Legendre polynomials](https://en.wikipedia.org/wiki/Legendre_polynomials):
```math
p_0(x) = 1, \; p_1(x) = x, \; p_2(x) = (3x^2 - 1)/2, \; \ldots
```
which satisfy the recurrence ([found on Wikipedia](https://en.wikipedia.org/wiki/Legendre_polynomials#Recurrence_relations) and in many other sources):
```math
(k+1)p_{k+1}(x) = (2k+1)x p_k(x) - k p_{k-1}(x) \, .
```
In the notation given above, that corresponds to coefficients $a_k = (2k+1)/(k+1)$, $b_k = 0$, and $c_k = k/(k+1)$,
or equivalently $\alpha_k = 0$ and $\beta_k = c_k/a_k a_{k-1} = k^2 / (4k^2 - 1)$, giving a Jacobi matrix:
```math
J = \begin{pmatrix}
0 & \sqrt{1/3} & & & \\
\sqrt{1/3} & 0 & \sqrt{4/15} & & \\
& \sqrt{4/15} & 0 & \sqrt{9/35} & \\
& & \ddots & \ddots & \ddots
\end{pmatrix} .
```
which can be constructed in Julia by
```
julia> using LinearAlgebra # for SymTridiagonal

julia> J(n) = SymTridiagonal(zeros(n), [sqrt(k^2/(4k^2-1)) for k=1:n-1]) # the n×n matrix Jₙ
J (generic function with 1 method)

julia> J(5)
5×5 SymTridiagonal{Float64, Vector{Float64}}:
 0.0      0.57735    ⋅         ⋅         ⋅
 0.57735  0.0       0.516398   ⋅         ⋅
  ⋅       0.516398  0.0       0.507093   ⋅
  ⋅        ⋅        0.507093  0.0       0.503953
  ⋅        ⋅         ⋅        0.503953  0.0
```
The unit integral is simply $I[1] = \int_{-1}^{+1} dx = 2$, so we can construct our $n$-point Gauss rule with, for example:
```
julia> x, w = gauss(J(5), 2); [x w]
5×2 Matrix{Float64}:
 -0.90618   0.236927
 -0.538469  0.478629
  0.0       0.568889
  0.538469  0.478629
  0.90618   0.236927
```
This is, of course, the same as the "standard" Gaussian quadrature rule, returned by `gauss(n)`:
```
julia> x, w = gauss(5); [x w]
5×2 Matrix{Float64}:
 -0.90618   0.236927
 -0.538469  0.478629
  0.0       0.568889
  0.538469  0.478629
  0.90618   0.236927
```
Similarly, the 5-point Gauss–Kronrod rule can be constructed from the $9\times 9$ Jacobi matrix ($9 = (3n+3)/2$):
```
julia> x, w, gw = kronrod(J(9), 5, 2); [x w]
11×2 Matrix{Float64}:
 -0.984085  0.042582
 -0.90618   0.115233
 -0.754167  0.186801
 -0.538469  0.24104
 -0.27963   0.27285
  0.0       0.282987
  0.27963   0.27285
  0.538469  0.24104
  0.754167  0.186801
  0.90618   0.115233
  0.984085  0.042582
```
which is the same as the "standard" Gauss–Kronrod rule returned by `kronrod(n)` (returning only the $x_i \le 0$ points) or `kronrod(n, -1, +1)` (returning all the points):
```
julia> x, w, gw = kronrod(5); [x w]
6×2 Matrix{Float64}:
 -0.984085  0.042582
 -0.90618   0.115233
 -0.754167  0.186801
 -0.538469  0.24104
 -0.27963   0.27285
  0.0       0.282987
```

Notice that, in this case, our Jacobi matrix had zero diagonal entries $\alpha_k = 0$.  It turns out that this *always* happens for a weight function $w(x)$ that is *symmetric* in the integration interval $(a,b)$, in this case meaning $w(x)=w(-x)$.   This is called a "hollow" tridiagonal matrix, and its eigenvalues always come in $\pm x_j$ pairs: the quadrature rule is has *symmetric points and weights*.   In this case QuadGK can do its computations a bit more efficiently, and only compute the non-redundant $x_i \le 0$ half of of the quadrature rule, if you represent $J_n$ with a special type [`QuadGK.HollowSymTridiagonal`](@ref) whose constructor only requires you to supply the off-diagonal elements $\sqrt{\beta_k}$:
```
julia> Jhollow(n) = QuadGK.HollowSymTridiagonal([sqrt(k^2/(4k^2-1)) for k=1:n-1])
Jhollow (generic function with 1 method)

julia> Jhollow(5)
5×5 QuadGK.HollowSymTridiagonal{Float64, Vector{Float64}}:
  ⋅       0.57735    ⋅         ⋅         ⋅
 0.57735   ⋅        0.516398   ⋅         ⋅
  ⋅       0.516398   ⋅        0.507093   ⋅
  ⋅        ⋅        0.507093   ⋅        0.503953
  ⋅        ⋅         ⋅        0.503953   ⋅

julia> x, w = gauss(Jhollow(5), 2); [x w]
5×2 Matrix{Float64}:
 -0.90618   0.236927
 -0.538469  0.478629
  0.0       0.568889
  0.538469  0.478629
  0.90618   0.236927

julia> x, w, gw = kronrod(Jhollow(9), 5, 2); [x w]
6×2 Matrix{Float64}:
 -0.984085  0.042582
 -0.90618   0.115233
 -0.754167  0.186801
 -0.538469  0.24104
 -0.27963   0.27285
  0.0       0.282987
```

### Gauss–Jacobi quadrature via the Jacobi matrix

## Arbitrary weight functions

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
