# Gaussian quadrature and arbitrary weight functions

The manual chapter on [Gauss and Gauss–Kronrod quadrature rules](@ref) explains the fundamentals
of numerical integration ("quadrature") to approximate $\int_a^b f(x) dx$ by a weighted sum of
$f(x_i)$ values at quadrature points $x_i$.  Make sure you understand that chapter before reading
this one!

More generally, one can compute quadrature rules for
a **weighted** integral:
```math
\int_a^b W(x) f(x) dx \approx \sum_{i=1}^n w_i f(x_i) \, ,
```
where the effect of **weight function** $W(x)$ (usually required to be $≥ 0$ in ``(a,b)``) is
included in the quadrature weights $w_i$ and points $x_i$.  The main motivation
for weighted quadrature rules is to handle *poorly behaved* integrands — singular, discontinuous, highly oscillatory, and so on — where the "bad" behavior is *known*
and can be *factored out* into $W(x)$.  By designing a quadrature rule with $W(x)$
taken into account, one can obtain fast convergence as long as the remaining
factor $f(x)$ is smooth, regardless of how "bad" $W(x)$ is.  Moreover, the rule
can be re-used for many different $f(x)$ as long as $W(x)$ remains the same.

Gaussian quadrature is ideally suited to designing weighted quadrature rules, and QuadGK
includes functions to construct the points $x_i$ and weights $w_i$ for nearly any desired weight
function $W(x) \ge 0$, in principle, to any desired precision.   The case of Gauss–Kronrod rules (if you
want an error estimate) is a bit trickier: it turns out that Gauss–Kronrod rules may not exist
for arbitrary weight functions \[see the review in [Notaris (2016)](https://etna.ricam.oeaw.ac.at/vol.45.2016/pp371-404.dir/pp371-404.pdf)\], but if a (real-valued) rule *does* exist then QuadGK can compute it for you (to arbitrary precision) using an algorithm by [Laurie (1997)](https://www.ams.org/journals/mcom/1997-66-219/S0025-5718-97-00861-2/S0025-5718-97-00861-2.pdf).  You can specify the weight function $W(x)$ and the interval $(a,b)$ in one of two
ways:

* Via the [Jacobi matrix](https://en.wikipedia.org/wiki/Jacobi_operator) of the [orthogonal polynomials](https://en.wikipedia.org/wiki/Orthogonal_polynomials) associated with this weighted integral.  That may sound complicated, but it turns out that these are tabulated for many important weighted integrals.  For example, all of the weighted integrals in the [FastGaussQuadrature.jl](https://github.com/JuliaApproximation/FastGaussQuadrature.jl) package are based on well-known recurrences that you can look up easily.
* By explicitly providing the weight function $W(x)$, in which case QuadGK can perform a sequence of numerical integrals of $W(x)$ against polynomials (using `quadgk`) to numerically construct the Jacobi matrix and hence the Gauss or Gauss–Kronrod quadrature rule.  (This can be computationally expensive, especially to attain high accuracy, but it can still be worthwhile if you re-use the quadrature rule for many different $f(x)$ and/or $f(x)$ is extremely computationally expensive.)

## Weight functions and Jacobi matrices

For any weighted integral $I[f] = \int_a^b W(x) f(x)$ with non-negative $W(x)$, there is an associated set of [orthogonal polynomials](https://en.wikipedia.org/wiki/Orthogonal_polynomials) $p_k(x)$ of degrees $k = 0,1,\ldots$, such that $I[p_j p_k] = 0$ for $j \ne k$.   Amazingly,
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

The common case of integrals $I[f] = \int_{-1}^{+1} f(x) dx$, corresponding to the weight function $W(x) = 1$ over
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

Notice that, in this case, our Jacobi matrix had zero diagonal entries $\alpha_k = 0$.  It turns out that this *always* happens when the integration is centered around zero (``a=-b``) and the weight function $W(x)$ that is *symmetric* (``W(x)=W(-x)``).   This is called a "hollow" tridiagonal matrix, and its eigenvalues always come in $\pm x_j$ pairs: the quadrature rule is has *symmetric points and weights*.   In this case QuadGK can do its computations a bit more efficiently, and only compute the non-redundant $x_i \le 0$ half of of the quadrature rule, if you represent $J_n$ with a special type [`QuadGK.HollowSymTridiagonal`](@ref) whose constructor only requires you to supply the off-diagonal elements $\sqrt{\beta_k}$:
```
julia> JholloW(n) = QuadGK.HollowSymTridiagonal([sqrt(k^2/(4k^2-1)) for k=1:n-1])
Jhollow (generic function with 1 method)

julia> JholloW(5)
5×5 QuadGK.HollowSymTridiagonal{Float64, Vector{Float64}}:
  ⋅       0.57735    ⋅         ⋅         ⋅
 0.57735   ⋅        0.516398   ⋅         ⋅
  ⋅       0.516398   ⋅        0.507093   ⋅
  ⋅        ⋅        0.507093   ⋅        0.503953
  ⋅        ⋅         ⋅        0.503953   ⋅

julia> x, w = gauss(JholloW(5), 2); [x w]
5×2 Matrix{Float64}:
 -0.90618   0.236927
 -0.538469  0.478629
  0.0       0.568889
  0.538469  0.478629
  0.90618   0.236927

julia> x, w, gw = kronrod(JholloW(9), 5, 2); [x w] # only returns xᵢ ≤ 0 points:
6×2 Matrix{Float64}:
 -0.984085  0.042582
 -0.90618   0.115233
 -0.754167  0.186801
 -0.538469  0.24104
 -0.27963   0.27285
  0.0       0.282987
```
(The `gauss` function returns all the points, albeit computed more efficiently, while the `kronrod` function returns only the $x_i \le 0$ points for a `HollowSymTridiagonal` Jacobi matrix.)

If you have the Jacobi matrix for one interval, but want QuadGK to rescale the quadrature points and weights to some other interval (rather than doing the change of variables yourself), you can use the method `gauss(J, unitintegral,  (a,b) => (newa, newb))`, where
`(newa,newb)` is the new interval and `unitintegral` is the integral of $f(x)=1$ over
the new interval, and similarly for `kronrod`.  For example, to rescale the Legendre $W(x)=1$ rule from $(-1,+1)$ to the interval $(4,7)$, with unit integral $7-4 = 3$, we could do:
```
julia> x, w = gauss(JholloW(5), 3, (-1,1) => (4,7)); [x w]
5×2 Matrix{Float64}:
 4.14073  0.35539
 4.6923   0.717943
 5.5      0.853333
 6.3077   0.717943
 6.85927  0.35539

julia> x, w = kronrod(JholloW(9), 5, 3, (-1,1) => (4,7)); [x w]
11×2 Matrix{Float64}:
 4.02387  0.0638731
 4.14073  0.17285
 4.36875  0.280201
 4.6923   0.361561
 5.08055  0.409275
 5.5      0.424481
 5.91945  0.409275
 6.3077   0.361561
 6.63125  0.280201
 6.85927  0.17285
 6.97613  0.0638731
```
(When the result is rescaled to a new interval, both functions return all of the points, but
they are still computed more efficiently for a `HollowSymTridiagonal` Jacobi matrix.)

### Gauss–Jacobi quadrature via the Jacobi matrix

A typical application of weighted quadrature rules is to accelerate convergence for
integrands that have power-law singularities at one or both of the endpoints.  Without
loss of generality, we can rescale the interval to $(-1,+1)$, in which case such
integrals are of the form:
```math
I[f] = \int_{-1}^{+1} \underbrace{(1-x)^\alpha (1+x)^\beta}_{W(x)} f(x) dx \, ,
```
where $\alpha > -1$ and $\beta > -1$ are the power laws at the two endpoints, which
we have factored out into a weight function $W(x) = (1+x)^\alpha (1-x)^\beta$ multiplied by some (hopefully smooth) function $f(x)$.  For
example, $\alpha = 0.5$ means that there is a square-root singularity at $x=+1$
(where the integrand is finite, but its slope blows up).  Or if $\beta = -0.1$ then
the integrand blows up at $x=-1$ but the integral is still finite (``1/x^{0.1}`` is an "integrable singularity").   This weight function is quite well known, in fact:
it yields [Gauss–Jacobi quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Jacobi_quadrature), with the corresponding orthogonal polynomials
being the [Jacobi polynomials](https://en.wikipedia.org/wiki/Jacobi_polynomials).

Again, we can simply look up the 3-term recurrence for the Jacobi polynomials $p_n$
corresponding to this weight function:
```math
2k (k + \alpha + \beta) (2k + \alpha + \beta - 2) p_k(x) = (2k+\alpha + \beta-1) \Big\{ (2k+\alpha + \beta)(2k+\alpha+\beta-2) x +  \alpha^2 - \beta^2 \Big\} p_{k-1}(x) - 2 (k+\alpha - 1) (k + \beta-1) (2k+\alpha + \beta) p_{k-2}(x),
```
giving $\alpha_k$ and $\beta_k$ by the earlier formulas, after a bit of algebra.
We also will need the unit integral
```math
I[1] = \int_{-1}^{+1} (1+x)^\alpha (1-x)^\beta dx  = \frac{2^{\alpha + \beta + 1}}{\alpha + \beta + 1} \frac{\Gamma(\alpha+1)\Gamma(\beta+1)}{\Gamma(\alpha+\beta+1)} \, ,
```
where $\Gamma$ is the [Gamma function](https://en.wikipedia.org/wiki/Gamma_function), computed in Julia by [SpecialFunctions.jl](https://github.com/JuliaMath/SpecialFunctions.jl).  This is all rather tedious, but fortunately exactly these expressions have already been worked out for us by the [FastGaussQuadrature.jl](https://github.com/JuliaApproximation/FastGaussQuadrature.jl) package, in undocumented functions `FastGaussQuadrature.jacobi_jacobimatrix(n, α, β)` (which computes the Jacobi matrix $J_n$) and `FastGaussQuadrature.jacobimoment(α, β)` (which computes $I[1]$).

We can use these to immediately compute the Gauss and Gauss–Kronrod points and weights for the Jacobi weight function, say for $\alpha = 0.5$, $\beta = -0.1$, and $n=5$:
```
julia> using FastGaussQuadrature, QuadGK

julia> α, β, n = 0.5, -0.1, 5;

julia> Jₙ = FastGaussQuadrature.jacobi_jacobimatrix(n, α, β)
5×5 LinearAlgebra.SymTridiagonal{Float64, Vector{Float64}}:
 -0.25       0.525105     ⋅            ⋅            ⋅
  0.525105  -0.0227273   0.506534      ⋅            ⋅
   ⋅         0.506534   -0.00852273   0.503003      ⋅
   ⋅          ⋅          0.503003    -0.00446429   0.501725
   ⋅          ⋅           ⋅           0.501725    -0.00274725

julia> I₁ = FastGaussQuadrature.jacobimoment(α, β)
2.012023098289125

julia> x, w = gauss(Jₙ, I₁); [x w]
5×2 Matrix{Float64}:
 -0.923234   0.372265
 -0.589357   0.610968
 -0.0806012  0.574759
  0.452539   0.349891
  0.852191   0.10414
```
(Notice that this weight function is *not* symmetric, and so the Jacobi matrix
is *not* hollow and the quadrature points and weights are asymmetrically distributed: the  points are denser near $x=-1$ where the weight function diverges.)
These are the same as the Gauss points and weights returned by the `gaussjacobi`
function in FastGaussQuadrature (which has fancy algorithms that scale better for
large `n` than those in QuadGK):
```
julia> xf, wf = FastGaussQuadrature.gaussjacobi(n, α, β); [xf wf]
5×2 Matrix{Float64}:
 -0.923234   0.372265
 -0.589357   0.610968
 -0.0806012  0.574759
  0.452539   0.349891
  0.852191   0.10414

julia> [x w] - [xf wf] # they are same points/weights to nearly machine precision
5×2 Matrix{Float64}:
  0.0           3.33067e-16
  0.0           0.0
 -1.38778e-17  -2.22045e-16
  5.55112e-17  -2.77556e-16
 -1.11022e-16   9.71445e-17
```
However, QuadGK can also return the 12-point Gauss–Kronrod rule, which embeds/
extends the 5-point Gauss-Jacobi rule in order to give you an error estimate:
```
julia> J₁₂ = FastGaussQuadrature.jacobi_jacobimatrix(12, α, β);

julia> kx, kw, gw = kronrod(J₁₂, n, I₁); [kx kw]
11×2 Matrix{Float64}:
 -0.988882   0.0723663
 -0.923234   0.181321
 -0.786958   0.264521
 -0.589357   0.306879
 -0.347734   0.311949
 -0.0806012  0.286857
  0.192962   0.238356
  0.452539   0.175128
  0.677987   0.109024
  0.852191   0.0520297
  0.962303   0.0135914

julia> [ kx[2:2:end] gw ]  # embedded Gauss–Jacobi rule is a subset x₂ᵢ of the points
5×2 Matrix{Float64}:
 -0.923234   0.372265
 -0.589357   0.610968
 -0.0806012  0.574759
  0.452539   0.349891
  0.852191   0.10414
```

The whole point of this is to accelerate convergence for smooth $f(x)$.  For
example, let's consider $f(x) = \cos(2x)$ with $\alpha = 0.5, \beta = -0.1$ as above.
In this case, according to Mathematica, the correct integral to 100 decimal places is
$I[\cos(2x)] \approx 0.9016684424525614794498545355301765224191593237834490575027527594933568786176710824696779907143025232764922385146156$, or about `0.9016684424525615` to machine precision.  If we use
the default `quadgk` function, which uses adaptive Gauss–Kronod quadrature that doesn't
have the singularity built-in, it takes about 1000 function evaluations to reach 9 digits of accuracy:
```
julia> exact = 0.9016684424525615;

julia> I, _ = quadgk_count(x -> (1-x)^α * (1+x)^β * cos(2x), -1, 1, rtol=1e-9)
(0.9016684425015659, 6.535590698106445e-10, 1125)

julia> I - exact
4.900435612853471e-11
```
(This isn't too terrible! If we plotted the points where `quadgk` evaluates our integrand, we would see that it concentrates points mostly close to the singularities at the boundaries.  To get a similar error from unweighted Gauss–Legendre quadrature requires about $n=10^5$ points, which is too slow with the `gauss(n)` function — it's only practical with `x, w = FastGaussQuadrature.gausslegendre(10^5)`, which uses a fancy $O(n)$ algorithm.  Ordinary Gaussian quadrature very slowly converging for non-smooth functions.)
In contrast, our 5-point Gauss–Jacobi quadrature rule from above gets about 6 digits:
```
julia> I = sum(@. cos(2x) * w)
0.9016690323443182

julia> I - exact
5.898917566637962e-7
```
and gets 10 digits with only 7 points:
```
julia> x, w = gauss(FastGaussQuadrature.jacobi_jacobimatrix(7, α, β), I₁);

julia> I = sum(@. cos(2x) * w)
0.9016684424777912

julia> I - exact
2.522970721230422e-11
```
This is not unexpected, because the fact that $f(x)$ is smooth means that Gaussian
quadrature converges exponentially fast, regardless of the weight function's endpoint singularities (which have been taken into account analytically by the quadrature rule).
The Gauss–Kronrod rule also converges exponentially, and gives us an error estimate
to give us added confidence in the result.  For example, with our 12-point Gauss–Kronrod rule we obtain the correct result to machine precision:
```
julia> Ik = sum(@. cos(2kx) * kw)
0.9016684424525613

julia> Ik - exact
-2.220446049250313e-16
```
while a subset `kx[2:2:end]` of the points (for which we could re-use the integrand evaluations if we wanted) gives us an embedded Gauss rule and an error bound:
```
julia> Ig = sum(@. cos(2kx[2:2:end]) * gw)
0.9016690323443182

julia> abs(Ik - Ig) # conservative error estimate: Kronrod - Gauss
5.898917568858408e-7
```
As usual, this error bound is quite conservative for smooth $f(x)$ where the quadrature rule is converging rapidly, since it is actually an error estimate for the 5-point `Ig` and not for the 12-point `Ik`.  But at least it gives you some indication as to whether you picked a sufficient number of points to integrate $f(x)$ sufficiently accurately.

For fun, let's do the same calculation to 100 digits with $n=11$, using `BigFloat` arithmetic.  (We simple need to pass `big"0.5"` and `big"-0.1"` for `α` and `β` to FastGaussQuadrature and it will construct the Jacobi matrix in `BigFloat` precision, which QuadGK will then turn into `BigFloat` Gauss/Gauss–Kronrod points and weights.)
```
julia> setprecision(100, base=10)
100

julia> bigexact = big"0.9016684424525614794498545355301765224191593237834490575027527594933568786176710824696779907143025232764922385146156"
0.901668442452561479449854535530176522419159323783449057502752759493356878617671082469677990714302523252

julia> bigJ = FastGaussQuadrature.jacobi_jacobimatrix(18, big"0.5", big"-0.1");

julia> bigI₁ = FastGaussQuadrature.jacobimoment(big"0.5", big"-0.1")
2.01202309828912479732166203322245014347199888907111184953347045850828228938420405746115463698460502738

julia> bigkx, bigkw, biggw = kronrod(bigJ, 11, bigI₁);

julia> bigIk = sum(@. cos(2bigkx) * bigkw)
0.901668442452561479449854535530176522419159355847389937592609530571246098597677208391320749304692456157

julia> Float64(bigIk - bigexact)
3.2063940880089856e-44
```
so the 18-point Gauss–Kronrod rule is accurate to about 43 digits, while the conservative error estimate (= error of embedded 11-point Gauss rule) is about 20 digits:
```
julia> bigIg = sum(@. cos(2bigkx[2:2:end]) * biggw)
0.901668442452561479451864506089616011729536201879100045019491743663463853013435135385929850010604582451

julia> Float64(abs(bigIk - bigIg))
2.0099705594394893e-21
```

## Arbitrary weight functions

Although analytical formulas for 3-term recurrences and Jacobi matrices are known for many common types of singularities that appear in integrals, this is certainly not universally true.   As a fallback, you can simply supply an arbitrary weight function $W(x)$ and let QuadGK compute everything for you numerically (essentially by a form of [Gram–Schmidt process](https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process) in which a basis of polynomials is orthonormalized with respect to $w$, using a sequence of $O(n)$ numerical integrals).   This is much more time consuming, especially if you want high accuracy (i.e. you specify a low tolerance for the numerical integrals), but can be worth it if your $f(x)$ is expensive and/or you need many integrals of a similar form:  the numerical integrals are againt cheap polynomial functions, and are only done once for all $f(x)$ with the same weight function.

For example:
```jl
using QuadGK
x, w = gauss(x -> exp(-x) / sqrt(x), 10, 0, -log(1e-10), rtol=1e-9)
```
computes the points and weights for performing `∫exp(-x)f(x)/√x dx` integrals from `0` to `-log(1e-10) ≈ 23`, so that there is a `1/√x` singularity in the integrand at `x=0` and a rapid decay for increasing `x`.  (The `gauss` function currently does not support infinite integration intervals, but for a rapidly decaying weight function you can approximate an infinite interval to any desired accuracy by a sufficiently broad interval, with a tradeoff in computational expense.)  For example, with `f(x) = sin(x)`, the exact answer is `0.570370556005742…`.  Using the points and weights above with `sum(sin.(x) .* w)`, we obtain `0.5703706212868831`, which is correct to 6–7 digits using only 10 `f(x)` evaluations.  Obtaining similar
accuracy for the same integral from `quadgk` requires nearly 300 function evaluations.   However, the
`gauss` function itself computes many (``2n``) numerical integrals of your weight function (multiplied
by polynomials), so this is only more efficient if your `f(x)` is very expensive or if you need
to compute a large number of integrals with the same `W`.  See the [`gauss`](@ref) documentation for more information.

Similarly, one can use the `kronrod(W, n, a, b, rtol=rtol)` function to construct Gauss–Kronrod rules
for arbitrary weight functions.   Unfortunately, it turns out that a Gauss–Kronrod rule does not exist for the weight function above, and the `kronrod` function consequently throws an error — probably because it is very similar to [Gauss–Laguerre quadrature](https://en.wikipedia.org/wiki/Gauss%E2%80%93Laguerre_quadrature) and Gauss–Kronrod rules are known to not exist for the Gauss–Laguerre problem [(Kahaner & Monegato, 1978)](https://doi.org/10.1007/BF01590820).   However, we can for example reproduce the points and weights from the Gauss–Jacobi weight function of the previous section, now computed completely numerically without supplying the analytical Jacobi matrix:
```
julia> kx, kw, gw = kronrod(x -> (1-x)^0.5 * (1+x)^-0.1, 5, -1, 1, rtol=1e-9); [kx kw]
11×2 Matrix{Float64}:
 -0.988882   0.0723663
 -0.923234   0.181321
 -0.786958   0.264521
 -0.589357   0.306879
 -0.347734   0.311949
 -0.0806012  0.286857
  0.192962   0.238356
  0.452539   0.175128
  0.677987   0.109024
  0.852191   0.0520297
  0.962303   0.0135914
```
(If you compare these more quantitatively to those in the previous section, you'll see that they
are accurate to about 10 digits, consistent with the `rtol=1e-9` that we passed as a tolerance
for the numerical integrals used in constructing the Jacobi matrix numerically.)

For a more practical example that can *only* be done numerically, see our tutorial using a [weight function interpolated from tabulated solar-spectrum data](https://nbviewer.jupyter.org/urls/math.mit.edu/~stevenj/Solar-Quadrature.ipynb), also described in [Johnson (2019)](https://arxiv.org/abs/1912.06870).
