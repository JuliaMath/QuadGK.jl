# Constructing Gaussian quadrature weights for an
# arbitrary weight function and integration bounds.

using Polynomials

"""
    gauss(W, N, a, b; rtol=sqrt(eps), quad=quadgk)

Return a pair `(x, w)` of `N` quadrature points `x[i]` and weights `w[i]` to
integrate functions on the interval `(a, b)` multiplied by the weight function
`W(x)`.  That is, `sum(w .* f.(x))` approximates the integral `∫ W(x)f(x)dx`
from `a` to `b`.

This function performs `2N` numerical integrals of polynomials against `W(x)`
using the integration function `quad` (defaults to `quadgk`) with relative tolerance `rtol`
(which defaults to half of the precision `eps` of the endpoints).
This is followed by an O(N²) calculations. So, using a large order `N` is expensive.

If `W` has lots of singularities that make it hard to integrate numerically,
you may need to decrease `rtol`.   You can also pass in a specialized quadrature routine
via the `quad` keyword argument, which should accept arguments `quad(f,a,b,rtol=_,atol=_)`
similar to `quadgk`.  (This is useful if your weight function has discontinuities, in which
case you might want to break up the integration interval at the discontinuities.)
"""
function gauss(W, N, a::Real,b::Real; rtol::Real=sqrt(eps(typeof(float(b-a)))), quad=quadgk)
    # Uses the Lanczos recurrence described in Trefethen & Bau,
    # Numerical Linear Algebra, to find the `N`-point Gaussian quadrature
    # using O(N) integrals and O(N²) operations.
    α = zeros(N)
    β = zeros(N+1)
    T = typeof(float(b-a))
    x = Poly(T[0, 1])
    q₀ = Poly(T[0])
    wint = first(quad(W, a, b, rtol=rtol))
    atol = rtol*wint
    q₁ = Poly(T[T(1)/sqrt(wint)])
    for n = 1:N
        v = x * q₁
        q₁v = q₁ * v
        α[n] = first(quad(x -> W(x) * q₁v(x), a, b, rtol=rtol, atol=atol))
        n == N && break
        v -= β[n]*q₀ + α[n]*q₁
        v² = v*v
        β[n+1] = sqrt(first(quad(x -> W(x) * v²(x), a, b, rtol=rtol, atol=atol)))
        q₀ = q₁
        q₁ = v / β[n+1]
    end

    # TODO: handle BigFloat etcetera — requires us to update eignewt() to
    #       support nonzero diagonal entries.
    E = eigen(SymTridiagonal(α, β[2:N]))

    return (E.values, wint .* abs2.(E.vectors[1,:]))
end
