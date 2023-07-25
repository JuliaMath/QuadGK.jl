# Constructing Gaussian quadrature weights for an
# arbitrary weight function and integration bounds.

# for numerical stability, we apply the usual Lanczos
# Gram–Schmidt procedure to the basis {T₀,T₁,T₂,…} of
# Chebyshev polynomials on [-1,1] rather than to the
# textbook monomial basis {1,x,x²,…}.

# evaluate Chebyshev polynomial p(x) with coefficients a[i]
# by a Clenshaw recurrence.
function chebeval(x, a)
    if length(a) ≤ 2
        length(a) == 1 && return a[1] + x * zero(a[1])
        return a[1]+x*a[2]
    end
    bₖ = a[end-1] + 2x*a[end]
    bₖ₊₁ = oftype(bₖ, a[end])
    for j = lastindex(a)-2:-1:2
        bⱼ = a[j] + 2x*bₖ - bₖ₊₁
        bₖ, bₖ₊₁ = bⱼ, bₖ
    end
    return a[1] + x*bₖ - bₖ₊₁
end

# if a[i] are coefficients of Chebyshev series, compute the coefficients xa (in-place)
# of the series multiplied by x, using recurrence xTₙ = 0.5 (Tₙ₊₁+Tₙ₋₁) for n > 0
function chebx!(xa, a)
    resize!(xa, length(a)+1)
    xa .= 0
    for n = 2:lastindex(a)
        c = 0.5*a[n]
        xa[n-1] += c
        xa[n+1] += c
    end
    if !isempty(a)
        xa[2] += a[1]
    end
    return xa
end

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

The precision of the calculations and return value is determined from the types of `a` and `b`.
"""
function gauss(W, N::Integer, a::Real,b::Real; rtol::Real=sqrt(eps(typeof(float(b-a)))), quad=quadgk)
    (isfinite(a) && isfinite(b)) || throw(ArgumentError("a finite interval is required"))

    # find the Jacobi matrix and apply the Golub–Welsh algorithm:
    J, xscale, wint = _jacobi(W, N, a, b, rtol, quad)
    x, w = gauss(J, wint)
    @. x = (x + 1) / xscale + a
    return (x, w)
end

"""
    kronrod(W, N, a, b; rtol=sqrt(eps), quad=quadgk)

Return a tuple `(x, w, wg)` of `N` quadrature points `x[i]` and weights `w[i]` to
integrate functions on the interval `(a, b)` multiplied by the weight function
`W(x)`, along with the weights `wg` of an embedded Gauss rule corresponding to `x[2:2:end]`,
similar to the `gauss(W, N, a, b)` function and analogous to `kronrod(N)` (which only
returns the `x ≤ 0` points for a constant weight function).

That is, `I = sum(w .* f.(x))` approximates the integral `∫ W(x)f(x)dx`
from `a` to `b`.  And an error estimate is `abs(I - Ig)`, where `Ig` is
the result `Ig = sum(wg .* f.(x[2:2:end]))` of the embedded Gauss rule.

This function performs `≈ 3N+3` numerical integrals of polynomials against `W(x)`
using the integration function `quad` (defaults to `quadgk`) with relative tolerance `rtol`
(which defaults to half of the precision `eps` of the endpoints).
This is followed by an O(N²) calculations. So, using a large order `N` is expensive.
"""
function kronrod(W, N::Integer, a::Real,b::Real; rtol::Real=sqrt(eps(typeof(float(b-a)))), quad=quadgk)
    (isfinite(a) && isfinite(b)) || throw(ArgumentError("a finite interval is required"))

    # find the Jacobi matrix and apply the Golub–Welsh algorithm:
    J, xscale, wint = _jacobi(W, div(3N+3,2), a, b, rtol, quad)
    x, w, wg = kronrod(J, N, wint)
    @. x = (x + 1) / xscale + a
    return (x, w, wg)
end

function _jacobi(W, N, a, b, rtol, quad)
    # Uses the Lanczos recurrence described in Trefethen & Bau,
    # Numerical Linear Algebra, to find the Jacobi matrix
    # (the 3-term recurrence) for W using O(N) integrals,
    # applied to Chebyshev basis:
    xscale = float(2/(b-a)) # scaling from (a,b) to (-1,1)
    T = typeof(xscale)
    α = zeros(T, N)
    β = zeros(T, N)
    q₀ = sizehint!(T[0], N+1) # 0 polynomial
    wint = first(quad(W, a, b, rtol=rtol))
    (wint isa Real && wint > 0) ||
        throw(ArgumentError("weight W must be real and positive"))
    atol = rtol*wint
    q₁ = sizehint!(T[1/sqrt(wint)],N+1) # normalized constant polynomial
    v = copy(q₀)
    for n = 1:N
        chebx!(v, q₁) # v = x * q₁
        α[n] = first(quad(a, b, rtol=rtol, atol=atol) do x
            t = (x-a)*xscale - 1
            W(x) * chebeval(t, q₁) * chebeval(t, v)
        end)
        n == N && break
        for j = 1:length(q₀); v[j] -= β[n]*q₀[j]; end
        for j = 1:length(q₁); v[j] -= α[n]*q₁[j]; end
        β[n+1] = sqrt(first(quad(a, b, rtol=rtol, atol=atol) do x
            W(x) * chebeval((x-a)*xscale - 1, v)^2
        end))
        v .*= inv(β[n+1])
        q₀,q₁,v = q₁,v,q₀
    end

    return SymTridiagonal(α, β[2:N]), xscale, wint
end
