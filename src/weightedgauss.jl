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
"""
gauss(W, N, a::Real,b::Real; rtol::Real=sqrt(eps(typeof(float(b-a)))), quad=quadgk) =
    handle_infinities(W, (a,b)) do W, ab, tfunc
        x, w = _gauss(W, N, tfunc, ab..., rtol, quad)
        return (x, w)
    end

function _gauss(W, N, tfunc, a, b, rtol, quad)
    # Uses the Lanczos recurrence described in Trefethen & Bau,
    # Numerical Linear Algebra, to find the `N`-point Gaussian quadrature
    # using O(N) integrals and O(N²) operations, applied to Chebyshev basis:
    xscale = 0.5*(b-a) # scaling from [-1,1] to (a,b)
    T = typeof(xscale)
    α = zeros(T, N)
    β = zeros(T, N+1)
    q₀ = sizehint!(T[0], N+1) # 0 polynomial
    # (note that our integrals are rescaled to [-1,1], with the Jacobian
    #  factor |xscale| absorbed into the definition of the inner product.)
    wint = first(quad(x -> W(a + (x+1)*xscale), T(-1), T(1), rtol=rtol))
    (wint isa Real && wint > 0) ||
        throw(ArgumentError("weight W must be real and positive"))
    atol = rtol*wint
    q₁ = sizehint!(T[1/sqrt(wint)],N+1) # normalized constant polynomial
    v = copy(q₀)
    for n = 1:N
        chebx!(v, q₁) # v = x * q₁
        α[n] = first(quad(T(-1), T(1), rtol=rtol, atol=atol) do x
            y = a + (x+1)*xscale
            W(y) * chebeval(x, q₁) * chebeval(x, v)
        end)
        n == N && break
        for j = 1:length(q₀); v[j] -= β[n]*q₀[j]; end
        for j = 1:length(q₁); v[j] -= α[n]*q₁[j]; end
        β[n+1] = sqrt(first(quad(T(-1), T(1), rtol=rtol, atol=atol) do x
            y = a + (x+1)*xscale
            W(y) * chebeval(x, v)^2
        end))
        v .*= inv(β[n+1])
        q₀,q₁,v = q₁,v,q₀
    end

    # TODO: handle BigFloat etcetera — requires us to update eignewt() to
    #       support nonzero diagonal entries.
    E = eigen(SymTridiagonal(α, β[2:N]))

    w = E.vectors[1,:]
    w .= (abs(xscale) * wint) .* abs2.(w)
    return (tfunc.(a .+ (E.values .+ 1) .* xscale), w)
end
