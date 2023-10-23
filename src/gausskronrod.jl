###########################################################################
# Gauss-Kronrod integration-weight computation for arbitrary floating-point
# types and precision, implemented based on the description in:
#
#    Dirk P. Laurie, "Calculation of Gauss-Kronrod quadrature rules,"
#    Mathematics of Computation, vol. 66, no. 219, pp. 1133-1145 (1997).
#
# for the Kronrod rule, and for the Gauss rule from the description in
#
#    Lloyd N. Trefethen and David Bau, Numerical Linear Algebra (SIAM, 1997).
#
# Arbitrary-precision eigenvalue (eignewt & eigpoly) and eigenvector
# (eigvec1) routines are written by SGJ, independent of the above sources.
#
# Another useful reference is the review article:
#
#    Sotirios E. Notaris, "Gauss–Kronrod quadrature formulae —
#    A survey of fifty years of research," Electronic Transactions
#    on Numerical Analysis vol. 45, pp. 374–404 (2016).
#    https://etna.ricam.oeaw.ac.at/vol.45.2016/pp371-404.dir/pp371-404.pdf

###########################################################################
# Eigensolver utilities:

# Implement a type for symmetric tridiagonal matrices with zero diagonals:
#
# For the common case of Gauss-Kronrod rules for the unit weight function,
# the diagonals of the Jacobi matrices are zero and certain things simplify
# compared to the general case of an arbitrary weight function.

# a type for us to dispatch on; we don't actually need the full AbstractMatrix functionality
"""
    QuadGK.HollowSymTridiagonal(ev::AbstractVector)

Construct a "hollow" symmetric tridiagonal matrix, whose diagonal entries are zero and
whose first sub/super-diagonal is `ev`.

The `HollowSymTridiagonal` type can be passed to [`gauss`](@ref) or [`kronrod`](@ref) for
Jacobi matrices to dispatch to specialized methods that exploit the special "hollow" structure
arising for symmetric weight functions, in order to generate symmetric quadrature rules more efficiently.
"""
struct HollowSymTridiagonal{T, V<:AbstractVector{T}} <: AbstractMatrix{T}
    ev::V # superdiagonal
    function HollowSymTridiagonal{T, V}(ev) where {T, V<:AbstractVector{T}}
        Base.require_one_based_indexing(ev)
        return new{T, V}(ev)
    end
end
HollowSymTridiagonal(ev::AbstractVector{T}) where {T} =
    HollowSymTridiagonal{T,typeof(ev)}(ev)
Base.size(A::HollowSymTridiagonal) = (length(A.ev)+1,length(A.ev)+1)
LinearAlgebra.diag(A::HollowSymTridiagonal) = zeros(eltype(A), size(A,1))
LinearAlgebra.SymTridiagonal(A::HollowSymTridiagonal{T}) where {T} = SymTridiagonal{T}(A)
LinearAlgebra.SymTridiagonal{T}(A::HollowSymTridiagonal) where {T} = SymTridiagonal(zeros(T, length(A.ev)+1), Vector{T}(A.ev))
function HollowSymTridiagonal(A::SymTridiagonal)
    iszero(A.dv) || throw(ArgumentError("expected zero diagonal"))
    return HollowSymTridiagonal(A.ev)
end
function HollowSymTridiagonal{T}(A::SymTridiagonal) where {T}
    iszero(A.dv) || throw(ArgumentError("expected zero diagonal"))
    return HollowSymTridiagonal(AbstractVector{T}(A.ev))
end
Base.Matrix(A::HollowSymTridiagonal) = Matrix(SymTridiagonal(A))
Base.Matrix{T}(A::HollowSymTridiagonal) where {T} = Matrix{T}(SymTridiagonal{T}(A))

const AbstractSymTri{T} = Union{HollowSymTridiagonal{T}, SymTridiagonal{T}}

# for display purposes:
function Base.replace_in_print_matrix(A::HollowSymTridiagonal, i::Integer, j::Integer, s::AbstractString)
    i==j-1||i==j+1 ? s : Base.replace_with_centered_mark(s)
end
@inline function Base.getindex(A::HollowSymTridiagonal{T}, i::Integer, j::Integer) where T
    @boundscheck checkbounds(A, i, j)
    if i == j + 1
        return copy(transpose(@inbounds A.ev[j])) # materialized for type stability
    elseif i + 1 == j
        return @inbounds A.ev[i]
    else
        return zero(T)
    end
end

# Given a symmetric tridiagonal matrix H with H[i,i] = 0 and
# H[i-1,i] = H[i,i-1] = b[i-1], compute p(z) = det(z I - H) and its
# derivative p'(z), returning (p,p').
function eigpoly(b::AbstractVector{<:Real},z::Number,m::Integer=length(b)+1)
    d1 = z
    d1deriv = d2 = one(z)
    d2deriv = zero(z)
    for i = 2:m
        b2 = b[i-1]^2
        d = z * d1 - b2 * d2
        dderiv = d1 + z * d1deriv - b2 * d2deriv
        d2 = d1
        d1 = d
        d2deriv = d1deriv
        d1deriv = dderiv
    end
    return (d1, d1deriv)
end
eigpoly(H::HollowSymTridiagonal{<:Real},z) = eigpoly(H.ev, z)

# as above, but for general symmetric tridiagonal (diagonal ≠ 0)
function eigpoly(H::SymTridiagonal{<:Real},z::Number)
    d1 = z - H.dv[1]
    d1deriv = d2 = one(d1)
    d2deriv = zero(d1)
    for i = 2:length(H.dv)
        b2 = H.ev[i-1]^2
        a = z - H.dv[i]
        d = a * d1 - b2 * d2
        dderiv = d1 + a * d1deriv - b2 * d2deriv
        d2 = d1
        d1 = d
        d2deriv = d1deriv
        d1deriv = dderiv
    end
    return (d1, d1deriv)
end

# compute the n smallest eigenvalues of the symmetric tridiagonal matrix H
# (defined from b as in eigpoly) using a Newton iteration
# on det(H - lambda I).  Unlike eig, handles BigFloat.
function eignewt(H::AbstractSymTri{T}, n::Integer) where {T<:Real}
    # get initial guess from eig on Float64 matrix
    lambda0 = eigvals(SymTridiagonal{Float64}(H), 1:n)
    lambda = Vector{float(T)}(lambda0)
    for i = 1:n
        for k = 1:1000
            (p,pderiv) = eigpoly(H,lambda[i])
            δλ = p / pderiv # may be NaN or Inf if pderiv underflows to 0.0
            if isfinite(δλ)
                lambda[i] -= δλ
                if abs(δλ) ≤ 10 * eps(lambda[i])
                    # do one final Newton iteration for luck and profit:
                    δλ = (/)(eigpoly(H,lambda[i])...) # = p / pderiv
                    isfinite(δλ) && (lambda[i] -= δλ)
                end
            else
                break
            end
        end
    end
    return lambda
end
function eignewt(b::AbstractVector{<:Real},m::Integer,n::Integer)
    m == length(b)+1 || throw(ArgumentError("$m != length(b)+1 = $(length(b)+1) unsupported"))
    eignewt(HollowSymTridiagonal(b), n)
end

# given an eigenvalue λ and the matrix H(b) from above, return
# the corresponding eigenvector, normalized to 1.
function eigvec1!(v::AbstractVector, H::HollowSymTridiagonal, λ::Number)
    # "cheat" and use the fact that our eigenvector v must have a
    # nonzero first entries (since it is a quadrature weight), so we
    # can set v[1] = 1 to solve for the rest of the components:.
    m = size(H, 1)
    m == length(v) || throw(DimensionMismatch())
    v[1] = 1
    if m > 1
        s = v[1]
        v[2] = λ * v[1] / H.ev[1]
        s += abs2(v[2])
        for i = 3:m
            v[i] = (λ*v[i-1] - H.ev[i-2]*v[i-2]) / H.ev[i-1]
            s += abs2(v[i])
        end
        rmul!(v, 1 / sqrt(s))
    end
    return v
end
function eigvec1!(v::AbstractVector, b::AbstractVector, λ::Number, m=length(b)+1)
    m == length(b)+1 || throw(ArgumentError("$m != length(b)+1 = $(length(b)+1) unsupported"))
    return eigvec1!(v, HollowSymTridiagonal(b), λ)
end
eigvec1(b::AbstractVector, λ::Number, m=length(b)+1) =
    eigvec1!(Vector{promote_type(eltype(b),typeof(λ))}(undef, m), b, λ, m)

# as above but for a general SymTridiagonal
function eigvec1!(v::AbstractVector, H::SymTridiagonal, λ::Number)
    # "cheat" and use the fact that our eigenvector v must have a
    # nonzero first entries (since it is a quadrature weight), so we
    # can set v[1] = 1 to solve for the rest of the components:.
    m = size(H,1)
    m == length(v) || throw(DimensionMismatch())
    v[1] = 1
    if m > 1
        s = v[1]
        v[2] = (λ - H.dv[1]) * v[1] / H.ev[1]
        s += abs2(v[2])
        for i = 3:m
            v[i] = ((λ - H.dv[i-1])*v[i-1] - H.ev[i-2]*v[i-2]) / H.ev[i-1]
            s += abs2(v[i])
        end
        rmul!(v, 1 / sqrt(s))
    end
    return v
end

###########################################################################
# Gauss–Kronrod rules for the unit weight function:

"""
    gauss([T,] n)
    gauss([T,] n, a, b)

Return a pair `(x, w)` of `n` quadrature points `x[i]` and weights `w[i]` to
integrate functions on the interval ``(a, b)``, which defaults to ``(-1,1)``,  i.e. `sum(w .* f.(x))`
approximates the integral ``\\int_a^b f(x) dx``.

Uses the Golub–Welch method described in Trefethen &
Bau, Numerical Linear Algebra, to find the `n`-point Gaussian quadrature
rule in O(`n`²) operations.

`T` is an optional parameter specifying the floating-point type, defaulting
to `Float64`. Arbitrary precision (`BigFloat`) is also supported.  If `T` is not supplied,
 but the interval `(a, b)` is passed, then the floating-point type is determined
 from the types of `a` and `b`.
"""
function gauss(::Type{T}, N::Integer) where T<:AbstractFloat
    if N < 1
        throw(ArgumentError("Gauss rules require positive order"))
    end
    o = one(T)
    b = T[ n / sqrt(4n^2 - o) for n = 1:N-1 ]
    return gauss(HollowSymTridiagonal(b), 2)
end

gauss(N::Integer) = gauss(Float64, N) # integration on the standard interval (-1,1)

# re-scaled to an arbitrary interval:
gauss(N::Integer, a::Real, b::Real) = gauss(typeof(float(b-a)), N, a, b)
function gauss(::Type{T}, N::Integer, a::Real, b::Real) where T<:AbstractFloat
    x, w = gauss(T, N)
    s = T(b-a)/2
    x .= a .+ (x .+ 1) .* s
    w .*= abs(s)
    return (x, w)
end

function symtri(A::AbstractMatrix)
    J = SymTridiagonal(A) # may silently discard off-diagonal elements
    J == A || throw(ArgumentError("Jacobi matrix must be symmetric tridiagonal"))
    return J
end

# Gauss rules for an arbitrary Jacobi matrix J
"""
    gauss(J::AbstractMatrix, unitintegral::Real=1, [ (a₀,b₀) => (a,b) ])

Construct the ``n``-point Gaussian quadrature rule for ``I[f] = \\int_a^b w(x) f(x) dx``
from the ``n \\times n``
symmetric tridiagonal Jacobi matrix `J` corresponding to the orthogonal
polynomials for that weighted integral.  The value of `unitintegral` should
be ``I[1]``, the integral of the weight function.

An optional argument `(a₀,b₀) => (a,b)` allows you to specify that `J` was originally
defined for a different interval ``(a_0, b_0)``, which you want to rescale to
a given ``(a, b)``.  (`gauss` will rescale the points and weights for you.)

Returns a pair `(x, w)` of ``n`` quadrature points `x[i]` and weights `w[i]` to
integrate functions, i.e. `sum(w .* f.(x))` approximates the integral ``I[f]``.
"""
gauss(A::AbstractMatrix{<:Real}, unitintegral::Real=1) =
    gauss(symtri(A), unitintegral)

function gauss(J::AbstractSymTri{<:Real}, unitintegral::Real=1)
    # Golub–Welch algorithm
    x = eignewt(J, size(J,1))
    v = Vector{promote_type(eltype(J),eltype(x))}(undef, size(J,1))
    w = [ unitintegral * abs2(eigvec1!(v,J,x[i])[1]) for i = 1:size(J,1) ]
    return (x, w)
end

# as above but rescaled to an arbitrary interval and unit integral
function gauss(J::AbstractMatrix{<:Real}, unitintegral::Real, xrescale::Pair{<:Tuple{Real,Real}, <:Tuple{Real,Real}})
    x, w = gauss(J, unitintegral)
    T = eltype(x)
    a0, b0 = xrescale.first
    a, b = xrescale.second
    xscale = (T(b) - T(a)) / (T(b0) - T(a0))
    x .= (x .- a0) .* xscale .+ a
    return x, w
end

"""
    kronrod([T,] n)
    kronrod([T,] n, a, b)

Compute ``2n+1`` Kronrod points `x[i]` and weights `w[i]` based on the description in
Laurie (1997), appendix A, for integrating on the interval ``(a,b)`` (defaulting to ``[-1,1]``).

If `a` and `b` are not passed, since the rule is symmetric,
this only returns the `n+1` points with `x <= 0`.
The function Also computes the embedded `n`-point Gauss quadrature weights `gw`
(again for `x <= 0` if `a` and `b` are not passed), corresponding to the points `x[2:2:end]`.
Returns `(x,w,wg)` in O(`n`²) operations.

`T` is an optional parameter specifying the floating-point type, defaulting
to `Float64`. Arbitrary precision (`BigFloat`) is also supported.

Given these points and weights, the estimated integral `I` and error `E` can
be computed for an integrand `f(x)` as follows:

    x, w, wg = kronrod(n)
    fx⁰ = f(x[end])                # f(0)
    x⁻ = x[1:end-1]                # the x < 0 Kronrod points
    fx = f.(x⁻) .+ f.((-).(x⁻))    # f(x < 0) + f(x > 0)
    I = sum(fx .* w[1:end-1]) + fx⁰ * w[end]
    if isodd(n)
        E = abs(sum(fx[2:2:end] .* wg[1:end-1]) + fx⁰*wg[end] - I)
    else
        E = abs(sum(fx[2:2:end] .* wg[1:end])- I)
    end
"""
function kronrod(::Type{T}, n::Integer) where T<:AbstractFloat
    n < 1 && throw(ArgumentError("Kronrod rules require positive order"))

    o = one(T)
    b = zeros(T, 2n)
    for j = 1:div(3n+1,2)
        b[j] = j^2 / (4j^2 - o)
    end
    x, w, v = _kronrod(HollowSymTridiagonal(b), b, Int(n), 2)

    # Get embedded Gauss rule from even-indexed points, using
    # the Golub–Welch method as described in Trefethen and Bau.
    # (we don't need the eigenvalues since we already have them).
    for j = 1:n-1
        b[j] = j / sqrt(4j^2 - o)
    end
    @views gw = T[ 2abs2(eigvec1!(v[1:n],b[1:n-1],x[i],n)[1]) for i = 2:2:n+1 ]

    return x, w, gw
end

kronrod(n::Integer) = kronrod(Float64, n)

# as above but allow you to pass the interval [a,b],
# and returns all the points not just half
function kronrod(n::Integer, a::Real, b::Real)
    x, w, gw = kronrod(typeof(float(b-a)), n)
    x = [x; rmul!(reverse!(x[1:end-1]), -1)]
    w = [w; reverse!(w[1:end-1])]
    gw = [gw; reverse!(gw[1:end-isodd(n)])]
    T = eltype(x)
    xscale = (T(b) - T(a)) / 2
    x .= (x .+ 1) .* xscale .+ a
    w .*= xscale
    gw .*= xscale
    return x, w, gw
end

# as above, but generalized to an arbitrary Jacobi matrix
"""
    kronrod(J::AbstractMatrix, n::Integer, unitintegral::Real=1, [ (a₀,b₀) => (a,b) ])

Construct the ``2n+1``-point Gauss–Kronrod quadrature rule for ``I[f] = \\int_a^b w(x) f(x) dx``
from the ``m \\times m``
symmetric tridiagonal Jacobi matrix `J` corresponding to the orthogonal
polynomials for that weighted integral, where `m ≥ (3n+3)÷2`.  The value of `unitintegral` should
be ``I[1]``, the integral of the weight function.

An optional argument `(a₀,b₀) => (a,b)` allows you to specify that `J` was originally
defined for a different interval ``(a_0, b_0)``, which you want to rescale to
a given ``(a, b)``.  (`gauss` will rescale the points and weights for you.)

Returns a tuple `(x, w, gw)` of ``n`` quadrature points `x[i]` and weights `w[i]` to
integrate functions, i.e. `sum(w .* f.(x))` approximates the integral ``I[f]``.  `gw`
are the weights of the embedded Gauss rule corresponding to the points `x[2:2:end]`,
which can be used for error estimation.
"""
kronrod(A::AbstractMatrix{<:Real}, n::Integer, unitintegral::Real=1) =
    kronrod(symtri(A), n, unitintegral)

function kronrod(J::AbstractSymTri{<:Real}, n::Integer, unitintegral::Real=1)
    x, w, v = _kronrod(J, _kronrod_b(J, n), Int(n), unitintegral)

    # Get embedded Gauss rule from even-indexed points
    Jsmall = if J isa SymTridiagonal
        @views SymTridiagonal(J.dv[1:n], J.ev[1:n-1])
    else
        @views HollowSymTridiagonal(J.ev[1:n-1])
    end
    @views gw = [ unitintegral*abs2(eigvec1!(v[1:n],Jsmall,x[i])[1]) for i = 2:2:length(x) ]

    return x, w, gw
end

function kronrod(J::AbstractMatrix{<:Real}, n::Integer, unitintegral::Real, xrescale::Pair{<:Tuple{Real,Real}, <:Tuple{Real,Real}})
    x, w, gw = kronrod(J, n, unitintegral)
    if J isa HollowSymTridiagonal
        x = [x; rmul!(reverse!(x[1:end-1]), -1)]
        w = [w; reverse!(w[1:end-1])]
        gw = [gw; reverse!(gw[1:end-isodd(n)])]
    end
    a0, b0 = xrescale.first
    a, b = xrescale.second
    T = eltype(x)
    xscale = (T(b) - T(a)) / (T(b0) - T(a0))
    x .= (x .- a0) .* xscale .+ a
    return x, w, gw
end

"""
    kronrodjacobi(J::Union{SymTridiagonal, QuadGK.HollowSymTridiagonal}, n::Integer)

Given a real-symmetric tridiagonal matrix `J`, return the symmetric tridiagonal
"Kronrod–Jacobi" matrix, whose eigenvalues and eigenvectors yield the Gauss–Kronrod
rule of order `n`, e.g. by calling `x, w = gauss(kronrodjacobi(n))`.

See also the [`kronrod`](@ref) function, which returns the Kronrod points and
weights directly, along with the weights of an embedded Gauss rule.

``O(n^2)`` algorithm from Dirk P. Laurie, "Calculation of Gauss-Kronrod quadrature rules,"
*Mathematics of Computation*, vol. 66, no. 219, pp. 1133-1145 (1997).
"""
kronrodjacobi(J::AbstractSymTri{<:Real}, n::Integer) =
    _kronrodjacobi(J, _kronrod_b(J, n), Int(n))

###########################################################################
# internal implementation of algorithm from Laurie (1997) to return the (x,w)
# of the order-n Gauss–Kronrod rule, given the Jacobi matrix J of the weight,
# the order n, and a vector b of length 2n that has ALREADY been initialized
# to J.ev[j]^2 for j=1:div(3n+1,2) and to 0 otherwise.  J.ev is NOT used.

# construct the b vector from J, for passing to the other _kronrod functions below.
function _kronrod_b(J::AbstractSymTri{<:Real}, n::Integer)
    n < 1 && throw(ArgumentError("Kronrod rules require positive order"))
    size(J,1) ≥ div(3n+3,2) || throw(ArgumentError("J size must be ≥ $(div(3n+3,2)) for n=$n"))

    b = zeros(float(eltype(J)), 2n)
    for j = 1:div(3n+1,2)
        b[j] = J.ev[j]^2
    end
    return b
end

# return the Kronrod–Jacobi matrix
function _kronrodjacobi(J::AbstractSymTri{<:Real}, b::AbstractVector{T}, n::Int) where {T<:AbstractFloat}
    # these are checked above:
    # size(J,1) > div(3n+1,2) || throw(ArgumentError("J size must be > $(div(3n+1,2)) for n=$n"))
    # length(b) == 2n || throw(DimensionMismatch())

    # construct a,b of Jacobi–Kronrod matrix:
    if J isa SymTridiagonal
        a = copyto!(zeros(T, 2n+1), 1, J.dv, 1, div(3n, 2) + 1)
        # (a is zero if J isa HollowSymTridiagonal, and is hence omitted).
    end
    s = zeros(T, div(n,2) + 2)
    t = zeros(T, length(s))
    t[2] = b[n+1]
    for m = 0:n-2
        u = zero(T)
        for k = div(m+1,2):-1:0
            u += b[k + n + 1]*s[k+1] - (m > k ? b[m - k]*s[k+2] : zero(T))
            if J isa SymTridiagonal
                u += (a[k+n+2] - a[(m-k)+1]) * t[k+2]
            end
            s[k+2] = u
        end
        s,t = t,s
    end
    for j = div(n,2):-1:0
        s[j+2] = s[j+1]
    end
    for m = n-1:2n-3
        u = zero(T)
        for k = m+1-n:div(m-1,2)
            j = n - (m - k) - 1
            u -= b[k + n + 1]*s[j+2] - b[m - k]*s[j+3]
            if J isa SymTridiagonal
                u -= (a[k+n+2] - a[(m-k)+1]) * t[j+2]
            end
            s[j+2] = u
        end
        k = div(m+1,2)
        j = n - (m - k + 2)
        if 2k != m
            b[k+n+1] = s[j+2] / s[j+3]
        elseif J isa SymTridiagonal
            a[k+n+2] = a[k+1] + (s[j+2] - b[k+n+1] * s[j+3]) / t[j+3]
        end
        s,t = t,s
    end
    if J isa SymTridiagonal
        a[2n+1] = a[n] - b[2n]*s[2]/t[2]
    end

    # Note: this sqrt.(b) step can fail if any element of b is negative.
    # This can happen because Gauss–Kronrod points and weights are not guaranteed
    # to be real for all orthogonal polynomials (all Jacobi matrices J)!
    # We won't try to handle this case in QuadGK for now.
    any(<(0), b) && throw(ArgumentError("real Gauss–Kronrod rule does not exist for this Jacobi matrix"))
    b .= sqrt.(b)

    # the Jacobi–Kronrod matrix:
    return J isa SymTridiagonal ? SymTridiagonal(a, b) : HollowSymTridiagonal(b)
end

# return the Kronrod weights and rule.  unitintegral should be the integral of the weight function
function _kronrod(J::AbstractSymTri{<:Real}, b::AbstractVector{T}, n::Int, unitintegral::Real=1) where {T<:AbstractFloat}
    # the Jacobi–Kronrod matrix:
    KJ = _kronrodjacobi(J, b, n)

    # now we just apply Golub–Welch to KJ:

    # get quadrature points x (negative points only for HollowSymTridiagonal)
    x = eignewt(KJ, J isa SymTridiagonal ? 2n+1 : n+1)

    v = Vector{promote_type(eltype(b),eltype(x))}(undef, 2n+1)

    # get quadrature weights
    w = T[ unitintegral * abs2(eigvec1!(v,KJ,λ)[1]) for λ in x ]

    return (x, w, v)
end

###########################################################################
# Type-stable cache of quadrature rule results, so that we don't
# repeat the kronrod calculation unnecessarily.

# precomputed n=7 rule in double precision (computed in 100-bit arithmetic),
# since this is the common case.
const xd7 = [-9.9145537112081263920685469752598e-01,
             -9.4910791234275852452618968404809e-01,
             -8.6486442335976907278971278864098e-01,
             -7.415311855993944398638647732811e-01,
             -5.8608723546769113029414483825842e-01,
             -4.0584515137739716690660641207707e-01,
             -2.0778495500789846760068940377309e-01,
             0.0]
const wd7 = [2.2935322010529224963732008059913e-02,
             6.3092092629978553290700663189093e-02,
             1.0479001032225018383987632254189e-01,
             1.4065325971552591874518959051021e-01,
             1.6900472663926790282658342659795e-01,
             1.9035057806478540991325640242055e-01,
             2.0443294007529889241416199923466e-01,
             2.0948214108472782801299917489173e-01]
const gwd7 = [1.2948496616886969327061143267787e-01,
              2.797053914892766679014677714229e-01,
              3.8183005050511894495036977548818e-01,
              4.1795918367346938775510204081658e-01]

# cache of T -> n -> (x,w,gw) Kronrod rules, to avoid recomputing them
# unnecessarily for repeated integration.   We initialize it with the
# default n=7 rule for double-precision calculations.  We use a cache
# of caches to allow us to evaluate the cache in a type-stable way with
# a generated function below.
const rulecache = Dict{Type,Dict}(
    Float64 => Dict{Int,NTuple{3,Vector{Float64}}}(7 => (xd7,wd7,gwd7)),
    Float32 => Dict{Int,NTuple{3,Vector{Float32}}}(7 => (xd7,wd7,gwd7)))

# for BigFloat rules, we need a separate cache keyed by (n,precision)
const bigrulecache = Dict{Tuple{Int,Int}, NTuple{3,Vector{BigFloat}}}()

function cachedrule(::Union{Type{BigFloat},Type{Complex{BigFloat}}}, n::Integer)
    key = (Int(n), precision(BigFloat))
    haskey(bigrulecache, key) ? bigrulecache[key] : (bigrulecache[key] = kronrod(BigFloat, Int(n)))
end

# use a generated function to make this type-stable
@generated function _cachedrule(::Type{TF}, n::Int) where {TF}
    cache = haskey(rulecache, TF) ? rulecache[TF] : (rulecache[TF] = Dict{Int,NTuple{3,Vector{TF}}}())
    :(haskey($cache, n) ? $cache[n] : ($cache[n] = kronrod($TF, n)))
end

cachedrule(::Type{T}, n::Integer) where {T<:Number} =
    _cachedrule(typeof(float(real(one(T)))), Int(n))
