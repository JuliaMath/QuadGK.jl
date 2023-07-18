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

###########################################################################
# Eigensolver utilities:

# Implement a type for symmetric tridiagonal matrices with zero diagonals:
#
# For the common case of Gauss-Kronrod rules for the unit weight function,
# the diagonals of the Jacobi matrices are zero and certain things simplify
# compared to the general case of an arbitrary weight function.

# a type for us to dispatch on; we don't actually need the full AbstractMatrix functionality
struct ZeroSymTridiagonal{T, V<:AbstractVector{T}} <: AbstractMatrix{T}
    ev::V # superdiagonal
    function ZeroSymTridiagonal{T, V}(ev) where {T, V<:AbstractVector{T}}
        Base.require_one_based_indexing(ev)
        return new{T, V}(ev)
    end
end
ZeroSymTridiagonal(ev::AbstractVector{T}) where {T<:Real} =
    ZeroSymTridiagonal{T,typeof(ev)}(ev)
Base.size(A::ZeroSymTridiagonal) = (length(A.ev)+1,length(A.ev)+1)
LinearAlgebra.diag(A::ZeroSymTridiagonal) = zeros(eltype(A), size(A,1))
LinearAlgebra.SymTridiagonal(A::ZeroSymTridiagonal{T}) where {T} = SymTridiagonal{T}(A)
LinearAlgebra.SymTridiagonal{T}(A::ZeroSymTridiagonal) where {T} = SymTridiagonal(zeros(T, length(A.ev)+1), Vector{T}(A.ev))
Base.Matrix(A::ZeroSymTridiagonal) = Matrix(SymTridiagonal(A))
Base.Matrix{T}(A::ZeroSymTridiagonal) where {T} = Matrix{T}(SymTridiagonal{T}(A))

const AbstractSymTri{T} = Union{ZeroSymTridiagonal{T}, SymTridiagonal{T}}

# for display purposes:
function Base.replace_in_print_matrix(A::ZeroSymTridiagonal, i::Integer, j::Integer, s::AbstractString)
    i==j-1||i==j+1 ? s : Base.replace_with_centered_mark(s)
end
@inline function Base.getindex(A::ZeroSymTridiagonal{T}, i::Integer, j::Integer) where T
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
eigpoly(H::ZeroSymTridiagonal{<:Real},z) = eigpoly(H.ev, z)

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
    eignewt(ZeroSymTridiagonal(b), n)
end

# given an eigenvalue λ and the matrix H(b) from above, return
# the corresponding eigenvector, normalized to 1.
function eigvec1!(v::AbstractVector, H::ZeroSymTridiagonal, λ::Number)
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
    return eigvec1!(v, ZeroSymTridiagonal(b), λ)
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
    gauss([T,] N, a=-1, b=1)

Return a pair `(x, w)` of `N` quadrature points `x[i]` and weights `w[i]` to
integrate functions on the interval `(a, b)`,  i.e. `sum(w .* f.(x))`
approximates the integral.  Uses the method described in Trefethen &
Bau, Numerical Linear Algebra, to find the `N`-point Gaussian quadrature
in O(`N`²) operations.

`T` is an optional parameter specifying the floating-point type, defaulting
to `Float64`. Arbitrary precision (`BigFloat`) is also supported.
"""
function gauss(::Type{T}, N::Integer) where T<:AbstractFloat
    if N < 1
        throw(ArgumentError("Gauss rules require positive order"))
    end
    o = one(T)
    b = T[ n / sqrt(4n^2 - o) for n = 1:N-1 ]
    return gauss(ZeroSymTridiagonal(b))
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

# Gauss rules for an arbitrary Jacobi matrix J
function gauss(J::AbstractSymTri{<:Real})
    # Golub–Welch algorithm
    x = eignewt(J, size(J,1))
    v = Vector{promote_type(eltype(J),eltype(x))}(undef, size(J,1))
    w = [ 2abs2(eigvec1!(v,J,x[i])[1]) for i = 1:size(J,1) ]
    return (x, w)
end

"""
    kronrod([T,] n)

Compute `2n+1` Kronrod points `x` and weights `w` based on the description in
Laurie (1997), appendix A, simplified for `a=0`, for integrating on `[-1,1]`.
Since the rule is symmetric, this only returns the `n+1` points with `x <= 0`.
The function Also computes the embedded `n`-point Gauss quadrature weights `gw`
(again for `x <= 0`), corresponding to the points `x[2:2:end]`. Returns `(x,w,wg)`
in O(`n`²) operations.

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
    if n < 1
        throw(ArgumentError("Kronrod rules require positive order"))
    end
    o = one(T)
    b = zeros(T, 2n+1)
    b[1] = 2*o
    for j = 1:div(3n+1,2)
        b[j+1] = j^2 / (4j^2 - o)
    end
    s = zeros(T, div(n,2) + 2)
    t = zeros(T, div(n,2) + 2)
    t[2] = b[n+2]
    for m = 0:n-2
        u = zero(T)
        for k = div(m+1,2):-1:0
            l = m - k + 1
            k1 = k + n + 2
            u += b[k1]*s[k+1] - b[l]*s[k+2]
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
            l = m - k + 1
            j = n - l
            k1 = k + n + 2
            u -= b[k1]*s[j+2] - b[l]*s[j+3]
            s[j+2] = u
        end
        k = div(m+1,2)
        if 2k != m
            j = n - (m - k + 2)
            b[k+n+2] = s[j+2] / s[j+3]
        end
        s,t = t,s
    end
    for j = 1:2n
        b[j] = sqrt(b[j+1])
    end
    resize!(b, 2n) # no longer need last point

    # get negative quadrature points x
    x = eignewt(b,2n+1,n+1) # x <= 0

    v = Vector{promote_type(eltype(b),eltype(x))}(undef, 2n+1)

    # get quadrature weights
    w = T[ 2abs2(eigvec1!(v,b,x[i],2n+1)[1]) for i in 1:n+1 ]

    # Get embedded Gauss rule from even-indexed points, using
    # the Golub–Welch method as described in Trefethen and Bau.
    for j = 1:n-1
        b[j] = j / sqrt(4j^2 - o)
    end
    @views gw = T[ 2abs2(eigvec1!(v[1:n],b[1:n-1],x[i],n)[1]) for i = 2:2:n+1 ]

    return (x, w, gw)
end

kronrod(N::Integer) = kronrod(Float64, N)

###########################################################################
# Gauss–Kronrod rules for an arbitrary Jacobi matrix J

function gauss(J::AbstractSymTri{<:Real})
    # Golub–Welch algorithm
    x = eignewt(J, size(J,1))
    v = Vector{promote_type(eltype(J),eltype(x))}(undef, size(J,1))
    w = [ 2abs2(eigvec1!(v,J,x[i])[1]) for i = 1:size(J,1) ]
    return (x, w)
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
    key = (n, precision(BigFloat))
    haskey(bigrulecache, key) ? bigrulecache[key] : (bigrulecache[key] = kronrod(BigFloat, n))
end

# use a generated function to make this type-stable
@generated function _cachedrule(::Type{TF}, n::Int) where {TF}
    cache = haskey(rulecache, TF) ? rulecache[TF] : (rulecache[TF] = Dict{Int,NTuple{3,Vector{TF}}}())
    :(haskey($cache, n) ? $cache[n] : ($cache[n] = kronrod($TF, n)))
end

cachedrule(::Type{T}, n::Integer) where {T<:Number} =
    _cachedrule(typeof(float(real(one(T)))), Int(n))
