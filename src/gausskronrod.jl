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
# Since we only implement Gauss-Kronrod rules for the unit weight function,
# the diagonals of the Jacobi matrices are zero and certain things simplify
# compared to the general case of an arbitrary weight function.

# Given a symmetric tridiagonal matrix H with H[i,i] = 0 and
# H[i-1,i] = H[i,i-1] = b[i-1], compute p(z) = det(z I - H) and its
# derivative p'(z), returning (p,p').
function eigpoly(b,z,m=length(b)+1)
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

# compute the n smallest eigenvalues of the symmetric tridiagonal matrix H
# (defined from b as in eigpoly) using a Newton iteration
# on det(H - lambda I).  Unlike eig, handles BigFloat.
function eignewt(b,m,n)
    # get initial guess from eig on Float64 matrix
    H = SymTridiagonal(zeros(m), Float64[ b[i] for i in 1:m-1 ])
    lambda0 = sort(eigvals(H))

    lambda = Array{eltype(b)}(undef, n)
    for i = 1:n
        lambda[i] = lambda0[i]
        for k = 1:1000
            (p,pderiv) = eigpoly(b,lambda[i],m)
            lambda[i] = (lamold = lambda[i]) - p / pderiv
            if abs(lambda[i] - lamold) < 10 * eps(lambda[i]) * abs(lambda[i])
                break
            end
        end
        # do one final Newton iteration for luck and profit:
        (p,pderiv) = eigpoly(b,lambda[i],m)
        lambda[i] = lambda[i] - p / pderiv
    end
    return lambda
end

# given an eigenvalue z and the matrix H(b) from above, return
# the corresponding eigenvector, normalized to 1.
function eigvec1(b,z::Number,m=length(b)+1)
    # "cheat" and use the fact that our eigenvector v must have a
    # nonzero first entries (since it is a quadrature weight), so we
    # can set v[1] = 1 to solve for the rest of the components:.
    v = Array{eltype(b)}(undef, m)
    v[1] = 1
    if m > 1
        s = v[1]
        v[2] = z * v[1] / b[1]
        s += v[2]^2
        for i = 3:m
            v[i] = - (b[i-2]*v[i-2] - z*v[i-1]) / b[i-1]
            s += v[i]^2
        end
        rmul!(v, 1 / sqrt(s))
    end
    return v
end

"""
    gauss([T,] N)

Return a pair `(x, w)` of `N` quadrature points `x[i]` and weights `w[i]` to
integrate functions on the interval `(-1, 1)`,  i.e. `sum(w .* f.(x))`
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
    x = eignewt(b,N,N)
    w = T[ 2*eigvec1(b,x[i])[1]^2 for i = 1:N ]
    return (x, w)
end

gauss(N::Integer) = gauss(Float64, N)

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

    # get negative quadrature points x
    x = eignewt(b,2n+1,n+1) # x <= 0

    # get quadrature weights
    w = T[ 2*eigvec1(b,x[i],2n+1)[1]^2 for i in 1:n+1 ]

    # Get embedded Gauss rule from even-indexed points, using
    # the method described in Trefethen and Bau.
    for j = 1:n-1
        b[j] = j / sqrt(4j^2 - o)
    end
    gw = T[ 2*eigvec1(b,x[i],n)[1]^2 for i = 2:2:n+1 ]

    return (x, w, gw)
end

kronrod(N::Integer) = kronrod(Float64, N)