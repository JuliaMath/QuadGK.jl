# Internal routine: integrate f over the union of the open intervals
# (s[1],s[2]), (s[2],s[3]), ..., (s[end-1],s[end]), using h-adaptive
# integration with the order-n Kronrod rule and weights of type Tw,
# with absolute tolerance atol and relative tolerance rtol,
# with maxevals an approximate maximum number of f evaluations.
function do_quadgk(f, s, n, ::Type{Tw}, atol, rtol, maxevals, nrm) where Tw
    if eltype(s) <: Real # check for infinite or semi-infinite intervals
        s1 = s[1]; s2 = s[end]; inf1 = isinf(s1); inf2 = isinf(s2)
        if inf1 || inf2
            if inf1 && inf2 # x = t/(1-t^2) coordinate transformation
                return do_quadgk(t -> begin t2 = t*t; den = 1 / (1 - t2);
                                            f(t*den) * (1+t2)*den*den; end,
                                 map(x -> isinf(x) ? copysign(one(x), x) : 2x / (1+hypot(1,2x)), s),
                                 n, Tw, atol, rtol, maxevals, nrm)
            end
            s0,si = inf1 ? (s2,s1) : (s1,s2)
            if si < 0 # x = s0 - t/(1-t)
                return do_quadgk(t -> begin den = 1 / (1 - t);
                                            f(s0 - t*den) * den*den; end,
                                 reverse!(map(x -> 1 / (1 + 1 / (s0 - x)), s)),
                                 n, Tw, atol, rtol, maxevals, nrm)
            else # x = s0 + t/(1-t)
                return do_quadgk(t -> begin den = 1 / (1 - t);
                                            f(s0 + t*den) * den*den; end,
                                 map(x -> 1 / (1 + 1 / (x - s0)), s),
                                 n, Tw, atol, rtol, maxevals, nrm)
            end
        end
    end

    key = rulekey(Tw,n)
    x,w,gw = haskey(rulecache, key) ? rulecache[key] :
       (rulecache[key] = kronrod(Tw, n))
    segs = Segment[]
    for i in 1:length(s) - 1
        heappush!(segs, evalrule(f, s[i],s[i+1], x,w,gw, nrm), Reverse)
    end
    numevals = (2n+1) * length(segs)
    I = segs[1].I
    E = segs[1].E
    for i in 2:length(segs)
        I += segs[i].I
        E += segs[i].E
    end
    # Pop the biggest-error segment and subdivide (h-adaptation)
    # until convergence is achieved or maxevals is exceeded.
    while E > atol && E > rtol * nrm(I) && numevals < maxevals
        s = heappop!(segs, Reverse)
        mid = (s.a + s.b) / 2
        s1 = evalrule(f, s.a, mid, x,w,gw, nrm)
        s2 = evalrule(f, mid, s.b, x,w,gw, nrm)
        heappush!(segs, s1, Reverse)
        heappush!(segs, s2, Reverse)
        I = (I - s.I) + s1.I + s2.I
        E = (E - s.E) + s1.E + s2.E
        numevals += 4n+2
    end
    # re-sum (paranoia about accumulated roundoff)
    I = segs[1].I
    E = segs[1].E
    for i in 2:length(segs)
        I += segs[i].I
        E += segs[i].E
    end
    return (I, E)
end

# handle keyword deprecation
function tols(atol,rtol,abstol,reltol)
    if !ismissing(abstol) || !ismissing(reltol)
        Base.depwarn("abstol and reltol keywords are now atol and rtol, respectively", :quadgk)
    end
    return coalesce(abstol,atol), coalesce(reltol,rtol)
end

# Gauss-Kronrod quadrature of f from a to b to c...

function quadgk(f, a::T,b::T,c::T...;
                atol=zero(T), rtol=sqrt(eps(T)), abstol=missing, reltol=missing,
                maxevals=10^7, order=7, norm=norm) where T<:AbstractFloat
    atol_,rtol_ = tols(atol,rtol,abstol,reltol)
    do_quadgk(f, [a, b, c...], order, T, atol_, rtol_, maxevals, norm)
end

function quadgk(f, a::Complex{T},
                b::Complex{T},c::Complex{T}...;
                atol=zero(T), rtol=sqrt(eps(T)), abstol=missing, reltol=missing,
                maxevals=10^7, order=7, norm=norm) where T<:AbstractFloat
    atol_,rtol_ = tols(atol,rtol,abstol,reltol)
    do_quadgk(f, [a, b, c...], order, T, atol_, rtol_, maxevals, norm)
end

# generic version: determine precision from a combination of
# all the integration-segment endpoints
"""
    quadgk(f, a,b,c...; rtol=sqrt(eps), atol=0, maxevals=10^7, order=7, norm=norm)

Numerically integrate the function `f(x)` from `a` to `b`, and optionally over additional
intervals `b` to `c` and so on. Keyword options include a relative error tolerance `rtol`
(defaults to `sqrt(eps)` in the precision of the endpoints), an absolute error tolerance
`atol` (defaults to 0), a maximum number of function evaluations `maxevals` (defaults to
`10^7`), and the `order` of the integration rule (defaults to 7).

Returns a pair `(I,E)` of the estimated integral `I` and an estimated upper bound on the
absolute error `E`. If `maxevals` is not exceeded then `E <= max(atol, rtol*norm(I))`
will hold. (Note that it is useful to specify a positive `atol` in cases where `norm(I)`
may be zero.)

The endpoints `a` et cetera can also be complex (in which case the integral is performed over
straight-line segments in the complex plane). If the endpoints are `BigFloat`, then the
integration will be performed in `BigFloat` precision as well.

!!! note
    It is advisable to increase the integration `order` in rough proportion to the
    precision, for smooth integrands.

More generally, the precision is set by the precision of the integration
endpoints (promoted to floating-point types).

The integrand `f(x)` can return any numeric scalar, vector, or matrix type, or in fact any
type supporting `+`, `-`, multiplication by real values, and a `norm` (i.e., any normed
vector space). Alternatively, a different norm can be specified by passing a `norm`-like
function as the `norm` keyword argument (which defaults to `norm`).

!!! note
    Only one-dimensional integrals are provided by this function. For multi-dimensional
    integration (cubature), there are many different algorithms (often much better than simple
    nested 1d integrals) and the optimal choice tends to be very problem-dependent. See the
    Julia external-package listing for available algorithms for multidimensional integration or
    other specialized tasks (such as integrals of highly oscillatory or singular functions).

The algorithm is an adaptive Gauss-Kronrod integration technique: the integral in each
interval is estimated using a Kronrod rule (`2*order+1` points) and the error is estimated
using an embedded Gauss rule (`order` points). The interval with the largest error is then
subdivided into two intervals and the process is repeated until the desired error tolerance
is achieved.

These quadrature rules work best for smooth functions within each interval, so if your
function has a known discontinuity or other singularity, it is best to subdivide your
interval to put the singularity at an endpoint. For example, if `f` has a discontinuity at
`x=0.7` and you want to integrate from 0 to 1, you should use `quadgk(f, 0,0.7,1)` to
subdivide the interval at the point of discontinuity. The integrand is never evaluated
exactly at the endpoints of the intervals, so it is possible to integrate functions that
diverge at the endpoints as long as the singularity is integrable (for example, a `log(x)`
or `1/sqrt(x)` singularity).

For real-valued endpoints, the starting and/or ending points may be infinite. (A coordinate
transformation is performed internally to map the infinite interval to a finite one.)
"""
function quadgk(f, a, b, c...; kws...)
    T = promote_type(typeof(float(a)), typeof(b))
    for x in c
        T = promote_type(T, typeof(x))
    end
    cT = map(T, c)
    quadgk(f, convert(T, a), convert(T, b), cT...; kws...)
end