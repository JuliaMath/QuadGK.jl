# Internal routine: integrate f over the union of the open intervals
# (s[1],s[2]), (s[2],s[3]), ..., (s[end-1],s[end]), using h-adaptive
# integration with the order-n Kronrod rule and weights of type Tw,
# with absolute tolerance atol and relative tolerance rtol,
# with maxevals an approximate maximum number of f evaluations.
function do_quadgk(f::F, s::NTuple{N,T}, n, atol, rtol, maxevals, nrm, segbuf) where {T,N,F}
    x,w,gw = cachedrule(T,n)

    @assert N ≥ 2
    if f isa BatchIntegrand
        segs = evalrules(f, s, x,w,gw, nrm)
    else
        segs = ntuple(Val{N-1}()) do i
            a, b = s[i], s[i+1]
            evalrule(f, a,b, x,w,gw, nrm)
        end
    end
    if f isa InplaceIntegrand
        I = f.I .= segs[1].I
        for i = 2:length(segs)
            I .+= segs[i].I
        end
    else
        I = sum(s -> s.I, segs)
    end
    E = sum(s -> s.E, segs)
    numevals = (2n+1) * (N-1)

    # logic here is mainly to handle dimensionful quantities: we
    # don't know the correct type of atol115, in particular, until
    # this point where we have the type of E from f.  Also, follow
    # Base.isapprox in that if atol≠0 is supplied by the user, rtol
    # defaults to zero.
    atol_ = something(atol, zero(E))
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(eltype(x)))) : zero(eltype(x)))

    # optimize common case of no subdivision
    if E ≤ atol_ || E ≤ rtol_ * nrm(I) || numevals ≥ maxevals
        return (I, E) # fast return when no subdivisions required
    end

    segheap = segbuf === nothing ? collect(segs) : (resize!(segbuf, N-1) .= segs)
    heapify!(segheap, Reverse)
    return resum(f, adapt(f, segheap, I, E, numevals, x,w,gw,n, atol_, rtol_, maxevals, nrm))
end

# internal routine to perform the h-adaptive refinement of the integration segments (segs)
function adapt(f::F, segs::Vector{T}, I, E, numevals, x,w,gw,n, atol, rtol, maxevals, nrm) where {F, T}
    # Pop the biggest-error segment and subdivide (h-adaptation)
    # until convergence is achieved or maxevals is exceeded.
    while E > atol && E > rtol * nrm(I) && numevals < maxevals
        next = refine(f, segs, I, E, numevals, x,w,gw,n, atol, rtol, maxevals, nrm)
        next isa Vector && return next # handle type-unstable functions
        I, E, numevals = next
    end
    return segs
end

# internal routine to refine the segment with largest error
function refine(f::F, segs::Vector{T}, I, E, numevals, x,w,gw,n, atol, rtol, maxevals, nrm) where {F, T}
    s = heappop!(segs, Reverse)
    mid = (s.a + s.b) / 2

    # early return if integrand evaluated at endpoints
    if check_endpoint_roundoff(s.a, mid, x) || check_endpoint_roundoff(mid, s.b, x)
        heappush!(segs, s, Reverse)
        return segs
    end

    s1 = evalrule(f, s.a, mid, x,w,gw, nrm)
    s2 = evalrule(f, mid, s.b, x,w,gw, nrm)

    if f isa InplaceIntegrand
        I .= (I .- s.I) .+ s1.I .+ s2.I
    else
        I = (I - s.I) + s1.I + s2.I
    end
    E = (E - s.E) + s1.E + s2.E
    numevals += 4n+2

    # handle type-unstable functions by converting to a wider type if needed
    Tj = promote_type(typeof(s1), promote_type(typeof(s2), T))
    if Tj !== T
        return adapt(f, heappush!(heappush!(Vector{Tj}(segs), s1, Reverse), s2, Reverse),
                     I, E, numevals, x,w,gw,n, atol, rtol, maxevals, nrm)
    end

    heappush!(segs, s1, Reverse)
    heappush!(segs, s2, Reverse)

    return I, E, numevals
end

# re-sum (paranoia about accumulated roundoff)
function resum(f, segs)
    if f isa InplaceIntegrand
        I = f.I .= segs[1].I
        E = segs[1].E
        for i in 2:length(segs)
            I .+= segs[i].I
            E += segs[i].E
        end
    else
        I = segs[1].I
        E = segs[1].E
        for i in 2:length(segs)
            I += segs[i].I
            E += segs[i].E
        end
    end
    return (I, E)
end

realone(x) = false
realone(x::Number) = one(x) isa Real

# transform f and the endpoints s to handle infinite intervals, if any,
# and pass transformed data to workfunc(f, s, tfunc)
function handle_infinities(workfunc, f, s)
    s1, s2 = s[1], s[end]
    if realone(s1) && realone(s2) # check for infinite or semi-infinite intervals
        inf1, inf2 = isinf(s1), isinf(s2)
        if inf1 || inf2
            if inf1 && inf2 # x = t/(1-t^2) coordinate transformation
                return workfunc(t -> begin t2 = t*t; den = 1 / (1 - t2);
                                            f(oneunit(s1)*t*den) * (1+t2)*den*den*oneunit(s1); end,
                                map(x -> isinf(x) ? (signbit(x) ? -one(x) : one(x)) : 2x / (oneunit(x)+hypot(oneunit(x),2x)), s),
                                t -> oneunit(s1) * t / (1 - t^2))
            end
            let (s0,si) = inf1 ? (s2,s1) : (s1,s2) # let is needed for JuliaLang/julia#15276
                if si < zero(si) # x = s0 - t/(1-t)
                    return workfunc(t -> begin den = 1 / (1 - t);
                                                f(s0 - oneunit(s1)*t*den) * den*den*oneunit(s1); end,
                                    reverse(map(x -> 1 / (1 + oneunit(x) / (s0 - x)), s)),
                                    t -> s0 - oneunit(s1)*t/(1-t))
                else # x = s0 + t/(1-t)
                    return workfunc(t -> begin den = 1 / (1 - t);
                                                f(s0 + oneunit(s1)*t*den) * den*den*oneunit(s1); end,
                                    map(x -> 1 / (1 + oneunit(x) / (x - s0)), s),
                                    t -> s0 + oneunit(s1)*t/(1-t))
                end
            end
        end
    end
    return workfunc(f, s, identity)
end

function handle_infinities(workfunc, f::InplaceIntegrand, s)
    s1, s2 = s[1], s[end]
    if realone(s1) && realone(s2) # check for infinite or semi-infinite intervals
        inf1, inf2 = isinf(s1), isinf(s2)
        if inf1 || inf2
            ftmp = f.fx # original integrand may have different units
            if inf1 && inf2 # x = t/(1-t^2) coordinate transformation
                return workfunc(InplaceIntegrand((v, t) -> begin t2 = t*t; den = 1 / (1 - t2);
                                            f.f!(ftmp, oneunit(s1)*t*den); v .= ftmp .* ((1+t2)*den*den*oneunit(s1)); end, f.I, f.fx * oneunit(s1)),
                                map(x -> isinf(x) ? (signbit(x) ? -one(x) : one(x)) : 2x / (oneunit(x)+hypot(oneunit(x),2x)), s),
                                t -> oneunit(s1) * t / (1 - t^2))
            end
            let (s0,si) = inf1 ? (s2,s1) : (s1,s2) # let is needed for JuliaLang/julia#15276
                if si < zero(si) # x = s0 - t/(1-t)
                    return workfunc(InplaceIntegrand((v, t) -> begin den = 1 / (1 - t);
                                            f.f!(ftmp, s0 - oneunit(s1)*t*den); v .= ftmp .* (den * den * oneunit(s1)); end, f.I, f.fx * oneunit(s1)),
                                    reverse(map(x -> 1 / (1 + oneunit(x) / (s0 - x)), s)),
                                    t -> s0 - oneunit(s1)*t/(1-t))
                else # x = s0 + t/(1-t)
                    return workfunc(InplaceIntegrand((v, t) -> begin den = 1 / (1 - t);
                                            f.f!(ftmp, s0 + oneunit(s1)*t*den); v .= ftmp .* (den * den * oneunit(s1)); end, f.I, f.fx * oneunit(s1)),
                                    map(x -> 1 / (1 + oneunit(x) / (x - s0)), s),
                                    t -> s0 + oneunit(s1)*t/(1-t))
                end
            end
        end
    end
    return workfunc(f, s, identity)
end

function check_endpoint_roundoff(a, b, x)
    c = convert(eltype(x), 0.5) * (b-a)
    eval_at_a = a == a + (1+x[1])*c
    eval_at_b = b == a + (1-x[1])*c
    return eval_at_a || eval_at_b
end

# Gauss-Kronrod quadrature of f from a to b to c...

"""
    quadgk(f, a,b,c...; rtol=sqrt(eps), atol=0, maxevals=10^7, order=7, norm=norm, segbuf=nothing)

Numerically integrate the function `f(x)` from `a` to `b`, and optionally over additional
intervals `b` to `c` and so on. Keyword options include a relative error tolerance `rtol`
(if `atol==0`, defaults to `sqrt(eps)` in the precision of the endpoints), an absolute error tolerance
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

In normal usage, `quadgk(...)` will allocate a buffer for segments. You can
instead pass a preallocated buffer allocated using `alloc_segbuf(...)` as the
`segbuf` argument. This buffer can be used across multiple calls to avoid
repeated allocation.
"""
quadgk(f, segs...; kws...) =
    quadgk(f, promote(segs...)...; kws...)

function quadgk(f, segs::T...;
       atol=nothing, rtol=nothing, maxevals=10^7, order=7, norm=norm, segbuf=nothing) where {T}
    handle_infinities(f, segs) do f, s, _
        do_quadgk(f, s, order, atol, rtol, maxevals, norm, segbuf)
    end
end

"""
    function alloc_segbuf(domain_type=Float64, range_type=Float64, error_type=Float64; size=1)

This helper will allocate a segment buffer for segments to a `quadgk(...)` call
with the given `domain_type`, which is the same as the type of the integration
limits, `range_type` i.e. the range of the function being integrated and
`error_type`, the type returned by the `norm` given to `quadgk(...)` and
starting with the given `size`. The buffer can then be reused across multiple
compatible calls to `quadgk(...)` to avoid repeated allocation.
"""
function alloc_segbuf(domain_type=Float64, range_type=Float64, error_type=Float64; size=1)
    Vector{Segment{domain_type, range_type, error_type}}(undef, size)
end

"""
    quadgk!(f!, result, a,b,c...; rtol=sqrt(eps), atol=0, maxevals=10^7, order=7, norm=norm)

Like `quadgk`, but make use of in-place operations for array-valued integrands (or other mutable
types supporting in-place operations).  In particular, there are two differences from `quadgk`:

1. The function `f!` should be of the form `f!(y, x) = y .= f(x)`.  That is, it writes the
   return value of the integand `f(x)` in-place into its first argument `y`.   (The return
   value of `f!` is ignored.)

2. Like `quadgk`, the return value is a tuple `(I,E)` of the estimated integral `I` and the
   estimated error `E`.   However, in `quadgk!` the estimated integral is written in-place
   into the `result` argument, so that `I === result`.

Otherwise, the behavior is identical to `quadgk`.

For integrands whose values are *small* arrays whose length is known at compile-time,
it is usually more efficient to use `quadgk` and modify your integrand to return
an `SVector` from the [StaticArrays.jl package](https://github.com/JuliaArrays/StaticArrays.jl).
"""
quadgk!(f!, result, segs...; kws...) =
    quadgk!(f!, result, promote(segs...)...; kws...)

function quadgk!(f!, result, a::T,b::T,c::T...; atol=nothing, rtol=nothing, maxevals=10^7, order=7, norm=norm, segbuf=nothing) where {T}
    fx = result / oneunit(T) # pre-allocate array of correct type for integrand evaluations
    f = InplaceIntegrand(f!, result, fx)
    return quadgk(f, a, b, c...; atol=atol, rtol=rtol, maxevals=maxevals, order=order, norm=norm, segbuf=segbuf)
end

"""
    quadgk_count(f, args...; kws...)

Identical to [`quadgk`](@ref) but returns a triple `(I, E, count)`
of the estimated integral `I`, the estimated error bound `E`, and a `count`
of the number of times the integrand `f` was evaluated.

The count of integrand evaluations is a useful performance metric: a large
number typically indicates a badly behaved integrand (with singularities,
discontinuities, sharp peaks, and/or rapid oscillations), in which case
it may be possible to mathematically transform the problem in some way
to improve the convergence rate.
"""
function quadgk_count(f, args...; kws...)
    count = 0
    i = quadgk(args...; kws...) do x
        count += 1
        f(x)
    end
    return (i..., count)
end

"""
    quadgk_print([io], f, args...; kws...)

Identical to [`quadgk`](@ref), but **prints** each integrand
evaluation to the stream `io` (defaults to `stdout`) in the form:
```
f(x1) = y1
f(x2) = y2
...
```
which is useful for pedagogy and debugging.

Also, like [`quadgk_count`](@ref), it returns a triple `(I, E, count)`
of the estimated integral `I`, the estimated error bound `E`, and a `count`
of the number of times the integrand `f` was evaluated.
"""
quadgk_print(io::IO, f, args...; kws...) = quadgk_count(args...; kws...) do x
    y = f(x)
    println(io, "f(", x, ") = ", y)
    y
end
quadgk_print(f, args...; kws...) = quadgk_print(stdout, f, args...; kws...)
