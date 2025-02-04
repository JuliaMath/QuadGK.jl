# the high-level quadgk* API functions and friends

"""
    quadgk(f, a,b,c...; rtol=sqrt(eps), atol=0, maxevals=10^7, order=7, norm=norm, segbuf=nothing, eval_segbuf=nothing)

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

Instead of passing the integration domain as separate arguments `a,b,c...`,
you can alternatively pass the domain as a single argument: an array of
endpoints `[a,b,c...]` or an array of interval tuples `[(a,b), (b,c)]`.
(The latter enables you to integrate over a disjoint domain if you want.)

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
`segbuf` argument.  Alternatively, one can replace the first `quadgk(...)`
call with `quadgk_segbuf(...)` to return the segment buffer from a given
call.  This buffer can be used across multiple calls to avoid
repeated allocation.   Upon return from `quadgk`, the `segbuf` array contains
an array of subintervals that were used for the final quadrature evaluation.

By passing `eval_segbuf=segbuf` to a subsequent call to `quadgk`, these subintervals
can be re-used as the starting point for the next integrand evaluation (over the
same domain), even for an integrand of a different result type; by also passing
`maxevals=0`, further refinement of these subintervals is prohibited, so that it
forces the same quadrature rule to be used (which is useful for evaluating e.g.
derivatives of the approximate integral).
"""
quadgk(f, segs...; kws...) =
    quadgk(f, promote(segs...)...; kws...)

function quadgk(f, segs::T...;
       atol=nothing, rtol=nothing, maxevals=10^7, order=7, norm=norm, segbuf=nothing, eval_segbuf=nothing) where {T}
    handle_infinities(f, segs) do f, s, _
        do_quadgk(f, s, order, atol, rtol, maxevals, norm, segbuf, eval_segbuf)
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

Alternatively, you can replace your first call to `quadgk(...)` with a
call to `quadgk_segbuf(...)`, which returns the computed segment buffer
from your first integration.  This saves you the trouble of figuring out
`domain_type` etc., which may not be obvious if the integrand is a variable.
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

function quadgk!(f!, result, a::T,b::T,c::T...; atol=nothing, rtol=nothing, maxevals=10^7, order=7, norm=norm, segbuf=nothing, eval_segbuf=nothing) where {T}
    fx = result / oneunit(T) # pre-allocate array of correct type for integrand evaluations
    f = InplaceIntegrand(f!, result, fx)
    return quadgk(f, a, b, c...; atol=atol, rtol=rtol, maxevals=maxevals, order=order, norm=norm, segbuf=segbuf, eval_segbuf=eval_segbuf)
end

struct Counter{F}
    f::F
    count::Base.RefValue{Int}
end
function (c::Counter{F})(args...) where F
    c.count[] += 1
    c.f(args...)
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
    counter = Counter(f, Ref(0))
    i = quadgk(counter, args...; kws...)
    return (i..., counter.count[])
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

# variants that also return a segment buffer:

"""
    quadgk_segbuf(args...; kws...)

Identical to `quadgk(args...; kws...)`, but returns a tuple
`(I, E, segbuf)` where `segbuf` is a segment buffer (storing the
subintervals used for the final integral evaluation) that can
be passed as a `segbuf` and/or `eval_segbuf` argument on subsequent
calls to `quadgk` and related functions.
"""
quadgk_segbuf(args...; segbuf=nothing, kws...) =
    quadgk(args...; segbuf=ReturnSegbuf(segbuf), kws...)


"""
    quadgk_segbuf_count(args...; kws...)

Identical to `quadgk_count(args...; kws...)`, but returns a tuple
`(I, E, segbuf, count)` where `segbuf` is a segment buffer (storing the
subintervals used for the final integral evaluation) that can
be passed as a `segbuf` and/or `eval_segbuf` argument on subsequent
calls to `quadgk` and related functions.
"""
quadgk_segbuf_count(args...; segbuf=nothing, kws...) =
    quadgk_count(args...; segbuf=ReturnSegbuf(segbuf), kws...)

"""
    quadgk_segbuf_print(args...; kws...)

Identical to `quadgk_print(args...; kws...)`, but returns a tuple
`(I, E, segbuf, count)` where `segbuf` is a segment buffer (storing the
subintervals used for the final integral evaluation) that can
be passed as a `segbuf` and/or `eval_segbuf` argument on subsequent
calls to `quadgk` and related functions.
"""
quadgk_segbuf_print(args...; segbuf=nothing, kws...) =
    quadgk_print(args...; segbuf=ReturnSegbuf(segbuf), kws...)

"""
    quadgk_segbuf!(args...; kws...)

Identical to `quadgk!(args...; kws...)`, but returns a tuple
`(I, E, segbuf)` where `segbuf` is a segment buffer (storing the
subintervals used for the final integral evaluation) that can
be passed as a `segbuf` and/or `eval_segbuf` argument on subsequent
calls to `quadgk` and related functions.
"""
quadgk_segbuf!(args...; segbuf=nothing, kws...) =
    quadgk!(args...; segbuf=ReturnSegbuf(segbuf), kws...)

# variants that take an array of points or an array of (a,b) tuples
# to specify the integration domain:

function quadgk(f, segs::Union{AbstractVector{<:Number},AbstractVector{<:Tuple{Number,Number}}}; kws...)
    segbuf, min, max = to_segbuf(segs)
    return quadgk(f, min, max; eval_segbuf=segbuf, kws...)
end
function quadgk!(f, result, segs::Union{AbstractVector{<:Number},AbstractVector{<:Tuple{Number,Number}}}; kws...)
    segbuf, min, max = to_segbuf(segs)
    return quadgk!(f, result, min, max; eval_segbuf=segbuf, kws...)
end

# helper function for above: convert array of points or intervals
# to array of Segment, along with min/max of all points so that
# quadgk gets the correct domain type and handles infinities
function to_segbuf(x::AbstractVector{<:Number})
    length(x) >= 2 || throw(ArgumentError("at least 2 endpoints are required"))
    Tx = float(isconcretetype(eltype(x)) ? eltype(x) : mapreduce(typeof, promote_type, x))
    segbuf = Segment{Tx,Nothing,Nothing}[Segment(Tx(x[i]), Tx(x[i+1])) for i in firstindex(x):lastindex(x)-1]
    return segbuf, Tx.(extrema(real, x))...
end
function to_segbuf(segments::AbstractVector{<:Tuple{Number,Number}})
    !isempty(segments) || throw(ArgumentError("at least 1 interval is required"))
    Tx = float(if isconcretetype(eltype(segments))
        promote_type(fieldtypes(eltype(segments))...)
    else
        mapreduce(seg -> promote_type(typeof.(seg)...), promote_type, segments)
    end)
    return Segment{Tx,Nothing,Nothing}[Segment(Tx(seg[1]), Tx(seg[2])) for seg in segments],
           Tx(minimum(seg -> min(real.(seg)...), segments)),
           Tx(maximum(seg -> max(real.(seg)...), segments))
end
