"""
    BatchIntegrand(f!, y::AbstractVector, [x::AbstractVector]; max_batch=typemax(Int))
    BatchIntegrand{Y,X}(f!; max_batch=typemax(Int)) where {Y,X}
    BatchIntegrand{Y}(f!; max_batch=typemax(Int)) where {Y}

Constructor for a `BatchIntegrand` accepting an integrand of the form `f!(y,x) = y .= f.(x)`
that can evaluate the integrand at multiple quadrature nodes using, for example, threads,
the GPU, or distributed memory. The arguments `y`, and optionally `x`, are pre-allocated
buffers of the correct element type for the domain and range of `f!`. Additionally, they
must be `resize!`-able since the number of evaluation points may vary between calls to `f!`.
Alternatively, the element types of those buffers, `Y` and optionally `X`, may be passed as
type parameters to the constructor. The `max_batch` keyword roughly limits the number of
nodes passed to the integrand, though at least `4*order+2` nodes will be used by the GK
rule.

## Internal allocations

If `x` or `X` are not specified, `quadgk` internally creates a new `BatchIntegrand` with the
user-supplied `y` buffer and a freshly-allocated `x` buffer based on the domain types. So,
if you want to reuse the `x` buffer between calls, supply `{Y,X}` or pass `y,x` explicitly.
"""
struct BatchIntegrand{Y,X,Ty<:AbstractVector{Y},Tx<:AbstractVector{X},F}
    # in-place function f!(y, x) that takes an array of x values and outputs an array of results in-place
    f!::F
    y::Ty
    x::Tx
    max_batch::Int # maximum number of x to supply in parallel
end

function BatchIntegrand(f!, y::AbstractVector, x::AbstractVector=similar(y, Nothing); max_batch::Integer=typemax(Int))
    max_batch > 0 || throw(ArgumentError("max_batch must be positive"))
    return BatchIntegrand(f!, y, x, max_batch)
end
BatchIntegrand{Y,X}(f!; kws...) where {Y,X} = BatchIntegrand(f!, Y[], X[]; kws...)
BatchIntegrand{Y}(f!; kws...) where {Y} = BatchIntegrand(f!, Y[]; kws...)

function batchevalrule(fx::AbstractVector{T}, a,b, x,w,gw, nrm) where {T}
    l = length(x)
    n = 2l - 1 # number of Kronrod points
    n1 = 1 - (l & 1) # 0 if even order, 1 if odd order
    s = convert(eltype(x), 0.5) * (b-a)
    if n1 == 0 # even: Gauss rule does not include x == 0
        Ik = fx[l] * w[end]
        Ig = zero(Ik)
    else # odd: don't count x==0 twice in Gauss rule
        f0 = fx[l]
        Ig = f0 * gw[end]
        Ik = f0 * w[end] + (fx[l-1] + fx[l+1]) * w[end-1]
    end
    for i = 1:length(gw)-n1
        fg = fx[2i] + fx[n-2i+1]
        fk = fx[2i-1] + fx[n-2i+2]
        Ig += fg * gw[i]
        Ik += fg * w[2i] + fk * w[2i-1]
    end
    Ik_s, Ig_s = Ik * s, Ig * s # new variable since this may change the type
    E = nrm(Ik_s - Ig_s)
    if isnan(E) || isinf(E)
        throw(DomainError(a+s, "integrand produced $E in the interval ($a, $b)"))
    end
    return Segment(oftype(s, a), oftype(s, b), Ik_s, E)
end

function evalrules(f::BatchIntegrand, s::NTuple{N}, x,w,gw, nrm) where {N}
    l = length(x)
    m = 2l-1    # evaluations per segment
    n = (N-1)*m # total evaluations
    resize!(f.x, n)
    resize!(f.y, n)
    for i in 1:(N-1)    # fill buffer with evaluation points
        a = s[i]; b = s[i+1]
        c = convert(eltype(x), 0.5) * (b-a)
        o = (i-1)*m
        f.x[l+o] = a + c
        for j in 1:l-1
            f.x[j+o] = a + (1 + x[j]) * c
            f.x[m+1-j+o] = a + (1 - x[j]) * c
        end
    end
    f.f!(f.y, f.x)  # evaluate integrand
    return ntuple(Val(N-1)) do i
        return batchevalrule(view(f.y, (1+(i-1)*m):(i*m)), s[i], s[i+1], x,w,gw, nrm)
    end
end

# we refine as many segments as we can fit into the buffer
function refine(f::BatchIntegrand, segs::Vector{T}, I, E, numevals, x,w,gw,n, atol, rtol, maxevals, nrm) where {T}
    tol = max(atol, rtol*nrm(I))
    nsegs = 0
    len = length(segs)
    l = length(x)
    m = 2l-1 # == 2n+1

    # collect as many segments that will have to be evaluated for the current tolerance
    # while staying under max_batch and maxevals
    while len > nsegs && 2m*nsegs < f.max_batch && E > tol && numevals < maxevals
        # same as heappop!, but moves segments to end of heap/vector to avoid allocations
        s = segs[1]
        y = segs[len-nsegs]
        segs[len-nsegs] = s
        nsegs += 1
        tol += s.E
        numevals += 2m
        len > nsegs && DataStructures.percolate_down!(segs, 1, y, Reverse, len-nsegs)
    end

    resize!(f.x, 2m*nsegs)
    resize!(f.y, 2m*nsegs)
    for i in 1:nsegs    # fill buffer with evaluation points
        s = segs[len-i+1]
        mid = (s.a+s.b)/2
        for (j,a,b) in ((2,s.a,mid), (1,mid,s.b))
            check_endpoint_roundoff(a, b, x) && return segs
            c = convert(eltype(x), 0.5) * (b-a)
            o = (2i-j)*m
            f.x[l+o] = a + c
            for k in 1:l-1
                # early return if integrand evaluated at endpoints
                f.x[k+o] = a + (1 + x[k]) * c
                f.x[m+1-k+o] = a + (1 - x[k]) * c
            end
        end
    end
    f.f!(f.y, f.x)

    resize!(segs, len+nsegs)
    for i in 1:nsegs    # evaluate segments and update estimates & heap
        s = segs[len-i+1]
        mid = (s.a + s.b)/2
        s1 = batchevalrule(view(f.y, 1+2(i-1)*m:(2i-1)*m), s.a,mid, x,w,gw, nrm)
        s2 = batchevalrule(view(f.y, 1+(2i-1)*m:2i*m), mid,s.b, x,w,gw, nrm)
        I = (I - s.I) + s1.I + s2.I
        E = (E - s.E) + s1.E + s2.E
        segs[len-i+1] = s1
        segs[len+i]   = s2
    end
    for i in 1:2nsegs
        DataStructures.percolate_up!(segs, len-nsegs+i, Reverse)
    end

    return I, E, numevals
end

function handle_infinities(workfunc, f::BatchIntegrand, s)
    s1, s2 = s[1], s[end]
    if realone(s1) && realone(s2) # check for infinite or semi-infinite intervals
        inf1, inf2 = isinf(s1), isinf(s2)
        if inf1 || inf2
            xtmp = f.x # buffer to store evaluation points
            ytmp = f.y # original integrand may have different units
            xbuf = similar(xtmp, typeof(one(eltype(f.x))))
            ybuf = similar(ytmp, typeof(oneunit(eltype(f.y))*oneunit(s1)))
            if inf1 && inf2 # x = t/(1-t^2) coordinate transformation
                return workfunc(BatchIntegrand((v, t) -> begin resize!(xtmp, length(t)); resize!(ytmp, length(v));
                                            f.f!(ytmp, xtmp .= oneunit(s1) .* t ./ (1 .- t .* t)); v .= ytmp .* (1 .+ t .* t) .* oneunit(s1) ./ (1 .- t .* t) .^ 2; end, ybuf, xbuf, f.max_batch),
                                map(x -> isinf(x) ? (signbit(x) ? -one(x) : one(x)) : 2x / (oneunit(x)+hypot(oneunit(x),2x)), s),
                                t -> oneunit(s1) * t / (1 - t^2))
            end
            let (s0,si) = inf1 ? (s2,s1) : (s1,s2) # let is needed for JuliaLang/julia#15276
                if si < zero(si) # x = s0 - t/(1-t)
                    return workfunc(BatchIntegrand((v, t) -> begin resize!(xtmp, length(t)); resize!(ytmp, length(v));
                                            f.f!(ytmp, xtmp .= s0 .- oneunit(s1) .* t ./ (1 .- t)); v .= ytmp .* oneunit(s1) ./ (1 .- t) .^ 2; end, ybuf, xbuf, f.max_batch),
                                    reverse(map(x -> 1 / (1 + oneunit(x) / (s0 - x)), s)),
                                    t -> s0 - oneunit(s1)*t/(1-t))
                else # x = s0 + t/(1-t)
                    return workfunc(BatchIntegrand((v, t) -> begin resize!(xtmp, length(t)); resize!(ytmp, length(v));
                                            f.f!(ytmp, xtmp .= s0 .+ oneunit(s1) .* t ./ (1 .- t)); v .= ytmp .* oneunit(s1) ./ (1 .- t) .^ 2; end, ybuf, xbuf, f.max_batch),
                                    map(x -> 1 / (1 + oneunit(x) / (x - s0)), s),
                                    t -> s0 + oneunit(s1)*t/(1-t))
                end
            end
        end
    end
    return workfunc(f, s, identity)
end

"""
    quadgk(f::BatchIntegrand, a,b,c...; kws...)

Like [`quadgk`](@ref), but batches evaluation points for an in-place integrand to evaluate
simultaneously. In particular, there are two differences from `quadgk`

1. The function `f.f!` should be of the form `f!(y, x) = y .= f.(x)`.  That is, it writes
   the return values of the integand `f(x)` in-place into its first argument `y`. (The
   return value of `f!` is ignored.) See [`BatchIntegrand`](@ref) for how to define the
   integrand.

2. `f.max_batch` changes how the adaptive refinement is done by batching multiple segments
   together. Choosing `max_batch<=4*order+2` will reproduce the result of
   non-batched `quadgk`, however if `max_batch=n*(4*order+2)` up to `2n` Kronrod rules will be evaluated
   together, which can produce slightly different results and sometimes require more
   integrand evaluations when using relative tolerances.
"""
function quadgk(f::BatchIntegrand{Y,Nothing}, segs::T...; kws...) where {Y,T}
    FT = float(T) # the gk points are floating-point
    g = BatchIntegrand(f.f!, f.y, similar(f.x, FT), f.max_batch)
    return quadgk(g, segs...; kws...)
end
