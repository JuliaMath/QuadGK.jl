
# This can be wrapped around the `segbuf` argument
# of do_quadgk to indicate that the segment buffer
# should be saved and returned as an extra return value.
# (Helps us re-use do_quadgk for both cases, while
#  remaining type-stable despite the varying return value.)
struct ReturnSegbuf{S<:Union{Nothing,<:AbstractVector{<:Segment}}}
    segbuf::S # either a pre-allocateed segment buffer or nothing
end

# Internal routine: integrate f over the union of the open intervals
# (s[1],s[2]), (s[2],s[3]), ..., (s[end-1],s[end]), using h-adaptive
# integration with the order-n Kronrod rule and weights of type Tw,
# with absolute tolerance atol and relative tolerance rtol,
# with maxevals an approximate maximum number of f evaluations.
function do_quadgk(f::F, s::NTuple{N,T}, n, atol, rtol, maxevals, nrm,
                   _segbuf::Union{Nothing,<:AbstractVector{<:Segment},ReturnSegbuf},
                   eval_segbuf::Union{Nothing,<:AbstractVector{<:Segment}}) where {T,N,F}
    x,w,wg = cachedrule(T,n)
    segbuf = _segbuf isa ReturnSegbuf ? _segbuf.segbuf : _segbuf

    if !isnothing(eval_segbuf) # contains initial quadrature intervals
        isempty(eval_segbuf) && throw(ArgumentError("eval_segbuf must be non-empty"))
        if f isa BatchIntegrand
            if isnothing(segbuf)
                segbuf2 = evalrules(f, eval_segbuf, x,w,wg, nrm)
            else
                segbuf2 = evalrules!(resize!(segbuf, length(eval_segbuf)),
                                     f, eval_segbuf, x,w,wg, nrm)
            end
        else
            if isnothing(segbuf)
                segbuf2 = map(eval_segbuf) do seg
                    evalrule(f, seg.a, seg.b, x,w,wg, nrm)
                end
            else
                segbuf2 = map!(resize!(segbuf, length(eval_segbuf)), eval_segbuf) do seg
                    evalrule(f, seg.a, seg.b, x,w,wg, nrm)
                end
            end
        end

        I, E = resum(f, segbuf2)
        numevals = (2n+1) * length(segbuf2)
    else
        @assert N ≥ 2
        if f isa BatchIntegrand
            segs = evalrules(f, s, x,w,wg, nrm)
        else
            segs = ntuple(Val{N-1}()) do i
                a, b = s[i], s[i+1]
                evalrule(f, a,b, x,w,wg, nrm)
            end
        end

        I, E = resum(f, segs)
        numevals = (2n+1) * (N-1)

        # save segs into segbuf if !== nothing, both for adaptive subdivision
        # and to return to the user for re-using with different integrands:
        isnothing(segbuf) || (resize!(segbuf, N-1) .= segs)
        segbuf2 = segbuf # to mirror eval_segbuf branch above
    end

    # logic here is mainly to handle dimensionful quantities: we
    # don't know the correct type of atol115, in particular, until
    # this point where we have the type of E from f.  Also, follow
    # Base.isapprox in that if atol≠0 is supplied by the user, rtol
    # defaults to zero.
    atol_ = something(atol, zero(E))
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(eltype(x)))) : zero(eltype(x)))

    # optimize common case of no subdivision
    if numevals ≥ maxevals || E ≤ atol_ || E ≤ rtol_ * nrm(I)
        # fast return when no subdivisions required
        if _segbuf isa ReturnSegbuf
            return (I, E, isnothing(segbuf2) ? collect(segs) : segbuf2)
        else
            return (I, E)
        end
    end

    segheap = segbuf2 === nothing ? collect(segs) : segbuf2
    heapify!(segheap, Reverse)
    I, E = resum(f, adapt(f, segheap, I, E, numevals, x,w,wg,n, atol_, rtol_, maxevals, nrm))
    return _segbuf isa ReturnSegbuf ? (I, E, segheap) : (I, E)
end

# internal routine to perform the h-adaptive refinement of the integration segments (segs)
function adapt(f::F, segs::Vector{T}, I, E, numevals, x,w,wg,n, atol, rtol, maxevals, nrm) where {F, T}
    # Pop the biggest-error segment and subdivide (h-adaptation)
    # until convergence is achieved or maxevals is exceeded.
    while E > atol && E > rtol * nrm(I) && numevals < maxevals
        next = refine(f, segs, I, E, numevals, x,w,wg,n, atol, rtol, maxevals, nrm)
        next isa Vector && return next # handle type-unstable functions
        I, E, numevals = next
    end
    return segs
end

# internal routine to refine the segment with largest error
function refine(f::F, segs::Vector{T}, I, E, numevals, x,w,wg,n, atol, rtol, maxevals, nrm) where {F, T}
    s = heappop!(segs, Reverse)
    mid = (s.a + s.b) / 2

    # early return if integrand evaluated at endpoints
    if check_endpoint_roundoff(s.a, mid, x) || check_endpoint_roundoff(mid, s.b, x)
        heappush!(segs, s, Reverse)
        return segs
    end

    s1 = evalrule(f, s.a, mid, x,w,wg, nrm)
    s2 = evalrule(f, mid, s.b, x,w,wg, nrm)

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
                     I, E, numevals, x,w,wg,n, atol, rtol, maxevals, nrm)
    end

    heappush!(segs, s1, Reverse)
    heappush!(segs, s2, Reverse)

    return I, E, numevals
end

# re-sum (paranoia about accumulated roundoff)
function resum(f, segs)
    if f isa InplaceIntegrand
        I = f.I .= segs[firstindex(segs)].I
        E = segs[firstindex(segs)].E
        for i in firstindex(segs)+1:lastindex(segs)
            I .+= segs[i].I
            E += segs[i].E
        end
    else
        I = segs[firstindex(segs)].I
        E = segs[firstindex(segs)].E
        for i in firstindex(segs)+1:lastindex(segs)
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
