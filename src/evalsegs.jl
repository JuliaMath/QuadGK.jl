# we go to great lengths to repeat the order of operations of evalrule, but for an integrand
# evaluated in batches and possibly over multiple segments. For this we need iteration
# specific to the order of operations in both the evalution and accumulation of the
# quadrature rule

struct XIterator{T,X,FX}
    x::Vector{T}
    a::X
    s::FX
end

Base.eltype(::Type{XIterator{T,X,FX}}) where {T,X,FX} = FX
Base.length(itr::XIterator) = 2*length(itr.x)-1
function Base.iterate(itr::XIterator)
    l = length(itr.x)
    n1 = 1 - (l & 1) # 0 if even order, 1 if odd order
    return itr.a + itr.s, (0, false, false, div(l-n1,2), n1)
end
function Base.iterate(itr::XIterator, state)
    i, flip, iskronrod, n, n1 = state
    x = itr.x
    a = itr.a
    s = itr.s
    if i > n
        return nothing
    elseif i == 0
        if n1 == 0 # even: Gauss rule does not include x == 0
            return iterate(itr, (i+1, flip,iskronrod, n, n1))
        else # odd: don't count x==0 twice in Gauss rule
            if flip
                y = a + (1-x[end-1])*s
                i += 1
            else
                y = a + (1+x[end-1])*s
            end
        end
    else
        if iskronrod
            if flip
                y = a + (1-x[2i-1])*s
                iskronrod = false
                i += 1
            else
                y = a + (1+x[2i-1])*s
            end
        else
            if flip
                y = a + (1-x[2i])*s
                iskronrod = true
            else
                y = a + (1+x[2i])*s
            end
        end
    end
    return y, (i, !flip, iskronrod, n, n1)
end

# this is a flattened product iterator over the GK points in the intervals (a,b) contained
# It would also be nice to return the intervals, since if segs is a stateful iterator (e.g.
# a heap being popped), then we would only have to visit each segment once. The difficulty
# comes with keeping a buffer of the popped intervals so that we can rescale Ik and Ig by s
# at the last step of evalrule
struct XSegIterator{T,S}
    x::Vector{T}
    segs::S
end

Base.length(itr::XSegIterator) = length(itr.segs) * (2length(itr.x)-1)
Base.eltype(::Type{XSegIterator{T,S}}) where {T,S} = typeof(one(T) * oneunit(eltype(eltype(S))))

function Base.iterate(itr::XSegIterator)
    next_seg = iterate(itr.segs)
    next_seg === nothing && return nothing
    seg, segstate = next_seg
    a, b = seg
    s = convert(eltype(itr.x), 0.5) * (b - a)
    xitr = XIterator(itr.x, a, s)
    x, xstate = iterate(xitr) # this is a type-stable call
    return x, (segstate, xitr, xstate)
end
# debating whether to return (x, (seg, segstate, xstate)) to provide seg to outer routine
function Base.iterate(itr::XSegIterator, (segstate, xitr, xstate))
    next_x = iterate(xitr, xstate)
    if next_x === nothing
        next_seg = iterate(itr.segs, segstate)
        next_seg === nothing && return nothing
        seg, segstate = next_seg
        a, b = seg
        s = convert(eltype(itr.x), 0.5) * (b - a)
        xitr = XIterator(itr.x, a, s)
        x, xstate = iterate(xitr) # this is a type-stable call
        return x, (segstate, xitr, xstate)
    else
        x, xstate = next_x
        return x, (segstate, xitr, xstate)
    end
end

struct BatchedSegmentIterator{E,F,Y,X,S,T,N}
    f::BatchIntegrand{F,Y,X}
    segs::S
    x::Vector{T}
    w::Vector{T}
    gw::Vector{T}
    nrm::N
    function BatchedSegmentIterator{E}(f::BatchIntegrand{F,Y,X}, segs::S, x::R, w::R, gw::R, nrm::N) where {E,F,Y,X,S,T,R<:Vector{T},N}
        new{E,F,Y,X,S,T,N}(f, segs, x,w,gw, nrm)
    end
end

# segs must have a length and eltype
# TODO: relax the assumption about length and resize batches as needed
function BatchedSegmentIterator(f, segs, x,w,gw, nrm)
    TX = eltype(f.x)
    # x,w,gw are unitless, but segs have units
    TI = typeof(oneunit(eltype(f.y)) * one(eltype(w)) * oneunit(eltype(eltype(segs))))
    TE = typeof(nrm(oneunit(TI)))
    return BatchedSegmentIterator{Segment{TX,TI,TE}}(f, segs, x,w,gw, nrm)
end

Base.length(sb::BatchedSegmentIterator) = length(sb.segs)
Base.eltype(::Type{<:BatchedSegmentIterator{E}}) where {E} = E

function fill_xbuf!!(ybuf, xbuf, max_batch, ix, itr::XSegIterator, next)
    # TODO: resize buffers based on expected length of itr, i.e. whether there are segments
    while next !== nothing
        ix == max_batch && break
        xbuf[ix += 1], state = next
        next = iterate(itr, state)
    end
    # truncate buffers when points are exhausted or limits reached
    if next === nothing || ix == max_batch
        # TODO don't let iterator type instability of next propagate to outer functions
        resize!(xbuf, ix)
        resize!(ybuf, ix)
        ix = 0
    end
    return ix, next
end

function evalrule(f::BatchIntegrand{F}, a,b, w,gw, nrm, ix,iy,itr, next=iterate(itr)) where {F}
    # these variables are the state of the accumulator
    i = 0
    flip = false
    iskronrod = false

    l = length(w)
    n1 = 1 - (l & 1)
    n = length(gw) - n1

    # evaluate the first batch if points have not yet been added to buffer
    if iy == 0 || iy == length(f.y)
        ix, next = fill_xbuf!!(f.y, f.x, f.max_batch, ix, itr, next)
        # next === nothing && TODO extract and save some of the intervals
        f.f!(f.y, f.x)
        iy = 0
    end

    # unroll first batch iteration to get types
    f0 = f.y[iy += 1]
    fk = zero(f0)
    fg = zero(f0)
    Ik = f0 * w[end]
    Ig = n1 == 0 ? zero(Ik) : f0 * gw[end]

    while i <= n || i == 0
        # refill points if they become exhausted
        if iy == length(f.y)
            ix, next = fill_xbuf!!(f.y, f.x, f.max_batch, ix, itr, next)
            # next === nothing && TODO extract and save some of the intervals
            f.f!(f.y, f.x)
            iy = 0
        end
        fi = f.y[iy += 1]
        if i == 0
            if n1 == 0 # even: Gauss rule does not include x == 0
                i += 1
                iy -= 1
                continue
            else # odd: don't count x==0 twice in Gauss rule
                if flip
                    fk += fi
                    Ik += fk * w[end-1]
                    i += 1
                else
                    fk = fi
                end
            end
        else
            if iskronrod
                if flip
                    fk += fi
                    Ig += fg * gw[i]
                    Ik += fg * w[2i] + fk * w[2i-1]
                    iskronrod = false
                    i += 1
                else
                    fk = fi
                end
            else
                if flip
                    fg += fi
                    iskronrod = true
                else
                    fg = fi
                end
            end
        end
        flip = !flip
    end

    s = convert(eltype(w), 0.5) * (b-a)
    Ik_s, Ig_s = Ik * s, Ig * s # new variable since this may change the type
    E = nrm(Ik_s - Ig_s)
    if isnan(E) || isinf(E)
        throw(DomainError(a+s, "integrand produced $E in the interval ($a, $b)"))
    end

    return Segment(oftype(s, a), oftype(s, b), Ik_s, E), (ix, iy, itr, next)
end


# internal routine to evaluate GK rule on a segment in a collection of segments
# analogous to evalrule, however the BatchIntegrand serves as a buffer of integrand
# evaluations that may contain the data need for less than or more than one segment
function Base.iterate(sb::BatchedSegmentIterator)
    f = sb.f; segitr = sb.segs; x = sb.x; w = sb.w; gw = sb.gw; nrm = sb.nrm
    l = length(x)
    n = 2l-1
    m = min(f.max_batch, n*length(segitr)) # TODO remove this resize and do it dynamically
    resize!(f.x, m)
    resize!(f.y, m)

    # fill the batch buffer (possibly less than or more than one segment)
    next_seg = iterate(segitr)
    next_seg === nothing && return nothing
    (a, b), segstate = next_seg
    # TODO find a way to rely only on XSegIterator so we use iterator only once
    itr = XSegIterator(x, segitr)

    seg, itrstate = evalrule(f, a,b, w,gw, nrm, 0,0,itr)
    return seg, (segstate, itrstate)
end

function Base.iterate(sb::BatchedSegmentIterator, state)
    f = sb.f; segitr = sb.segs; x = sb.x; w = sb.w; gw = sb.gw; nrm = sb.nrm
    segstate, itrstate = state

    next_seg = iterate(segitr, segstate)
    next_seg === nothing && return nothing
    (a, b), segstate = next_seg

    seg, itrstate = evalrule(f, a,b, w,gw, nrm, itrstate...)
    return seg, (segstate, itrstate)
end

# TODO: we could just return segitr instead of collecting it into a tuple, but then the code
# that handles the iterator needs to know it is a stateful iterator that should only be
# iterated over once
function evalsegs(f::BatchIntegrand{F}, segs, x,w,gw, nrm) where {F}
    segitr = BatchedSegmentIterator(f, segs, x,w,gw, nrm)
    return evalsegs_(Val{length(segs)}(), segitr)
end
evalsegs_(::Val{0}, itr, next) = ()
function evalsegs_(::Val{N}, itr, next=iterate(itr)) where {N}
    next === nothing && throw(ArgumentError("exhausted segments"))
    item, state = next
    return (item, evalsegs_(Val{N-1}(), itr, iterate(itr, state))...)
end
