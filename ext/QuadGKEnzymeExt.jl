
module QuadGKEnzymeExt

using QuadGK, Enzyme, LinearAlgebra

function Enzyme.EnzymeRules.augmented_primal(config, ofunc::Const{typeof(quadgk)}, ::Type{RT}, f, segs::Annotation{T}...; kws...) where {RT, T}
    prims = map(x->x.val, segs)

    retres, segbuf = if f isa Const
        if EnzymeRules.needs_primal(config)
            quadgk(f.val, prims...; kws...), nothing
        else
            nothing
        end
    else
        I, E, segbuf = quadgk_segbuf(f.val, prims...; kws...)
        if EnzymeRules.needs_primal(config)
            (I, E), segbuf
        else
            nothing, segbuf
        end
    end

    dres = if !Enzyme.EnzymeRules.needs_shadow(config)
        nothing
    elseif EnzymeRules.width(config) == 1
        zero.(res...)
    else
        ntuple(Val(EnzymeRules.width(config))) do i
            Base.@_inline_meta
            zero.(res...)
        end
    end

    cache = if RT <: Duplicated || RT <: DuplicatedNoNeed || RT <: BatchDuplicated || RT <: BatchDuplicatedNoNeed
        dres
    else
        nothing
    end
    cache2 = segbuf, cache

    return Enzyme.EnzymeRules.AugmentedReturn{
        Enzyme.EnzymeRules.needs_primal(config) ? eltype(RT) : Nothing,
        Enzyme.EnzymeRules.needs_shadow(config) ? (Enzyme.EnzymeRules.width(config) == 1 ? eltype(RT) : NTuple{Enzyme.EnzymeRules.width(config), eltype(RT)}) : Nothing,
        typeof(cache2)
    }(retres, dres, cache2)
end

function call(f, x)
    f(x)
end

# Wrapper around a function f that allows it to act as a vector space, and hence be usable as
# an integrand, where the vector operations act on the closed-over parameters of f that are
# begin differentiated with respect to.   In particular, if we have a closure f = x -> g(x, p), and we want
# to differentiate with respect to p, then our reverse (vJp) rule needs an integrand given by the
# Jacobian-vector product (pullback) vᵀ∂g/∂p.  But Enzyme wraps this in a closure so that it is the
# same "shape" as f, whereas to integrate it we need to be able to treat it as a vector space.
# ClosureVector calls Enzyme.Compiler.recursive_add, which is an internal function that "unwraps"
# the closure to access the internal state, which can then be added/subtracted/scaled.
struct ClosureVector{F}
    f::F
end

@inline function guaranteed_nonactive(::Type{T}) where T
    rt = Enzyme.Compiler.active_reg_inner(T, (), nothing)
    return rt == Enzyme.Compiler.AnyState || rt == Enzyme.Compiler.DupState
end

function Base.:+(a::CV, b::CV) where {CV <: ClosureVector}
    Enzyme.Compiler.recursive_add(a, b, identity, guaranteed_nonactive)::CV
end

function Base.:-(a::CV, b::CV) where {CV <: ClosureVector}
    Enzyme.Compiler.recursive_add(a, b, x->-x, guaranteed_nonactive)::CV
end

function Base.:*(a::Number, b::CV) where {CV <: ClosureVector}
    # b + (a-1) * b = a * b
    Enzyme.Compiler.recursive_add(b, b, x->(a-1)*x, guaranteed_nonactive)::CV
end

function Base.:*(a::ClosureVector, b::Number)
    return b*a
end

struct MixedClosureVector{F}
    f::Base.RefValue{F}
end

function Base.:+(a::CV, b::CV) where {CV <: MixedClosureVector}
    res = deepcopy(a)::CV
    Enzyme.Compiler.recursive_accumulate(res, b, identity)::CV
end

function Base.:-(a::CV, b::CV) where {CV <: MixedClosureVector}
    res = deepcopy(a)::CV
    Enzyme.Compiler.recursive_accumulate(res, b, x->-x)::CV
end

function Base.:*(a::Number, b::CV) where {CV <: MixedClosureVector}
    # b + (a-1) * b = a * b
    res = deepcopy(b)::CV
    Enzyme.Compiler.recursive_accumulate(res, b, x->(a-1)*x)::CV
end

function Base.:*(a::MixedClosureVector, b::Number)
    return b*a
end

function Enzyme.EnzymeRules.reverse(config, ofunc::Const{typeof(quadgk)}, dres::Active, cache, f::Union{Const, Active, MixedDuplicated}, segs::Annotation{T}...; kws...) where {T}
    df = if f isa Const
        nothing
    elseif f isa Active
        segbuf = cache[1]
        fwd, rev = Enzyme.autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(call)}, Active, typeof(f), Const{T})
        _df, _ = quadgk(map(x->x.val, segs)...; kws..., eval_segbuf=segbuf, maxevals=0, norm=f->0) do x
            tape, prim, shad = fwd(Const(call), f, Const(x))
            drev = rev(Const(call), f, Const(x), dres.val[1], tape)
            return ClosureVector(drev[1][1])
        end
        _df.f
    elseif f isa MixedDuplicated
        segbuf = cache[1]
        fwd, rev = Enzyme.autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(call)}, Active, typeof(f), Const{T})
        _df, _ = quadgk(map(x->x.val, segs)...; kws..., eval_segbuf=segbuf, maxevals=0, norm=f->0) do x
            fshadow = Ref(Enzyme.make_zero(f.val))
            tape, prim, shad = fwd(Const(call), MixedDuplicated(f.val, fshadow), Const(x))
            drev = rev(Const(call), f, Const(x), dres.val[1], tape)
            return MixedClosureVector(fshadow)
        end
        Enzyme.Compiler.recursive_accumulate(f.dval, _df.f)
        nothing
    end
    dsegs1 = segs[1] isa Const ? nothing : -LinearAlgebra.dot(f.val(segs[1].val), dres.val[1])
    dsegsn = segs[end] isa Const ? nothing : LinearAlgebra.dot(f.val(segs[end].val), dres.val[1])
    return (df, # f
            dsegs1,
            ntuple(i -> nothing, Val(length(segs)-2))...,
            dsegsn)
end

function Enzyme.EnzymeRules.reverse(config, ofunc::Const{typeof(quadgk)}, dres::Type{<:Union{Duplicated, BatchDuplicated}}, cache, f::Union{Const, Active}, segs::Annotation{T}...; kws...) where {T}
    dres = cache[2]
    df = if f isa Const
        nothing
    else
        segbuf = cache[1]
        fwd, rev = Enzyme.autodiff_thunk(ReverseSplitNoPrimal, Const{typeof(call)}, Active, typeof(f), Const{T})
        _df, _ = quadgk(map(x->x.val, segs)...; kws..., eval_segbuf=segbuf, maxevals=0, norm=f->0) do x
            tape, prim, shad = fwd(Const(call), f, Const(x))
            shad .= dres
            drev = rev(Const(call), f, Const(x), tape)
            return ClosureVector(drev[1][1])
        end
        _df.f
    end
    dsegs1 = segs[1] isa Const ? nothing : -LinearAlgebra.dot(f.val(segs[1].val), dres)
    dsegsn = segs[end] isa Const ? nothing : LinearAlgebra.dot(f.val(segs[end].val), dres)
    Enzyme.make_zero!(dres)
    return (df, # f
            dsegs1,
            ntuple(i -> nothing, Val(length(segs)-2))...,
            dsegsn)
end

end # module
