###########################################################################

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

# integration segment (a,b), estimated integral I, and estimated error E
struct Segment
    a::Number
    b::Number
    I
    E
end
Base.isless(i::Segment, j::Segment) = isless(i.E, j.E)


# Internal routine: approximately integrate f(x) over the interval (a,b)
# by evaluating the integration rule (x,w,gw). Return a Segment.
function evalrule(f, a,b, x,w,gw, nrm)
    # Ik and Ig are integrals via Kronrod and Gauss rules, respectively
    s = convert(eltype(x), 0.5) * (b-a)
    n1 = 1 - (length(x) & 1) # 0 if even order, 1 if odd order
    # unroll first iterationof loop to get correct type of Ik and Ig
    fg = f(a + (1+x[2])*s) + f(a + (1-x[2])*s)
    fk = f(a + (1+x[1])*s) + f(a + (1-x[1])*s)
    Ig = fg * gw[1]
    Ik = fg * w[2] + fk * w[1]
    for i = 2:length(gw)-n1
        fg = f(a + (1+x[2i])*s) + f(a + (1-x[2i])*s)
        fk = f(a + (1+x[2i-1])*s) + f(a + (1-x[2i-1])*s)
        Ig += fg * gw[i]
        Ik += fg * w[2i] + fk * w[2i-1]
    end
    if n1 == 0 # even: Gauss rule does not include x == 0
        Ik += f(a + s) * w[end]
    else # odd: don't count x==0 twice in Gauss rule
        f0 = f(a + s)
        Ig += f0 * gw[end]
        Ik += f0 * w[end] +
            (f(a + (1+x[end-1])*s) + f(a + (1-x[end-1])*s)) * w[end-1]
    end
    Ik *= s
    Ig *= s
    E = nrm(Ik - Ig)
    if isnan(E) || isinf(E)
        throw(DomainError(a+s, "integrand produced $E in the interval ($a, $b)"))
    end
    return Segment(a, b, Ik, E)
end

rulekey(::Type{BigFloat}, n) = (BigFloat, precision(BigFloat), n)
rulekey(T,n) = (T,n)

cachedrule(::Type{T}, n::Integer) where {T<:Number} = cachedrule(typeof(float(real(one(T)))), n::Integer)

function cachedrule(::Type{BigFloat}, n::Integer)
    key = (n, precision(BigFloat))
    haskey(bigrulecache, key) ? bigrulecache[key] : (bigrulecache[key] = kronrod(BigFloat, n))
end

# use a generated function to make this type-stable
@generated function cachedrule(::Type{T}, n::Integer) where {T<:AbstractFloat}
    cache = haskey(rulecache, T) ? rulecache[T] : (rulecache[T] = Dict{Int,NTuple{3,Vector{T}}}())
    :(haskey($cache, n) ? $cache[n] : ($cache[n] = kronrod($T, n)))
end