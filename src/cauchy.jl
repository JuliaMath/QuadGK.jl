# Internal routine: integrate f over the the union of the open intervals
# (s[1],s[2]), (s[2],s[3]), ..., (s[end-1],s[end]), using h-adaptive
# integration
function do_cauchy(fs::NTuple{N,F}, s::NTuple{M,T}, cs::NTuple{N,U}, n_gk, n_cc, atol, rtol, maxevals, nrm) where {F,T,U,N,M}
  gk_rule = cachedrule(eltype(s), n_gk)
  cc_rule = cachedpoints(eltype(s), n_cc); cc_rule = cc_rule[2:2:end], cc_rule[1:2:end]

  @assert M == N + 1
  segs = ntuple(i -> evalrule_cauchy(fs[i], s[i], s[i+1], cs[i], gk_rule, cc_rule, nrm), Val{N}())
  I = sum(s -> s.I, segs)
  E = sum(s -> s.E, segs)
  numevals = n_cc * N # Because it will definitely be a Clenshaw-Curtis evaluation

  # logic here is mainly to handle dimensionful quantities: we
  # don't know the correct type of atol, in particular, until
  # this point where we have the type of E from f.  Also, follow
  # Base.isapprox in that if atol≠0 is supplied by the user, rtol
  # defaults to zero.
  atol_ = isnothing(atol) ? sqrt(eps(typeof(E))) : atol
  rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(eltype(gk_rule[1])))) : zero(eltype(gk_rule[1])))

  if E ≤ atol_ || E ≤ rtol_ * nrm(I) || numevals ≥ maxevals
    return (I, E)
  end
  return adapt_cauchy(heapify!(collect(segs), Reverse), I, E, numevals, n_gk, n_cc, gk_rule, cc_rule, atol_, rtol_, maxevals, nrm)
end

# internal routine to perform the h-adaptive refinement of the integration segments (segs)
function adapt_cauchy(segs::Vector{T}, I, E, numevals, n_gk, n_cc, gk_rule, cc_rule, atol, rtol, maxevals, nrm) where {T}
  # Pop the biggest-error segment and subdivide (h-adaptation)
  # until convergence is achieved or maxevals is exceeded.
  while E > atol && E > rtol * nrm(I) && numevals < maxevals
    s = heappop!(segs, Reverse)
    
    # Assert refinement does not occur at the pole
    b1 = 0.5 * (s.a + s.b)
    a2 = b1
    
    if s.c > s.a && s.c <= b1
      b1 = 0.5 * (s.c + s.b)
      a2 = b1
    elseif s.c > b1 && s.c < s.b
      b1 = 0.5 * (s.a + s.c)
      a2 = b1
    end
    
    s1 = evalrule_cauchy(s.f, s.a, b1, s.c, gk_rule,cc_rule, nrm)
    s2 = evalrule_cauchy(s.f, a2, s.b, s.c, gk_rule,cc_rule, nrm)
    I = (I - s.I) + s1.I + s2.I
    E = (E - s.E) + s1.E + s2.E
    numevals += 2n_cc
    
    # handle type-unstable functions by converting to a wider type if needed
    Tj = promote_type(typeof(s1), promote_type(typeof(s2), T))
    if Tj !== T
      return adapt_cauchy(heappush!(heappush!(Vector{Tj}(segs), s1, Reverse), s2, Reverse),
                  I, E, numevals, n_cc, n_gk, gk_rule, cc_rule, atol, rtol, maxevals, nrm)
    end
    
    heappush!(segs, s1, Reverse)
    heappush!(segs, s2, Reverse)
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

function dct2(x::Vector, p::Vector, k, n)
  return 0.5 * (x[1] + (-1)^(k+1) * x[end]) + sum(x[j] * cos((k-1) * p[j]) for j in 2:n-1)
end

function dct1(x::Vector, p::Vector, k, n)
  return sum(x[j] * cos((k-1) * p[j]) for j in 1:n)
end

function evalrule_cauchy(fn, a::T, b::T, c::T, rk_rule, cc_rule, nrm) where {T}
  d = T((2 * c - b - a) / (b - a)) # Determine how close we are to the pole

  if abs(d) > 1.1 # Use Gauss-Kronrod if far away
    seg = evalrule(x -> fn(x) / (x - c), a, b, rk_rule[1], rk_rule[2], rk_rule[3], nrm)
    return CauchySegment(fn, c, oftype(d, a), oftype(d, b), seg.I, seg.E)
  
  else # Use modified Clenshaw-Curtis if close
    n₂ = length(cc_rule[2])
    n₁ = length(cc_rule[1]) + n₂

    # Initial values for the modified moments
    μⱼ = log(abs((1.0 - d) / (1.0 + d)))

    # Function evaluation
    f2 = fn.(b .+ (1 .- cos.(cc_rule[2])) .* (a-b)/2)
    f1 = fn.(b .+ (1 .- cos.(cc_rule[1])) .* (a-b)/2)

    # unroll first loop iteration
    cs₂ = dct2(f2, cc_rule[2], 1, n₂)
    cs₁ = dct1(f1, cc_rule[1], 1, n₂-1) + cs₂

    # Integrals with (n+1)/2-point and n-point Clenshaw-Curtis quadrature
    I₂ = 1.0/(n₂-1) * cs₂ * μⱼ
    I₁ = 1.0/(n₁-1) * cs₁ * μⱼ

    # update moments
    μⱼ, μⱼ₋₁ = μⱼ * d + 2.0, μⱼ
    
    for j in 2:n₂-1
      # calculate unnormalized coefficients
      cs₂ = dct2(f2, cc_rule[2], j, n₂)
      cs₁ = dct1(f1, cc_rule[1], j, n₂-1) + cs₂

      I₂ += 2.0/(n₂-1) * cs₂ * μⱼ
      I₁ += 2.0/(n₁-1) * cs₁ * μⱼ

      # update moments
      cst = isodd(j) ? T(4)/T(1 - (j-1)^2) : 0.0
      μⱼ, μⱼ₋₁ = 2d * μⱼ - μⱼ₋₁ + cst, μⱼ
    end

    cs₂ = dct2(f2, cc_rule[2], n₂, n₂)
    I₂ += 1.0/(n₂-1) * cs₂ * μⱼ

    for j in n₂:n₁-1
      cs₁ = dct2(f2, cc_rule[2], j, n₂) + dct1(f1, cc_rule[1], j, n₂-1)
      I₁ += 2.0/(n₁-1) * cs₁ * μⱼ

      # update moments
      cst = isodd(j) ? T(4)/T(1 - (j-1)^2) : 0.0
      μⱼ, μⱼ₋₁ = 2d * μⱼ - μⱼ₋₁ + cst, μⱼ
    end

    cs₁ = dct2(f2, cc_rule[2], n₁, n₂) + dct1(f1, cc_rule[1], n₁, n₂-1)
    I₁ += 1.0/(n₁-1) * cs₁ * μⱼ

    return CauchySegment(fn, c, oftype(d, a), oftype(d, b), I₁, abs(I₁ - I₂))
  end
end

cachedpoints(::Type{T}, n::Integer) where T<:Number = n == 1 ? error() : float(T)[π * k / (n-1) for k=0:n-1]

struct CauchySegment{TF,TX,TI,TE}
  f::TF
  c::TX
  a::TX
  b::TX
  I::TI
  E::TE
end
Base.isless(i::CauchySegment, j::CauchySegment) = isless(i.E, j.E)

@generated function unroll_poles(z, seq::NTuple{N,T}) where {N,T} 
  expand(i) = i == 0 ? :1 : :(*(z - seq[$i], $(expand(i-1))))
  return expand(N)
end

@inline init(t::Tuple) = _init(t...)
_init() = throw(ArgumentError("Cannot call init on an empty tuple"))
_init(v) = ()
@inline _init(v, t...) = (v, _init(t...)...)

"""
  cauchy(f, a,c1,...,b; atol=sqrt(eps), rtol=0, maxevals=10^7, order_gk=7, order_cc=25, norm=norm)

Numerically computes the Cauchy principal value integral of the function `f(x)/ Π_i (x - cᵢ)`
from `a` to `b` for simple poles located at `c1`, `c2` and so on.

Returns a pair `(I,E)` of the estimated integral `I` and an estimated upper bound on the
absolute error `E`. If `maxevals` is not exceeded then `E <= max(atol, rtol*norm(I))`
will hold.
"""
cauchy(f, a, bs...; kws...) = cauchy(f, promote(a,bs...)..., kws...)

function cauchy(f, a::T, bs::Vararg{T,N};
                atol=nothing, rtol=nothing, maxevals=10^7, order_gk=7, order_cc=25, norm=norm) where {T,N}
  
  cs, b = init(bs), last(bs)

  if cs != tuple(unique(cs)...)
    error("can only integrate simple poles")
  end

  cs = tuple(sort([cs...])...)

  if cs[1] ≤ a || cs[end] ≥ b
    error("singularities must lie inside the integration boundaries")
  end

  if iseven(order_cc)
    error("order_cc must be an odd number")
  elseif order_cc == 1
    error("order_cc must be > 2")
  end
  
  # Create segments between each pole
  segs = tuple(a, ntuple(i -> 0.5 * (cs[i] + cs[i+1]), Val{N-2}())..., b)

  # Generate functions without the simple pole of each segment
  if isone(length(cs))
    fs = tuple(f)
  else
    zs = ntuple(i -> tuple(cs[1:i-1]..., cs[i+1:end]...), Val{N-1}())
    fs = ntuple(i -> (z -> f(z) / unroll_poles(z, zs[i])), Val{N-1}())
  end
  do_cauchy(fs, convert.(float(T), segs), convert.(float(T), cs), order_gk, order_cc, atol, rtol, maxevals, norm)
end
