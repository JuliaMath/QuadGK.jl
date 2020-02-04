# Internal routine: integrate f over the interval (s[1], s[2]) using 
# h-adaptive integration
function do_cauchy(segs::NTuple{N,T}, n_gk, n_cc, atol, rtol, maxevals, nrm) where {T,N}
  gk_rule = cachedrule(eltype(atol), n_gk)
  cc_rule = clenshawcurtisnodes(eltype(atol), n_cc)

  segs = ntuple(i -> evalrule_cauchy(segs[i].f, segs[i].a, segs[i].b, segs[i].c, gk_rule, cc_rule, nrm), Val{N}())
  I = sum(s -> s.I, segs)
  E = sum(s -> s.E, segs)
  numevals = (2n_cc+1) * N # Because it will definitely be a Clenshaw-Curtis evaluation

  # logic here is mainly to handle dimensionful quantities: we
  # don't know the correct type of atol, in particular, until
  # this point where we have the type of E from f.  Also, follow
  # Base.isapprox in that if atol≠0 is supplied by the user, rtol
  # defaults to zero.
  atol_ = something(atol, E)
  rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(one(eltype(gk_rule[1])))) : zero(eltype(gk_rule[1])))

  if E ≤ atol_ || E ≤ rtol_ * nrm(I) || numevals ≥ maxevals
    return (I, E)
  end
  return adapt_cauchy(heapify!(collect(segs), Reverse), I, E, numevals, n_gk, gk_rule, cc_rule, atol_, rtol_, maxevals, nrm)
end

function adapt_cauchy(seg::T, I, E, numevals, n_gk, gk_rule, cc_rule, atol, rtol, maxevals, nrm) where {T}
  # Pop the biggest-error segment and subdivide (h-adaptation)
  # until convergence is achieved or maxevals is exceeded.
  while E > atol && E > rtol * nrm(I) && numevals < maxevals
    s = heappop!(seg, Reverse)
    
    a1 = s.a
    b1 = 0.5 * (s.a + s.b)
    a2 = b1
    b2 = s.b
    
    if s.c > a1 && s.c <= b1
      b1 = 0.5 * (s.c + b2)
      a2 = b1
    elseif (s.c > b1 && s.c < b2)
      b1 = 0.5 * (a1 + s.c)
      a2 = b1
    end
    
    s1 = evalrule_cauchy(s.f, a1, b1, s.c, gk_rule,cc_rule, nrm)
    s2 = evalrule_cauchy(s.f, a2, b2, s.c, gk_rule,cc_rule, nrm)
    numevals += 4*length(cc_rule)+2

    I = (I - s.I) + s1.I + s2.I
    E = (E - s.E) + s1.E + s2.E
    
    # handle type-unstable functions by converting to a wider type if needed
    Tj = promote_type(typeof(s1), promote_type(typeof(s2), T))
    if Tj !== T
      return adapt_cauchy(heappush!(heappush!(Vector{Tj}(seg), s1, Reverse), s2, Reverse),
                  I, E, numevals, n_gk, gk_rule, cc_rule, atol, rtol, maxevals, nrm)
    end
    
    heappush!(seg, s1, Reverse)
    heappush!(seg, s2, Reverse)
  end
  
  # re-sum (paranoia about accumulated roundoff)
  I = seg[1].I
  E = seg[1].E
  for i in 2:length(seg)
      I += seg[i].I
      E += seg[i].E
  end
  return (I, E)
end

# When close to the singularity c, use a special modified Clenshaw-Curtis rule
# otherwise, stick with Gauss-Kronrod
function evalrule_cauchy(f, a, b, c, rk_rule, cc_rule, nrm)

  # Determine how close we are to the singularity
  d = (2 * c - b - a) / (b - a)

  # Use Gauss-Kronrod
  if abs(d) > 1.1
    seg = evalrule(x -> f(x) / (x - c), a, b, rk_rule..., nrm)
    I, E = seg.I, seg.E

  # Use modified Clenshaw-Curtis
  else
    f_nodes = f.(b .+ (1 .- cc_rule) .* (a-b)/2)

    cheb = clenshawcurtisweights(f_nodes)
    cheb₂ = clenshawcurtisweights(f_nodes[1:2:end])

    μ = compute_moments(d, length(cc_rule))

    I₂ = cheb₂' * μ[1:length(cheb₂)]
    I = cheb' * μ
    E = abs(I - I₂)
  end
  return CauchySegment(f, c, oftype(d, a), oftype(d, b), I, E)
end

function compute_moments(cc::T, n::Int) where T
  μ = zeros(T, n)
  n > 0 && (μ[1] = log(abs((1.0 - cc) / (1.0 + cc))))
  if n > 1
    μ[2] = μ[1] * cc + 2
    for i=2:n
      cst = isodd(i) ? T(4)/T(1 - (i-1)^2) : 0.0
      @inbounds μ[i+1] = 2cc * μ[i] - μ[i-1] + cst
    end
  end
  μ
end

struct CauchySegment{TX,TI,TE}
  f::Any
  c::TX
  a::TX
  b::TX
  I::TI
  E::TE
end
Base.isless(i::CauchySegment, j::CauchySegment) = isless(i.E, j.E)

function gen_funcs(f::Function, cs::NTuple{N,T}) where {N,T}
  if N == 1
    return (f, )
  end

  funcs = tuple()
  for c in cs
    non_singularities = tuple(setdiff(cs, c)...)
    ex = :(*())
    for ns in non_singularities
      push!(ex.args, :(x - $ns))
    end
    funcs = tuple(funcs..., mk_function(:(x -> $f(x) / $ex)))
  end

  return funcs
end


"""
This function computes the Cauchy principal value of the integral of f over 
(a,b), with a singularity at c
"""
cauchy(f, a, b, cs...; kws...) = cauchy(f, promote(a,b,cs...)..., kws...)

function cauchy(f, a::T, b::T, cs::Vararg{T,N};
                atol=sqrt(eps(T)), rtol=nothing, maxevals=10^7, order_gk=7, order_cc=25, norm=norm) where {T,N}
  
  if cs != tuple(unique(cs)...)
    error("can only integrate simple poles")
  end

  cs = tuple(sort([cs...])...)

  if cs[1] ≤ a || cs[end] ≥ b
    error("singularities must lie inside the integration boundaries")
  end

  if iseven(order_cc)
    error("order_cc must be an odd number")
  end
  
  # Create segments with singularities in between
  segs = tuple(a, ntuple(i -> 0.5 * (cs[i] + cs[i+1]), Val{N-1}())..., b)
  segs = ntuple(i -> (segs[i], segs[i+1]), Val{N}())

  funs = gen_funcs(f, cs)

  segs = (CauchySegment(f, c, s[1], s[2], 0.0, 0.0) for (f, c, s) in zip(funs, cs, segs))

  do_cauchy(tuple(segs...), order_gk, order_cc, atol, rtol, maxevals, norm)
end