# Internal routine: integrate f over the interval (s[1], s[2]) using 
# h-adaptive integration
function do_cauchy(f::F, s::NTuple{2,T}, c::T, n_gk, n_cc, atol, rtol, maxevals, nrm) where {T,F}
  gk_rule = cachedrule(eltype(s), n_gk)
  cc_rule = clenshawcurtisnodes(eltype(s), n_cc)

  seg = evalrule_cauchy(f, s[1], s[2], c, gk_rule, cc_rule, nrm)
  I = seg.I
  E = seg.E
  numevals = (2n_gk+1)

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
  return adapt_cauchy(f, heapify!([seg], Reverse), c, I, E, numevals, n_gk, gk_rule, cc_rule, atol_, rtol_, maxevals, nrm)
end

function adapt_cauchy(f, segs::T, c, I, E, numevals, n_gk, gk_rule, cc_rule, atol, rtol, maxevals, nrm) where {T}
    # Pop the biggest-error segment and subdivide (h-adaptation)
    # until convergence is achieved or maxevals is exceeded.
    while E > atol && E > rtol * nrm(I) && numevals < maxevals
        s = heappop!(segs, Reverse)
        
        a1 = s.a
        b1 = 0.5*(s.a + s.b)
        a2 = b1
        b2 = s.b
        
        if c > a1 && c <= b1
          b1 = 0.5 * (c + b2) ;
          a2 = b1;
        elseif (c > b1 && c < b2)
          b1 = 0.5 * (a1 + c) ;
          a2 = b1;
        end
        
        s1 = evalrule_cauchy(f, a1, b1, c, gk_rule,cc_rule, nrm)
        s2 = evalrule_cauchy(f, a2, b2, c, gk_rule,cc_rule, nrm)

        I = (I - s.I) + s1.I + s2.I
        E = (E - s.E) + s1.E + s2.E
        numevals += 4n_gk+2
        # handle type-unstable functions by converting to a wider type if needed
        Tj = promote_type(typeof(s1), promote_type(typeof(s2), T))
        if Tj !== T
            return adapt_cauchy(f, heappush!(heappush!(Vector{Tj}(segs), s1, Reverse), s2, Reverse),
                         c, I, E, numevals, n_gk, gk_rule, cc_rule, atol, rtol, maxevals, nrm)
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

# When close to the singularity c, use a special modified Clenshaw-Curtis rule
# otherwise, stick with Gauss-Kronrod
function evalrule_cauchy(f, a, b, c, rk_rule, cc_rule, nrm)

  # Determine how close we are to the singularity
  d = (2 * c - b - a) / (b - a)

  # Use Gauss-Kronrod
  if abs(d) > 1.1
    return evalrule(x -> f(x) / (x - c), a, b, rk_rule..., nrm)

  # Use modified Clenshaw-Curtis
  else
    f_nodes = f.(b .+ (1 .- cc_rule) .* (a-b)/2)

    cheb = clenshawcurtisweights(f_nodes)
    cheb₂ = clenshawcurtisweights(f_nodes[1:2:end])

    μ = compute_moments(d, length(cc_rule))

    I = cheb' * μ
    I₂ = cheb₂' * μ[1:length(cheb₂)]
    
    E = abs(I - I₂)
    return Segment(oftype(d, a), oftype(d, b), I, E)
  end
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


"""
This function computes the Cauchy principal value of the integral of f over 
(a,b), with a singularity at c
"""
cauchy(f, a, c, b; kws...) = cauchy(f, promote(a,c,b)..., kws...)

function cauchy(f, a::T, c::T, b::T;
                atol=eps(T), rtol=nothing, maxevals=10^7, order_gk=7, order_cc=25, norm=norm) where {T}

  if c == a || c == b
    error("cannot integrate with singularity on endpoint")
  end

  if iseven(order_cc)
      error("order_cc must be an odd number")
  end
  
  handle_infinities(f, (a, b)) do f, s, _
    if >(s...)
      s = reverse(s)
      f = x -> -f(x)
    end
    
    do_cauchy(f, s, c, order_gk, order_cc, atol, rtol, maxevals, norm)
  end
end
