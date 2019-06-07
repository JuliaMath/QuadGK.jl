# integration segment (a,b), estimated integral I, and estimated error E
struct Segment{TX,TI,TE}
    a::TX
    b::TX
    I::TI
    E::TE
end
Base.@pure Base.promote_rule(::Type{Segment{TX,TI,TE}}, ::Type{Segment{TX′,TI′,TE′}}) where {TX,TI,TE,TX′,TI′,TE′} =
    Segment{promote_type(TX,TX′), promote_type(TI,TI′), promote_type(TE,TE′)}
Base.convert(::Type{T}, s::Segment) where {T<:Segment} = T(s.a,s.b,s.I,s.E)
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
    Ik_s, Ig_s = Ik * s, Ig * s # new variable since this may change the type
    E = nrm(Ik_s - Ig_s)
    if isnan(E) || isinf(E)
        throw(DomainError(a+s, "integrand produced $E in the interval ($a, $b)"))
    end
    return Segment(oftype(s, a), oftype(s, b), Ik_s, E)
end
