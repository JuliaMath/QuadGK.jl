# This file contains code that was formerly part of Julia. License is MIT: http://julialang.org/license

using QuadGK, LinearAlgebra, Test

≅(x::Tuple, y::Tuple; kws...) = all(a -> isapprox(a[1],a[2]; kws...), zip(x,y))

@testset "quadgk" begin
    @test quadgk(cos, 0,0.7,1)[1] ≈ sin(1)
    @test quadgk(x -> exp(im*x), 0,0.7,1)[1] ≈ (exp(1im)-1)/im
    @test quadgk(x -> exp(im*x), 0,1im)[1] ≈ -1im*expm1(-1)
    @test isapprox(quadgk(cos, 0,BigFloat(1),order=40)[1], sin(BigFloat(1)),
                   atol=1000*eps(BigFloat))
    @test quadgk(x -> exp(-x), 0,0.7,Inf)[1] ≈ 1.0
    @test quadgk(x -> exp(x), -Inf,0)[1] ≈ 1.0
    @test quadgk(x -> exp(-x^2), -Inf,Inf)[1] ≈ sqrt(pi)
    @test quadgk(x -> [exp(-x), exp(-2x)], 0, Inf)[1] ≈ [1,0.5]
    @test quadgk(cos, 0,0.7,1, norm=abs)[1] ≈ sin(1)

    # Test a function that is only implemented for Float32 values
    cos32(x::Float32) = cos(20x)
    @test quadgk(cos32, 0f0, 1f0)[1]::Float32 ≈ sin(20f0)/20

    # test integration of a type-unstable function where the instability is only detected
    # during refinement of the integration interval:
    @test quadgk(x -> x > 0.01 ? sin(10(x-0.01)) : 1im, 0,1.01, rtol=1e-4, order=3)[1] ≈ (1 - cos(10))/10+0.01im rtol=1e-4

    # order=1 (issue #66)
    @test quadgk_count(x -> 1, 0, 1, order=1) == (1.0, 0.0, 3)

    # empty or nearly empty intervals (issue #97)
    @test quadgk(x -> 1, 0,0) == quadgk(x -> 1, 0,0,0) == (0.0,0.0)
    @test quadgk(x -> 1, 1,nextfloat(1.0)) == (eps(),0.0)
end

module Test19626
    using QuadGK
    using Test

    # Define a mock physical quantity type
    struct MockQuantity <: Number
        val::Float64
    end

    # Following definitions needed for quadgk to work with MockQuantity
    import Base: +, -, *, abs, isnan, isinf, isless, float
    +(a::MockQuantity, b::MockQuantity) = MockQuantity(a.val+b.val)
    -(a::MockQuantity, b::MockQuantity) = MockQuantity(a.val-b.val)
    *(a::MockQuantity, b::Number) = MockQuantity(a.val*b)
    abs(a::MockQuantity) = MockQuantity(abs(a.val))
    float(a::MockQuantity) = a
    isnan(a::MockQuantity) = isnan(a.val)
    isinf(a::MockQuantity) = isinf(a.val)
    isless(a::MockQuantity, b::MockQuantity) = isless(a.val, b.val)

    # isapprox only needed for test purposes
    Base.isapprox(a::MockQuantity, b::MockQuantity) = isapprox(a.val, b.val)

    # Test physical quantity-valued functions
    @test QuadGK.quadgk(x->MockQuantity(x), 0.0, 1.0, atol=MockQuantity(0.0))[1] ≈
        MockQuantity(0.5)
end

@testset "inference" begin
    @test @inferred(QuadGK.cachedrule(Float16, 3)) == (Float16[-0.96, -0.7744, -0.434, 0.0], Float16[0.1047, 0.269, 0.4014, 0.4504], Float16[0.555, 0.8896])
    @test @inferred(QuadGK.cachedrule(Complex{BigFloat}, 3)) isa NTuple{3,Vector{BigFloat}}
    @test @inferred(quadgk(x -> exp(-x^2), 0, Inf, rtol=1e-8)) isa Tuple{Float64,Float64}
    @test @inferred(quadgk(x -> exp(-x^2), 0, 1, rtol=1e-8)) isa Tuple{Float64,Float64}
    @test @inferred(quadgk(x -> 1, 0, 1im)) === (1.0im, 0.0)
    @test @inferred(quadgk(x -> sin(10x), 0,1))[1] ≈ (1 - cos(10))/10
end

@testset "gauss" begin
    x,w = gauss(10, -1, 1)
    x′,w′ = @inferred gauss(x->1, 10, -1, 1)
    @test x ≈ x′ ≈ [-0.9739065285171717, -0.8650633666889845, -0.6794095682990244, -0.4333953941292472, -0.14887433898163124, 0.14887433898163124, 0.4333953941292472, 0.6794095682990244, 0.8650633666889845, 0.9739065285171717]
    @test w ≈ w′ ≈ [0.06667134430868811, 0.14945134915058064, 0.2190863625159821, 0.26926671930999635, 0.29552422471475276, 0.29552422471475276, 0.26926671930999635, 0.2190863625159821, 0.14945134915058064, 0.06667134430868811]

    # Gauss–Hermite quadrature:
    # we don't support infinite intervals, but any sufficiently broad interval
    # for a decaying weight function should be equivalent to infinite (to any desired precision).
    xH,wH = @inferred gauss(x->exp(-x^2), 5, -10, 10, rtol=1e-10)
    @test sum(xH.^4 .* wH) ≈ quadgk(x -> x^4 * exp(-x^2), -Inf, Inf)[1] ≈ 3sqrt(π)/4
    @test xH ≈ [-2.020182870456085632929,  -0.9585724646138185071128, 0, 0.9585724646138185071128, 2.020182870456085632929]
    @test wH ≈ [0.01995324205904591320774, 0.3936193231522411598285, 0.9453087204829418812257, 0.393619323152241159829, 0.01995324205904591320774]

    x,w = gauss(200, 2, 3)
    x′,w′ = gauss(x->1, 200, 2, 3, rtol=1e-11)
    @test x ≈ x′
    @test w ≈ w′

    xf,wf = gauss(20, 2.0f0, 3.0f0)
    xf′,wf′ = gauss(x->1, 20, 2.0f0, 3.0f0, rtol=1e-5)
    @test eltype(xf′) == eltype(wf′) == eltype(xf) == eltype(wf) == Float32
    @test xf ≈ xf′
    @test wf ≈ wf′

    # check for underflow bug: https://discourse.julialang.org/t/nan-returned-by-gauss/68260
    x,w = gauss(1093)
    @test all(isfinite, x) && all(isfinite, w)
end

# check some arbitrary precision (Maple) values from https://keisan.casio.com/exec/system/1289382036
@testset "kronrod" begin
    setprecision(200) do
        for (n,x0,w0,gw0) in (
            (1, [big"-0.77459666924148337703585307995647992216658434105832", big"0.0"],
                [big"0.55555555555555555555555555555555555555555555555556", big"0.88888888888888888888888888888888888888888888888889"],
                [big"2.0"]),
            (2, [big"-0.92582009977255146156656677658399952252931490100834", big"-0.57735026918962576450914878050195745564760175127013",  big"0.0"],
                [big"0.1979797979797979797979797979797979797979797979798", big"0.49090909090909090909090909090909090909090909090909", big"0.6222222222222222222222222222222222222222222222222"],
                [big"1.0"]),
            (3, [big"-0.96049126870802028342350709262907996266978223636529", big"-0.77459666924148337703585307995647992216658434105832", big"-0.43424374934680255800207150284462781728289855695503", big"0.0"],
                [big"0.1046562260264672651938238571920730382422021606251", big"0.26848808986833344072856928066670962476104560001718", big"0.4013974147759622229050518186184318787274230022865", big"0.45091653865847414234511008704557091653865847414235"],
                [big"0.55555555555555555555555555555555555555555555555556", big"0.88888888888888888888888888888888888888888888888889"]),
            (4, [big"-0.97656025073757311153450535936991962683375905330236", big"-0.86113631159405257522394648889280950509572537962972", big"-0.64028621749630998240468902315749201835600980018698", big"-0.33998104358485626480266575910324468720057586977091", big"0.0"],
                [big"0.06297737366547301476549248855281867632941703873821", big"0.1700536053357227268027388532962065872619470485872", big"0.266798340452284448032770628417855662476650377333", big"0.32694918960145162955845946561731918577437380049408", big"0.34644298189013636168107712823159977631522346969501"],
                [big"0.3478548451374538573730639492219994072353486958339", big"0.65214515486254614262693605077800059276465130416611"]),
            (5,[big"-0.98408536009484246449617293463613949958055282418847", big"-0.90617984593866399279762687829939296512565191076253", big"-0.75416672657084922044081716694611586638629980437148", big"-0.53846931010568309103631442070020880496728660690556", big"-0.27963041316178319341346652274897743624211881535617", big"0.0"],
                [big"0.04258203675108183286450945084767009187528571052993", big"0.11523331662247339402462684588057353916959629218019", big"0.1868007965564926574678000268784859712873998237471", big"0.24104033922864758669994261122326211129607983509941", big"0.27284980191255892234099326448445551826261012746316", big"0.28298741785749121320425560137110553621805642196044"],
                [big"0.23692688505618908751426404071991736264326000221241", big"0.47862867049936646804129151483563819291229555334314", big"0.56888888888888888888888888888888888888888888888889"]),
            ), which in (1,2,3)
            if which < 3
                m = div(3n+3,2)
                b = [ k / sqrt(4k^2 - big"1.0") for k = 1:m-1 ]
                if which == 1
                    # test generic Kronrod algorithm that doesn't
                    # assume a Jacobi matrix with zero diagonals
                    J = SymTridiagonal(zeros(BigFloat, m), b)
                    x,w,gw = kronrod(Matrix(J), n, 2)
                    @test kronrod(Matrix(J), n, 2, (-1,1)=>(-1,1)) ≅ (x,w,gw) atol=1e-55
                    @test kronrod(QuadGK.HollowSymTridiagonal(b), n, 2, (-1,1)=>(-1,1)) ≅ (x,w,gw) atol=1e-55
                    @test kronrod(n, big"-1.", big"1.") ≅ (x,w,gw) atol=1e-55
                    # check symmetric rule & remove redundant points/weights
                    @test x[1:n] ≈ -reverse(x[n+2:end]) atol=1e-55
                    @test w[1:n] ≈ reverse(w[n+2:end]) atol=1e-55
                    resize!(x, n+1)
                    resize!(w, n+1)
                    resize!(gw, length(2:2:n+1))
                else
                    # test generic HollowSymTridiagonal method
                    J = QuadGK.HollowSymTridiagonal(b)
                    x,w,gw = kronrod(J, n, 2)
                end
            else
                x,w,gw = kronrod(BigFloat, n)
            end
            @test (x,w) ≅ (x0,w0) atol=1e-49
            @test gw ≈ gw0 atol=1e-49

            xg, wg = gauss(n, big"-1", big"+1")
            nn = length(2:2:n+1)
            nn0 = length(2:2:n)
            @test wg[1:nn0] ≈ reverse(wg[nn+1:end]) atol=1e-55
            @test wg[1:nn] ≈ gw0 atol=1e-49
        end
    end

    # x -> 1 weight function:
    let (x, w, wg) = kronrod(x -> 1, 7, -1, 1)
        x0, w0, wg0 = kronrod(7)
        @test x ≈ [x0; -reverse(x0[1:end-1])]
        @test w ≈ [w0; reverse(w0[1:end-1])]
        @test wg ≈ [wg0; reverse(wg0[1:end-1])]
    end

    # non-symmetric Gauss–Kronrod rule for an arbitrary weight function
    let (x, w, wg) = kronrod(x -> 1+x^2, 7, 0, 1)
        # integral of 1 should be 4/3:
        @test sum(w) ≈ 4/3
        @test sum(wg) ≈ 4/3
        # should hold for well-behaved weights:
        @test all(>(0), w)
        @test all(>(0), wg)
        @test all(x -> 0 ≤ x ≤ 1, x)

        xg, wg2 = gauss(x -> 1+x^2, 7, 0, 1)
        @test xg ≈ x[2:2:end]
        @test wg2 ≈ wg

        # test against results of Laurie implementatation by Gautschi
        # in Matlab (from the OPQ suite https://www.cs.purdue.edu/archives/2002/wxg/codes/OPQ.html),
        # for the same Jacobi matrix:
        x0 = [0.00438238617866954, 0.02614951914104513, 0.06952432823447702, 0.1331034150629803, 0.2132620595793294, 0.306005657934478, 0.4072933420568336, 0.5125123454997229, 0.6164666827040001, 0.7142807062929434, 0.8021670261548075, 0.8770965454790314, 0.9359980967418293, 0.9759635237163186, 0.9959709148947024]
        w0 = [0.01177172100204654, 0.03248033795172398, 0.05424896760483217, 0.07382275351146311, 0.0910708306136658, 0.1068683178722224, 0.1213569758811105, 0.1331660440143274, 0.1402336045988007, 0.1410339077741439, 0.1345976965140659, 0.1193309881617451, 0.09351598256107904, 0.05828394027207477, 0.02155126500003199]
        @test (x,w) ≅ (x0,w0) atol=1e-14
    end

    # Gauss–Jacobi quadrature for α=0.5, β=-0.2:
    let
        # from FastGaussQuadrature.jacobi_jacobimatrix(12, 0.5, -0.2):
        dv = [-0.30434782608695654, -0.021233569261880688, -0.007751937984496124, -0.004016064257028112, -0.0024564276523570006, -0.0016575893914278945, -0.0011939280231963157, -0.0009009395512462996, -0.000704012873378256, -0.0005652911249293386, -0.0004638936137312509, -0.00038753252505120965]
        ev = [0.524367555787415, 0.5060015044290427, 0.5027136949904828, 0.5015465849161829, 0.5009991342001523, 0.5006986460562385, 0.5005159945713471, 0.5003966898762826, 0.5003144745319204, 0.5002554183632608, 0.5002115697583008]
        J = SymTridiagonal(dv, ev)
        wint = 2.1775041171955682 # FastGaussQuadrature.jacobimoment(0.5, -0.2)
        # from FastGaussQuadrature.gaussjacobi(12, 0.5, -0.2):
        x0 = [-0.9864058663156023, -0.9165946746181465, -0.7905606636553969, -0.6160030510532921, -0.40362885238758006, -0.16646844386286386, 0.08092631663971639, 0.3233754136167141, 0.5460022311541176, 0.7351464193955097, 0.8792021083194245, 0.9693300504217204]
        w0 = [0.1332843914263806, 0.22507385576322209, 0.27777923749595884, 0.3009074312028019, 0.2983522399648924, 0.2741818901904956, 0.23357001605691405, 0.1827277942958943, 0.12846478443650275, 0.07758611879734549, 0.036243907354385235, 0.00933245021077474]

        x, w = gauss(Matrix(J), wint, (-1,1) => (0,1))
        @test (x .* 2 .- 1, w) ≅ (x0, w0) atol=2e-14

        x, w, wg = kronrod(J, 7)
        @test (x, w) ≅ gauss(QuadGK.kronrodjacobi(J, 7)) atol=1e-14
    end

    # check that non-symtridiagonal matrices throw error
    for A in ((1:3) * (1:3)', (1:3) * (4:7)', Tridiagonal(1:2, 3:5, 6:7))
        @test_throws ArgumentError gauss(A)
    end
end

@testset "HollowSymTridiagonal" begin
    H = QuadGK.HollowSymTridiagonal(1:4)
    T = SymTridiagonal(zeros(Int, 5), [1:4;])
    @test H isa QuadGK.HollowSymTridiagonal{Int}
    @test size(H) == (5,5)
    @test diag(H)::Vector{Int} == zeros(Int, 5)
    @test H == T
    @test QuadGK.HollowSymTridiagonal(T)::QuadGK.HollowSymTridiagonal{Int} == T
    @test QuadGK.HollowSymTridiagonal{Int8}(T)::QuadGK.HollowSymTridiagonal{Int8} == T
    @test_throws ArgumentError QuadGK.HollowSymTridiagonal(SymTridiagonal(1:4, 1:3))
    @test_throws ArgumentError QuadGK.HollowSymTridiagonal{Int8}(SymTridiagonal(1:4, 1:3))
    @test SymTridiagonal(H)::SymTridiagonal{Int, Vector{Int}} == T
    @test SymTridiagonal{Int8}(H)::SymTridiagonal{Int8, Vector{Int8}} == T
    @test Matrix(H)::Matrix{Int} == T == collect(H)
    @test Matrix{Int8}(H)::Matrix{Int8} == T
    @test replace(repr("text/plain", H), ",U"=>", U") == "5×5 QuadGK.HollowSymTridiagonal{$Int, UnitRange{$Int}}:\n ⋅  1  ⋅  ⋅  ⋅\n 1  ⋅  2  ⋅  ⋅\n ⋅  2  ⋅  3  ⋅\n ⋅  ⋅  3  ⋅  4\n ⋅  ⋅  ⋅  4  ⋅"

    λ = [-5.16351661076931, -1.8270457603216725, 0, 1.8270457603216728, 5.16351661076931]
    v1 = [0.03655465066022525, -0.18875054588494244, 0.469030964154226, -0.681449360868536, 0.5278955504450348]
    for A in (H, T)
        @test A*v1 ≈ λ[1]*v1
        @test eigvals(Hermitian(Matrix(A))) ≈ λ ≈ QuadGK.eignewt(A, 5)
        @test QuadGK.eigvec1!(zeros(5), A, λ[1]) ≈ v1
    end
    @test QuadGK.eigvec1(1:4, λ[1]) ≈ v1
end

@testset "inplace" begin
    I = [0., 0.]
    I′,E′ = quadgk!(I, 0, 1) do r,x
        r[1] = cos(100x)
        r[2] = sin(30x)
    end
    @test quadgk(x -> [cos(100x), sin(30x)], 0, 1) ≅ (I′,E′) ≅ ([-0.005063656411097513, 0.028191618337080532], 4.2100180879009775e-10)
    I″,E″= quadgk!(I, 0, 1.0) do r,x # check mixed-type argument promotion
        r[1] = cos(100x)
        r[2] = sin(30x)
    end
    @test (I″,E″) ≅ (I′,E′)
    @test I === I′ # result is written in-place to I

    # even orders
    @test quadgk(x -> [cos(100x), sin(30x)], 0, 1, order=8)[1] ≈ I′ ≈
          quadgk!((r,x) -> (r[1]=cos(100x); r[2]=sin(30x)), I, 0, 1, order=8)[1]

    # order=1 (issue #66)
    @test ([1.0], 0.0) == quadgk!((r,x) -> r[1] = 1.0, [0.], 0, 1, order=1)
end

@testset "inplace Inf" begin
    f!(v, x) = v .= exp(-x^2)
    @test quadgk!(f!, [0.], 1., Inf)[1] ≈ quadgk(x -> [exp(-x^2)], 1., Inf)[1]
    @test quadgk!(f!, [0.], -Inf, 1.)[1] ≈ quadgk(x -> [exp(-x^2)], -Inf, 1.)[1]
    @test quadgk!(f!, [0.], -Inf, Inf)[1] ≈ quadgk(x -> [exp(-x^2)], -Inf, Inf)[1]
end

# This is enough for allocation currently caused by the do-lambda in quadgk(...)
const smallallocbytes = 500

@testset "segbuf" begin
    # Should not need subdivision
    function id(x::Float64)::Float64
        1.0
    end
    # Should need subdivision
    function osc(x::Float64)::Float64
        (x - 0.3)^2 * sin(87(x + 0.07))
    end
    no_subdiv() = @timed quadgk(id, -1.0, 1.0)
    subdiv_alloc() = @timed quadgk(osc, -1.0, 1.0)
    segbuf = alloc_segbuf(size=1)
    subdiv_alloc_segbuf() = @timed quadgk(osc, -1.0, 1.0, segbuf=segbuf)
    no_subdiv() # warmup
    @test no_subdiv()[3] < smallallocbytes # [3] == .bytes starting in Julia 1.5
    subdiv_alloc() # warmup
    @test subdiv_alloc()[3] > smallallocbytes
    subdiv_alloc_segbuf() # warmup
    @test subdiv_alloc_segbuf()[3] < smallallocbytes
end

@testset "quadgk_count and quadgk_print" begin
    I, E, count = quadgk_count(x->cos(200x), 0,1)
    @test I ≈ -0.004366486486069923
    @test E ≈ 2.552995927726856e-13 atol=abs(I)*1e-8
    @test count == 1905
    @test sprint(io -> quadgk_print(io, x -> x^2, 0, 1, order=2), context=:compact=>true) ==
        "f(0.5) = 0.25\nf(0.211325) = 0.0446582\nf(0.788675) = 0.622008\nf(0.03709) = 0.00137566\nf(0.96291) = 0.927196\n"
end

@testset "batch" begin
    f(x) = min(abs(x), 1) # integrand requiring exactly three levels of refinement on [-2,2]
    gcnt = fill(0)
    f!(y, x) = (gcnt[] += 1; y .= f.(x))
    for order=1:7
        for max_batch in (4*order+2, 8*order+4)
            gcnt[] = 0
            g = QuadGK.BatchIntegrand{Float64}(f!, max_batch=max_batch)
            I′,E′ = quadgk(g, -2, 2, order=order)
            @test quadgk(f, -2, 2, order=order) == (I′,E′)
            @test gcnt[] == 2 + 1 + (max_batch < 8*order+4) # check if calls were batched
        end
    end

    # test constructors
    ref = BatchIntegrand(f!, Float64[], Nothing[], typemax(Int))
    for b in (
        BatchIntegrand(f!, Float64[]),
        BatchIntegrand(f!, Float64[], Nothing[]),
        BatchIntegrand{Float64}(f!),
        BatchIntegrand{Float64,Nothing}(f!),
    )
        for name in (:f!, :y, :x, :max_batch)
            @test getproperty(ref, name) == getproperty(b, name)
        end
    end
end

@testset "batch Inf" begin
    f!(v, x) = v .= exp.(-1 .* x .^ 2)
    g = QuadGK.BatchIntegrand{Float64}(f!)
    @test quadgk(g, 1., Inf)[1] ≈ quadgk(x -> exp(-x^2), 1., Inf)[1]
    @test quadgk(g, -Inf, 1.)[1] ≈ quadgk(x -> exp(-x^2), -Inf, 1.)[1]
    @test quadgk(g, -Inf, Inf)[1] ≈ quadgk(x -> exp(-x^2), -Inf, Inf)[1]
end

# issue 89: callable objects that are also arrays
struct Test89 <: AbstractVector{Float64}
end
(::Test89)(x::Real) = float(x)
@testset "issue 89" begin
    A = Test89()
    @test quadgk(A, 0, 10) == (50, 0)
end

# issue 86: nodes roundoff to endpoints
@testset "issue 86" begin
    I, = quadgk(y->10*y^9/(y^10)^1.1,1,Inf)[1]
    @test quadgk(x->1/x^1.1,1,Inf)[1] ≈ I rtol=0.05
    @test quadgk!((y,x)-> y .= 1/x^1.1,[0.0],1,Inf)[1][1] ≈ I rtol=0.05
    @test quadgk(BatchIntegrand{Float64}((y,x)-> @.(y = 1/x^1.1)),1,Inf)[1] ≈ I rtol=0.05
end
