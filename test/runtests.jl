# This file contains code that was formerly part of Julia. License is MIT: http://julialang.org/license

using QuadGK, Test

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
            )
            x,w,gw = kronrod(BigFloat, n)
            @test x ≈ x0 atol=1e-49
            @test w ≈ w0 atol=1e-49
            @test gw ≈ gw0 atol=1e-49
        end
    end
end

≅(x::Tuple, y::Tuple) = all(a -> isapprox(a[1],a[2]), zip(x,y))

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
