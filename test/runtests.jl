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
end