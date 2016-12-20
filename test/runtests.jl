# This file contains code that was formerly part of Julia. License is MIT: http://julialang.org/license

using QuadGK
using Base.Test

@testset "quadgk" begin
    @test QuadGK.quadgk(cos, 0,0.7,1)[1] ≈ sin(1)
    @test QuadGK.quadgk(x -> exp(im*x), 0,0.7,1)[1] ≈ (exp(1im)-1)/im
    @test QuadGK.quadgk(x -> exp(im*x), 0,1im)[1] ≈ -1im*expm1(-1)
    @test_approx_eq_eps QuadGK.quadgk(cos, 0,BigFloat(1),order=40)[1] sin(BigFloat(1)) 1000*eps(BigFloat)
    @test QuadGK.quadgk(x -> exp(-x), 0,0.7,Inf)[1] ≈ 1.0
    @test QuadGK.quadgk(x -> exp(x), -Inf,0)[1] ≈ 1.0
    @test QuadGK.quadgk(x -> exp(-x^2), -Inf,Inf)[1] ≈ sqrt(pi)
    @test QuadGK.quadgk(x -> [exp(-x), exp(-2x)], 0, Inf)[1] ≈ [1,0.5]
    @test QuadGK.quadgk(cos, 0,0.7,1, norm=abs)[1] ≈ sin(1)
end
