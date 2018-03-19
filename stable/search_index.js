var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Home",
    "title": "Home",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#QuadGK.jl-1",
    "page": "Home",
    "title": "QuadGK.jl",
    "category": "section",
    "text": "This package provides support for one-dimensional numerical integration in Julia using adaptive Gauss-Kronrod quadrature. The code was originally part of Base Julia."
},

{
    "location": "index.html#QuadGK.quadgk",
    "page": "Home",
    "title": "QuadGK.quadgk",
    "category": "function",
    "text": "quadgk(f, a,b,c...; reltol=sqrt(eps), abstol=0, maxevals=10^7, order=7, norm=vecnorm)\n\nNumerically integrate the function f(x) from a to b, and optionally over additional intervals b to c and so on. Keyword options include a relative error tolerance reltol (defaults to sqrt(eps) in the precision of the endpoints), an absolute error tolerance abstol (defaults to 0), a maximum number of function evaluations maxevals (defaults to 10^7), and the order of the integration rule (defaults to 7).\n\nReturns a pair (I,E) of the estimated integral I and an estimated upper bound on the absolute error E. If maxevals is not exceeded then E <= max(abstol, reltol*norm(I)) will hold. (Note that it is useful to specify a positive abstol in cases where norm(I) may be zero.)\n\nThe endpoints a et cetera can also be complex (in which case the integral is performed over straight-line segments in the complex plane). If the endpoints are BigFloat, then the integration will be performed in BigFloat precision as well.\n\nnote: Note\nIt is advisable to increase the integration order in rough proportion to the precision, for smooth integrands.\n\nMore generally, the precision is set by the precision of the integration endpoints (promoted to floating-point types).\n\nThe integrand f(x) can return any numeric scalar, vector, or matrix type, or in fact any type supporting +, -, multiplication by real values, and a norm (i.e., any normed vector space). Alternatively, a different norm can be specified by passing a norm-like function as the norm keyword argument (which defaults to vecnorm).\n\nnote: Note\nOnly one-dimensional integrals are provided by this function. For multi-dimensional integration (cubature), there are many different algorithms (often much better than simple nested 1d integrals) and the optimal choice tends to be very problem-dependent. See the Julia external-package listing for available algorithms for multidimensional integration or other specialized tasks (such as integrals of highly oscillatory or singular functions).\n\nThe algorithm is an adaptive Gauss-Kronrod integration technique: the integral in each interval is estimated using a Kronrod rule (2*order+1 points) and the error is estimated using an embedded Gauss rule (order points). The interval with the largest error is then subdivided into two intervals and the process is repeated until the desired error tolerance is achieved.\n\nThese quadrature rules work best for smooth functions within each interval, so if your function has a known discontinuity or other singularity, it is best to subdivide your interval to put the singularity at an endpoint. For example, if f has a discontinuity at x=0.7 and you want to integrate from 0 to 1, you should use quadgk(f, 0,0.7,1) to subdivide the interval at the point of discontinuity. The integrand is never evaluated exactly at the endpoints of the intervals, so it is possible to integrate functions that diverge at the endpoints as long as the singularity is integrable (for example, a log(x) or 1/sqrt(x) singularity).\n\nFor real-valued endpoints, the starting and/or ending points may be infinite. (A coordinate transformation is performed internally to map the infinite interval to a finite one.)\n\n\n\n\n\n"
},

{
    "location": "index.html#QuadGK.gauss",
    "page": "Home",
    "title": "QuadGK.gauss",
    "category": "function",
    "text": "gauss([T,] N)\n\nReturn a pair (x, w) of N quadrature points x[i] and weights w[i] to integrate functions on the interval (-1, 1),  i.e. sum(w .* f.(x)) approximates the integral.  Uses the method described in Trefethen & Bau, Numerical Linear Algebra, to find the N-point Gaussian quadrature in O(N²) operations.\n\nT is an optional parameter specifying the floating-point type, defaulting to Float64. Arbitrary precision (BigFloat) is also supported.\n\n\n\n\n\n"
},

{
    "location": "index.html#QuadGK.kronrod",
    "page": "Home",
    "title": "QuadGK.kronrod",
    "category": "function",
    "text": "kronrod([T,] n)\n\nCompute 2n+1 Kronrod points x and weights w based on the description in Laurie (1997), appendix A, simplified for a=0, for integrating on [-1,1]. Since the rule is symmetric, this only returns the n+1 points with x <= 0. The function Also computes the embedded n-point Gauss quadrature weights gw (again for x <= 0), corresponding to the points x[2:2:end]. Returns (x,w,wg) in O(n²) operations.\n\nT is an optional parameter specifying the floating-point type, defaulting to Float64. Arbitrary precision (BigFloat) is also supported.\n\nGiven these points and weights, the estimated integral I and error E can be computed for an integrand f(x) as follows:\n\nx, w, wg = kronrod(n)\nfx⁰ = f(x[end])                # f(0)\nx⁻ = x[1:end-1]                # the x < 0 Kronrod points\nfx = f.(x⁻) .+ f.((-).(x⁻))    # f(x < 0) + f(x > 0)\nI = sum(fx .* w[1:end-1]) + fx⁰ * w[end]\nif isodd(n)\n    E = abs(sum(fx[2:2:end] .* wg[1:end-1]) + fx⁰*wg[end] - I)\nelse\n    E = abs(sum(fx[2:2:end] .* wg[1:end])- I)\nend\n\n\n\n\n\n"
},

{
    "location": "index.html#Functions-1",
    "page": "Home",
    "title": "Functions",
    "category": "section",
    "text": "QuadGK.quadgk\nQuadGK.gauss\nQuadGK.kronrod"
},

]}
