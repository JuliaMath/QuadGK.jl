using Documenter, QuadGK

makedocs(
    modules = [QuadGK],
    clean = false,
    sitename = "QuadGK.jl",
    authors = "Steven G. Johnson and contributors.",
    pages = [
        "Home" => "index.md",
        "Examples" => "quadgk-examples.md",
        "Quadrature rules" => "gauss-kronrod.md",
        "Weighted quadrature" => "weighted-gauss.md",
        "Reference" => "functions.md",
    ],
)

deploydocs(
    repo = "github.com/JuliaMath/QuadGK.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
)
