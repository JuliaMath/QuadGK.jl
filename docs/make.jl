using Documenter, QuadGK

makedocs(
    format = :html,
    sitename = "QuadGK.jl",
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(
    repo = "github.com/ararslan/QuadGK.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
)
