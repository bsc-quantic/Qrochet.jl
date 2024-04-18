using Pkg
Pkg.activate(@__DIR__)
Pkg.develop(path = joinpath(@__DIR__, ".."))
Pkg.instantiate()

using Documenter
using Qrochet

DocMeta.setdocmeta!(Qrochet, :DocTestSetup, :(using Qrochet); recursive = true)

makedocs(
    modules = [Qrochet],
    sitename = "Qrochet.jl",
    authors = "Sergio Sánchez Ramírez and contributors",
    pages = Any[
        "Home"=>"index.md",
        "Quantum"=>"quantum.md",
        "Ansatzes"=>["`Product` ansatz" => "ansatz/product.md", "`Chain` ansatz" => "ansatz/chain.md"],
    ],
    format = Documenter.HTML(prettyurls = false, assets = ["assets/youtube.css"]),
    plugins = [],
    checkdocs = :exports,
    warnonly = true,
)

deploydocs(repo = "github.com/bsc-quantic/Qrochet.jl.git", devbranch = "master", push_preview = true)
