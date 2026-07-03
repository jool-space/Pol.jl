using Pol
using Documenter

DocMeta.setdocmeta!(Pol, :DocTestSetup, :(using Pol); recursive=true)

makedocs(;
    modules=[Pol],
    authors="AntonOresten <antonoresten@proton.me> and contributors",
    sitename="Pol.jl",
    format=Documenter.HTML(;
        canonical="https://jool-space.github.io/Pol.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/jool-space/Pol.jl",
    devbranch="main",
)
