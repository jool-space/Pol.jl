using Pol
using Documenter

DocMeta.setdocmeta!(Pol, :DocTestSetup, :(using Pol); recursive=true)

makedocs(;
    modules=[Pol],
    authors="AntonOresten <antonoresten@proton.me> and contributors",
    sitename="Pol.jl",
    format=Documenter.HTML(;
        canonical="https://docs.jool.space/Pol.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => [
            "Spaces" => "spaces.md",
            "Arenas" => "arenas.md",
            "Scratchspaces" => "frames.md",
            "Descriptions" => "descriptions.md",
            "Shadows" => "shadows.md",
        ],
        "API reference" => "reference.md",
    ],
)

deploydocs(;
    repo="github.com/jool-space/Pol.jl",
    deploy_repo="github.com/jool-space/docs",
    devbranch="main",
    dirname="Pol.jl",
)
