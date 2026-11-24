using Documenter
using EpidemicTrajectories

makedocs(
    modules = [EpidemicTrajectories],
    sitename = "EpidemicTrajectories.jl",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", "false") == "true"),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
    checkdocs = :none,
    warnonly = true,
)

deploydocs(
    repo = "github.com/EvoArt/EpidemicTrajectories.git",
    devbranch = "master",
    devurl = "dev",
    push_preview = false,
)
