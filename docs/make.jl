using Documenter
using Literate
using EpidemicTrajectories

# Turn the runnable Literate tutorial script into an executed markdown page. The
# code shown on the page is exactly the code in docs/literate/cattle_iffbs.jl, so
# a reader can copy it straight into the REPL.
const LITERATE_DIR = joinpath(@__DIR__, "literate")
const TUT_OUT = joinpath(@__DIR__, "src", "tutorials")

Literate.markdown(
    joinpath(LITERATE_DIR, "cattle_iffbs.jl"), TUT_OUT;
    documenter = true,
    # Collapse the data-simulation cell block so the page leads with the model,
    # not the setup. (Reader can expand it to see how the synthetic data is made.)
)

makedocs(
    modules = [EpidemicTrajectories],
    sitename = "EpidemicTrajectories.jl",
    format = Documenter.HTML(prettyurls = get(ENV, "CI", "false") == "true"),
    pages = [
        "Home" => "index.md",
        "Tutorials" => [
            "Fitting a model with a PPL" => "tutorials/cattle_iffbs.md",
        ],
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
