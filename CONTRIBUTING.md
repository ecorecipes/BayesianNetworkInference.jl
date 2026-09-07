# Contributing to BayesianNetworkInference.jl

## Local setup

The ecosystem packages depend on each other through `[sources]` entries that point at
sibling directories (`../OtherPackage.jl`). Clone every package you need side by side:

```sh
mkdir -p bbn && cd bbn
git clone https://github.com/ecorecipes/FiniteKernels.jl
git clone https://github.com/ecorecipes/BayesianNetworkFormats.jl
git clone https://github.com/ecorecipes/BayesianNetworks.jl
git clone https://github.com/ecorecipes/BayesianNetworkInference.jl
cd BayesianNetworkInference.jl
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

`Pkg.test()` is the only run that counts as "the suite passes".

## Style

- Julia ≥ 1.12. Every dependency, including stdlibs and test-only extras, has a `[compat]` entry.
- Format with JuliaFormatter using the repository `.JuliaFormatter.toml` (`style = "yas"`).
- `snake_case` for functions and variables, `CamelCase` for types, a leading underscore for internal helpers.
- Public functions have docstrings with a signature line, a one-sentence summary, and an example where practical.
- Errors are typed exceptions carrying the offending variable / mechanism names, never bare `error("...")` in library code.
- Every optimised code path is tested against a slower reference implementation on small models.

## Adding a vignette

1. Create `vignettes/NN_short_name/NN_short_name.qmd` with front matter including `engine: julia` and
   `pdf: default` under `format:` (copy an existing vignette's front matter).
2. Add any new dependencies to `vignettes/Project.toml`.
3. `cd vignettes && quarto render` renders HTML, GitHub-flavoured Markdown and PDF (the PDF needs
   `lualatex` and the JuliaMono font in `../fonts/JuliaMono/`; see `vignettes/README.md`). Commit the
   `.qmd`, the rendered `.md`, `.html` and `.pdf`, and `NN_short_name_files/`.
4. `julia scripts/sync_vignettes.jl` and commit `docs/src/tutorials/`.

The `Vignettes` workflow re-renders everything weekly and on demand; it is not a required check because
rendering is slow and depends on a Quarto installation. The `vignette-sync` job in CI is required.

## Architecture decision records

Significant design decisions are recorded in `docs/adr/NNNN-title.md` (context, decision, consequences).
Add a new record rather than editing an old one when a decision changes.

## Pull requests

Branch from `main`, keep PRs focused, and make sure CI is green (tests on Julia 1.12 and latest,
vignette sync check, docs build).
