# Vignettes

Each vignette lives in its own directory `NN_name/NN_name.qmd` and is rendered with
[Quarto](https://quarto.org) using the native Julia engine to three formats:
HTML (`NN_name.html`, self-contained), GitHub-flavoured Markdown (`NN_name.md` plus
`NN_name_files/`) and PDF (`NN_name.pdf`):

```sh
cd vignettes
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
quarto render            # renders every */*.qmd to HTML, GFM and PDF
quarto render --to gfm   # a single format
```

All three rendered outputs are committed. `julia scripts/sync_vignettes.jl` copies the
Markdown into `docs/src/tutorials/` for Documenter, and CI runs
`sync_vignettes.jl --check` and fails if the copies are stale.

Every `.qmd` must carry `engine: julia` and list `pdf: default` under `format:` in its
front matter (a document-level `format:` block otherwise hides the project-level PDF
format). Make sure the package is precompiled before rendering, otherwise Julia's
precompilation output is captured into the first cell of the rendered file.

## PDF

The PDF format (configured in `_quarto.yml`) uses `lualatex` with STIX Two Text/Math for
body text and mathematics and [JuliaMono](https://juliamono.netlify.app) (SIL OFL) for
code, so that the Unicode the vignettes print (`⊗ → ≈ ⊣ ─ │ ├` and friends) appears in
the PDF. JuliaMono is not installed system-wide: fontspec loads it from
`../../fonts/JuliaMono/` relative to this directory (the `fonts/` directory at the root of
the `bbn` checkout, which holds the four `.ttf` faces and the font's `LICENSE`). To
render elsewhere, download
[`JuliaMono-ttf.tar.gz`](https://github.com/cormullion/juliamono/releases/latest) and
unpack it there, or point `monofontoptions: Path=` at an absolute directory. TeX
requirements: `lualatex`, `koma-script`, `fvextra`, `stix2-otf` (TinyTeX installs the
rest on demand) and `rsvg-convert` (librsvg) for the SVG figures produced by
`to_graphviz`.
