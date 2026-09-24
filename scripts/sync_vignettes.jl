# Copy rendered vignettes (vignettes/NN_name/NN_name.md + NN_name_files/) into
# docs/src/tutorials/ so Documenter can serve them.
#
#   julia scripts/sync_vignettes.jl          # sync
#   julia scripts/sync_vignettes.jl --check  # exit 1 if docs/src/tutorials is stale
const ROOT = normpath(joinpath(@__DIR__, ".."))
const VIG = joinpath(ROOT, "vignettes")
const OUT = joinpath(ROOT, "docs", "src", "tutorials")
const CHECK = "--check" in ARGS

function vignette_dirs()
    sort(filter(readdir(VIG)) do d
             return occursin(r"^\d+_", d) && isfile(joinpath(VIG, d, d * ".md"))
         end)
end

function files_under(dir)
    out = Dict{String,Vector{UInt8}}()
    isdir(dir) || return out
    for (root, _, files) in walkdir(dir), f in files
        p = joinpath(root, f)
        out[relpath(p, dir)] = read(p)
    end
    return out
end

"""
Rewrite links between vignettes for the flattened `docs/src/tutorials/` layout.

In the vignette tree a sibling is `../NN_other/NN_other.md`; after syncing, every
tutorial sits in one directory, so the link must be plain `NN_other.md`. Doing this
here keeps both layouts correct: the committed vignettes stay browsable on GitHub and
Documenter resolves the cross-references.
"""
function rewrite_sibling_links(text::AbstractString)
    return replace(text, r"\]\(\.\./(\d+_[A-Za-z0-9_]+)/\1\.(md|qmd)\)" => s"](\1.md)")
end

stale = String[]
expected = Set{String}()
for d in vignette_dirs()
    md_src = joinpath(VIG, d, d * ".md")
    md_dst = joinpath(OUT, d * ".md")
    fig_src = joinpath(VIG, d, d * "_files")
    fig_dst = joinpath(OUT, d * "_files")
    push!(expected, d * ".md")
    isdir(fig_src) && push!(expected, d * "_files")
    if CHECK
        expected_md = rewrite_sibling_links(read(md_src, String))
        (isfile(md_dst) && read(md_dst, String) == expected_md) ||
            push!(stale, d * ".md")
        files_under(fig_src) == files_under(fig_dst) || push!(stale, d * "_files/")
    else
        mkpath(OUT)
        write(md_dst, rewrite_sibling_links(read(md_src, String)))
        rm(fig_dst; recursive=true, force=true)
        isdir(fig_src) && cp(fig_src, fig_dst)
    end
end
# orphans: tutorials with no matching vignette
if isdir(OUT)
    for f in readdir(OUT)
        f == ".gitkeep" && continue
        f in expected && continue
        if CHECK
            push!(stale, "orphan: " * f)
        else
            rm(joinpath(OUT, f); recursive=true, force=true)
        end
    end
end
if CHECK
    if isempty(stale)
        println("docs/src/tutorials is up to date")
    else
        println("docs/src/tutorials is stale; run `julia scripts/sync_vignettes.jl`:")
        foreach(s -> println("  ", s), stale)
        exit(1)
    end
else
    println("synced ", length(expected), " item(s) into docs/src/tutorials")
end
