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

"""
Make the raw HTML in Quarto's GitHub-flavoured Markdown readable by Documenter.

Pandoc writes a div with attributes as raw HTML lines (the reference list's
`<div id="refs" ...>`, one `<div id="ref-..." class="csl-entry">` per reference, `</div>`),
and citeproc protects the case of a name with `<span class="nocase">`. GitHub renders that
HTML, but Documenter's Markdown parser escapes raw HTML, so the tags would show as text.
Outside fenced code, each `<div ...>` or `</div>` line becomes a one-line `@raw html`
block, which keeps the `ref-...` anchors that citation links point to, and each span is
replaced by its content.
"""
function documenter_html(text::AbstractString)
    out = IOBuffer()
    prose = IOBuffer()
    function flush_prose()
        s = String(take!(prose))
        s = replace(s, r"<span[^>]*>(.*?)</span>"s => s"\1")
        s = replace(s,
                    r"^[ \t]*(?:<div\b[^>]*>|</div>)[ \t]*$"m => tag -> "```@raw html\n" *
                                                                        strip(tag) *
                                                                        "\n```")
        return print(out, s)
    end
    fence = ""
    for line in eachline(IOBuffer(text); keep=true)
        m = match(r"^[ \t]*(`{3,}|~{3,})", line)
        if isempty(fence)
            if m === nothing
                print(prose, line)
            else
                flush_prose()
                fence = m.captures[1]
                print(out, line)
            end
        else
            print(out, line)
            if m !== nothing && strip(line) == m.captures[1] &&
               first(m.captures[1]) == first(fence) &&
               length(m.captures[1]) >= length(fence)
                fence = ""
            end
        end
    end
    flush_prose()
    return String(take!(out))
end

tutorial_markdown(text::AbstractString) = documenter_html(rewrite_sibling_links(text))

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
        expected_md = tutorial_markdown(read(md_src, String))
        (isfile(md_dst) && read(md_dst, String) == expected_md) ||
            push!(stale, d * ".md")
        files_under(fig_src) == files_under(fig_dst) || push!(stale, d * "_files/")
    else
        mkpath(OUT)
        write(md_dst, tutorial_markdown(read(md_src, String)))
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
