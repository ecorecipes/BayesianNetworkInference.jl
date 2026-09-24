# Junction-tree inference: a clique tree from CliqueTrees.jl, calibrated by
# Shafer-Shenoy message passing (Shafer and Shenoy 1990; collect, then
# distribute), so that every single-variable marginal comes from one pass over
# the tree. Cited as `ShaferShenoy1990` in the docstrings below.

"""
    JunctionTree(; order=MinFill())

Exact inference on a junction (clique) tree. The tree decomposition of the
interaction graph is built by `CliqueTrees.cliquetree(graph; alg, snd=Maximal())`
with the fill-reducing algorithm of the [`EliminationStrategy`](@ref) `order`
(see [`build_junction_tree`](@ref)), and calibrated by Shafer-Shenoy message
passing [ShaferShenoy1990](@cite) (see [`calibrate`](@ref)). Use it through [`infer`](@ref) for a query
that lies in one clique, [`all_marginals`](@ref) for every single-variable
posterior in one pass, and [`clique_beliefs`](@ref) for the clique posteriors.
"""
struct JunctionTree{O<:EliminationStrategy} <: InferenceBackend
    order::O
end
JunctionTree(; order::EliminationStrategy=MinFill()) = JunctionTree(order)

"""
    CompiledJunctionTree

The tree decomposition of a [`FactorGraph`](@ref) built by
[`build_junction_tree`](@ref). Fields:

- `cliques[i]`: the variables of clique `i`;
- `separators[i]`: `cliques[i] ∩ cliques[parent[i]]` (empty for a root);
- `parent[i]` (`0` for a root), `children[i]`, `roots` (several for a
  disconnected interaction graph), and `postorder` (every clique after its
  children);
- `assignment[j]`: the clique that factor `j` of the graph is assigned to, the
  smallest clique containing its scope (`1` for a factor with empty scope,
  `0` only when the graph has no variables);
- `home[v]`: the smallest clique containing variable `v`;
- `order`: the elimination order the tree was built from;
- `treewidth`: the width of the decomposition (largest clique size minus one;
  `-1` for a graph without variables).

The decomposition depends only on the factor scopes and axes, so it is
computed once per factor graph and strategy and cached. The cache is keyed on
the **identity** of the graph's factor vector (`objectid`), not on its
content: two independently built factor graphs with equal factors therefore
get two decompositions instead of sharing one, which trades a little repeated
work for a cache lookup that is `O(1)` rather than a full table comparison of
every factor. Entries are held through a `WeakRef` and dropped once the
factor vector is unreachable. Do not mutate `fg.factors` after building; the
cached tree would no longer describe the graph.
"""
struct CompiledJunctionTree
    cliques::Vector{Vector{Symbol}}
    separators::Vector{Vector{Symbol}}
    parent::Vector{Int}
    children::Vector{Vector{Int}}
    roots::Vector{Int}
    postorder::Vector{Int}
    assignment::Vector{Int}
    home::Dict{Symbol,Int}
    order::Vector{Symbol}
    treewidth::Int
end

Base.length(jt::CompiledJunctionTree) = length(jt.cliques)

function Base.show(io::IO, jt::CompiledJunctionTree)
    return print(io, "CompiledJunctionTree with ", length(jt), " cliques (treewidth ",
                 jt.treewidth, ")")
end

function Base.show(io::IO, ::MIME"text/plain", jt::CompiledJunctionTree)
    show(io, jt)
    for i in eachindex(jt.cliques)
        print(io, "\n  ", i, ": ", jt.cliques[i])
        if jt.parent[i] != 0
            print(io, "  --", jt.separators[i], "--> ", jt.parent[i])
        end
    end
    return nothing
end

# The junction-tree cache: `objectid(fg.factors)` to the weak reference that
# produced it and the trees built from it, one per strategy. Keying on
# identity (rather than on the factor vector itself, whose `hash`/`isequal`
# compare every table) keeps a cache hit `O(1)`; the stored `WeakRef` is
# compared with `===` on lookup, so a recycled `objectid` can never return
# another graph's tree.
const _JunctionTreeEntry = Tuple{WeakRef,Dict{EliminationStrategy,CompiledJunctionTree}}
const _JUNCTION_TREE_CACHE = Dict{UInt,_JunctionTreeEntry}()
const _JUNCTION_TREE_LOCK = ReentrantLock()
const _JUNCTION_TREE_WATERMARK = Ref(16)

# Drop the entries whose factor vector has been collected, amortised: the
# sweep runs only when the cache has doubled since the last one.
function _prune_junction_tree_cache!()
    length(_JUNCTION_TREE_CACHE) <= 2 * _JUNCTION_TREE_WATERMARK[] && return nothing
    filter!(kv -> kv[2][1].value !== nothing, _JUNCTION_TREE_CACHE)
    _JUNCTION_TREE_WATERMARK[] = max(16, length(_JUNCTION_TREE_CACHE))
    return nothing
end

# The `alg` handed to `CliqueTrees.cliquetree`: the CliqueTrees algorithm of
# the strategy, or an explicit permutation for the strategies that need one
# (ExactTreewidth runs per component, UserOrder is user-supplied).
_tree_algorithm(graph, vars, index, strategy::EliminationStrategy) = _algorithm(strategy)
function _tree_algorithm(graph, vars, index, strategy::Union{ExactTreewidth,UserOrder})
    order = _elimination_order(graph, vars, index, strategy, Symbol[])
    return Int[index[v] for v in order]
end

"""
    build_junction_tree(fg::FactorGraph; order=MinFill()) -> CompiledJunctionTree

The junction tree of `fg`: `CliqueTrees.cliquetree(graph; alg, snd=Maximal())`
on the interaction graph, with `alg` the CliqueTrees algorithm of the
[`EliminationStrategy`](@ref) `order` (an explicit permutation for
[`ExactTreewidth`](@ref) and [`UserOrder`](@ref)). The maximal supernode
partition gives one clique per maximal clique of the triangulated graph; the
clique of every tree node is `residual ∪ separator` and the separator is the
intersection with its parent. Every factor is assigned to the smallest clique
containing its scope, which exists because each scope is a clique of the
interaction graph. The result is cached per factor-graph identity and
strategy (see [`CompiledJunctionTree`](@ref)). Clique trees and the
junction-tree property are set out in [KollerFriedman2009](@cite)
(chapter 10).

```jldoctest
julia> a = FiniteAxis(:A, [:x, :y]); b = FiniteAxis(:B, [:x, :y]); c = FiniteAxis(:C, [:x, :y]);

julia> fg = FactorGraph([Factor([a, b], ones(2, 2)), Factor([b, c], ones(2, 2))]);

julia> jt = build_junction_tree(fg);

julia> sort(jt.cliques)
2-element Vector{Vector{Symbol}}:
 [:A, :B]
 [:B, :C]

julia> jt.treewidth
1
```
"""
function build_junction_tree(fg::FactorGraph; order::EliminationStrategy=MinFill())
    lock(_JUNCTION_TREE_LOCK) do
        key = objectid(fg.factors)
        entry = get(_JUNCTION_TREE_CACHE, key, nothing)
        per_graph = if entry === nothing || entry[1].value !== fg.factors
            fresh = Dict{EliminationStrategy,CompiledJunctionTree}()
            _prune_junction_tree_cache!()
            _JUNCTION_TREE_CACHE[key] = (WeakRef(fg.factors), fresh)
            fresh
        else
            entry[2]
        end
        return get!(per_graph, order) do
            return _build_junction_tree(fg, order)
        end
    end
end

function _build_junction_tree(fg::FactorGraph, strategy::EliminationStrategy)
    graph, vars, index = interaction_graph(fg)
    alg = _tree_algorithm(graph, vars, index, strategy)
    label, tree = CliqueTrees.cliquetree(graph; alg=alg, snd=CliqueTrees.Maximal())
    n = length(tree)
    name(v) = vars[label[v]]
    byindex(vs) = sort!(Symbol[name(v) for v in vs]; by=v -> index[v])
    cliques = [byindex(tree[i]) for i in 1:n]
    separators = [byindex(CliqueTrees.separator(tree[i])) for i in 1:n]
    parent = [something(CliqueTrees.parentindex(tree, i), 0) for i in 1:n]
    children = [collect(Int, CliqueTrees.childindices(tree, i)) for i in 1:n]
    roots = collect(Int, CliqueTrees.rootindices(tree))
    postorder = _postorder(children, roots)
    width = n == 0 ? -1 : CliqueTrees.treewidth(tree)
    # factor assignment and home cliques
    member = Dict{Symbol,Vector{Int}}(v => Int[] for v in vars)
    for (i, c) in enumerate(cliques), v in c
        push!(member[v], i)
    end
    size_of(i) = prod(length(fg.axes[v]) for v in cliques[i]; init=1)
    home = Dict{Symbol,Int}(v => argmin(i -> size_of(i), member[v]) for v in vars)
    assignment = map(fg.factors) do f
        isempty(f.vars) && return n == 0 ? 0 : 1
        candidates = reduce(intersect, (member[v] for v in f.vars))
        isempty(candidates) &&
            throw(ScopeError(:build_junction_tree,
                             "no clique contains the scope of a factor", f.vars))
        return argmin(i -> size_of(i), candidates)
    end
    return CompiledJunctionTree(cliques, separators, parent, children, roots, postorder,
                                assignment, home, Symbol[name(v) for v in 1:length(vars)],
                                width)
end

# Every node after all of its children (iterative, so deep trees are fine).
function _postorder(children::Vector{Vector{Int}}, roots::Vector{Int})
    order = Int[]
    stack = Tuple{Int,Bool}[(r, false) for r in reverse(roots)]
    while !isempty(stack)
        node, expanded = pop!(stack)
        if expanded
            push!(order, node)
        else
            push!(stack, (node, true))
            for c in reverse(children[node])
                push!(stack, (c, false))
            end
        end
    end
    return order
end

"""
    CalibratedJunctionTree{T}

A [`CompiledJunctionTree`](@ref) after [`calibrate`](@ref): `beliefs[i]` is the
unnormalised clique belief (its scope is `cliques[i]` minus the evidence
variables, in some order), `potentials[i]` the product of the conditioned
factors assigned to clique `i`, `evidence` the evidence used,
`evidence_probability` the global partition mass after conditioning (equal to
`P(evidence)` when the original factors form a normalized joint),
and `n_messages` the number of separator messages passed.
"""
struct CalibratedJunctionTree{T<:Real}
    tree::CompiledJunctionTree
    evidence::Dict{Symbol,Symbol}
    potentials::Vector{Factor{T}}
    beliefs::Vector{Factor{T}}
    evidence_probability::T
    n_messages::Int
end

function Base.show(io::IO, c::CalibratedJunctionTree{T}) where {T}
    return print(io, "CalibratedJunctionTree{", T, "} with ", length(c.beliefs),
                 " cliques, P(evidence) = ", c.evidence_probability)
end

# Sum out everything in the scope of `f` that is not in `keep`.
_project(f::Factor, keep::Vector{Symbol}) = marginalize(f, setdiff(f.vars, keep))

function _junction_tree_messages(jt, potentials::Vector{F}, separators, product,
                                 project) where {F}
    n = length(jt)
    up, down, beliefs = (Vector{F}(undef, n) for _ in 1:3)
    n_messages = 0
    for i in jt.postorder
        jt.parent[i] == 0 && continue
        incoming = F[potentials[i]]
        append!(incoming, (up[k] for k in jt.children[i]))
        up[i] = project(product(incoming), separators[i])
        n_messages += 1
    end
    for i in Iterators.reverse(jt.postorder)
        incoming = F[potentials[i]]
        jt.parent[i] == 0 || push!(incoming, down[i])
        append!(incoming, (up[k] for k in jt.children[i]))
        beliefs[i] = product(copy(incoming))
        for child in jt.children[i]
            outgoing = F[potentials[i]]
            jt.parent[i] == 0 || push!(outgoing, down[i])
            append!(outgoing, (up[k] for k in jt.children[i] if k != child))
            down[child] = project(product(outgoing), separators[child])
            n_messages += 1
        end
    end
    return beliefs, n_messages
end

"""
    calibrate(fg::FactorGraph, jt=build_junction_tree(fg; order); evidence=Dict{Symbol,Symbol}(), order=MinFill())
        -> CalibratedJunctionTree

Shafer-Shenoy message passing [ShaferShenoy1990](@cite) on the junction
tree `jt` of `fg`. Every
factor is conditioned on `evidence` and multiplied into the potential of its
clique; messages are then passed from the leaves to the roots (collect) and
back (distribute), the message from clique `i` to neighbour `j` being the
product of `i`'s potential with the messages from its other neighbours,
summed down to the separator. The belief of a clique is its potential times
all incoming messages, and equals the unnormalized marginal of that connected component. For a connected
graph this is `P(clique, evidence)`; in a forest, `evidence_probability` multiplies
the component masses and posterior entry points check that global mass. There is no
separator division, so factors with zero entries (deterministic mechanisms)
need no special treatment -- unlike the division-based architectures of
[LauritzenSpiegelhalter1988](@cite) and its Hugin refinement
[JensenLauritzenOlesen1990](@cite), which divide by the separator potential
and pay for it with more storage but fewer multiplications.

Throws [`ScopeError`](@ref) for evidence on unknown variables and
`FiniteKernels.InvalidAxisError` for a label a variable does not have.
"""
function calibrate(fg::FactorGraph{T}, jt::CompiledJunctionTree;
                   evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}()) where {T}
    _check_query(fg, Symbol[], evidence)
    ev = Dict{Symbol,Symbol}(evidence)
    n = length(jt)
    length(jt.assignment) == length(fg.factors) ||
        throw(ShapeError(:calibrate,
                         "the junction tree was built for a different factor graph",
                         length(jt.assignment), length(fg.factors)))
    lists = [Factor{T}[] for _ in 1:n]
    constant = unit_factor(T)
    for (f, c) in zip(fg.factors, jt.assignment)
        g = condition(f, ev)
        if c == 0
            constant = multiply(constant, g)
        else
            push!(lists[c], g)
        end
    end
    potentials = Factor{T}[_product!(lists[c]) for c in 1:n]
    seps = [Symbol[v for v in s if !haskey(ev, v)] for s in jt.separators]
    beliefs, n_messages = _junction_tree_messages(jt, potentials, seps, _product!, _project)
    mass = constant.table[]
    for r in jt.roots
        mass *= sum(beliefs[r].table)
    end
    return CalibratedJunctionTree{T}(jt, ev, potentials, beliefs, mass, n_messages)
end
function calibrate(fg::FactorGraph;
                   evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
                   order::EliminationStrategy=MinFill())
    return calibrate(fg, build_junction_tree(fg; order); evidence)
end

"""
    JunctionTreeDiagnostics(clique, n_cliques, treewidth, max_clique_size, n_messages, fallback=false)

What a [`JunctionTree`](@ref) run of [`infer`](@ref) did: the index of the
`clique` the query was read from (`0` when there was none), the number of
cliques, the width of the decomposition, the number of entries of the largest
clique belief (of the largest intermediate factor when `fallback` is true),
the number of separator messages passed (`0` when `fallback` is true), and
whether the query was answered by the variable-elimination fallback because
no single clique contained it.

[`infer`](@ref) with the [`JunctionTree`](@ref) backend always returns this
type, so the diagnostics of a query that falls back can be told from the
`fallback` field rather than from the concrete type of the result.
"""
struct JunctionTreeDiagnostics
    clique::Int
    n_cliques::Int
    treewidth::Int
    max_clique_size::Int
    n_messages::Int
    fallback::Bool
end
function JunctionTreeDiagnostics(clique::Integer, n_cliques::Integer, treewidth::Integer,
                                 max_clique_size::Integer, n_messages::Integer)
    return JunctionTreeDiagnostics(clique, n_cliques, treewidth, max_clique_size,
                                   n_messages, false)
end

function _diagnostics(cal::CalibratedJunctionTree, clique::Int)
    largest = maximum(length, cal.beliefs; init=0)
    return JunctionTreeDiagnostics(clique, length(cal.tree), cal.tree.treewidth, largest,
                                   cal.n_messages, false)
end

# The smallest clique containing every variable of `query`, or nothing.
function _containing_clique(jt::CompiledJunctionTree, axes, query)
    best = 0
    best_size = typemax(Int)
    for (i, c) in enumerate(jt.cliques)
        all(in(c), query) || continue
        s = prod(length(axes[v]) for v in c; init=1)
        if s < best_size
            best, best_size = i, s
        end
    end
    return best == 0 ? nothing : best
end

# Callers check the global mass before reading a single component's posterior.
function _belief_marginal(belief::Factor, query)
    return normalize(reorder(_project(belief, collect(Symbol, query)), query))
end

function _infer(backend::JunctionTree, fg::FactorGraph{T}, query, evidence) where {T}
    _check_query(fg, query, evidence)
    jt = build_junction_tree(fg; order=backend.order)
    if isempty(query)
        cal = calibrate(fg, jt; evidence)
        return _factor(Symbol[], FiniteAxis[], fill(cal.evidence_probability)),
               _diagnostics(cal, 0)
    end
    c = _containing_clique(jt, fg.axes, query)
    if c === nothing
        @warn "JunctionTree: the query $(collect(query)) does not lie in one clique; falling back to variable elimination"
        f, ve = variable_elimination(fg, query; evidence, order=backend.order)
        return f,
               JunctionTreeDiagnostics(0, length(jt), jt.treewidth, ve.max_factor_size, 0,
                                       true)
    end
    cal = calibrate(fg, jt; evidence)
    _require_evidence_mass(cal.evidence_probability, evidence)
    return _belief_marginal(cal.beliefs[c], query), _diagnostics(cal, c)
end

"""
    clique_beliefs(fg::FactorGraph; evidence=Dict{Symbol,Symbol}(), backend=JunctionTree())
        -> Vector{Factor}

The posterior `P(clique | evidence)` of every clique of the junction tree of
`fg` (in the order of `build_junction_tree(fg; order=backend.order).cliques`),
as normalised factors whose scopes are the cliques minus the evidence
variables. Throws `FiniteKernels.KernelNormalizationError` if the evidence has
probability zero.
"""
function clique_beliefs(fg::FactorGraph;
                        evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
                        backend::InferenceBackend=JunctionTree())
    return _clique_beliefs(backend, fg, evidence)
end

function _clique_beliefs(backend::JunctionTree, fg::FactorGraph, evidence)
    cal = calibrate(fg, build_junction_tree(fg; order=backend.order); evidence)
    _require_evidence_mass(cal.evidence_probability, evidence)
    return map(normalize, cal.beliefs)
end

"""
    all_marginals(fg::FactorGraph; evidence=Dict{Symbol,Symbol}(), backend=JunctionTree())
        -> Dict{Symbol,Factor}

The posterior marginal `P(v | evidence)` of every variable `v` of `fg` as a
normalised one-variable factor; an observed variable maps to the point mass
at its observed label. With [`JunctionTree`](@ref) (the default) the tree is
calibrated once and every marginal is read off the smallest clique containing
its variable; with [`BeliefPropagation`](@ref) one run of
[`belief_propagation`](@ref) provides them all (exact on a tree-structured
factor graph, approximate otherwise); with [`VariableElimination`](@ref) each
marginal is a separate [`variable_elimination`](@ref) run, which is the
oracle the other two are tested against.
"""
function all_marginals(fg::FactorGraph;
                       evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
                       backend::InferenceBackend=JunctionTree())
    _check_query(fg, Symbol[], evidence)
    return _all_marginals(backend, fg, Dict{Symbol,Symbol}(evidence))
end

function _all_marginals(backend::JunctionTree, fg::FactorGraph{T}, evidence) where {T}
    jt = build_junction_tree(fg; order=backend.order)
    cal = calibrate(fg, jt; evidence)
    _require_evidence_mass(cal.evidence_probability, evidence)
    R = _division_type(T)
    out = Dict{Symbol,Factor{R}}()
    for (v, ax) in fg.axes
        if haskey(evidence, v)
            out[v] = _point_mass(R, ax, evidence[v])
        else
            out[v] = _belief_marginal(cal.beliefs[jt.home[v]], [v])
        end
    end
    return out
end

function _all_marginals(backend::VariableElimination, fg::FactorGraph{T},
                        evidence) where {T}
    if all(v -> haskey(evidence, v), keys(fg.axes))
        mass = variable_elimination(fg, Symbol[]; evidence, order=backend.order)[1].table[]
        _require_evidence_mass(mass, evidence)
    end
    R = _division_type(T)
    out = Dict{Symbol,Factor{R}}()
    for (v, ax) in fg.axes
        if haskey(evidence, v)
            out[v] = _point_mass(R, ax, evidence[v])
        else
            out[v] = variable_elimination(fg, [v]; evidence, order=backend.order)[1]
        end
    end
    return out
end

# The factor that puts all mass on `label` of `ax`.
function _point_mass(::Type{T}, ax::FiniteAxis, label::Symbol) where {T}
    table = zeros(T, length(ax))
    table[label_index(ax, label)] = one(T)
    return _factor([ax.name], [ax], table)
end
