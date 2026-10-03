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

The tree is checked once, when it is built, against the hypotheses of the
calibration proof (`Good` and `checkAssignment` in the Lean project): throws
[`ScopeError`](@ref) if CliqueTrees returns a tree that is not a rooted forest
(parent, children, roots and postorder disagree, or the parent links cycle), whose
separators are not the intersections of cliques with their parents, in which the
cliques holding a variable are not connected (running intersection), that misses
or repeats a variable, or on which a factor's scope lies in no clique. This is a
runtime check of CliqueTrees' output, not a proof of it.

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
    # `0` for a variable in no clique, or a factor whose scope no clique holds: both
    # are reported by `_check_junction_tree` below, with the condition named.
    home = Dict{Symbol,Int}(v => isempty(member[v]) ? 0 : argmin(size_of, member[v])
                            for v in vars)
    assignment = map(fg.factors) do f
        isempty(f.vars) && return n == 0 ? 0 : 1
        candidates = reduce(intersect, (member[v] for v in f.vars))
        return isempty(candidates) ? 0 : argmin(size_of, candidates)
    end
    _check_junction_tree(cliques, separators, parent, children, roots, postorder,
                         assignment, [f.vars for f in fg.factors], vars)
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

_tree_error(msg, vars=Symbol[]) = throw(ScopeError(:build_junction_tree, msg, vars))

# A runtime check that the tree CliqueTrees returned, and the factor assignment computed
# from it, satisfy the hypotheses under which `calibrate` is proved correct; it is not a
# proof that CliqueTrees is correct, and it says nothing about the arithmetic. It mirrors
# `Good` and `checkAssignment` of `BayesianNetworks.jl/proofs/BayesianNetworksProofs/
# Finite/JunctionTree.lean` (`calibrate_correct`, `calibrate_eq_ve`), translated from
# Lean's binary trees to the n-ary forest here through `graft_good` and `forest_good`:
#
# - the parent/children/roots/postorder data form a rooted forest (the Lean `Tree` is a
#   tree by construction; `forest_good` hangs its components under an empty virtual root,
#   which is what the scalar root masses of `calibrate` are);
# - running intersection: `Good` asks, at every node with bag `B`, that each child
#   subtree meets `B` only inside the child's bag (`l.vars ∩ B ⊆ l.bag`), that two child
#   subtrees meet only inside `B` (`l.vars ∩ r.vars ⊆ B`, `graft_good`'s pairwise
#   premise), and `forest_good` that components share no variable. Over a forest these
#   together say that the cliques containing each variable form one connected subtree,
#   which is checked in the equivalent form "every variable has exactly one top clique",
#   a clique holding it whose parent does not (or that is a root);
# - every separator is its clique intersected with its parent's clique, empty at a root:
#   Lean's `prepare` projects the upward message onto the parent's whole bag, and under
#   running intersection that projection keeps exactly this intersection;
# - every factor's scope lies in its clique (`(localFactor F fs).scope ⊆ B`); an
#   empty-scope factor may sit in any clique, or in `0`, the scalar `constant` of
#   `_calibrate` (the empty virtual root of `forest_good`);
# - `checkAssignment`'s "every factor index occurs exactly once" holds by the shape of
#   `assignment` (one entry per factor), checked by its length;
# - every interaction-graph variable is in some clique and no clique repeats one (Lean's
#   bags are `Finset`s); `calibrate_eq_ve`'s `hvars` holds because the variables are
#   read off the factor scopes. Linear in the total clique size.
function _check_junction_tree(cliques, separators, parent, children, roots, postorder,
                              assignment, scopes, vars)
    n = length(cliques)
    length(separators) == length(parent) == length(children) == n ||
        _tree_error("the clique, separator, parent and children lists differ in length")
    # forest: parents in range, no cycles, roots and children agree with parents
    for i in 1:n
        0 <= parent[i] <= n && parent[i] != i ||
            _tree_error("clique $i has parent $(parent[i]), which is not another clique",
                        cliques[i])
    end
    state = zeros(Int8, n)               # 0 unvisited, 1 on the current path, 2 done
    path = Int[]
    for i in 1:n
        j = i
        while j != 0 && state[j] == 0
            state[j] = 1
            push!(path, j)
            j = parent[j]
        end
        j != 0 && state[j] == 1 &&
            _tree_error("the parent links contain a cycle through clique $j", cliques[j])
        foreach(k -> state[k] = 2, path)
        empty!(path)
    end
    isroot = falses(n)
    for r in roots
        1 <= r <= n && parent[r] == 0 && !isroot[r] ||
            _tree_error("root $r is not a distinct clique without a parent")
        isroot[r] = true
    end
    for i in 1:n
        parent[i] == 0 && !isroot[i] &&
            _tree_error("clique $i has no parent but is not listed as a root", cliques[i])
    end
    listed = falses(n)
    for i in 1:n, c in children[i]
        1 <= c <= n && parent[c] == i && !listed[c] ||
            _tree_error("children of clique $i disagree with the parent links: " *
                        "child $c", cliques[i])
        listed[c] = true
    end
    for i in 1:n
        parent[i] == 0 || listed[i] ||
            _tree_error("clique $i is missing from the children of its parent " *
                        "$(parent[i])", cliques[i])
    end
    position = zeros(Int, n)
    length(postorder) == n ||
        _tree_error("the postorder does not visit every clique exactly once")
    for (k, i) in enumerate(postorder)
        1 <= i <= n && position[i] == 0 ||
            _tree_error("the postorder does not visit every clique exactly once")
        position[i] = k
    end
    for i in 1:n
        parent[i] == 0 || position[i] < position[parent[i]] ||
            _tree_error("the postorder visits clique $(parent[i]) before its child $i",
                        cliques[i])
    end
    # bags: no repeats, graph variables only, separators are intersections with parents
    known = Set{Symbol}(vars)
    sets = [Set{Symbol}(c) for c in cliques]
    for i in 1:n
        length(sets[i]) == length(cliques[i]) ||
            _tree_error("clique $i repeats a variable", cliques[i])
        issubset(sets[i], known) ||
            _tree_error("clique $i holds a variable outside the interaction graph",
                        setdiff(cliques[i], known))
        p = parent[i]
        sep = separators[i]
        ok = if p == 0
            isempty(sep)
        else
            length(sep) == length(Set(sep)) && all(in(sets[i]), sep) &&
                all(in(sets[p]), sep) && count(in(sets[p]), cliques[i]) == length(sep)
        end
        ok || _tree_error("the separator of clique $i is not its intersection with its " *
                          "parent clique $p (empty for a root)", sep)
    end
    # running intersection and coverage: exactly one top clique per variable
    top = Dict{Symbol,Int}(v => 0 for v in vars)
    for i in 1:n
        p = parent[i]
        for v in cliques[i]
            p != 0 && v in sets[p] && continue
            top[v] == 0 ||
                _tree_error("the cliques containing a variable are not connected " *
                            "(running intersection fails at cliques $(top[v]) and $i)",
                            [v])
            top[v] = i
        end
    end
    for v in vars
        top[v] == 0 && _tree_error("a variable of the interaction graph is in no clique",
                                   [v])
    end
    # factor assignment
    length(assignment) == length(scopes) ||
        _tree_error("the assignment does not give exactly one clique per factor")
    for (j, (s, c)) in enumerate(zip(scopes, assignment))
        if isempty(s)
            0 <= c <= n || _tree_error("empty-scope factor $j is assigned to clique $c, " *
                                       "which does not exist")
        else
            1 <= c <= n && all(in(sets[c]), s) ||
                _tree_error("no clique contains the scope of a factor: factor $j is " *
                            "assigned to clique $c, which does not hold its scope",
                            collect(Symbol, s))
        end
    end
    return nothing
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

# Shafer-Shenoy calibration in the arithmetic `A` (arithmetic.jl), shared by `calibrate` and
# `log_calibrate`: condition every factor on the evidence and multiply it into its clique's
# potential (or into `constant`, for a factor with no variable left), then pass the collect
# and distribute messages. What the beliefs' masses mean is the caller's.
function _calibrate(A::_Arithmetic, fg::FactorGraph{T}, jt::CompiledJunctionTree, evidence,
                    operation::Symbol) where {T}
    _check_query(fg, Symbol[], evidence)
    length(jt.assignment) == length(fg.factors) ||
        throw(ShapeError(operation,
                         "the junction tree was built for a different factor graph",
                         length(jt.assignment), length(fg.factors)))
    ev = Dict{Symbol,Symbol}(evidence)
    F = _factor_type(A, T)
    lists = [F[] for _ in 1:length(jt)]
    constant = _unit(A, T)
    for (f, c) in zip(fg.factors, jt.assignment)
        g = _conditioned(A, f, ev)
        if c == 0
            constant = _multiply(A, constant, g)
        else
            push!(lists[c], g)
        end
    end
    potentials = F[_product!(A, lists[c]) for c in 1:length(jt)]
    seps = [Symbol[v for v in s if !haskey(ev, v)] for s in jt.separators]
    beliefs, n_messages = _junction_tree_messages(jt, potentials, seps,
                                                  fs -> _product!(A, fs),
                                                  (f, keep) -> _project(A, f, keep))
    return ev, potentials, constant, beliefs, n_messages
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
`FiniteKernels.InvalidAxisError` for a label a variable does not have. The beliefs have the
graph's element type, so an integer or rational graph whose products or sums overflow that
type raises [`FactorDomainError`](@ref) (see [`multiply`](@ref)); the posterior entry points
built on calibration answer such a graph exactly instead.
"""
function calibrate(fg::FactorGraph, jt::CompiledJunctionTree;
                   evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}())
    return _calibrated(_Linear(), fg, jt, evidence)
end

# `calibrate` in the linear arithmetic `A`: unchecked for `calibrate` itself, and with the
# trust test on every product, the mass's included, for a posterior (`_Trusted`). Integer
# and rational masses are summed and multiplied in checked arithmetic (`_overflowed`).
function _calibrated(A::_Linear, fg::FactorGraph{T}, jt::CompiledJunctionTree,
                     evidence) where {T}
    ev, potentials, constant, beliefs, n_messages = _calibrate(A, fg, jt, evidence,
                                                               :calibrate)
    mass = constant.table[]
    for r in jt.roots
        total = _total(beliefs[r], :calibrate)
        total isa FactorDomainError && _overflowed(A, total)
        mass = _scalar_product(A, mass, total)
    end
    return CalibratedJunctionTree{T}(jt, ev, potentials, beliefs, mass, n_messages)
end

function _scalar_product(A::_Linear, x, y)
    p, overflow = _mul(promote(x, y)...)
    overflow &&
        _overflowed(A, _overflow(:calibrate, Symbol[], (), _exact(x) * _exact(y)))
    return p
end
function _scalar_product(::_Trusted, x::T, y::T) where {T<:Base.IEEEFloat}
    p = x * y
    _lost(x, y, p) && _untrusted()
    return p
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
no single clique contained it. `exact_fallback` is true when the run was not
trusted -- a binary64 evidence mass that was not a normal positive number, a product of
nonzero values below `floatmin` on the way, or an integer or rational product, sum or
quotient that overflowed its element type -- and the posterior was recomputed in exact
arithmetic, each cell correctly rounded (ADR 0014, ADR 0016): by recalibrating the tree,
or, for a query that no clique contains, by variable elimination's own exact fallback.

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
    exact_fallback::Bool
end
function JunctionTreeDiagnostics(clique::Integer, n_cliques::Integer, treewidth::Integer,
                                 max_clique_size::Integer, n_messages::Integer,
                                 fallback::Bool=false)
    return JunctionTreeDiagnostics(clique, n_cliques, treewidth, max_clique_size,
                                   n_messages, fallback, false)
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

# The posterior of `query` read off a clique belief, in either arithmetic. Callers check the
# global mass first; `_normalized` checks the belief's own linear mass again, so a clique
# whose sum is zero although the global mass was positive raises the same
# `ImpossibleEvidenceError` rather than `normalize`'s `ArgumentError`.
function _belief_marginal(A::_Arithmetic, belief, query, evidence)
    q = collect(Symbol, query)
    return _normalized(A, _reorder(A, _project(A, belief, q), q), evidence)
end

function _infer(backend::JunctionTree, fg::FactorGraph, query, evidence)
    _check_query(fg, query, evidence)
    isempty(query) && return _infer_binary64(backend, fg, query, evidence)
    return _resolving_mass(() -> _infer_binary64(backend, fg, query, evidence),
                           evidence) do
        return _exact_junction_tree_query(backend, fg, query, evidence)
    end
end

# The exact fallback of the junction tree (ADR 0014, ADR 0016): the shared calibration in
# exact dyadic arithmetic, with the global mass decided exactly (the constant and every
# component root) and each posterior cell correctly rounded.
function _exact_calibration(fg::FactorGraph, jt::CompiledJunctionTree, evidence)
    _, _, constant, beliefs, n_messages = _calibrate(_Dyadic(), fg, jt, evidence,
                                                     :calibrate)
    (_exactly_zero(constant) || any(r -> _exactly_zero(beliefs[r]), jt.roots)) &&
        _impossible(evidence)
    return beliefs, n_messages
end

# Reached only for a query that one clique contains: a query spanning cliques goes to
# `variable_elimination`, which resolves its own untrusted runs (`_infer_binary64`).
function _exact_junction_tree_query(backend::JunctionTree, fg::FactorGraph, query, evidence)
    jt = build_junction_tree(fg; order=backend.order)
    c = _containing_clique(jt, fg.axes, query)
    beliefs, n_messages = _exact_calibration(fg, jt, evidence)
    largest = maximum(b -> length(b.factor), beliefs; init=0)
    return _belief_marginal(_Dyadic(), beliefs[c], query, evidence),
           JunctionTreeDiagnostics(c, length(jt), jt.treewidth, largest, n_messages, false,
                                   true)
end

function _exact_all_marginals(order::EliminationStrategy, fg::FactorGraph, evidence)
    jt = build_junction_tree(fg; order)
    beliefs, _ = _exact_calibration(fg, jt, evidence)
    return _collect_marginals(fg, evidence, Float64) do v
        return _belief_marginal(_Dyadic(), beliefs[jt.home[v]], [v], evidence)
    end
end

function _infer_binary64(backend::JunctionTree, fg::FactorGraph{T}, query,
                         evidence) where {T}
    jt = build_junction_tree(fg; order=backend.order)
    if isempty(query)
        cal = calibrate(fg, jt; evidence)
        return _factor(Symbol[], FiniteAxis[], fill(cal.evidence_probability)),
               _diagnostics(cal, 0)
    end
    c = _containing_clique(jt, fg.axes, query)
    if c === nothing
        @warn "JunctionTree: the query $(collect(query)) does not lie in one clique; falling back to variable elimination"
        # `variable_elimination` resolves its own untrusted runs, and says whether it did.
        f, ve = variable_elimination(fg, query; evidence,
                                     order=_keeping(backend.order, query))
        return f,
               JunctionTreeDiagnostics(0, length(jt), jt.treewidth, ve.max_factor_size, 0,
                                       true, ve.exact_fallback)
    end
    cal = _calibrated(_Trusted(), fg, jt, evidence)
    _require_evidence_mass(cal.evidence_probability, evidence)
    return _belief_marginal(_Trusted(), cal.beliefs[c], query, evidence),
           _diagnostics(cal, c)
end

# The elimination order of a junction tree's strategy for a query that variable elimination
# answers instead. A tree's `UserOrder` lists every variable of the graph, as
# `build_junction_tree` requires, but variable elimination keeps the query, so the query is
# taken out of it and the rest keep the user's order.
_keeping(strategy::EliminationStrategy, query) = strategy
function _keeping(strategy::UserOrder, query)
    return UserOrder([v for v in strategy.vars if !(v in query)])
end

"""
    clique_beliefs(fg::FactorGraph; evidence=Dict{Symbol,Symbol}(), backend=JunctionTree())
        -> Vector{Factor}

The posterior `P(clique | evidence)` of every clique of the junction tree of
`fg` (in the order of `build_junction_tree(fg; order=backend.order).cliques`),
as normalised factors whose scopes are the cliques minus the evidence
variables. Throws `BayesianNetworks.ImpossibleEvidenceError` if the evidence
has probability exactly zero; a binary64 mass that underflowed is recomputed in exact
arithmetic and each belief correctly rounded (ADR 0014, ADR 0016).
"""
function clique_beliefs(fg::FactorGraph;
                        evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
                        backend::InferenceBackend=JunctionTree())
    return _clique_beliefs(backend, fg, evidence)
end

function _clique_beliefs(backend::JunctionTree, fg::FactorGraph, evidence)
    ordinary = function ()
        cal = _calibrated(_Trusted(), fg, build_junction_tree(fg; order=backend.order),
                          evidence)
        _require_evidence_mass(cal.evidence_probability, evidence)
        return [_normalized(_Trusted(), belief, evidence) for belief in cal.beliefs]
    end
    return _resolving_mass(ordinary, evidence) do
        beliefs, _ = _exact_calibration(fg, build_junction_tree(fg; order=backend.order),
                                        evidence)
        return [_normalized(_Dyadic(), belief, evidence) for belief in beliefs]
    end
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

The marginals have the element type of [`infer`](@ref)'s posteriors on `fg`. When the
exact backends do not trust a run (see [`variable_elimination`](@ref)), every marginal is
recomputed in exact arithmetic and returned as a correctly rounded `Factor{Float64}`, as
`infer` returns that posterior: the junction tree recalibrates exactly, and variable
elimination recomputes every marginal exactly as soon as one of its runs is untrusted, so
the dictionary never mixes the two.
"""
function all_marginals(fg::FactorGraph;
                       evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
                       backend::InferenceBackend=JunctionTree())
    _check_query(fg, Symbol[], evidence)
    return _all_marginals(backend, fg, Dict{Symbol,Symbol}(evidence))
end

function _all_marginals(backend::JunctionTree, fg::FactorGraph, evidence)
    return _resolving_mass(() -> _all_marginals_binary64(backend, fg, evidence),
                           evidence) do
        return _exact_all_marginals(backend.order, fg, evidence)
    end
end

function _all_marginals_binary64(backend::JunctionTree, fg::FactorGraph{T},
                                 evidence) where {T}
    jt = build_junction_tree(fg; order=backend.order)
    cal = _calibrated(_Trusted(), fg, jt, evidence)
    _require_evidence_mass(cal.evidence_probability, evidence)
    return _collect_marginals(fg, evidence, _division_type(T)) do v
        return _belief_marginal(_Trusted(), cal.beliefs[jt.home[v]], [v], evidence)
    end
end

# One variable-elimination run per marginal, all in the graph's arithmetic or, as soon as one
# of them is not trusted, all in exact arithmetic (review of 2026-10-02, finding 4): the
# exact marginals are `Factor{Float64}`, as `infer` returns them, and a dictionary of the
# graph's division type could not hold them.
function _all_marginals(backend::VariableElimination, fg::FactorGraph{T},
                        evidence) where {T}
    all(v -> haskey(evidence, v), keys(fg.axes)) &&
        _require_feasible(fg, evidence, backend.order)
    ordinary = function ()
        return _collect_marginals(fg, evidence, _division_type(T)) do v
            return first(_variable_elimination(fg, [v], evidence, backend.order, nothing))
        end
    end
    return _resolving_mass(ordinary, evidence) do
        return _collect_marginals(fg, evidence, Float64) do v
            return first(_exact_variable_elimination(fg, [v], evidence, backend.order))
        end
    end
end

# The factor that puts all mass on `label` of `ax`.
function _point_mass(::Type{T}, ax::FiniteAxis, label::Symbol) where {T}
    table = zeros(T, length(ax))
    table[label_index(ax, label)] = one(T)
    return _factor([ax.name], [ax], table)
end
