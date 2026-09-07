# Elimination-order strategies backed by CliqueTrees.jl.

"""
    EliminationStrategy

Abstract supertype of the strategies accepted by [`elimination_order`](@ref)
and [`treewidth`](@ref): [`MinFill`](@ref), [`MinDegree`](@ref),
[`ExactTreewidth`](@ref), [`AMDOrder`](@ref) and [`UserOrder`](@ref). Every
strategy but [`UserOrder`](@ref) delegates to a fill-reducing algorithm of
`CliqueTrees.jl` [CliqueTrees](@cite).
"""
abstract type EliminationStrategy end

"""
    MinFill()

Greedy minimum-fill ordering (`CliqueTrees.MF()`). The default strategy.
"""
struct MinFill <: EliminationStrategy end

"""
    MinDegree()

Multiple minimum-degree ordering (`CliqueTrees.MMD()`).
"""
struct MinDegree <: EliminationStrategy end

"""
    ExactTreewidth()

Exact minimum-treewidth ordering by the Bouchitte-Todinca minimal-separator
algorithm (`CliqueTrees.BT()` of [CliqueTrees](@cite), provided by
`TreeWidthSolver.jl`). Exponential in the
number of variables; use it on small graphs and as a reference in tests.
"""
struct ExactTreewidth <: EliminationStrategy end

"""
    AMDOrder()

Approximate minimum-degree (AMD) ordering (`CliqueTrees.AMD()` of
[CliqueTrees](@cite), backed by SuiteSparse through `AMD.jl`).
"""
struct AMDOrder <: EliminationStrategy end

"""
    UserOrder(vars::Vector{Symbol})

An explicit elimination order. [`elimination_order`](@ref) checks that it
lists every variable of the graph outside `keep` exactly once and throws
[`ScopeError`](@ref) otherwise.
"""
struct UserOrder <: EliminationStrategy
    vars::Vector{Symbol}
    function UserOrder(vars::AbstractVector{Symbol})
        allunique(vars) ||
            throw(ScopeError(:UserOrder, "elimination order repeats variables",
                             collect(vars)))
        return new(collect(Symbol, vars))
    end
end

_algorithm(::MinFill) = CliqueTrees.MF()
_algorithm(::MinDegree) = CliqueTrees.MMD()
_algorithm(::ExactTreewidth) = CliqueTrees.BT()
_algorithm(::AMDOrder) = CliqueTrees.AMD()

function _keep_indices(op::Symbol, index::Dict{Symbol,Int}, keep)
    allunique(keep) || throw(ScopeError(op, "kept variables repeat", collect(keep)))
    missing_vars = [v for v in keep if !haskey(index, v)]
    isempty(missing_vars) ||
        throw(ScopeError(op, "kept variables are not variables of the graph", missing_vars))
    return Int[index[v] for v in keep]
end

# Order the vertices of `graph` (labelled by `vars`) for elimination, returning
# the variables to eliminate with those in `keep` excluded.
function _elimination_order(graph::SimpleGraph, vars::Vector{Symbol},
                            index::Dict{Symbol,Int}, strategy::EliminationStrategy,
                            keep)
    keep_idx = _keep_indices(:elimination_order, index, keep)
    isempty(vars) && return Symbol[]
    order = _permutation(graph, keep_idx, strategy)
    # CompositeRotations was verified to place the kept vertices last; the
    # filter below makes the result independent of that guarantee.
    keepset = Set(keep_idx)
    return Symbol[vars[i] for i in order if !(i in keepset)]
end

# Vertex order of `graph` with the vertices in `keep_idx` forced to the end:
# `CliqueTrees.permutation(graph; alg=CompositeRotations(keep_idx, alg))`.
function _permutation(graph::SimpleGraph, keep_idx::Vector{Int},
                      strategy::EliminationStrategy)
    return _rotated_permutation(graph, keep_idx, _algorithm(strategy))
end

function _rotated_permutation(graph::SimpleGraph, keep_idx::Vector{Int},
                              alg::CliqueTrees.EliminationAlgorithm)
    isempty(keep_idx) || (alg = CliqueTrees.CompositeRotations(keep_idx, alg))
    order, _ = CliqueTrees.permutation(graph; alg=alg)
    return order
end

# TreeWidthSolver's BT fails on graphs with isolated vertices, so the exact
# strategy is applied one connected component at a time (the components are
# independent, so any interleaving of their orders has the same width).
function _permutation(graph::SimpleGraph, keep_idx::Vector{Int}, strategy::ExactTreewidth)
    keepset = Set(keep_idx)
    order = Int[]
    for comp in Graphs.connected_components(graph)
        if length(comp) == 1
            push!(order, comp[1])
            continue
        end
        sub, vmap = Graphs.induced_subgraph(graph, comp)
        local_keep = Int[i for (i, v) in enumerate(vmap) if v in keepset]
        append!(order, vmap[_rotated_permutation(sub, local_keep, _algorithm(strategy))])
    end
    # Kept vertices last overall, preserving their relative order.
    return vcat(filter(i -> !(i in keepset), order), filter(i -> i in keepset, order))
end

function _elimination_order(graph::SimpleGraph, vars::Vector{Symbol},
                            index::Dict{Symbol,Int}, strategy::UserOrder, keep)
    _keep_indices(:elimination_order, index, keep)
    keepset = Set(keep)
    order = strategy.vars
    bad = [v for v in order if !haskey(index, v)]
    isempty(bad) ||
        throw(ScopeError(:elimination_order, "user order names unknown variables", bad))
    kept = [v for v in order if v in keepset]
    isempty(kept) ||
        throw(ScopeError(:elimination_order, "user order eliminates kept variables", kept))
    absent = [v for v in vars if !(v in keepset) && !(v in order)]
    isempty(absent) ||
        throw(ScopeError(:elimination_order, "user order omits variables", absent))
    return copy(order)
end

"""
    elimination_order(fg::FactorGraph, strategy=MinFill(); keep=Symbol[]) -> Vector{Symbol}

The variables of `fg` in the order in which `strategy` eliminates them,
excluding `keep` (the query variables, which are forced to the end of the
underlying `CliqueTrees.permutation` through `CompositeRotations(keep, alg)`
and then dropped). Throws [`ScopeError`](@ref) if `keep` names unknown or
repeated variables, or if a [`UserOrder`](@ref) is not a permutation of the
remaining variables.
"""
function elimination_order(fg::FactorGraph, strategy::EliminationStrategy=MinFill();
                           keep::AbstractVector{Symbol}=Symbol[])
    graph, vars, index = interaction_graph(fg)
    return _elimination_order(graph, vars, index, strategy, keep)
end

"""
    treewidth(fg::FactorGraph, strategy=MinFill()) -> Int

The width of the elimination order that `strategy` produces on the
interaction graph of `fg` (one less than the largest clique of the induced
chordal graph); the treewidth itself when `strategy = ExactTreewidth()`.
Extends `CliqueTrees.treewidth`. A graph without variables has width `-1`,
following CliqueTrees.
"""
function CliqueTrees.treewidth(fg::FactorGraph, strategy::EliminationStrategy=MinFill())
    graph, vars, index = interaction_graph(fg)
    return _treewidth(graph, vars, index, strategy)
end

function _treewidth(graph::SimpleGraph, vars, index, strategy::EliminationStrategy)
    return CliqueTrees.treewidth(graph; alg=_algorithm(strategy))
end
function _treewidth(graph::SimpleGraph, vars, index, strategy::UserOrder)
    order = _elimination_order(graph, vars, index, strategy, Symbol[])
    return CliqueTrees.treewidth(graph; alg=Int[index[v] for v in order])
end
function _treewidth(graph::SimpleGraph, vars, index, strategy::ExactTreewidth)
    Graphs.nv(graph) == 0 && return -1
    width = 0
    for comp in Graphs.connected_components(graph)
        length(comp) == 1 && continue
        sub, _ = Graphs.induced_subgraph(graph, comp)
        width = max(width, CliqueTrees.treewidth(sub; alg=_algorithm(strategy)))
    end
    return width
end
