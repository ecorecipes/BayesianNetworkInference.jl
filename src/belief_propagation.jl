# Sum-product belief propagation on the factor graph (SPEC section 20):
# exact on tree-structured graphs, loopy with convergence diagnostics otherwise.

"""
    BeliefPropagation(; damping=0.0, tol=1e-8, maxiter=200, schedule=:flooding)

Sum-product message passing on the factor graph (see
[`belief_propagation`](@ref)). Messages are damped towards their previous
value with weight `damping` (in `[0, 1)`), iteration stops when the largest
change of any factor-to-variable message is below `tol` or after `maxiter`
sweeps, and `schedule` is `:flooding` (every message updated from the
previous sweep's messages) or `:sequential` (factor by factor, each update
seeing the latest messages). The result is exact when the factor graph
conditioned on the evidence is a tree ([`is_tree`](@ref)) and approximate
(loopy belief propagation) otherwise; [`BPDiagnostics`](@ref) reports
convergence. Throws `ArgumentError` for parameters outside these ranges.
"""
struct BeliefPropagation <: InferenceBackend
    damping::Float64
    tol::Float64
    maxiter::Int
    schedule::Symbol
    function BeliefPropagation(damping::Real, tol::Real, maxiter::Integer, schedule::Symbol)
        0 <= damping < 1 ||
            throw(ArgumentError("damping must lie in [0, 1), got damping = $damping"))
        tol > 0 || throw(ArgumentError("tol must be positive, got tol = $tol"))
        maxiter >= 1 ||
            throw(ArgumentError("maxiter must be at least 1, got maxiter = $maxiter"))
        schedule in (:flooding, :sequential) ||
            throw(ArgumentError("schedule must be :flooding or :sequential, got schedule = $(repr(schedule))"))
        return new(damping, tol, maxiter, schedule)
    end
end
function BeliefPropagation(; damping::Real=0.0, tol::Real=1e-8, maxiter::Integer=200,
                           schedule::Symbol=:flooding)
    return BeliefPropagation(damping, tol, maxiter, schedule)
end

"""
    BPDiagnostics(iterations, converged, max_residual, tree)

What a run of [`belief_propagation`](@ref) did: the number of sweeps
performed, whether the largest change of a factor-to-variable message fell
below the tolerance, that largest change at the last sweep, and whether the
factor graph conditioned on the evidence was a tree.

The field is `tree`, not `exact`, because being a tree is a property of the
graph while exactness is a property of the iterate. On a tree the fixed
point of the message equations is the exact marginal, so `tree && converged`
means the returned marginals are exact up to the convergence tolerance: they
differ from the fixed point by an amount of the order of `max_residual`,
which with `damping > 0` is `backend.tol` rather than machine precision.
"""
struct BPDiagnostics
    iterations::Int
    converged::Bool
    max_residual::Float64
    tree::Bool
end

"""
    is_tree(fg::FactorGraph) -> Bool

Whether the bipartite factor graph (one node per variable, one per factor,
an edge for every scope membership) has no cycles, i.e. is a forest. On such
a graph [`belief_propagation`](@ref) is exact after a number of sweeps
bounded by the diameter. A compiled Bayesian network is a tree here exactly
when its DAG is a polytree. Extends `Graphs.is_tree`.
"""
Graphs.is_tree(fg::FactorGraph) = _is_forest(fg.factors)

function _is_forest(factors::AbstractVector{<:Factor})
    vars = _variables(factors)
    index = Dict{Symbol,Int}(v => i for (i, v) in enumerate(vars))
    nv = length(vars)
    g = SimpleGraph(nv + length(factors))
    for (j, f) in enumerate(factors), v in f.vars
        Graphs.add_edge!(g, index[v], nv + j)
    end
    return Graphs.ne(g) == Graphs.nv(g) - length(Graphs.connected_components(g))
end

# Message state of one run: factor `f` has one message in each direction for
# every position `k` of its scope; `incidences[x]` lists the `(f, k)` pairs of
# variable `x`.
struct _BPState{T}
    factors::Vector{Factor{T}}
    vars::Vector{Symbol}
    axes::Vector{FiniteAxis}
    incidences::Vector{Vector{Tuple{Int,Int}}}
    to_var::Vector{Vector{Vector{T}}}      # to_var[f][k]: factor f to variable at position k
    to_factor::Vector{Vector{Vector{T}}}   # to_factor[f][k]: that variable to factor f
end

function _BPState(factors::Vector{Factor{T}}, axes::Dict{Symbol,FiniteAxis}) where {T}
    vars = _variables(factors)
    index = Dict{Symbol,Int}(v => i for (i, v) in enumerate(vars))
    incidences = [Tuple{Int,Int}[] for _ in vars]
    for (f, fac) in enumerate(factors), (k, v) in enumerate(fac.vars)
        push!(incidences[index[v]], (f, k))
    end
    uniform(fac, k) = fill(one(T) / size(fac.table, k), size(fac.table, k))
    to_var = [[uniform(fac, k) for k in 1:ndims(fac)] for fac in factors]
    to_factor = [[uniform(fac, k) for k in 1:ndims(fac)] for fac in factors]
    return _BPState{T}(factors, vars, FiniteAxis[axes[v] for v in vars], incidences, to_var,
                       to_factor)
end

# Normalise a message in place. An all-zero message means that the supports of
# the incoming messages and of the factor do not intersect, i.e. the
# configuration has probability zero under the evidence; that is a hard
# failure, reported exactly as variable elimination and the junction tree
# report impossible evidence.
function _normalize!(m::Vector, what::AbstractString)
    s = sum(m)
    iszero(s) &&
        throw(KernelNormalizationError("belief propagation cannot normalise the $what: it is identically zero, so the evidence has probability zero",
                                       1.0, 0.0))
    m ./= s
    return m
end

# m_{x -> f}(x) = prod_{g in N(x) \ f} m_{g -> x}(x), for the incidence (f, k) of x.
function _variable_message(st::_BPState{T}, x::Int, f::Int, k::Int) where {T}
    m = ones(T, length(st.axes[x]))
    for (g, l) in st.incidences[x]
        (g == f && l == k) && continue
        m .*= st.to_var[g][l]
    end
    return _normalize!(m, "message from variable $(repr(st.vars[x])) to factor $f")
end

# m_{f -> x}(x) = sum_{x_f \ x} f(x_f) prod_{y != x} m_{y -> f}(y), x at position k.
function _factor_message(fac::Factor{T}, incoming::Vector{Vector{T}}, k::Int) where {T}
    nd = ndims(fac)
    nd == 1 && return copy(vec(fac.table))
    sz = size(fac)
    t = fac.table
    for j in 1:nd
        j == k && continue
        shape = ntuple(i -> i == j ? sz[j] : 1, nd)
        t = t .* reshape(incoming[j], shape)
    end
    dims = Tuple(j for j in 1:nd if j != k)
    return vec(sum(t; dims=dims))
end

# Update the messages out of factor `f`, returning the largest change.
function _update_factor!(st::_BPState{T}, f::Int, damping) where {T}
    fac = st.factors[f]
    residual = 0.0
    for k in 1:ndims(fac)
        new = _normalize!(_factor_message(fac, st.to_factor[f], k),
                          "message from factor $f to variable $(repr(fac.vars[k]))")
        old = st.to_var[f][k]
        iszero(damping) || (new .= (1 - damping) .* new .+ damping .* old)
        residual = max(residual, maximum(abs.(new .- old)))
        st.to_var[f][k] = new
    end
    return residual
end

function _sweep!(st::_BPState, damping, schedule::Symbol)
    residual = 0.0
    if schedule === :flooding
        for (x, inc) in enumerate(st.incidences), (f, k) in inc
            st.to_factor[f][k] = _variable_message(st, x, f, k)
        end
        for f in eachindex(st.factors)
            residual = max(residual, _update_factor!(st, f, damping))
        end
    else
        index = Dict{Symbol,Int}(v => i for (i, v) in enumerate(st.vars))
        for f in eachindex(st.factors)
            for (k, v) in enumerate(st.factors[f].vars)
                st.to_factor[f][k] = _variable_message(st, index[v], f, k)
            end
            residual = max(residual, _update_factor!(st, f, damping))
        end
    end
    return residual
end

function _belief(st::_BPState{T}, x::Int) where {T}
    m = ones(T, length(st.axes[x]))
    for (g, l) in st.incidences[x]
        m .*= st.to_var[g][l]
    end
    return normalize(_factor([st.vars[x]], [st.axes[x]], m))
end

"""
    belief_propagation(fg::FactorGraph, backend=BeliefPropagation(); evidence=Dict{Symbol,Symbol}())
        -> (marginals::Dict{Symbol,Factor}, diagnostics::BPDiagnostics)

Sum-product belief propagation on `fg` conditioned on `evidence`. Factors
are conditioned first; a factor left with an empty scope only scales the
joint, so it is dropped from the message passing after its value has been
multiplied into the scale `P(evidence)` contributed by such factors.
Messages start uniform and are updated by the sum-product equations of
SPEC section 20 until the largest change of a factor-to-variable message is
below `backend.tol` or `backend.maxiter` sweeps have run. The marginal of
every unobserved variable is the normalised product of its incoming
messages; an observed variable maps to the point mass at its label. On a
tree-structured graph ([`is_tree`](@ref) after conditioning) the fixed point
is the exact marginal and `diagnostics.tree` is true; otherwise the
marginals are the loopy approximation and `diagnostics.converged` says
whether the fixed point was reached.

The update equations are Pearl's belief propagation [Pearl1988](@cite) in
the factor-graph (sum-product) form of [Kschischang2001](@cite); the
behaviour of the loopy fixed point on graphs with cycles is the empirical
study of [MurphyWeissJordan1999](@cite).

Impossible evidence is an error, as it is for [`variable_elimination`](@ref)
and the [`JunctionTree`](@ref) backend: when a dropped scalar factor is
zero, when a message is identically zero, or when the belief of a variable
has zero mass, `FiniteKernels.KernelNormalizationError` is thrown rather
than a normalised belief returned.

Throws [`ScopeError`](@ref) for evidence on unknown variables and
`FiniteKernels.InvalidAxisError` for a label a variable does not have.
"""
function belief_propagation(fg::FactorGraph{T},
                            backend::BeliefPropagation=BeliefPropagation();
                            evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}()) where {T}
    _check_query(fg, Symbol[], evidence)
    ev = Dict{Symbol,Symbol}(evidence)
    factors = Factor{T}[]
    scale = one(T)
    for f in fg.factors
        g = condition(f, ev)
        isempty(g.vars) ? (scale *= g.table[]) : push!(factors, g)
    end
    iszero(scale) &&
        throw(KernelNormalizationError("belief propagation: the evidence $(sort!(collect(ev), by=first)) has probability zero (a factor conditioned to the empty scope is zero)",
                                       1.0, 0.0))
    st = _BPState(factors, fg.axes)
    iterations = 0
    residual = Inf
    converged = false
    while iterations < backend.maxiter
        residual = _sweep!(st, backend.damping, backend.schedule)
        iterations += 1
        if residual < backend.tol
            converged = true
            break
        end
    end
    marginals = Dict{Symbol,Factor{T}}()
    for (x, v) in enumerate(st.vars)
        marginals[v] = _belief(st, x)
    end
    for (v, l) in ev
        marginals[v] = _point_mass(T, fg.axes[v], l)
    end
    return marginals, BPDiagnostics(iterations, converged, residual, _is_forest(factors))
end

function _infer(backend::BeliefPropagation, fg::FactorGraph, query, evidence)
    _check_query(fg, query, evidence)
    length(query) == 1 ||
        throw(ScopeError(:infer,
                         "BeliefPropagation answers single-variable queries only; use VariableElimination or JunctionTree for a joint query",
                         collect(query)))
    marginals, diagnostics = belief_propagation(fg, backend; evidence)
    return marginals[query[1]], diagnostics
end

function _all_marginals(backend::BeliefPropagation, fg::FactorGraph, evidence)
    return belief_propagation(fg, backend; evidence)[1]
end
