# Sum-product belief propagation on the factor graph (SPEC section 20):
# exact on tree-structured graphs, loopy with convergence diagnostics otherwise.

"""
    BeliefPropagation(; damping=0.0, tol=1e-8, maxiter=200, schedule=:flooding, check_evidence=false)

Sum-product message passing on the factor graph (see
[`belief_propagation`](@ref)). Messages are damped towards their previous
value with weight `damping` (in `[0, 1)`), iteration stops when the largest
undamped message-equation residual at the returned iterate is below `tol` or after `maxiter`
sweeps, and `schedule` is `:flooding` (every message updated from the
previous sweep's messages) or `:sequential` (factor by factor, each update
seeing the latest messages). A fixed point on a feasible tree has the exact
marginals; a finite tolerance is not itself a marginal-error bound.
`check_evidence=true` first uses exact variable elimination to check global
evidence feasibility, potentially at exponential cost. Without that opt-in,
local zero-support errors are detected but global feasibility is not certified.
[`BPDiagnostics`](@ref) reports both convergence and whether evidence was checked.
Throws `ArgumentError` for parameters outside these ranges.
"""
struct BeliefPropagation <: InferenceBackend
    damping::Float64
    tol::Float64
    maxiter::Int
    schedule::Symbol
    check_evidence::Bool
    function BeliefPropagation(damping::Real, tol::Real, maxiter::Integer,
                               schedule::Symbol, check_evidence::Bool)
        0 <= damping < 1 ||
            throw(ArgumentError("damping must lie in [0, 1), got damping = $damping"))
        isfinite(tol) && tol > 0 ||
            throw(ArgumentError("tol must be finite and positive, got tol = $tol"))
        maxiter >= 1 ||
            throw(ArgumentError("maxiter must be at least 1, got maxiter = $maxiter"))
        schedule in (:flooding, :sequential) ||
            throw(ArgumentError("schedule must be :flooding or :sequential, got schedule = $(repr(schedule))"))
        return new(damping, tol, maxiter, schedule, check_evidence)
    end
end
function BeliefPropagation(damping::Real, tol::Real, maxiter::Integer, schedule::Symbol)
    return BeliefPropagation(damping, tol, maxiter, schedule, false)
end
function BeliefPropagation(; damping::Real=0.0, tol::Real=1e-8, maxiter::Integer=200,
                           schedule::Symbol=:flooding, check_evidence::Bool=false)
    return BeliefPropagation(damping, tol, maxiter, schedule, check_evidence)
end

"""
    BPDiagnostics(iterations, converged, max_residual, tree, evidence_checked=false)

What a run of [`belief_propagation`](@ref) did: the number of sweeps
performed, whether the undamped message-equation residual fell below the
tolerance, that residual at the returned iterate, whether the conditioned
factor graph was a tree, and whether global evidence feasibility was checked.

The field is `tree`, not `exact`, because being a tree is a property of the
graph while exactness is a property of the iterate. On a tree the fixed
point of the message equations gives exact marginals for feasible evidence.
`max_residual` is not a general bound on marginal error. In particular,
`converged` does not certify global evidence feasibility: unless
`evidence_checked` is true, that status is unknown.
"""
struct BPDiagnostics
    iterations::Int
    converged::Bool
    max_residual::Float64
    tree::Bool
    evidence_checked::Bool
end
function BPDiagnostics(iterations::Integer, converged::Bool, max_residual::Real, tree::Bool)
    return BPDiagnostics(iterations, converged, max_residual, tree, false)
end

"""
    is_tree(fg::FactorGraph) -> Bool

Whether the bipartite factor graph (one node per variable, one per factor,
an edge for every scope membership) has no cycles, i.e. is a forest. On such
a graph undamped messages reach the exact fixed point after sufficiently many
sweeps when evidence is feasible. A compiled Bayesian network is a tree here exactly
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
# variable `x`. `evidence` is carried so that a zero message or belief can report it.
struct _BPState{T}
    factors::Vector{Factor{T}}
    vars::Vector{Symbol}
    axes::Vector{FiniteAxis}
    incidences::Vector{Vector{Tuple{Int,Int}}}
    to_var::Vector{Vector{Vector{T}}}      # to_var[f][k]: factor f to variable at position k
    to_factor::Vector{Vector{Vector{T}}}   # to_factor[f][k]: that variable to factor f
    evidence::Dict{Symbol,Symbol}
end

function _BPState(factors::Vector{Factor{T}}, axes::Dict{Symbol,FiniteAxis},
                  evidence::Dict{Symbol,Symbol}) where {T}
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
                       to_factor, evidence)
end

# Normalise a message in place. An all-zero message means that the supports of
# the incoming messages and of the factor do not intersect, i.e. the
# configuration has probability zero under the evidence; that is a hard
# failure, reported exactly as variable elimination and the junction tree
# report impossible evidence: `ImpossibleEvidenceError` (ADR 0012).
function _normalize!(m::Vector, evidence)
    s = sum(m)
    _require_evidence_mass(s, evidence)
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
    return _normalize!(m, st.evidence)
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

# Update the messages out of factor `f`; convergence is measured separately.
function _update_factor!(st::_BPState{T}, f::Int, damping) where {T}
    fac = st.factors[f]
    residual = 0.0
    for k in 1:ndims(fac)
        new = _normalize!(_factor_message(fac, st.to_factor[f], k), st.evidence)
        old = st.to_var[f][k]
        residual = max(residual, maximum(abs.(new .- old)))
        iszero(damping) || (new .= (1 - damping) .* new .+ damping .* old)
        st.to_var[f][k] = new
    end
    return residual
end

function _fixed_point_residual!(st::_BPState)
    for (x, inc) in enumerate(st.incidences), (f, k) in inc
        st.to_factor[f][k] = _variable_message(st, x, f, k)
    end
    residual = 0.0
    for (f, fac) in enumerate(st.factors), k in 1:ndims(fac)
        target = _normalize!(_factor_message(fac, st.to_factor[f], k), st.evidence)
        residual = max(residual, maximum(abs.(target .- st.to_var[f][k])))
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
    return _posterior_normalize(_factor([st.vars[x]], [st.axes[x]], m), st.evidence)
end

"""
    belief_propagation(fg::FactorGraph, backend=BeliefPropagation(); evidence=Dict{Symbol,Symbol}())
        -> (marginals::Dict{Symbol,Factor}, diagnostics::BPDiagnostics)

Sum-product belief propagation on `fg` conditioned on `evidence`. Factors
are conditioned first; a factor left with an empty scope only scales the
joint, so it is dropped from the message passing after its value has been
multiplied into the scale `P(evidence)` contributed by such factors.
Messages use a floating-point type (integer factors are promoted), start
uniform and follow the sum-product equations of SPEC section 20 until the
largest undamped message-equation residual at the current iterate is
below `backend.tol` or `backend.maxiter` sweeps have run. The marginal of
every unobserved variable is the normalised product of its incoming
messages; an observed variable maps to the point mass at its label. On a
tree-structured graph ([`is_tree`](@ref) after conditioning) the fixed point
is the exact marginal and `diagnostics.tree` is true; otherwise the
marginals are the loopy approximation and `diagnostics.converged` says
whether the residual tolerance was reached, not a bound on marginal error.

The update equations are Pearl's belief propagation [Pearl1988](@cite) in
the factor-graph (sum-product) form of [Kschischang2001](@cite); the
behaviour of the loopy fixed point on graphs with cycles is the empirical
study of [MurphyWeissJordan1999](@cite).

Detectable impossible evidence is an error: when a dropped scalar factor is
zero, when a message is identically zero, or when the belief of a variable
has zero mass, `BayesianNetworks.ImpossibleEvidenceError` is thrown rather
than a normalised belief returned, as variable elimination and the junction
tree do. Local support is not a global feasibility test on loopy graphs,
and damped iterates may retain tiny positive support. Set
`backend.check_evidence=true` to run an exact VE feasibility check first
(it throws the same error for zero evidence mass);
`diagnostics.evidence_checked` records that opt-in.

Throws [`ScopeError`](@ref) for evidence on unknown variables and
`FiniteKernels.InvalidAxisError` for a label a variable does not have.
"""
function belief_propagation(fg::FactorGraph{T},
                            backend::BeliefPropagation=BeliefPropagation();
                            evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}()) where {T}
    _check_query(fg, Symbol[], evidence)
    ev = Dict{Symbol,Symbol}(evidence)
    if backend.check_evidence
        mass = variable_elimination(fg, Symbol[]; evidence=ev)[1].table[]
        _require_evidence_mass(mass, ev)
    end
    R = typeof(float(one(T)))
    factors = Factor{R}[]
    scale = one(R)
    for f in fg.factors
        g = _convert_factor(R, condition(f, ev))
        isempty(g.vars) ? (scale *= g.table[]) : push!(factors, g)
    end
    # a factor conditioned to the empty scope is zero: the evidence has probability zero
    _require_evidence_mass(scale, ev)
    st = _BPState(factors, fg.axes, ev)
    iterations = 0
    residual = Inf
    converged = false
    while iterations < backend.maxiter
        _sweep!(st, backend.damping, backend.schedule)
        residual = _fixed_point_residual!(st)
        iterations += 1
        if residual < backend.tol
            converged = true
            break
        end
    end
    marginals = Dict{Symbol,Factor{R}}()
    for (x, v) in enumerate(st.vars)
        marginals[v] = _belief(st, x)
    end
    for (v, l) in ev
        marginals[v] = _point_mass(R, fg.axes[v], l)
    end
    return marginals,
           BPDiagnostics(iterations, converged, residual, _is_forest(factors),
                         backend.check_evidence)
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
