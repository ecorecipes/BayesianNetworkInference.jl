# Variable elimination and the brute-force oracle it is tested against.

"""
    InferenceBackend

Abstract supertype of inference backends accepted by [`infer`](@ref).
"""
abstract type InferenceBackend end

"""
    VariableElimination(; order=MinFill())

Exact inference by variable elimination, with the elimination order chosen by
an [`EliminationStrategy`](@ref).
"""
struct VariableElimination{O<:EliminationStrategy} <: InferenceBackend
    order::O
end
VariableElimination(; order::EliminationStrategy=MinFill()) = VariableElimination(order)

"""
    InferenceDiagnostics(order, max_factor_size, n_multiplications, treewidth)

What a run of [`variable_elimination`](@ref) did: the elimination `order`
used, the number of entries of the largest intermediate factor, the number of
pairwise factor products, and the width of the elimination order (largest
intermediate scope minus one, which equals the treewidth of the order on the
interaction graph of the conditioned factors).
"""
struct InferenceDiagnostics
    order::Vector{Symbol}
    max_factor_size::Int
    n_multiplications::Int
    treewidth::Int
end

function _check_query(fg::FactorGraph, query, evidence)
    allunique(query) || throw(ScopeError(:infer, "query variables repeat", collect(query)))
    unknown = [v for v in query if !haskey(fg.axes, v)]
    isempty(unknown) ||
        throw(ScopeError(:infer, "query variables are not variables of the factor graph",
                         unknown))
    unknown = [v for v in keys(evidence) if !haskey(fg.axes, v)]
    isempty(unknown) ||
        throw(ScopeError(:infer, "evidence variables are not variables of the factor graph",
                         unknown))
    both = [v for v in query if haskey(evidence, v)]
    isempty(both) ||
        throw(ScopeError(:infer, "query variables cannot also carry evidence", both))
    return nothing
end

# Strategy restricted to the variables that survive conditioning.
_restrict(strategy::EliminationStrategy, evidence) = strategy
function _restrict(strategy::UserOrder, evidence)
    return UserOrder([v for v in strategy.vars if !haskey(evidence, v)])
end

"""
    variable_elimination(fg::FactorGraph, query; evidence=Dict{Symbol,Symbol}(), order=MinFill())
        -> (posterior::Factor, diagnostics::InferenceDiagnostics)

The posterior `P(query | evidence)` as a normalised factor whose scope is
`query` in the given order, computed by conditioning every factor on the
evidence, eliminating the remaining non-query variables in the order chosen
by `order` (an [`EliminationStrategy`](@ref)), multiplying what is left and
normalising. With an empty `query` the returned scalar factor is the
unnormalised probability of the evidence, `P(evidence)`.

The algorithm is bucket elimination [Dechter1999](@cite) in the factor
presentation of [KollerFriedman2009](@cite) (chapter 9); pushing each sum
past the factors that do not mention the eliminated variable is the
optimisation of [ZhangPoole1994](@cite).

Throws [`ScopeError`](@ref) for unknown or repeated query variables, unknown
evidence variables, or a query variable that also carries evidence.
"""
function variable_elimination(fg::FactorGraph{T}, query::AbstractVector{Symbol};
                              evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
                              order::EliminationStrategy=MinFill()) where {T}
    _check_query(fg, query, evidence)
    factors = Factor{T}[condition(f, evidence) for f in fg.factors]
    vars = _variables(factors)
    graph, vars, index = _interaction_graph(factors, vars)
    elim = _elimination_order(graph, vars, index, _restrict(order, evidence), query)
    max_size = 0
    n_mult = 0
    width = 0
    for v in elim
        touching = Factor{T}[]
        rest = Factor{T}[]
        for f in factors
            push!(v in f.vars ? touching : rest, f)
        end
        prod = _product!(touching)
        n_mult += max(length(touching) - 1, 0)
        max_size = max(max_size, length(prod))
        width = max(width, ndims(prod) - 1)
        push!(rest, marginalize(prod, v))
        factors = rest
    end
    result = _product!(factors)
    n_mult += max(length(factors) - 1, 0)
    max_size = max(max_size, length(result))
    width = max(width, ndims(result) - 1)
    result = reorder(result, query)
    isempty(query) || (result = normalize(result))
    return result, InferenceDiagnostics(elim, max_size, n_mult, width)
end
function variable_elimination(fg::FactorGraph, query::Symbol; kwargs...)
    return variable_elimination(fg, [query]; kwargs...)
end

# Product of a list of factors, multiplying the smallest scopes first.
function _product!(fs::Vector{<:Factor})
    isempty(fs) && return unit_factor()
    sort!(fs; by=ndims)
    return reduce(multiply, fs)
end

"""
    infer(fg::FactorGraph, query; evidence=Dict{Symbol,Symbol}(), backend=VariableElimination())
        -> (posterior::Factor, diagnostics)

Generic inference entry point: `P(query | evidence)` computed by `backend`.
`query` is a vector of variables (or a single symbol) and `evidence` maps
observed variables to labels. With [`VariableElimination`](@ref) this is
[`variable_elimination`](@ref) with the backend's ordering strategy.

The model bridge adds the method `infer(model::BayesModel, query; evidence,
backend)` that compiles the model to a [`FactorGraph`](@ref) first; this
method is the one it delegates to.
"""
function infer(fg::FactorGraph, query::AbstractVector{Symbol};
               evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
               backend::InferenceBackend=VariableElimination())
    return _infer(backend, fg, query, evidence)
end
infer(fg::FactorGraph, query::Symbol; kwargs...) = infer(fg, [query]; kwargs...)

function _infer(backend::VariableElimination, fg::FactorGraph, query, evidence)
    return variable_elimination(fg, query; evidence, order=backend.order)
end

# Brute-force oracle
# ------------------

"""
    joint_factor(fg::FactorGraph) -> Factor

The product of every factor of `fg`: the full (unnormalised) joint table, with
scope in [`variables`](@ref) order. Exponential in the number of variables;
this is the oracle that variable elimination is tested against.
"""
function joint_factor(fg::FactorGraph)
    isempty(fg.factors) && return unit_factor(eltype(fg))
    return reorder(multiply(fg.factors), variables(fg))
end

"""
    brute_force_marginal(fg::FactorGraph, query; evidence=Dict{Symbol,Symbol}()) -> Factor

`P(query | evidence)` by conditioning and summing the full joint table.
Same conventions as [`variable_elimination`](@ref) (including the
unnormalised `P(evidence)` for an empty query), without diagnostics.
"""
function brute_force_marginal(fg::FactorGraph, query::AbstractVector{Symbol};
                              evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}())
    _check_query(fg, query, evidence)
    j = condition(joint_factor(fg), evidence)
    m = reorder(marginalize(j, setdiff(j.vars, query)), query)
    return isempty(query) ? m : normalize(m)
end
function brute_force_marginal(fg::FactorGraph, query::Symbol; kwargs...)
    return brute_force_marginal(fg, [query]; kwargs...)
end
