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
    log_fallback::Bool
end
function InferenceDiagnostics(order::AbstractVector{Symbol}, max_factor_size::Integer,
                              n_multiplications::Integer, treewidth::Integer)
    return InferenceDiagnostics(order, max_factor_size, n_multiplications, treewidth, false)
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

# Evidence mass (ADR 0012, ADR 0014). A binary64 mass is trusted only when it is a normal
# positive number. Zero, a subnormal, a negative or a non-finite mass does not say whether
# the evidence is impossible -- a positive probability that underflowed is zero too -- so
# the binary64 path throws the internal `_UnresolvedMass` signal (src/errors.jl) instead of deciding. Every
# public entry point catches it (`_resolving_mass`) and recomputes in the log domain, where
# only an exact zero gives `-Inf` and so `ImpossibleEvidenceError`. Exact element types
# (integers, rationals) decide directly.

_impossible(evidence) = throw(ImpossibleEvidenceError(Dict{Symbol,Symbol}(evidence)))

function _require_evidence_mass(mass::AbstractFloat, evidence)
    (isfinite(mass) && mass >= floatmin(typeof(mass))) && return nothing
    return throw(_UnresolvedMass(Dict{Symbol,Symbol}(evidence)))
end
function _require_evidence_mass(mass::Real, evidence)
    mass > 0 && return nothing
    mass == 0 && _impossible(evidence)
    return throw(IndeterminatePosteriorError(Dict{Symbol,Symbol}(evidence),
                                             "the evidence mass is negative ($(mass))"))
end

# Normalise a posterior factor after checking its mass, so that posterior code never
# reaches the `ArgumentError` of `normalize(::Factor)`. A negative posterior cell can come
# only from tolerated entries in [-atol, 0); its sign is an artefact of the rounding.
function _posterior_normalize(f::Factor, evidence)
    _require_evidence_mass(sum(f.table), evidence)
    p = normalize(f)
    v = minimum(p.table; init=zero(eltype(p.table)))
    v < 0 &&
        throw(IndeterminatePosteriorError(Dict{Symbol,Symbol}(evidence),
                                          "a posterior cell is negative ($(v))"))
    return p
end

# Run `ordinary`; if its binary64 mass was not trustworthy, run `fallback` (a log-domain
# computation of the same answer) instead. A tolerated negative entry is outside the log
# domain, and with a mass that small the posterior is indeterminate. Called with a `do`
# block, which Julia passes first: `_resolving_mass(() -> ordinary, evidence) do ... end`.
function _resolving_mass(fallback, ordinary, evidence)
    try
        return ordinary()
    catch e
        e isa _UnresolvedMass || rethrow()
    end
    try
        return fallback()
    catch e
        (e isa FactorDomainError && e.backend === :log_domain && isfinite(e.value) &&
         e.value < 0) || rethrow()
        throw(IndeterminatePosteriorError(Dict{Symbol,Symbol}(evidence),
                                          "the binary64 evidence mass is below the normal range and a tolerated negative entry ($(e.value)) has no logarithm"))
    end
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
evidence variables, or a query variable that also carries evidence, and
`BayesianNetworks.ImpossibleEvidenceError` for a non-empty `query` when the
evidence has probability exactly zero. A binary64 mass that is zero, subnormal or
non-finite does not decide that -- a positive probability can underflow -- so the
posterior is then recomputed by [`log_variable_elimination`](@ref) and returned, with
`diagnostics.log_fallback == true` (ADR 0014). A model with tolerated entries in
`[-atol, 0)` raises `BayesianNetworks.IndeterminatePosteriorError` when a posterior cell
comes out negative or the log domain meets such an entry.
"""
function variable_elimination(fg::FactorGraph{T}, query::AbstractVector{Symbol};
                              evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
                              order::EliminationStrategy=MinFill()) where {T}
    isempty(query) && return _variable_elimination(fg, query, evidence, order, nothing)
    return _resolving_mass(() -> _variable_elimination(fg, query, evidence, order, nothing),
                           evidence) do
        f, d = log_variable_elimination(fg, query; evidence, order)
        return f,
               InferenceDiagnostics(d.order, d.max_factor_size, d.n_multiplications,
                                    d.treewidth, true)
    end
end

# The elimination-order cache. `graph`, `vars` and `index` exist only to produce `elim`,
# and everything after it depends on `elim` alone, so the order is what is worth keeping.
#
# It is a deterministic function of the *conditioned* factor scopes, the strategy and the
# query. Conditioning removes the observed variables from every scope, so the scopes depend
# on which variables are observed but not on their values -- which is why the key carries
# the evidence's keys and not the evidence. That is the case that matters: `predict` scores
# thousands of cases with the same evidence variables and different values, and they all
# share one entry.
#
# Keyed on the identity of the factor vector, with the same discipline as the junction-tree
# cache in `junction_tree.jl`: the stored `WeakRef` is compared with `===` on lookup, so a
# recycled `objectid` can never return another graph's order, and an entry is dropped once
# the factor vector is unreachable. Do not mutate `fg.factors` after a query; the cached
# order would no longer describe the graph.
const _OrderKey = Tuple{Vector{Symbol},EliminationStrategy,Vector{Symbol}}
const _OrderEntry = Tuple{WeakRef,Dict{_OrderKey,Vector{Symbol}}}
const _ORDER_CACHE = Dict{UInt,_OrderEntry}()

# A workload that sweeps many distinct evidence patterns would otherwise grow an entry per
# pattern; drop the graph's orders wholesale rather than grow without bound.
const _ORDER_CACHE_LIMIT = 256

function _cached_elimination_order(factors::Vector{<:Factor}, owner, query, evidence,
                                   order)
    key = (sort!(collect(Symbol, keys(evidence))), order, collect(Symbol, query))
    id = objectid(owner)
    entry = get(_ORDER_CACHE, id, nothing)
    if entry !== nothing && entry[1].value === owner
        cached = get(entry[2], key, nothing)
        cached === nothing || return cached
    else
        entry = (WeakRef(owner), Dict{_OrderKey,Vector{Symbol}}())
        _ORDER_CACHE[id] = entry
    end
    vars = _variables(factors)
    graph, vars, index = _interaction_graph(factors, vars)
    elim = _elimination_order(graph, vars, index, _restrict(order, evidence), query)
    length(entry[2]) < _ORDER_CACHE_LIMIT || empty!(entry[2])
    entry[2][key] = elim
    return elim
end

# Bucket elimination in the arithmetic `A` (arithmetic.jl), shared by `variable_elimination`
# and `log_variable_elimination`: condition every factor on the evidence, eliminate the
# non-query variables in the (cached) order, multiply what is left and put it in `query`
# order. Returns that unnormalised factor with the order, the largest factor, the number of
# multiplications and the width. Normalising, and deciding what the mass means, is the
# caller's. `observer` sees the conditioned inputs, each bucket and the final product; only
# the linear execution trace passes one.
function _eliminate(A::_Arithmetic, fg::FactorGraph{T}, query, evidence, order,
                    observer=nothing) where {T}
    _check_query(fg, query, evidence)
    F = _factor_type(A, T)
    factors = F[_conditioned(A, f, evidence) for f in fg.factors]
    observer === nothing || observer(:conditioned, factors)
    elim = _cached_elimination_order(Factor[_plain(f) for f in factors], fg.factors, query,
                                     evidence, order)
    max_size = 0
    n_mult = 0
    width = 0
    for v in elim
        touching = F[]
        rest = F[]
        for f in factors
            push!(v in _plain(f).vars ? touching : rest, f)
        end
        prod = _product!(A, touching)
        n_mult += max(length(touching) - 1, 0)
        max_size = max(max_size, length(_plain(prod)))
        width = max(width, ndims(_plain(prod)) - 1)
        reduced = _sum_out(A, prod, v)
        if observer !== nothing
            indices = findall(f -> v in _plain(f).vars, factors)
            observer(:bucket, (variable=v, inputs=indices, product=prod, result=reduced))
        end
        push!(rest, reduced)
        factors = rest
    end
    result = _product!(A, factors)
    n_mult += max(length(factors) - 1, 0)
    max_size = max(max_size, length(_plain(result)))
    width = max(width, ndims(_plain(result)) - 1)
    result = _reorder(A, result, query)
    observer === nothing || observer(:final_product, result)
    return result, elim, max_size, n_mult, width
end

function _variable_elimination(fg::FactorGraph, query, evidence, order, observer)
    result, elim, max_size, n_mult, width = _eliminate(_Linear(), fg, query, evidence,
                                                       order,
                                                       observer)
    isempty(query) || (result = _normalized(_Linear(), result, evidence))
    observer === nothing || observer(:result, result)
    return result, InferenceDiagnostics(elim, max_size, n_mult, width)
end
function variable_elimination(fg::FactorGraph, query::Symbol; kwargs...)
    return variable_elimination(fg, [query]; kwargs...)
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
    isempty(query) && return m
    return _resolving_mass(() -> _posterior_normalize(m, evidence), evidence) do
        return first(log_variable_elimination(fg, query; evidence))
    end
end
function brute_force_marginal(fg::FactorGraph, query::Symbol; kwargs...)
    return brute_force_marginal(fg, [query]; kwargs...)
end
