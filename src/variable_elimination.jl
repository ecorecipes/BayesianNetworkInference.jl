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
interaction graph of the conditioned factors). `exact_fallback` is true when the run was
not trusted -- a binary64 evidence mass that is not a normal positive number, a product of
nonzero values below `floatmin`, or an integer or rational product, sum or quotient that
overflowed its element type -- and the posterior was recomputed in exact arithmetic and
correctly rounded to Float64 (ADR 0014, ADR 0016).
"""
struct InferenceDiagnostics
    order::Vector{Symbol}
    max_factor_size::Int
    n_multiplications::Int
    treewidth::Int
    exact_fallback::Bool
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

# Evidence mass (ADR 0012, ADR 0014). A binary64 run is trusted only when its mass is a
# normal positive number and no product it computed from nonzero operands fell below
# `floatmin` (the trust test, `_check_product` in arithmetic.jl). Zero, a subnormal, a
# negative or a non-finite mass does not say whether the evidence is impossible -- a
# positive probability that underflowed is zero too -- and a normal mass built from
# underflowed products can be far from the exact one, so an untrusted run throws the
# internal `_UnresolvedMass` signal (src/errors.jl) instead of deciding. Every public entry
# point catches it (`_resolving_mass`) and recomputes in exact dyadic arithmetic
# (arithmetic.jl, ADR 0016), where only an exact zero is zero and so
# `ImpossibleEvidenceError`, and each posterior cell is rounded once to the nearest Float64.
# Exact element types (integers, rationals) decide directly.

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

# Tolerated entries (ADR 0007, ADR 0011, ADR 0014 decision 4, ADR 0016 decision 4), under the
# rule that BayesianNetworks, InfluenceDiagrams and this package share: a tolerated entry in
# [-atol, 0) takes part in a posterior when it lies on a configuration consistent with the
# evidence whose other entries are all nonzero; an entry multiplied by an exact zero
# contributes nothing. When one takes part, the posterior is indeterminate if the evidence
# mass is not larger than the budget (1 + atol)^n - 1, or if a cell of the queried posterior
# is negative; cells of the joint that the query sums out do not count. A prior (empty
# evidence) is exempt, and the binary64 and exact runs decide by the same rule.
#
# Only a model knows `atol`, so the model-level entry points apply the rule
# (`_under_tolerance`, model_inference.jl): they decide whether an entry takes part and check
# the budget, then ask the backend for the posterior inside `_MODEL_QUERY`. There a prior's
# cells are returned as computed, and the exact fallback takes negative entries at their
# exact value -- one that takes part shows in a negative exact cell -- instead of rejecting
# them. The factor-graph methods run outside it: a negative posterior cell is indeterminate,
# and the exact fallback rejects any negative entry (ADR 0016 decision 4).
struct _ModelQuery
    prior::Bool
end
const _MODEL_QUERY = Base.ScopedValues.ScopedValue{Union{Nothing,_ModelQuery}}(nothing)
_in_model_query() = _MODEL_QUERY[] !== nothing
_model_prior() = (q = _MODEL_QUERY[]; q !== nothing && q.prior)

# Whether a tolerated negative entry takes part in the posterior under `evidence`: whether
# some configuration consistent with the evidence has every entry nonzero and one negative.
# The shared driver in the support arithmetic (arithmetic.jl), over the schedule of `order`,
# so it costs one elimination of Boolean tables and never enumerates configurations.
function _takes_part(fg::FactorGraph, evidence, order::EliminationStrategy)
    result = first(_eliminate(_Support(), fg, Symbol[], evidence, order))
    return (result.table[] & 0x02) != 0
end

# Normalise a posterior factor after checking its mass, so that posterior code never
# reaches the `ArgumentError` of `normalize(::Factor)`. A negative posterior cell can come
# only from tolerated entries in [-atol, 0); its sign is an artefact of the rounding, unless
# the query is a model's prior. An integer or rational total or quotient that overflows its
# element type is the arithmetic `A`'s to resolve (`_overflowed`, arithmetic.jl).
function _posterior_normalize(f::Factor, evidence, A::_Linear=_Linear())
    s = _total(f, :normalize)
    s isa FactorDomainError && _overflowed(A, s)
    _require_evidence_mass(s, evidence)
    p = _normalize(f)
    p isa FactorDomainError && _overflowed(A, p)
    v = minimum(p.table; init=zero(eltype(p.table)))
    v < 0 && !_model_prior() &&
        throw(IndeterminatePosteriorError(Dict{Symbol,Symbol}(evidence),
                                          "a posterior cell is negative ($(v))"))
    return p
end

# Run `ordinary`; if its binary64 mass was not trustworthy, run `fallback` (an exact
# computation of the same answer) instead. Called with a `do` block, which Julia passes
# first: `_resolving_mass(() -> ordinary, evidence) do ... end`.
function _resolving_mass(fallback, ordinary, evidence)
    try
        return ordinary()
    catch e
        e isa _UnresolvedMass || rethrow()
    end
    return fallback()
end

# The fallback of variable elimination (ADR 0014, ADR 0016): the shared driver in exact
# dyadic arithmetic, each posterior cell correctly rounded.
function _exact_variable_elimination(fg::FactorGraph, query, evidence, order)
    result, elim, max_size, n_mult, width = _eliminate(_Dyadic(), fg, query, evidence,
                                                       order)
    return _normalized(_Dyadic(), result, evidence),
           InferenceDiagnostics(elim, max_size, n_mult, width, true)
end

# Whether the evidence has probability exactly zero, decided exactly.
function _exactly_impossible(fg::FactorGraph, evidence,
                             order::EliminationStrategy=MinFill())
    return _exactly_zero(first(_eliminate(_Dyadic(), fg, Symbol[], evidence, order)))
end

# `ImpossibleEvidenceError` unless the evidence has positive mass: decided from the empty
# query's binary64 mass when that is a normal positive number, and exactly otherwise. An
# integer or rational graph decides exactly from the start, since the empty query's mass in
# its own element type can overflow (`FactorDomainError`). Belief propagation's opt-in check
# and the variable-elimination `all_marginals` of fully observed evidence use it.
function _require_feasible(fg::FactorGraph{T}, evidence,
                           order::EliminationStrategy) where {T}
    if _checked(T)
        _exactly_impossible(fg, evidence, order) && _impossible(evidence)
        return nothing
    end
    mass = first(_variable_elimination(fg, Symbol[], evidence, order, nothing)).table[]
    _resolving_mass(() -> _require_evidence_mass(mass, evidence), evidence) do
        _exactly_impossible(fg, evidence, order) && _impossible(evidence)
        return nothing
    end
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
evidence variables, or a query variable that also carries evidence, and
`BayesianNetworks.ImpossibleEvidenceError` for a non-empty `query` when the
evidence has probability exactly zero. A binary64 run does not decide that when its mass
is zero, subnormal or non-finite -- a positive probability can underflow -- and its
posterior is not trusted when a product of nonzero values fell below `floatmin` on the way
(it came out subnormal, or rounded to zero), even if the mass is normal. Nor is a run on an
integer or rational graph whose product, sum or quotient overflowed the element type: such
tables are computed in checked arithmetic, never wrapped. The posterior is
then recomputed in exact arithmetic, every entry at its exact value whatever the graph's
element type, each cell rounded once to the Float64 nearest the
exact posterior of the graph as bound, and returned, as a `Factor{Float64}`, with
`diagnostics.exact_fallback == true` (ADR 0014, ADR 0016). A model with tolerated entries
in `[-atol, 0)` raises `BayesianNetworks.IndeterminatePosteriorError` when a posterior cell
comes out negative, or when the exact fallback is needed. An empty `query` returns the mass
in the graph's own element type, so an integer mass that overflows it is a
[`FactorDomainError`](@ref) instead.

The elimination order is cached per graph (see the source); `diagnostics.order` is the
caller's own copy, so changing it changes nothing else.
"""
function variable_elimination(fg::FactorGraph{T}, query::AbstractVector{Symbol};
                              evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
                              order::EliminationStrategy=MinFill()) where {T}
    isempty(query) && return _variable_elimination(fg, query, evidence, order, nothing)
    return _resolving_mass(() -> _variable_elimination(fg, query, evidence, order, nothing),
                           evidence) do
        return _exact_variable_elimination(fg, query, evidence, order)
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
# cache in `junction_tree.jl` (review of 2026-10-02, finding 2):
#
# - the stored `WeakRef` is compared with `===` on lookup, so a recycled `objectid` can never
#   return another graph's order;
# - the entries whose factor vector has been collected are dropped, amortised: the sweep
#   runs when a graph is added and the cache has doubled since the last sweep, so a stream
#   of graphs compiled per query (every model-level `infer` compiles one) keeps the cache
#   bounded by about twice the graphs alive;
# - every access holds `_ORDER_LOCK`, since queries may run on several threads at once;
#   computing an order does not, so that one slow ordering does not hold up the others
#   (two threads that miss on the same key compute the same order, and the second store
#   wins);
# - the cached vector is never handed out: a caller gets a copy, which becomes its
#   diagnostics' `order`, so changing that changes nothing in the cache.
#
# Do not mutate `fg.factors` after a query; the cached order would no longer describe the
# graph.
const _OrderKey = Tuple{Vector{Symbol},EliminationStrategy,Vector{Symbol}}
const _OrderEntry = Tuple{WeakRef,Dict{_OrderKey,Vector{Symbol}}}
const _ORDER_CACHE = Dict{UInt,_OrderEntry}()
const _ORDER_LOCK = ReentrantLock()
const _ORDER_WATERMARK = Ref(16)

# A workload that sweeps many distinct evidence patterns would otherwise grow an entry per
# pattern; drop the graph's orders wholesale rather than grow without bound.
const _ORDER_CACHE_LIMIT = 256

# Drop the entries whose factor vector has been collected; the caller holds `_ORDER_LOCK`.
function _prune_order_cache!()
    length(_ORDER_CACHE) <= 2 * _ORDER_WATERMARK[] && return nothing
    filter!(kv -> kv[2][1].value !== nothing, _ORDER_CACHE)
    _ORDER_WATERMARK[] = max(16, length(_ORDER_CACHE))
    return nothing
end

function _cached_elimination_order(factors::Vector{<:Factor}, owner, query, evidence,
                                   order)
    key = (sort!(collect(Symbol, keys(evidence))), order, collect(Symbol, query))
    id = objectid(owner)
    cached = lock(_ORDER_LOCK) do
        entry = get(_ORDER_CACHE, id, nothing)
        (entry === nothing || entry[1].value !== owner) && return nothing
        return get(entry[2], key, nothing)
    end
    cached === nothing || return copy(cached)
    vars = _variables(factors)
    graph, vars, index = _interaction_graph(factors, vars)
    elim = _elimination_order(graph, vars, index, _restrict(order, evidence), query)
    lock(_ORDER_LOCK) do
        entry = get(_ORDER_CACHE, id, nothing)
        if entry === nothing || entry[1].value !== owner
            _prune_order_cache!()
            entry = (WeakRef(owner), Dict{_OrderKey,Vector{Symbol}}())
            _ORDER_CACHE[id] = entry
        end
        length(entry[2]) < _ORDER_CACHE_LIMIT || empty!(entry[2])
        entry[2][key] = copy(elim)
        return nothing
    end
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

# A posterior run applies the trust test to every product (`_Trusted`); an empty query
# returns the unnormalised mass, which may legally be tiny (ADR 0011), and checks nothing.
function _variable_elimination(fg::FactorGraph, query, evidence, order, observer)
    A = isempty(query) ? _Linear() : _Trusted()
    result, elim, max_size, n_mult, width = _eliminate(A, fg, query, evidence, order,
                                                       observer)
    isempty(query) || (result = _normalized(A, result, evidence))
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
this is the oracle that variable elimination is tested against. Like [`multiply`](@ref),
it raises [`FactorDomainError`](@ref) when an integer or rational product overflows the
element type.
"""
function joint_factor(fg::FactorGraph)
    isempty(fg.factors) && return unit_factor(eltype(fg))
    return reorder(multiply(fg.factors), variables(fg))
end

"""
    brute_force_marginal(fg::FactorGraph, query; evidence=Dict{Symbol,Symbol}()) -> Factor

`P(query | evidence)` by conditioning and summing the full joint table.
Same conventions as [`variable_elimination`](@ref) (including the
unnormalised `P(evidence)` for an empty query, and the exact fallback when the binary64
run is not trusted), without diagnostics.
"""
function brute_force_marginal(fg::FactorGraph, query::AbstractVector{Symbol};
                              evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}())
    _check_query(fg, query, evidence)
    isempty(query) && return _brute_force(_Linear(), fg, query, evidence)
    ordinary = () -> _normalized(_Trusted(), _brute_force(_Trusted(), fg, query, evidence),
                                 evidence)
    return _resolving_mass(ordinary, evidence) do
        return first(_exact_variable_elimination(fg, query, evidence, MinFill()))
    end
end

# The unnormalised `P(query, evidence)` from the full joint table, multiplied in the order
# of `joint_factor` (in the arithmetic `A`, which for a posterior checks every product).
function _brute_force(A::_Linear, fg::FactorGraph, query, evidence)
    joint = isempty(fg.factors) ? unit_factor(eltype(fg)) :
            reorder(reduce((a, b) -> _multiply(A, a, b), fg.factors), variables(fg))
    j = condition(joint, evidence)
    return reorder(_project(A, j, collect(Symbol, query)), query)
end
function brute_force_marginal(fg::FactorGraph, query::Symbol; kwargs...)
    return brute_force_marginal(fg, [query]; kwargs...)
end
