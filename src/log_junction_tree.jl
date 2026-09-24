"""
    LogJunctionTree(; order=MinFill())

Shafer-Shenoy junction-tree propagation with centered log-domain potentials and
messages. Reuses the ordinary structural clique-tree cache and propagation
schedule, but never multiplies small probability masses in ordinary arithmetic.
All singleton marginals come from one calibration. Queries spanning cliques
fall back explicitly to [`LogVariableElimination`](@ref).
"""
struct LogJunctionTree{O<:EliminationStrategy} <: InferenceBackend
    order::O
end
LogJunctionTree(; order::EliminationStrategy=MinFill()) = LogJunctionTree(order)

"""
    LogCalibratedJunctionTree

A log-domain calibration: `tree`, `evidence`, centered-log `potentials` and
`beliefs`, global `log_evidence_probability`, and `n_messages`. Each internal
log factor stores its centered table separately from its common log scale.
As with ordinary calibration, forest beliefs are component-local; posterior
entry points check the separate global evidence mass.
"""
struct LogCalibratedJunctionTree
    tree::CompiledJunctionTree
    evidence::Dict{Symbol,Symbol}
    potentials::Vector{_LogFactor}
    beliefs::Vector{_LogFactor}
    log_evidence_probability::Float64
    n_messages::Int
end

"""
    LogJunctionTreeDiagnostics

The ordinary junction-tree diagnostic fields (`clique`, `n_cliques`,
`treewidth`, `max_clique_size`, `n_messages`, `fallback`) together with
`log_evidence_probability` and `mass_status`. An underflowed ordinary mass
is not confused with truly zero support.
"""
struct LogJunctionTreeDiagnostics
    clique::Int
    n_cliques::Int
    treewidth::Int
    max_clique_size::Int
    n_messages::Int
    fallback::Bool
    log_evidence_probability::Float64
    mass_status::Symbol
end

function _log_project(factor::_LogFactor, keep::Vector{Symbol})
    result = factor
    for variable in setdiff(factor.factor.vars, keep)
        result = _log_sum_out(result, variable)
    end
    return result
end

function _log_mass(factor::_LogFactor)
    total = sum(exp, factor.factor.table)
    return iszero(total) ? -Inf : factor.log_scale + log(total)
end

function _log_mass_status(log_mass)
    log_mass == -Inf && return :zero
    mass = exp(log_mass)
    return iszero(mass) ? :underflow : isinf(mass) ? :overflow : :finite
end

"""
    log_calibrate(fg, jt=build_junction_tree(fg; order); evidence=Dict(), order=MinFill())

Calibrate a clique tree using log-domain products and log-sum-exp projections.
Finite nonnegative inputs are validated before evidence conditioning, including
off-evidence entries. Exactly zero support gives log mass `-Inf`; tiny positive
evidence retains a finite log mass. This is a numerical backend, not a universal
floating-point error bound.
"""
function log_calibrate(fg::FactorGraph, jt::CompiledJunctionTree;
                       evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}())
    _check_query(fg, Symbol[], evidence)
    length(jt.assignment) == length(fg.factors) ||
        throw(ShapeError(:log_calibrate, "the tree was built for a different factor graph",
                         length(jt.assignment), length(fg.factors)))
    ev = Dict{Symbol,Symbol}(evidence)
    lists = [_LogFactor[] for _ in 1:length(jt)]
    constant = _log_product(_LogFactor[])
    for (factor, clique) in zip(fg.factors, jt.assignment)
        logged = _as_log_factor(factor)
        conditioned = _center_log(condition(logged.factor, ev), logged.log_scale)
        if clique == 0
            constant = _log_multiply(constant, conditioned)
        else
            push!(lists[clique], conditioned)
        end
    end
    potentials = _LogFactor[_log_product(items) for items in lists]
    separators = [Symbol[v for v in scope if !haskey(ev, v)] for scope in jt.separators]
    beliefs, count = _junction_tree_messages(jt, potentials, separators, _log_product,
                                             _log_project)
    masses = [_log_mass(constant); [_log_mass(beliefs[root]) for root in jt.roots]]
    log_mass = any(==(-Inf), masses) ? -Inf : sum(masses)
    return LogCalibratedJunctionTree(jt, ev, potentials, beliefs, log_mass, count)
end

function log_calibrate(fg::FactorGraph;
                       evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
                       order::EliminationStrategy=MinFill())
    return log_calibrate(fg, build_junction_tree(fg; order); evidence)
end

log_evidence_probability(cal::LogCalibratedJunctionTree) = cal.log_evidence_probability

function _diagnostics(cal::LogCalibratedJunctionTree, clique::Int)
    largest = maximum(f -> length(f.factor), cal.beliefs; init=0)
    return LogJunctionTreeDiagnostics(clique, length(cal.tree), cal.tree.treewidth, largest,
                                      cal.n_messages, false, cal.log_evidence_probability,
                                      _log_mass_status(cal.log_evidence_probability))
end

function _log_belief_marginal(belief::_LogFactor, query)
    projected = _log_project(belief, collect(Symbol, query))
    result = reorder(projected.factor, query)
    weights = similar(result.table)
    weights .= exp.(result.table)
    weights ./= sum(weights)
    return _factor(result.vars, result.axes, weights)
end

function _infer(backend::LogJunctionTree, fg::FactorGraph, query, evidence)
    _check_query(fg, query, evidence)
    tree = build_junction_tree(fg; order=backend.order)
    clique = _containing_clique(tree, fg.axes, query)
    if !isempty(query) && clique === nothing
        @warn "LogJunctionTree: the query $(collect(query)) does not lie in one clique; falling back to log-domain variable elimination"
        result, info = log_variable_elimination(fg, query; evidence, order=backend.order)
        return result,
               LogJunctionTreeDiagnostics(0, length(tree), tree.treewidth,
                                          info.max_factor_size, 0, true,
                                          info.log_evidence_probability, info.mass_status)
    end
    cal = log_calibrate(fg, tree; evidence)
    if isempty(query)
        return _factor(Symbol[], FiniteAxis[], fill(exp(cal.log_evidence_probability))),
               _diagnostics(cal, 0)
    end
    cal.log_evidence_probability == -Inf && _require_evidence_mass(0.0, evidence)
    return _log_belief_marginal(cal.beliefs[clique], query), _diagnostics(cal, clique)
end

function _all_marginals(backend::LogJunctionTree, fg::FactorGraph, evidence)
    cal = log_calibrate(fg; evidence, order=backend.order)
    cal.log_evidence_probability == -Inf && _require_evidence_mass(0.0, evidence)
    return Dict{Symbol,Factor{Float64}}(variable => haskey(evidence, variable) ?
                                                    _point_mass(Float64, axis,
                                                                evidence[variable]) :
                                                    _log_belief_marginal(cal.beliefs[cal.tree.home[variable]],
                                                                         [variable])
                                        for (variable, axis) in fg.axes)
end

function _clique_beliefs(backend::LogJunctionTree, fg::FactorGraph, evidence)
    cal = log_calibrate(fg; evidence, order=backend.order)
    cal.log_evidence_probability == -Inf && _require_evidence_mass(0.0, evidence)
    return [_log_belief_marginal(belief, belief.factor.vars) for belief in cal.beliefs]
end
