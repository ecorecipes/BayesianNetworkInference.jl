"""
    LogVariableElimination(; order=MinFill())

Variable elimination with centered log-domain factors. Structural zeros remain
`-Inf`; products cannot turn positive finite inputs into zero before posterior
normalization. Returns Float64 factors and [`LogInferenceDiagnostics`](@ref).
The ordinary [`VariableElimination`](@ref) backend remains the default.
"""
struct LogVariableElimination{O<:EliminationStrategy} <: InferenceBackend
    order::O
end
LogVariableElimination(; order::EliminationStrategy=MinFill()) = LogVariableElimination(order)

"""
    LogFactorDomainError(vars, index, value)

A log-domain input factor has a negative/nonfinite entry, or an entry whose
logarithm cannot be represented. The offending scope and table index are retained.
Small negative values are rejected, not clamped to zero.
"""
struct LogFactorDomainError <: Exception
    vars::Vector{Symbol}
    index::Tuple
    value::Real
end
function Base.showerror(io::IO, error::LogFactorDomainError)
    return print(io, "LogFactorDomainError: factor over ", error.vars, " at ",
                 error.index, " has unsupported value ", error.value)
end

"""
    LogInferenceDiagnostics(order, max_factor_size, n_multiplications, treewidth,
                            log_evidence_probability, mass_status)

Log-domain elimination diagnostics. `mass_status` is `:finite`, `:underflow`,
`:overflow`, or `:zero`. A finite log mass with `:underflow` is not impossible
evidence. The log mass is unnormalized for a general factor graph, matching the
ordinary empty-query convention.
"""
struct LogInferenceDiagnostics
    order::Vector{Symbol}
    max_factor_size::Int
    n_multiplications::Int
    treewidth::Int
    log_evidence_probability::Float64
    mass_status::Symbol
end

struct _LogFactor
    factor::Factor{Float64}
    log_scale::Float64
end

function _center_log(factor::Factor{Float64}, scale::Float64)
    peak = maximum(factor.table)
    peak == -Inf && return _LogFactor(factor, scale)
    centered = similar(factor.table)
    centered .= factor.table .- peak
    return _LogFactor(_factor(factor.vars, factor.axes, centered), scale + peak)
end

function _as_log_factor(factor::Factor)
    table = Array{Float64}(undef, size(factor))
    for index in CartesianIndices(factor.table)
        value = factor.table[index]
        isfinite(value) && value >= 0 ||
            throw(LogFactorDomainError(copy(factor.vars), Tuple(index), value))
        if iszero(value)
            table[index] = -Inf
        else
            logarithm = Float64(log(value))
            isfinite(logarithm) ||
                throw(LogFactorDomainError(copy(factor.vars), Tuple(index), value))
            table[index] = logarithm
        end
    end
    return _center_log(_factor(factor.vars, factor.axes, table), 0.0)
end

function _log_multiply(left::_LogFactor, right::_LogFactor)
    vars, axes = _union_axes(left.factor, right.factor, :log_multiply)
    shape = Tuple(length(axis) for axis in axes)
    table = Array{Float64}(undef, shape)
    table .= _broadcastable(left.factor, vars, shape) .+ _broadcastable(right.factor, vars, shape)
    return _center_log(_factor(vars, axes, table), left.log_scale + right.log_scale)
end

function _log_product(factors::Vector{_LogFactor})
    isempty(factors) && return _LogFactor(_factor(Symbol[], FiniteAxis[], fill(0.0)), 0.0)
    sort!(factors; by=factor -> ndims(factor.factor))
    return reduce(_log_multiply, factors)
end

function _log_sum_out(source::_LogFactor, variable::Symbol)
    factor = source.factor
    position = _position(factor, :log_marginalize, variable)
    keep = [index for index in eachindex(factor.vars) if index != position]
    shape = Tuple(size(factor)[index] for index in keep)
    table = Array{Float64}(undef, shape)
    for index in CartesianIndices(table)
        coordinates = Tuple(index)
        full(state) = ntuple(i -> i == position ? state : coordinates[i < position ? i : i - 1], ndims(factor))
        peak = maximum(factor.table[full(state)...] for state in 1:size(factor.table, position))
        if peak == -Inf
            table[index] = -Inf
        else
            total = sum(exp(factor.table[full(state)...] - peak) for state in 1:size(factor.table, position))
            table[index] = peak + log(total)
        end
    end
    return _center_log(_factor(factor.vars[keep], factor.axes[keep], table), source.log_scale)
end

function _log_eliminate(fg::FactorGraph, query, evidence, order)
    _check_query(fg, query, evidence)
    factors = _LogFactor[]
    for factor in fg.factors
        logged = _as_log_factor(factor)
        push!(factors, _center_log(condition(logged.factor, evidence), logged.log_scale))
    end
    ordinary = Factor{Float64}[factor.factor for factor in factors]
    vars = _variables(ordinary)
    graph, vars, index = _interaction_graph(ordinary, vars)
    elimination = _elimination_order(graph, vars, index, _restrict(order, evidence), query)
    largest, multiplications, width = 0, 0, 0
    for variable in elimination
        touched = _LogFactor[factor for factor in factors if variable in factor.factor.vars]
        factors = _LogFactor[factor for factor in factors if !(variable in factor.factor.vars)]
        product = _log_product(touched)
        largest = max(largest, length(product.factor))
        width = max(width, ndims(product.factor) - 1)
        multiplications += max(length(touched) - 1, 0)
        push!(factors, _log_sum_out(product, variable))
    end
    result = _log_product(factors)
    multiplications += max(length(factors) - 1, 0)
    largest = max(largest, length(result.factor))
    width = max(width, ndims(result.factor) - 1)
    result = _LogFactor(reorder(result.factor, query), result.log_scale)
    total = sum(exp, result.factor.table)
    log_mass = iszero(total) ? -Inf : result.log_scale + log(total)
    mass = exp(log_mass)
    status = log_mass == -Inf ? :zero : iszero(mass) ? :underflow : isinf(mass) ? :overflow : :finite
    return result, total, LogInferenceDiagnostics(elimination, largest, multiplications,
                                                 width, log_mass, status)
end

"""
    log_variable_elimination(fg, query; evidence=Dict{Symbol,Symbol}(), order=MinFill())

Compute a posterior using centered log-domain products and log-sum-exp
marginalization. Positive common scales are kept outside the tables so they
cannot erase small differences between posterior cells. For an empty query,
return the ordinary unnormalized mass (possibly zero by underflow); the finite
log mass remains available in the diagnostics. A nonempty query with truly
zero support throws `KernelNormalizationError`.
"""
function log_variable_elimination(fg::FactorGraph, query::AbstractVector{Symbol};
                                  evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
                                  order::EliminationStrategy=MinFill())
    result, total, diagnostics = _log_eliminate(fg, query, evidence, order)
    if isempty(query)
        return _factor(Symbol[], FiniteAxis[], fill(exp(diagnostics.log_evidence_probability))), diagnostics
    end
    diagnostics.mass_status == :zero && _require_evidence_mass(0.0, evidence)
    return _factor(result.factor.vars, result.factor.axes, exp.(result.factor.table) ./ total), diagnostics
end
log_variable_elimination(fg::FactorGraph, query::Symbol; kwargs...) =
    log_variable_elimination(fg, [query]; kwargs...)

function _infer(backend::LogVariableElimination, fg::FactorGraph, query, evidence)
    return log_variable_elimination(fg, query; evidence, order=backend.order)
end

"""
    log_evidence_probability(fg; evidence=Dict{Symbol,Symbol}(), order=MinFill())
    log_evidence_probability(model::BayesModel; evidence=Dict(), order=MinFill(), atol=DEFAULT_ATOL)
    log_evidence_probability(calibration::LogCalibratedJunctionTree)

Return the log unnormalized evidence mass without exponentiating it. Exactly
unsupported evidence gives `-Inf`; positive but unrepresentable ordinary mass
has a finite log value. Model evidence is merged with explicit evidence as in
`infer`; `atol` is the compilation/normalization tolerance, not a log cutoff.
The calibration method reads the already-computed log mass without rerunning
message passing.
"""
function log_evidence_probability(fg::FactorGraph;
                                  evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
                                  order::EliminationStrategy=MinFill())
    return last(_log_eliminate(fg, Symbol[], evidence, order)).log_evidence_probability
end

function _all_marginals(backend::LogVariableElimination, fg::FactorGraph, evidence)
    log_evidence_probability(fg; evidence, order=backend.order) == -Inf &&
        _require_evidence_mass(0.0, evidence)
    result = Dict{Symbol,Factor{Float64}}()
    for (variable, axis) in fg.axes
        result[variable] = haskey(evidence, variable) ? _point_mass(Float64, axis, evidence[variable]) :
                           first(log_variable_elimination(fg, [variable]; evidence, order=backend.order))
    end
    return result
end
