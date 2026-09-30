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
function LogVariableElimination(; order::EliminationStrategy=MinFill())
    return LogVariableElimination(order)
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

# Log-domain factors and their operations are in arithmetic.jl; the elimination itself is
# the shared driver `_eliminate` (variable_elimination.jl).
function _log_eliminate(fg::FactorGraph, query, evidence, order)
    result, elimination, largest, multiplications, width = _eliminate(_LogDomain(), fg,
                                                                      query, evidence,
                                                                      order)
    log_mass = _log_mass(result)
    return result,
           LogInferenceDiagnostics(elimination, largest, multiplications, width, log_mass,
                                   _log_mass_status(log_mass))
end

"""
    log_variable_elimination(fg, query; evidence=Dict{Symbol,Symbol}(), order=MinFill())

Compute a posterior using centered log-domain products and log-sum-exp
marginalization. Positive common scales are kept outside the tables so they
cannot erase small differences between posterior cells. For an empty query,
return the ordinary unnormalized mass (possibly zero by underflow); the finite
log mass remains available in the diagnostics. A nonempty query with truly
zero support (`mass_status == :zero`) throws
`BayesianNetworks.ImpossibleEvidenceError`; an underflowed mass does not.
"""
function log_variable_elimination(fg::FactorGraph, query::AbstractVector{Symbol};
                                  evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
                                  order::EliminationStrategy=MinFill())
    result, diagnostics = _log_eliminate(fg, query, evidence, order)
    if isempty(query)
        return _factor(Symbol[], FiniteAxis[],
                       fill(exp(diagnostics.log_evidence_probability))), diagnostics
    end
    diagnostics.mass_status == :zero && _impossible(evidence)
    return _normalized(_LogDomain(), result, evidence), diagnostics
end
function log_variable_elimination(fg::FactorGraph, query::Symbol; kwargs...)
    return log_variable_elimination(fg, [query]; kwargs...)
end

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
        _impossible(evidence)
    return _collect_marginals(fg, evidence, Float64) do v
        return first(log_variable_elimination(fg, [v]; evidence, order=backend.order))
    end
end
