mutable struct _VETraceRecorder
    data::Dict{String,Any}
    remaining::Int
end

function _trace_factor!(recorder::_VETraceRecorder, factor::Factor{Float64})
    length(factor) <= recorder.remaining ||
        throw(ScopeError(:trace_variable_elimination, "the trace cell budget was exceeded", copy(factor.vars)))
    recorder.remaining -= length(factor)
    return Dict{String,Any}("scope" => String.(factor.vars),
                           "values" => [string(reinterpret(UInt64, value); base=16, pad=16)
                                        for value in vec(factor.table)])
end

function (recorder::_VETraceRecorder)(kind::Symbol, value)
    if kind == :conditioned
        recorder.data["conditioned"] = [_trace_factor!(recorder, factor) for factor in value]
    elseif kind == :bucket
        push!(recorder.data["steps"], Dict{String,Any}(
            "variable" => String(value.variable), "inputs" => value.inputs .- 1,
            "product" => _trace_factor!(recorder, value.product),
            "result" => _trace_factor!(recorder, value.result)))
    else
        recorder.data[String(kind)] = _trace_factor!(recorder, value)
    end
    return nothing
end

"""
    trace_variable_elimination(fg, query; evidence=Dict(), order=MinFill(), max_entries=1_000_000)
    trace_variable_elimination(model::BayesModel, query; evidence=Dict(), order=MinFill(), atol=DEFAULT_ATOL, max_entries=1_000_000)
        -> (factor, diagnostics, trace)

Capture the actual ordinary variable-elimination execution, using the same
driver as [`variable_elimination`](@ref), not a replay substituted for it.
The JSON-compatible trace contains original/conditioned Float64 factors,
every actual bucket product and reduction, zero-based active-bag input indices,
the final product and the returned result. Tables are flattened first-axis
fastest with explicit ordered scopes and state labels.

Inputs must be finite nonnegative Float64 factors. `max_entries` caps copied
trace cells; input/query errors and impossible nonempty conditionals retain
the ordinary API behavior. A trace is not itself a proof: an independent exact
consumer may reject numerical drift or nonfinite intermediates. Model-level
capture starts after [`compile`](@ref), so it does not certify the compiler or
source-CPT transcription. Metadata is descriptive, not installation attestation.
"""
function trace_variable_elimination(fg::FactorGraph{Float64}, query::AbstractVector{Symbol};
                                    evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
                                    order::EliminationStrategy=MinFill(),
                                    max_entries::Integer=1_000_000)
    0 < max_entries <= typemax(Int) ||
        throw(ArgumentError("max_entries must be a positive representable integer"))
    _check_query(fg, query, evidence)
    for factor in fg.factors
        all(value -> isfinite(value) && value >= 0, factor.table) ||
            throw(ScopeError(:trace_variable_elimination,
                             "trace inputs must be finite and nonnegative", copy(factor.vars)))
    end
    data = Dict{String,Any}(
        "format" => "ecorecipes.ve-execution-trace", "version" => 1,
        "layout" => "first-axis-fastest", "arithmetic" => "binary64-observed-exact-shadow-v1",
        "variables" => [Dict("id" => String(name), "states" => String.(labels(fg.axes[name])))
                        for name in variables(fg)],
        "evidence" => Dict(String(name) => String(state) for (name, state) in evidence),
        "query" => String.(query), "steps" => Any[],
        "metadata" => Dict("producer" => "BayesianNetworkInference.trace_variable_elimination",
                           "runtime_version" => string(VERSION), "package_version" => string(Base.pkgversion(@__MODULE__))))
    recorder = _VETraceRecorder(data, Int(max_entries))
    data["inputs"] = [_trace_factor!(recorder, factor) for factor in fg.factors]
    result, diagnostics = _variable_elimination(fg, query, evidence, order, recorder)
    return result, diagnostics, data
end

function trace_variable_elimination(fg::FactorGraph, query::AbstractVector{Symbol}; kwargs...)
    throw(ScopeError(:trace_variable_elimination, "the v1 trace profile requires Float64 factors", variables(fg)))
end

trace_variable_elimination(fg::FactorGraph, query::Symbol; kwargs...) =
    trace_variable_elimination(fg, [query]; kwargs...)

function trace_variable_elimination(model::BayesModel, query;
                                    evidence=Dict{Symbol,Symbol}(),
                                    order::EliminationStrategy=MinFill(),
                                    atol::Real=BayesianNetworks.DEFAULT_ATOL,
                                    max_entries::Integer=1_000_000)
    return trace_variable_elimination(compile(model; atol), query;
                                      evidence=_model_evidence(model, evidence), order, max_entries)
end
