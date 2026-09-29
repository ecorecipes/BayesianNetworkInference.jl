# Model-level inference and sampling: `infer`, `posterior` and
# `ancestral_sample` on a `BayesModel`, through `compile`.

_evidence_dict(ev::AbstractDict{Symbol,Symbol}) = Dict{Symbol,Symbol}(ev)
_evidence_dict(ev::AbstractVector{<:Pair{Symbol,Symbol}}) = Dict{Symbol,Symbol}(ev)
_evidence_dict(ev::Pair{Symbol,Symbol}) = Dict{Symbol,Symbol}(ev)

# The evidence recorded on the model merged with the explicit evidence; the
# explicit entries win.
function _model_evidence(m::BayesModel, ev)
    return merge(Dict{Symbol,Symbol}(BayesianNetworks.evidence(m)), _evidence_dict(ev))
end

# The tolerance budget (ADR 0014, and BayesianNetworks' `marginal`): a validated model may
# hold entries in [-atol, 0), and when it does, evidence whose mass is within the budget
# (1 + atol)^n - 1 of zero leaves the posterior's sign to the rounding. Only the model-level
# entry points know `atol`, so they check it; the check costs one elimination and runs only
# for a model with a negative entry.
function _check_tolerance_budget(fg::FactorGraph, ev::AbstractDict, atol::Real)
    isempty(ev) && return nothing
    any(f -> any(<(0), f.table), fg.factors) || return nothing
    mass = variable_elimination(fg, Symbol[]; evidence=ev)[1].table[]
    budget = BayesianNetworks._joint_atol(atol, length(fg.factors))
    (isfinite(mass) && mass > budget) && return nothing
    return throw(IndeterminatePosteriorError(Dict{Symbol,Symbol}(ev),
                                             "the evidence mass $(mass) is within the tolerance budget $(budget) of zero"))
end

"""
    infer(m::BayesModel, query; evidence=Dict{Symbol,Symbol}(), backend=VariableElimination(), atol=DEFAULT_ATOL)
        -> (posterior::Factor, diagnostics::InferenceDiagnostics)

`P(query | evidence)` on a `BayesianNetworks.BayesModel`: the model is
[`compile`](@ref)d to a [`FactorGraph`](@ref) and the query is answered by
`backend` (see the [`FactorGraph`](@ref) method of `infer`).
`query` is a variable or a vector of variables; the posterior is a normalised
factor with that scope (a one-variable factor for a single symbol).
`atol` controls kernel-normalization validation during compilation, not the
convergence tolerance of an approximate backend. The other model-facing
inference, sampling, scoring and sensitivity entry points accept it too.

The evidence used is the evidence recorded on the model by `observe` merged
with the explicit `evidence` (a dictionary, a vector of pairs or a single
`:X => :x` pair), the explicit entries winning; to ignore the recorded
evidence, query `unobserve(m)`. Interventions are part of the model's syntax,
so `infer(do_intervention(m, :X => :x), :Y)` is the interventional
distribution `P(Y | do(X = x))`, while `infer(observe(m, :X => :x), :Y)` is
the conditional `P(Y | X = x)`.

Throws [`CompileError`](@ref) (or a `BayesianNetworks` exception) when the
model cannot be compiled, [`ScopeError`](@ref) for unknown or repeated
query variables, unknown evidence variables, or a query variable that also
carries evidence, and `BayesianNetworks.ImpossibleEvidenceError` for a
non-empty query when the evidence has probability exactly zero, as
`BayesianNetworks.marginal` does. Evidence whose binary64 mass underflows is
answered by the log-domain fallback (ADR 0014). When the model holds tolerated
entries in `[-atol, 0)` and the evidence mass is within the tolerance budget
`(1 + atol)^n - 1` of zero, `BayesianNetworks.IndeterminatePosteriorError` is raised.

```jldoctest
julia> using BayesianNetworks

julia> m = reference_habitat_model();

julia> p, diag = infer(m, :Occupancy);

julia> round.(p.table; digits=4)
2-element Vector{Float64}:
 0.5238
 0.4762

julia> p ≈ Factor(marginal(m, :Occupancy), :Occupancy)
true

julia> infer(observe(m, :Vegetation => :dense), :Occupancy)[1] ≈
       infer(m, :Occupancy; evidence=Dict(:Vegetation => :dense))[1]
true
```
"""
function infer(m::BayesModel, query::AbstractVector{Symbol};
               evidence=Dict{Symbol,Symbol}(),
               backend::InferenceBackend=VariableElimination(),
               atol::Real=BayesianNetworks.DEFAULT_ATOL)
    fg = compile(m; atol=atol)
    ev = _model_evidence(m, evidence)
    # An empty query returns the unnormalised mass (ADR 0011), which may be tiny or negative.
    isempty(query) || _check_tolerance_budget(fg, ev, atol)
    return infer(fg, query; evidence=ev, backend)
end
infer(m::BayesModel, query::Symbol; kwargs...) = infer(m, [query]; kwargs...)

function log_evidence_probability(m::BayesModel; evidence=Dict{Symbol,Symbol}(),
                                  order::EliminationStrategy=MinFill(),
                                  atol::Real=BayesianNetworks.DEFAULT_ATOL)
    return log_evidence_probability(compile(m; atol); evidence=_model_evidence(m, evidence),
                                    order)
end

"""
    all_marginals(m::BayesModel; evidence=Dict{Symbol,Symbol}(), backend=JunctionTree())
        -> Dict{Symbol,Factor}

The posterior marginal of every variable of a `BayesianNetworks.BayesModel`:
the model is [`compile`](@ref)d and the [`FactorGraph`](@ref) method of
[`all_marginals`](@ref) is applied with the same evidence semantics as the
model-level [`infer`](@ref) (recorded evidence merged with the explicit
evidence, explicit entries winning). Observed variables map to point masses.

```jldoctest
julia> using BayesianNetworks

julia> ms = all_marginals(reference_habitat_model(); evidence=Dict(:Vegetation => :dense));

julia> round.(ms[:Occupancy].table; digits=4)
2-element Vector{Float64}:
 0.3325
 0.6675

julia> ms[:Vegetation].table
3-element Vector{Float64}:
 0.0
 0.0
 1.0
```
"""
function all_marginals(m::BayesModel; evidence=Dict{Symbol,Symbol}(),
                       backend::InferenceBackend=JunctionTree(),
                       atol::Real=BayesianNetworks.DEFAULT_ATOL)
    fg = compile(m; atol=atol)
    ev = _model_evidence(m, evidence)
    _check_tolerance_budget(fg, ev, atol)
    return all_marginals(fg; evidence=ev, backend)
end

"""
    posterior(m::BayesModel, var::Symbol; evidence=Dict{Symbol,Symbol}(), backend=VariableElimination())
        -> Dict{Symbol,Float64}

The posterior marginal of one variable as a dictionary from state to
probability: [`infer`](@ref) on the model without the diagnostics, for reading off numbers. Same keyword arguments and
evidence semantics as `infer`.

```jldoctest
julia> using BayesianNetworks

julia> p = posterior(reference_habitat_model(), :Occupancy; evidence=Dict(:Vegetation => :dense));

julia> round(p[:present]; digits=4)
0.6675
```
"""
function posterior(m::BayesModel, var::Symbol; kwargs...)
    f, _ = infer(m, [var]; kwargs...)
    return Dict{Symbol,Float64}(l => p for (l, p) in zip(axis(f, var).labels, f.table))
end

# Sampling
# --------

"""
    ancestral_sample(m::BayesModel, n; rng=Random.default_rng()) -> AncestralSamples

`n` ancestral samples of a closed `BayesianNetworks.BayesModel`: the kernels
of its mechanisms are drawn in topological order, each given the sampled
parents (SPEC section 18), by the kernel-level [`ancestral_sample`](@ref).
Interventions are respected (they are mechanisms); evidence is ignored, so
this samples the prior or interventional distribution, never the posterior.
The columns of the result follow the topological order of the model. Throws
[`CompileError`](@ref) when the model is open or lacks kernels.

`BayesianNetworks.sample(m, n)` draws the same distribution as a vector of
dictionaries; convert its output with `AncestralSamples(m, samples)` to compare it with exact posteriors through [`empirical_marginal`](@ref).
"""
function ancestral_sample(m::BayesModel, n::Integer; rng::AbstractRNG=default_rng(),
                          atol::Real=BayesianNetworks.DEFAULT_ATOL)
    kernels, order, parent_names, _ = _model_kernels(m; atol=atol)
    return ancestral_sample(kernels, order, parent_names, n; rng)
end

"""
    AncestralSamples(m::BayesModel, samples::AbstractVector{<:AbstractDict{Symbol,Symbol}})
        -> AncestralSamples

The samples returned by `BayesianNetworks.sample(m, n)` (one dictionary from
variable to state per draw) as an [`AncestralSamples`](@ref) table over the
variables of `m` in topological order, so that [`empirical_marginal`](@ref)
can turn them into factors. Every dictionary must assign every variable
([`ScopeError`](@ref) otherwise) one of its states
(`FiniteKernels.InvalidAxisError` otherwise).
"""
function AncestralSamples(m::BayesModel,
                          samples::AbstractVector{<:AbstractDict{Symbol,Symbol}})
    bn = syntax(m)
    order = Symbol[variable_name(bn, v) for v in topological_order(bn)]
    axes = FiniteAxis[FiniteAxis(x, states(bn, x)) for x in order]
    n = length(samples)
    states_ = Matrix{Symbol}(undef, n, length(order))
    for (i, s) in enumerate(samples)
        absent = [x for x in order if !haskey(s, x)]
        isempty(absent) ||
            throw(ScopeError(:AncestralSamples, "sample $i does not assign every variable",
                             absent))
        for (j, x) in enumerate(order)
            label_index(axes[j], s[x])    # validates the state
            states_[i, j] = s[x]
        end
    end
    return AncestralSamples(order, axes, states_)
end

"""
    empirical_marginal(m::BayesModel, samples, vars) -> Factor

The relative frequencies of `vars` (a variable or a vector of variables) among
`samples` drawn by `BayesianNetworks.sample(m, n)`, as a normalised factor:
`empirical_marginal(AncestralSamples(m, samples), vars)`.
"""
function empirical_marginal(m::BayesModel,
                            samples::AbstractVector{<:AbstractDict{Symbol,Symbol}},
                            vars::AbstractVector{Symbol})
    return empirical_marginal(AncestralSamples(m, samples), vars)
end
function empirical_marginal(m::BayesModel,
                            samples::AbstractVector{<:AbstractDict{Symbol,Symbol}},
                            var::Symbol)
    return empirical_marginal(m, samples, [var])
end
