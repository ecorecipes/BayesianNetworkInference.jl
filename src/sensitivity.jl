# Information-theoretic sensitivity analysis: the entropy of a posterior, the
# mutual information of two variables under the model, the ranking of every
# variable by how much it would reduce the uncertainty of a target (the
# entropy-reduction metric of Marcot 2012, "variance reduction"'s discrete
# counterpart), and a findings tornado.
#
# Cited from the docstrings below with the keys of the shared bibliography
# (`docs/references.bib`): Marcot2012, ChenPollino2012. The information
# measures themselves are standard (Cover and Thomas 2006).
#
# A sensitivity analysis says which variables the model's answer depends on.
# It is not a validation: see `scores.jl` for out-of-sample scoring.

# Shannon entropy of a probability vector, with 0 log 0 = 0.
function _entropy(p::AbstractArray{<:Real}, base::Real)
    total = 0.0
    for q in p
        q > 0 && (total -= q * log(q))
    end
    return total / log(base)
end

"""
    entropy(m::BayesModel, x::Symbol; evidence=Dict{Symbol,Symbol}(), backend=VariableElimination(), base=2)
        -> Float64
    entropy(fg::FactorGraph, x; evidence=..., backend=..., base=2)

The Shannon entropy `H(x | evidence)` of the posterior of `x`, in units of
`log(base)` (bits by default, nats with `base = exp(1)`). `x` may be a single
variable or a vector, in which case the joint posterior is used, so that
`H(x, y) = entropy(m, [x, y])`. The convention `0 log 0 = 0` applies.

The posterior comes from [`infer`](@ref), so the evidence semantics are the
model-level ones: the evidence recorded by `observe` merged with the explicit
`evidence`. So is the rule for tolerated entries in `[-atol, 0)`: on the model method a
posterior is indeterminate (`BayesianNetworks.IndeterminatePosteriorError`) exactly when the
model-level `infer` finds it so, here and in [`mutual_information`](@ref),
[`sensitivity`](@ref) and [`tornado`](@ref).

```jldoctest
julia> using BayesianNetworks

julia> round(entropy(reference_habitat_model(), :Occupancy); digits=4)
0.9984
```
"""
function entropy(fg::FactorGraph, x::AbstractVector{Symbol};
                 evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
                 backend::InferenceBackend=VariableElimination(), base::Real=2)
    base > 1 || throw(ArgumentError("base must be greater than one, got base = $base"))
    f, _ = infer(fg, x; evidence, backend)
    return _entropy(f.table, base)
end
function entropy(fg::FactorGraph, x::Symbol; kwargs...)
    return entropy(fg, [x]; kwargs...)
end
function entropy(m::BayesModel, x::AbstractVector{Symbol}; evidence=Dict{Symbol,Symbol}(),
                 backend::InferenceBackend=VariableElimination(),
                 atol::Real=BayesianNetworks.DEFAULT_ATOL, kwargs...)
    fg = compile(m; atol=atol)
    ev = _model_evidence(m, evidence)
    _check_labels(m, x, ev)
    return _under_tolerance(fg, ev, _tolerance_budget(fg, atol), backend) do
        return entropy(fg, x; evidence=ev, backend, kwargs...)
    end
end
entropy(m::BayesModel, x::Symbol; kwargs...) = entropy(m, [x]; kwargs...)

"""
    mutual_information(m::BayesModel, x::Symbol, y::Symbol; evidence=Dict{Symbol,Symbol}(),
                       backend=VariableElimination(), base=2) -> Float64
    mutual_information(fg::FactorGraph, x, y; evidence=..., backend=..., base=2)

The mutual information

```math
I(x; y) = \\sum_{a, b} P(x = a, y = b) \\log \\frac{P(x = a, y = b)}{P(x = a) P(y = b)}
```

of two variables under the model, conditioned on `evidence`, in units of
`log(base)`. It is computed from the joint posterior returned by
[`infer`](@ref), so it is exact, symmetric, non-negative, zero exactly when
`x` and `y` are conditionally independent given the evidence (in particular
when they are d-separated by it), and equal to the entropy reduction
`H(x) - H(x | y)` averaged over `y`.

Throws [`ScopeError`](@ref) if `x == y`, if either variable carries evidence,
or if either is unknown to a factor graph; on the model method an unknown variable is
`BayesianNetworks.UnknownVariableError` and an unknown evidence state
`BayesianNetworks.UnknownStateError`, as for [`infer`](@ref) (ADR 0015).
"""
function mutual_information(fg::FactorGraph, x::Symbol, y::Symbol;
                            evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
                            backend::InferenceBackend=VariableElimination(), base::Real=2)
    base > 1 || throw(ArgumentError("base must be greater than one, got base = $base"))
    x == y &&
        throw(ScopeError(:mutual_information,
                         "mutual information needs two distinct variables", [x]))
    joint, _ = infer(fg, [x, y]; evidence, backend)
    px = marginalize(joint, y).table
    py = marginalize(joint, x).table
    total = 0.0
    for i in axes(joint.table, 1), j in axes(joint.table, 2)
        p = joint.table[i, j]
        # A positive cell has positive marginals unless a cell is negative, which only a
        # model's prior can return (tolerance included); such a term is skipped, as
        # `entropy` skips a cell that is not positive.
        p > 0 && px[i] > 0 && py[j] > 0 && (total += p * log(p / (px[i] * py[j])))
    end
    return max(total / log(base), 0.0)
end
function mutual_information(m::BayesModel, x::Symbol, y::Symbol;
                            evidence=Dict{Symbol,Symbol}(),
                            backend::InferenceBackend=VariableElimination(),
                            atol::Real=BayesianNetworks.DEFAULT_ATOL, kwargs...)
    fg = compile(m; atol=atol)
    ev = _model_evidence(m, evidence)
    _check_labels(m, (x, y), ev)
    return _under_tolerance(fg, ev, _tolerance_budget(fg, atol), backend) do
        return mutual_information(fg, x, y; evidence=ev, backend, kwargs...)
    end
end

"""
    sensitivity(m::BayesModel, target::Symbol; evidence=Dict{Symbol,Symbol}(),
                variables=nothing, backend=VariableElimination(), base=2)
        -> Vector{@NamedTuple{variable::Symbol, mutual_information::Float64, entropy_reduction::Float64}}
    sensitivity(fg::FactorGraph, target; ...)

Rank every other variable of the model by its
[`mutual_information`](@ref) with `target`, largest first. This is the
entropy-reduction (mutual information) sensitivity metric of
[Marcot2012](@cite): for each
variable `X`, `mutual_information` is `I(target; X | evidence)` in units of
`log(base)`, and `entropy_reduction` is the same quantity as a fraction of
`H(target | evidence)`, that is the proportion of the target's remaining
uncertainty that observing `X` would remove on average.

Variables carrying evidence, and the target itself, are skipped;
`variables` restricts the ranking to a chosen list. A variable d-separated
from the target by the evidence scores exactly zero. On the model method an unknown
target or listed variable is `BayesianNetworks.UnknownVariableError` (ADR 0015); on the
factor-graph method it is [`ScopeError`](@ref).

A sensitivity ranking describes the model, not the world: it says which
observations would change the answer, not whether the answer is right. Pair it
with [`evaluate`](@ref) ([ChenPollino2012](@cite)).

```jldoctest
julia> using BayesianNetworks

julia> s = sensitivity(reference_habitat_model(), :Occupancy);

julia> first(s).variable
:HabitatQuality
```
"""
function sensitivity(fg::FactorGraph, target::Symbol;
                     evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
                     variables::Union{Nothing,AbstractVector{Symbol}}=nothing,
                     backend::InferenceBackend=VariableElimination(), base::Real=2)
    haskey(fg.axes, target) ||
        throw(ScopeError(:sensitivity, "the target is not a variable of the factor graph",
                         [target]))
    candidates = variables === nothing ? _variables(fg.factors) :
                 collect(Symbol, variables)
    candidates = Symbol[v for v in candidates if v != target && !haskey(evidence, v)]
    h = entropy(fg, target; evidence, backend, base)
    rows = map(candidates) do v
        mi = mutual_information(fg, target, v; evidence, backend, base)
        return (variable=v, mutual_information=mi,
                entropy_reduction=h > 0 ? mi / h : 0.0)
    end
    return sort!(collect(rows); by=r -> (-r.mutual_information, r.variable))
end
function sensitivity(m::BayesModel, target::Symbol; evidence=Dict{Symbol,Symbol}(),
                     variables::Union{Nothing,AbstractVector{Symbol}}=nothing,
                     backend::InferenceBackend=VariableElimination(),
                     atol::Real=BayesianNetworks.DEFAULT_ATOL, kwargs...)
    fg = compile(m; atol=atol)
    ev = _model_evidence(m, evidence)
    _check_labels(m, variables === nothing ? [target] : vcat(target, variables), ev)
    # Every entropy and mutual information below conditions on the same evidence.
    return _under_tolerance(fg, ev, _tolerance_budget(fg, atol), backend) do
        return sensitivity(fg, target; evidence=ev, variables, backend, kwargs...)
    end
end

"""
    tornado(m::BayesModel, target::Symbol, state::Symbol; evidence=Dict{Symbol,Symbol}(),
            variables=nothing, backend=VariableElimination())
        -> Vector{@NamedTuple{variable::Symbol, low::Float64, high::Float64, range::Float64,
                              low_state::Symbol, high_state::Symbol}}
    tornado(fg::FactorGraph, target, state; ...)

The data for a tornado diagram of *findings*: for every other variable `X`,
the smallest and largest value of `P(target = state | evidence, X = x)` as `x`
ranges over the states of `X`, the width of that interval, and the states
attaining the two ends. Rows are sorted by decreasing `range`, which is the
order a tornado plot draws them in.

Where [`sensitivity`](@ref) averages over what `X` might turn out to be, this
reports the extremes, so it answers "how far could one observation move the
answer?" rather than "how much would it tell me on average". A state whose
finding is impossible given the evidence (its query throws
`BayesianNetworks.ImpossibleEvidenceError`) is skipped, and a variable with no
possible state is omitted. The evidence itself is checked first: if it is
impossible, `ImpossibleEvidenceError` is thrown rather than an empty table
returned. On the model method an unknown `target` or listed variable is
`BayesianNetworks.UnknownVariableError` and an unknown `state`
`BayesianNetworks.UnknownStateError` (ADR 0015); on the factor-graph method they are
[`ScopeError`](@ref) and `FiniteKernels`' `InvalidAxisError`. The model method asks for the
base query and every finding, whose mass can be far smaller than the evidence's, under the
rule for tolerated entries of the model-level [`infer`](@ref), and raises
`BayesianNetworks.IndeterminatePosteriorError` for one that is indeterminate.

This is a sensitivity to findings, not to parameters: it does not perturb any
CPT.
"""
function tornado(fg::FactorGraph, target::Symbol, state::Symbol;
                 evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}(),
                 variables::Union{Nothing,AbstractVector{Symbol}}=nothing,
                 backend::InferenceBackend=VariableElimination())
    return _tornado(fg, target, state, evidence, variables, backend, nothing)
end

# `budget` is the model's tolerance budget (`_tolerance_budget`): the base query and every
# finding, whose mass can be far smaller, are asked for under the rule for tolerated
# entries; `nothing` on a factor graph, which has no tolerance.
function _tornado(fg::FactorGraph, target::Symbol, state::Symbol, evidence, variables,
                  backend::InferenceBackend, budget)
    haskey(fg.axes, target) ||
        throw(ScopeError(:tornado, "the target is not a variable of the factor graph",
                         [target]))
    k = label_index(fg.axes[target], state)
    candidates = variables === nothing ? _variables(fg.factors) :
                 collect(Symbol, variables)
    candidates = Symbol[v for v in candidates if v != target && !haskey(evidence, v)]
    # Impossible base evidence makes every finding impossible, and skipping them all would
    # answer, with an empty table, a question that has no answer. Query it once, so that
    # the skip below only ever drops findings that the evidence rules out.
    _under_tolerance(fg, evidence, budget, backend) do
        return infer(fg, [target]; evidence, backend)
    end
    rows = NamedTuple{(:variable, :low, :high, :range, :low_state, :high_state),
                      Tuple{Symbol,Float64,Float64,Float64,Symbol,Symbol}}[]
    for v in candidates
        best = nothing
        worst = nothing
        for label in fg.axes[v].labels
            ev = merge(Dict{Symbol,Symbol}(evidence), Dict(v => label))
            p = try
                _under_tolerance(fg, ev, budget, backend) do
                    return first(infer(fg, [target]; evidence=ev, backend)).table[k]
                end
            catch e
                e isa ImpossibleEvidenceError && continue   # impossible finding
                rethrow()
            end
            (worst === nothing || p < worst[1]) && (worst = (p, label))
            (best === nothing || p > best[1]) && (best = (p, label))
        end
        best === nothing && continue
        push!(rows,
              (variable=v, low=worst[1], high=best[1], range=best[1] - worst[1],
               low_state=worst[2], high_state=best[2]))
    end
    return sort!(rows; by=r -> (-r.range, r.variable))
end
function tornado(m::BayesModel, target::Symbol, state::Symbol;
                 evidence=Dict{Symbol,Symbol}(),
                 variables::Union{Nothing,AbstractVector{Symbol}}=nothing,
                 backend::InferenceBackend=VariableElimination(),
                 atol::Real=BayesianNetworks.DEFAULT_ATOL)
    fg = compile(m; atol=atol)
    ev = _model_evidence(m, evidence)
    _check_labels(m, variables === nothing ? Symbol[] : variables, ev)
    _check_state(m, target, state)
    return _tornado(fg, target, state, ev, variables, backend, _tolerance_budget(fg, atol))
end
