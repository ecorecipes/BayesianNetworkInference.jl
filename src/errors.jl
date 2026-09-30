# Exceptions of BayesianNetworkInference (ADR 0013). Every exception the package defines
# lives in this file, with its `showerror` method, and subtypes `InferenceError`, the
# package's own root, which subtypes `BayesianNetworks`' `BayesNetError`: this package is
# built on `BayesianNetworks`, so its errors belong to that tier. The module includes this
# file first; every field type comes from Base.
#
# - `ScopeError` is a frozen name. The pinned constructor-schedules checker compares the
#   recorded `string(typeof(e))` with "ScopeError", so the type keeps its name, its
#   defining module (this one) and its unqualified printing. Only its supertype may change.
# - Carry the offending variables in the fields and the message. Library code never calls
#   a bare `error("...")`.
# - Invalid arguments and keywords raise `ArgumentError`, not a type from this file;
#   `normalize(::Factor)` on a zero total is one (ADR 0012).
# - Typed errors of lower packages pass through unchanged where nothing is added:
#   `FiniteKernels`' `InvalidAxisError` for an unknown label and its kernel errors from
#   `FiniteKernel(f, inputs, outputs)`, `BayesianNetworks.validate`'s errors from `compile`,
#   and `BayesianNetworks`' `ImpossibleEvidenceError` for evidence of zero mass. The module
#   re-exports their roots, the FiniteKernels types and `ImpossibleEvidenceError`, each as
#   its owner's binding.
# - A docstring names another package's type as a code span, never with `@ref`.

"""
    InferenceError

Abstract supertype of every exception that BayesianNetworkInference defines:
[`ScopeError`](@ref), [`ShapeError`](@ref), [`CompileError`](@ref),
[`FactorDomainError`](@ref), [`FactorEntryError`](@ref) and [`TraceLimitError`](@ref). It subtypes `BayesianNetworks`' `BayesNetError`, the root
of `BayesianNetworks` and of every package built on it (ADR 0013). A root marks the
dependency tier that introduced an error, not a kind of failure, so a `catch` on
`BayesNetError` covers these errors too.

Not every exception that this package's functions raise is an `InferenceError`. Where the
package adds nothing to a typed error of a lower package, that error passes through
unchanged:

- `BayesianNetworks`' `ImpossibleEvidenceError`, for evidence of probability exactly
  zero, from every posterior entry point (ADRs 0012 and 0014), and its
  `IndeterminatePosteriorError`, when tolerated negative entries leave the posterior
  undetermined;
- the exceptions of `BayesianNetworks.validate`, from [`compile`](@ref) and the
  model-level entry points;
- `FiniteKernels`' errors, under its root `FiniteKernelsError`: `InvalidAxisError` for a
  label that a variable does not have, and the kernel errors of
  `FiniteKernel(f, inputs, outputs)`, such as `KernelNormalizationError`.

`BayesNetError`, `ImpossibleEvidenceError`, `FiniteKernelsError` and its five types,
`BayesianNetworkFormatsError` (the root only), and `AnyBayesNetError`, which catches every
typed exception of the ecosystem, are re-exported.
Invalid arguments and keywords raise Base's `ArgumentError`, as does [`normalize`](@ref)
on a factor whose total is zero; it is outside every root.

Two errors of the same concrete type are `==` when their fields are pairwise `isequal`, by
`BayesNetError`'s structural equality, and they hash alike.

```jldoctest
julia> f = Factor([FiniteAxis(:A, [:a0, :a1])], [0.3, 0.7]);

julia> try
           marginalize(f, [:B])
       catch e
           (typeof(e), e isa InferenceError, e isa BayesNetError)
       end
(ScopeError, true, true)
```
"""
abstract type InferenceError <: BayesNetError end

"""
    ScopeError(operation, msg, vars)

Thrown when the variables handed to `operation` (a `Symbol`) are not compatible
with the scope of a factor: unknown variables, duplicates, or an order that is
not a permutation of the scope. `vars` lists the offending variables.
"""
struct ScopeError <: InferenceError
    operation::Symbol
    msg::String
    vars::Vector{Symbol}
end

function Base.showerror(io::IO, e::ScopeError)
    return print(io, "ScopeError in ", e.operation, ": ", e.msg, " (variables ", e.vars,
                 ")")
end

"""
    ShapeError(operation, msg, expected, got)

Thrown when a table does not have the shape implied by its axes, or when two
factors disagree about the states of a shared variable.
"""
struct ShapeError <: InferenceError
    operation::Symbol
    msg::String
    expected::Any
    got::Any
end

function Base.showerror(io::IO, e::ShapeError)
    return print(io, "ShapeError in ", e.operation, ": ", e.msg, " (expected ", e.expected,
                 ", got ", e.got, ")")
end

"""
    CompileError(msg, variables)

Thrown by [`compile`](@ref) when a `BayesModel` cannot be turned into a factor
graph: it is open (`variables` are the exogenous variables, which have no
mechanism) or some mechanisms have no kernel (`variables` are their targets,
see `BayesianNetworks.missing_kernels`). Structural problems (cycles,
duplicate names) and ill-fitting kernels are reported by
`BayesianNetworks.validate` with its own typed exceptions.
"""
struct CompileError <: InferenceError
    msg::String
    variables::Vector{Symbol}
end

function Base.showerror(io::IO, e::CompileError)
    return print(io, "CompileError: ", e.msg, " (variables ", e.variables, ")")
end

"""
    FactorDomainError(backend, vars, index, value)

A factor entry that `backend`'s arithmetic cannot take (ADR 0015). The entry is valid in a
factor graph, which accepts finite entries down to `-atol` ([`FactorEntryError`](@ref)
otherwise), but this backend needs more:

- `:log_domain` ([`LogVariableElimination`](@ref), [`LogJunctionTree`](@ref),
  [`log_evidence_probability`](@ref)) needs finite, nonnegative entries whose logarithm is
  representable;
- `:trace_variable_elimination` ([`trace_variable_elimination`](@ref)) needs finite,
  nonnegative inputs;
- `:stable_decision_elimination` (InfluenceDiagrams' exact decision elimination) needs
  finite, nonnegative probabilities, since exact arithmetic does not accept tolerated
  negative entries.

`vars` is the factor's scope, `index` the entry's position in its table and `value` the
entry. Small negative values are rejected, not clamped to zero. It replaces
`LogFactorDomainError`, which only the log domain raised.
"""
struct FactorDomainError <: InferenceError
    backend::Symbol
    vars::Vector{Symbol}
    index::Tuple
    value::Real
end
function Base.showerror(io::IO, e::FactorDomainError)
    return print(io, "FactorDomainError: the ", e.backend,
                 " backend cannot take the entry ",
                 e.value, " at ", e.index, " of the factor over ", e.vars)
end

"""
    TraceLimitError(trace, limit, detail, vars)

An execution trace cannot record this run (ADR 0015). `trace` is the tracing function
(`:trace_variable_elimination`, or InfluenceDiagrams' `:trace_decision_elimination`),
`limit` names the limit that was reached and `detail` says why:

- `:cells`, `:compilation_cells`: a table would exceed the `max_entries` budget;
- `:rational_digits`: an exact rational would exceed the profile's digit limit;
- `:scalar_type`: the v1 profile records Float64 factors only;
- `:evidence_underflow`: the evidence mass underflowed, and the v1 profile records Float64
  execution, so it cannot fall back to the log domain as the backends do (ADR 0014).

`vars` names the factor or variables concerned. These are limits of the trace format, not
properties of the model: the same query without a trace succeeds.
"""
struct TraceLimitError <: InferenceError
    trace::Symbol
    limit::Symbol
    detail::String
    vars::Vector{Symbol}
end
function Base.showerror(io::IO, e::TraceLimitError)
    return print(io, "TraceLimitError in ", e.trace, " (", e.limit, "): ", e.detail,
                 isempty(e.vars) ? "" : string(" [", join(e.vars, ", "), "]"))
end

"""
    FactorEntryError(vars, index, value, atol)

A factor handed to [`FactorGraph`](@ref) has an entry that is not finite or is below
`-atol`. `vars` is the factor's scope, `index` the entry's position in its table (in
scope order) and `value` the entry. `FactorGraph(...; check = false)` skips the check.
"""
struct FactorEntryError <: InferenceError
    vars::Vector{Symbol}
    index::CartesianIndex
    value::Float64
    atol::Float64
end

function Base.showerror(io::IO, e::FactorEntryError)
    return print(io, "FactorEntryError: the factor over ", e.vars, " has the entry ",
                 e.value,
                 " at ", Tuple(e.index),
                 "; entries of a factor graph must be finite and at least -",
                 e.atol, " (pass check = false to skip this)")
end

# Internal control-flow signal, never raised to a caller (ADR 0014): the binary64 path found
# an evidence mass that is not a normal positive number, which does not decide whether the
# evidence is impossible. Every public entry point catches it and recomputes in the log
# domain (`_resolving_mass` in variable_elimination.jl).
struct _UnresolvedMass <: InferenceError
    evidence::Dict{Symbol,Symbol}
end
