# The bridge from BayesianNetworks.jl: compile a `BayesModel` to a `FactorGraph`,
# one factor per mechanism (SPEC section 15).

"""
    FactorGraphBackend()

The compilation target of [`compile`](@ref): a [`FactorGraph`](@ref) with one
factor per mechanism of the model.
"""
struct FactorGraphBackend end

"""
    CompileError(msg, variables)

Thrown by [`compile`](@ref) when a `BayesModel` cannot be turned into a factor
graph: it is open (`variables` are the exogenous variables, which have no
mechanism) or some mechanisms have no kernel (`variables` are their targets,
see `BayesianNetworks.missing_kernels`). Structural problems (cycles,
duplicate names) and ill-fitting kernels are reported by
`BayesianNetworks.validate` with its own typed exceptions.
"""
struct CompileError <: Exception
    msg::String
    variables::Vector{Symbol}
end

function Base.showerror(io::IO, e::CompileError)
    return print(io, "CompileError: ", e.msg, " (variables ", e.variables, ")")
end

# The kernels, topological order and parent names of a closed model with full
# semantics, in the shape that `ancestral_sample` and `compile` consume.
# Throws `CompileError` for an open model or missing kernels, and the typed
# exceptions of `BayesianNetworks.validate` for anything else.
function _model_kernels(m::BayesModel)
    bn = syntax(m)
    open_vars = Symbol[variable_name(bn, v) for v in exogenous(bn)]
    isempty(open_vars) ||
        throw(CompileError("the model is open: every variable needs a mechanism",
                           open_vars))
    absent = missing_kernels(m)
    isempty(absent) ||
        throw(CompileError("the model has mechanisms without a kernel; bind them with bind_kernel or bind_cpt",
                           absent))
    BayesianNetworks.validate(m; closed=true, unique_names=true, semantics=true)
    order = Symbol[variable_name(bn, v) for v in topological_order(bn)]
    kernels = Dict{Symbol,FiniteKernel}()
    parent_names = Dict{Symbol,Vector{Symbol}}()
    mechanism = Dict{Symbol,Tuple{Symbol,Int}}()
    for x in order
        mech = mechanism_of(bn, x)
        kernels[x] = kernel(m, x)
        parent_names[x] = Symbol[variable_name(bn, p) for p in inputs(bn, mech)]
        mechanism[x] = (mechanism_name(bn, mech), mech)
    end
    return kernels, order, parent_names, mechanism
end

"""
    compile(m::BayesModel, backend=FactorGraphBackend()) -> FactorGraph

Compile a closed `BayesianNetworks.BayesModel` with full semantics to a
[`FactorGraph`](@ref): for every mechanism `kappa_X(x | p_1, ..., p_k)` the
factor `Factor(kernel(m, X), parents, X)` with scope `(p_1, ..., p_k, X)`
(parents in `input_position` order) and the parents-first table `cpt(k)`
(ADR 0002). Factors are listed in topological order of their targets, so
[`variables`](@ref)`(fg)` is a topological order of the model, and
`fg.provenance[i]` is the named tuple `(variable, mechanism, id)` with the
target, the mechanism name and its part id.

Hard interventions (`do_intervention`) are ordinary mechanisms with a
point-mass reference, materialised by `BayesianNetworks.kernel`, so an
intervened model compiles unchanged; the evidence recorded by `observe` is
not part of the factor graph (it is applied by [`infer`](@ref)).

Throws [`CompileError`](@ref) naming the offending variables when the model is
open or a mechanism has no kernel, and the exceptions of
`BayesianNetworks.validate` for structural or binding problems.

```jldoctest
julia> using BayesianNetworks

julia> fg = compile(reference_habitat_model())
FactorGraph{Float64} with 7 factors over 7 variables

julia> scope(fg.factors[3])
3-element Vector{Symbol}:
 :Climate
 :Irrigation
 :SoilMoisture

julia> fg.provenance[1]
(variable = :Climate, mechanism = :Climate_mechanism, id = 5)
```
"""
function compile(m::BayesModel, ::FactorGraphBackend=FactorGraphBackend())
    kernels, order, parent_names, mechanism = _model_kernels(m)
    factors = Factor{Float64}[]
    provenance = Any[]
    for x in order
        f = Factor(kernels[x], parent_names[x], x)
        push!(factors, _convert_factor(Float64, f))
        name, id = mechanism[x]
        push!(provenance, (variable=x, mechanism=name, id=id))
    end
    return FactorGraph{Float64}(factors, provenance)
end
