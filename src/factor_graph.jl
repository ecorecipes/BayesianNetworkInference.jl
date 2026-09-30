# A bag of factors over shared variables, and its interaction graph.

"""
    FactorGraph(factors; provenance = nothing, check = true, atol = DEFAULT_ATOL)
    FactorGraph{T}(factors, provenance)

A collection of factors whose product is an (unnormalised) joint distribution.
`axes` maps every variable to its axis; factors that disagree about the states
of a shared variable are rejected with [`ShapeError`](@ref). `provenance[i]`
records where `factors[i]` came from (a mechanism or variable id, filled in by
the model bridge; `nothing` when built by hand).

With `check = true` (the default) every entry must be finite and at least `-atol`,
else [`FactorEntryError`](@ref): the same contract a `FiniteKernel` has (ADR 0007), since
a factor graph's product is meant to be a measure. The check runs once per graph, not per
query. Entries in `[-atol, 0)` are tolerated, as rounded tables produce them;
the log-domain backends still reject them with [`FactorDomainError`](@ref), because
they have no logarithm. `check = false` skips the check, and the inner constructor
`FactorGraph{T}(factors, provenance)` never runs it: [`compile`](@ref) uses it, because
the model was already validated at its own `atol`, and so does a caller whose factors are
not a measure (InfluenceDiagrams orders signed utility potentials with one).

```jldoctest
julia> a = FiniteAxis(:A, [:a0, :a1]); b = FiniteAxis(:B, [:b0, :b1]);

julia> fg = FactorGraph([Factor(a, [0.3, 0.7]), Factor([a, b], [0.9 0.1; 0.2 0.8])]);

julia> variables(fg)
2-element Vector{Symbol}:
 :A
 :B
```
"""
struct FactorGraph{T<:Real}
    factors::Vector{Factor{T}}
    axes::Dict{Symbol,FiniteAxis}
    provenance::Vector{Any}
    function FactorGraph{T}(factors::AbstractVector{<:Factor},
                            provenance::AbstractVector) where {T<:Real}
        length(provenance) == length(factors) ||
            throw(ShapeError(:FactorGraph, "one provenance entry per factor",
                             length(factors), length(provenance)))
        axes = Dict{Symbol,FiniteAxis}()
        for f in factors, (v, a) in zip(f.vars, f.axes)
            b = get(axes, v, nothing)
            if b === nothing
                axes[v] = a
            elseif b != a
                throw(ShapeError(:FactorGraph,
                                 "factors disagree about the states of $(repr(v))",
                                 b.labels, a.labels))
            end
        end
        fs = Factor{T}[_convert_factor(T, f) for f in factors]
        return new{T}(fs, axes, collect(Any, provenance))
    end
end

_convert_factor(::Type{T}, f::Factor{T}) where {T} = f
function _convert_factor(::Type{T}, f::Factor) where {T}
    return _factor(f.vars, f.axes, convert(Array{T}, f.table))
end

function FactorGraph(factors::AbstractVector{<:Factor};
                     provenance::Union{Nothing,AbstractVector}=nothing, check::Bool=true,
                     atol::Real=BayesianNetworks.DEFAULT_ATOL)
    T = isempty(factors) ? Float64 : mapreduce(eltype, promote_type, factors)
    prov = provenance === nothing ? fill(nothing, length(factors)) : provenance
    check && foreach(f -> _check_factor_entries(f, atol), factors)
    return FactorGraph{T}(factors, prov)
end

function _check_factor_entries(f::Factor, atol::Real)
    for ci in CartesianIndices(f.table)
        v = f.table[ci]
        (isfinite(v) && v >= -atol) ||
            throw(FactorEntryError(copy(f.vars), ci, Float64(v), Float64(atol)))
    end
    return nothing
end

Base.length(fg::FactorGraph) = length(fg.factors)
Base.eltype(::Type{FactorGraph{T}}) where {T} = T

function Base.show(io::IO, fg::FactorGraph{T}) where {T}
    return print(io, "FactorGraph{", T, "} with ", length(fg.factors), " factors over ",
                 length(fg.axes), " variables")
end

"""
    variables(fg::FactorGraph) -> Vector{Symbol}

The variables of a factor graph in a fixed order: first appearance across the
factor scopes. This is the vertex order of [`interaction_graph`](@ref).
"""
variables(fg::FactorGraph) = _variables(fg.factors)

function _variables(factors::AbstractVector{<:Factor})
    vars = Symbol[]
    seen = Set{Symbol}()
    for f in factors, v in f.vars
        v in seen && continue
        push!(seen, v)
        push!(vars, v)
    end
    return vars
end

"""
    interaction_graph(fg::FactorGraph) -> (graph, vars, index)

The undirected interaction (Markov) graph of the factors: one vertex per
variable in the order of [`variables`](@ref) and an edge between every pair of
variables that share a factor. Because each factor of a compiled Bayesian
network spans a child and all of its parents, this is already the moral graph.
Returns the `Graphs.SimpleGraph`, the vertex-to-variable vector `vars`, and
the variable-to-vertex `index::Dict{Symbol,Int}`.
"""
interaction_graph(fg::FactorGraph) = _interaction_graph(fg.factors, variables(fg))

function _interaction_graph(factors::AbstractVector{<:Factor}, vars::Vector{Symbol})
    index = Dict{Symbol,Int}(v => i for (i, v) in enumerate(vars))
    n = length(vars)
    I = Int[]
    J = Int[]
    for f in factors
        ids = [index[v] for v in f.vars]
        for a in ids, b in ids
            a == b && continue
            push!(I, a)
            push!(J, b)
        end
    end
    A = sparse(I, J, trues(length(I)), n, n, (x, y) -> x | y)
    return SimpleGraph(A), vars, index
end
