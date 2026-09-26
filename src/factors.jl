# Finite factors: unnormalised tensors over an ordered variable scope, and the
# (multiply, marginalise, condition, ...) algebra used by variable elimination.

"""
    ScopeError(operation, msg, vars)

Thrown when the variables handed to `operation` (a `Symbol`) are not compatible
with the scope of a factor: unknown variables, duplicates, or an order that is
not a permutation of the scope. `vars` lists the offending variables.
"""
struct ScopeError <: Exception
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
struct ShapeError <: Exception
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
    Factor{T<:Real}(vars::Vector{Symbol}, axes::Vector{FiniteAxis}, table::Array{T})
    Factor(axes::Vector{FiniteAxis}, table::Array)
    Factor(k::FiniteKernel, inputs::Vector{Symbol}, output::Symbol)

A finite factor: a non-negative (in practice) table `table` indexed by an
ordered **scope** `vars`, with `axes[i]` carrying the state labels of `vars[i]`.
The invariants are `length(vars) == length(axes) == ndims(table)`,
`size(table) == Tuple(length.(axes))`, `axes[i].name == vars[i]`, and unique
variables. A factor with empty scope is a scalar (`table` is a 0-dimensional
array).

Factors are deliberately distinct from `FiniteKernel`s: a kernel is a
morphism with an input/output partition and a normalisation invariant, whereas
a factor is an unnormalised tensor closed under pointwise product and
summation. Convert explicitly with `Factor(k, inputs, output)` and
`FiniteKernel(f, inputs, outputs)`.

```jldoctest
julia> a = FiniteAxis(:A, [:a0, :a1]); b = FiniteAxis(:B, [:b0, :b1]);

julia> f = Factor([a, b], [0.9 0.1; 0.2 0.8]);

julia> scope(f), size(f)
([:A, :B], (2, 2))

julia> marginalize(f, :B).table
2-element Vector{Float64}:
 1.0
 1.0
```
"""
struct Factor{T<:Real}
    vars::Vector{Symbol}
    axes::Vector{FiniteAxis}
    table::Array{T}
    function Factor{T}(vars::AbstractVector{Symbol}, axes::AbstractVector{FiniteAxis},
                       table::Array{T}) where {T<:Real}
        n = length(vars)
        length(axes) == n ||
            throw(ShapeError(:Factor, "one axis per scope variable", n, length(axes)))
        ndims(table) == n ||
            throw(ShapeError(:Factor, "table rank must equal the scope length", n,
                             ndims(table)))
        allunique(vars) ||
            throw(ScopeError(:Factor, "scope variables must be unique", collect(vars)))
        for i in 1:n
            axes[i].name == vars[i] ||
                throw(ScopeError(:Factor,
                                 "axis $(repr(axes[i].name)) does not carry the name of scope variable $(repr(vars[i]))",
                                 [vars[i]]))
        end
        expected = Tuple(map(length, axes))
        size(table) == expected ||
            throw(ShapeError(:Factor, "table size must match the axis lengths", expected,
                             size(table)))
        return new{T}(collect(Symbol, vars), collect(FiniteAxis, axes), table)
    end
end

function Factor(vars::AbstractVector{Symbol}, axes::AbstractVector{FiniteAxis},
                table::AbstractArray{T}) where {T<:Real}
    return Factor{T}(vars, axes, convert(Array{T}, table))
end
function Factor(axes::AbstractVector{FiniteAxis}, table::AbstractArray{<:Real})
    return Factor([a.name for a in axes], axes, table)
end
Factor(axis::FiniteAxis, table::AbstractArray{<:Real}) = Factor([axis], table)

# Internal constructor for results whose invariants hold by construction.
function _factor(vars::Vector{Symbol}, axes::Vector{FiniteAxis},
                 table::Array{T}) where {T<:Real}
    return Factor{T}(vars, axes, table)
end

"""
    unit_factor(T=Float64) -> Factor{T}

The multiplicative identity: a scalar factor with empty scope and value one.
"""
function unit_factor(::Type{T}=Float64) where {T<:Real}
    return _factor(Symbol[], FiniteAxis[], fill(one(T)))
end

"""
    scope(f::Factor) -> Vector{Symbol}

The ordered variable scope of a factor.
"""
scope(f::Factor) = f.vars

Base.size(f::Factor) = size(f.table)
Base.ndims(f::Factor) = ndims(f.table)
Base.length(f::Factor) = length(f.table)
Base.eltype(::Type{Factor{T}}) where {T} = T
Base.eltype(::Factor{T}) where {T} = T

_division_type(::Type{T}) where {T<:Real} = typeof(one(T) / one(T))

"""
    axis(f::Factor, var::Symbol) -> FiniteAxis

The axis (state labels) of `var` in the scope of `f`; throws
[`ScopeError`](@ref) if `var` is not in scope.
"""
axis(f::Factor, var::Symbol) = f.axes[_position(f, :axis, var)]

function _position(f::Factor, op::Symbol, var::Symbol)
    i = findfirst(==(var), f.vars)
    i === nothing &&
        throw(ScopeError(op, "variable $(repr(var)) is not in the scope $(f.vars)", [var]))
    return i
end

function _positions(f::Factor, op::Symbol, vars)
    allunique(vars) || throw(ScopeError(op, "variables must be unique", collect(vars)))
    return [_position(f, op, v) for v in vars]
end

"""
    ==(f::Factor, g::Factor)
    isapprox(f::Factor, g::Factor; kwargs...)

Equality modulo axis order: `f` and `g` must have the same scope as a set with
identical axes, and tables that agree (exactly, or within `isapprox` tolerances)
once `g` is reordered to the scope order of `f`.
"""
function Base.:(==)(f::Factor, g::Factor)
    _same_scope(f, g) || return false
    return f.table == reorder(g, f.vars).table
end
function Base.isapprox(f::Factor, g::Factor; kwargs...)
    _same_scope(f, g) || return false
    return isapprox(f.table, reorder(g, f.vars).table; kwargs...)
end
function _same_scope(f::Factor, g::Factor)
    length(f.vars) == length(g.vars) || return false
    for (v, a) in zip(f.vars, f.axes)
        j = findfirst(==(v), g.vars)
        (j !== nothing && g.axes[j] == a) || return false
    end
    return true
end
Base.hash(f::Factor, h::UInt) = hash(Set(f.vars), hash(:Factor, h))

function Base.show(io::IO, f::Factor{T}) where {T}
    return print(io, "Factor{", T, "} over ", Tuple(f.vars), " with size ", size(f))
end

# Multiplication
# --------------

# Table of `h` permuted into the order of `vars` (a superset of h.vars) and
# reshaped with singleton axes so that it broadcasts against a table of `sz`.
function _broadcastable(h::Factor, vars::Vector{Symbol}, sz::Tuple)
    present = [i for i in eachindex(vars) if vars[i] in h.vars]
    perm = [findfirst(==(vars[i]), h.vars) for i in present]
    t = perm == 1:length(perm) ? h.table : permutedims(h.table, perm)
    shape = ntuple(i -> vars[i] in h.vars ? sz[i] : 1, length(vars))
    return reshape(t, shape)
end

function _union_axes(f::Factor, g::Factor, operation::Symbol)
    vars = copy(f.vars)
    axes = copy(f.axes)
    for (v, a) in zip(g.vars, g.axes)
        i = findfirst(==(v), vars)
        if i === nothing
            push!(vars, v)
            push!(axes, a)
        elseif axes[i] != a
            throw(ShapeError(operation,
                             "factors disagree about the states of $(repr(v))",
                             axes[i].labels, a.labels))
        end
    end
    return vars, axes
end

# Stride of each result axis inside `h`'s table, zero where `h` lacks that variable, so
# that a linear walk over the result can index `h` without permuting or reshaping it.
# `vec` of a table whose rank is not known statically is still a `Vector`, so the kernel
# below is type stable where broadcasting over an `Array{T}` of unknown rank is not --
# the same reason `BayesianNetworks`' `_Factor` carries explicit strides.
function _result_strides(h::Factor, vars::Vector{Symbol})
    own = Vector{Int}(undef, length(h.vars))
    acc = 1
    for k in eachindex(h.vars)
        own[k] = acc
        acc *= length(h.axes[k])
    end
    out = zeros(Int, length(vars))
    for (i, v) in enumerate(vars)
        k = findfirst(==(v), h.vars)
        k === nothing || (out[i] = own[k])
    end
    return out
end

# out[i] = a[.] * b[.] over the result's joint states, walking an odometer over `sz`.
function _product_into!(out::Vector{T}, sz::Vector{Int}, a::Vector{S}, as::Vector{Int},
                        b::Vector{U}, bs::Vector{Int}) where {T,S,U}
    d = length(sz)
    pos = zeros(Int, d)
    ai = 1
    bi = 1
    @inbounds for i in eachindex(out)
        out[i] = a[ai] * b[bi]
        for k in 1:d
            pos[k] += 1
            ai += as[k]
            bi += bs[k]
            pos[k] < sz[k] && break
            pos[k] = 0
            ai -= as[k] * sz[k]
            bi -= bs[k] * sz[k]
        end
    end
    return out
end

"""
    multiply(f::Factor, g::Factor) -> Factor
    multiply(fs::Factor...) -> Factor
    f * g

Pointwise product. The scope of the result is `f.vars` followed by the
variables of `g` not already present, so multiplication is commutative modulo
axis order. Shared variables must carry identical axes ([`ShapeError`](@ref)
otherwise). The element type is the promotion of the two element types.
"""
function multiply(f::Factor{S}, g::Factor{U}) where {S,U}
    vars, axes = _union_axes(f, g, :multiply)
    sz = Int[length(a) for a in axes]
    T = promote_type(S, U)
    table = Array{T}(undef, Tuple(sz))
    isempty(table) ||
        _product_into!(vec(table), sz, vec(f.table), _result_strides(f, vars),
                       vec(g.table), _result_strides(g, vars))
    return _factor(vars, axes, table)
end
multiply(f::Factor) = f
function multiply(f::Factor, g::Factor, h::Factor, hs::Factor...)
    return multiply(multiply(f, g), h, hs...)
end
function multiply(fs::AbstractVector{<:Factor})
    isempty(fs) && return unit_factor()
    return reduce(multiply, fs)
end
Base.:*(f::Factor, g::Factor) = multiply(f, g)

# Marginalisation and maximisation
# --------------------------------

function _reduce_out(op, name::Symbol, f::Factor, vars)
    isempty(vars) && return f
    dims = Tuple(_positions(f, name, vars))
    keep = setdiff(1:ndims(f), dims)
    t = dropdims(op(f.table; dims=dims); dims=dims)
    return _factor(f.vars[keep], f.axes[keep], t)
end

"""
    marginalize(f::Factor, vars) -> Factor

Sum out `vars` (a vector of symbols, or one symbol); the remaining scope keeps
its order. Throws [`ScopeError`](@ref) for variables outside the scope.
"""
function marginalize(f::Factor, vars::AbstractVector{Symbol})
    return _reduce_out(sum, :marginalize, f, vars)
end
marginalize(f::Factor, var::Symbol) = marginalize(f, [var])

"""
    maximize(f::Factor, vars) -> Factor

Max out `vars`: the counterpart of [`marginalize`](@ref) for the
(×, max) semiring used by decision variable elimination.
"""
maximize(f::Factor, vars::AbstractVector{Symbol}) = _reduce_out(maximum, :maximize, f, vars)
maximize(f::Factor, var::Symbol) = maximize(f, [var])

"""
    argmax_table(f::Factor, var::Symbol) -> Array{Symbol}

For every configuration of the other variables (indexed in the order of
`setdiff(scope(f), [var])`), the label of `var` at which `f` is maximal. Ties
resolve to the first label. This is the policy table of a decision variable
in decision variable elimination.
"""
function argmax_table(f::Factor, var::Symbol)
    d = _position(f, :argmax_table, var)
    labs = f.axes[d].labels
    idx = dropdims(map(ci -> ci[d], argmax(f.table; dims=d)); dims=d)
    return map(i -> labs[i], idx)
end

# Conditioning, normalisation, reordering
# ---------------------------------------

"""
    condition(f::Factor, evidence::AbstractDict{Symbol,Symbol}) -> Factor

Slice `f` at the observed labels of every scope variable that appears in
`evidence`, dropping those axes; evidence on variables outside the scope is
ignored. Throws `FiniteKernels.InvalidAxisError` for a label the variable
does not have.
"""
function condition(f::Factor, evidence::AbstractDict{Symbol,Symbol})
    any(v -> haskey(evidence, v), f.vars) || return f
    idx = ntuple(i -> haskey(evidence, f.vars[i]) ?
                      label_index(f.axes[i], evidence[f.vars[i]]) : Colon(), ndims(f))
    keep = [i for i in 1:ndims(f) if !haskey(evidence, f.vars[i])]
    t = f.table[idx...]
    table = t isa AbstractArray ? t : fill(t)
    return _factor(f.vars[keep], f.axes[keep], table)
end

"""
    normalize(f::Factor) -> Factor

Divide by the total mass so that the table sums to one. Throws
`FiniteKernels.KernelNormalizationError` when the total is zero. Extends
`LinearAlgebra.normalize`, as `FiniteKernels` does for kernels.
"""
function normalize(f::Factor)
    s = sum(f.table)
    iszero(s) &&
        throw(KernelNormalizationError("cannot normalise a factor over $(f.vars) with zero total mass",
                                       1.0, 0.0))
    # `f.table ./ s` returns a *scalar* for an empty scope, where the table is a 0-d array,
    # so divide in place: an empty-scope factor arises from any maximal clique that evidence
    # fully instantiates. The log-domain path already does this.
    table = similar(f.table, typeof(one(eltype(f.table)) / s))
    table .= f.table ./ s
    return _factor(f.vars, f.axes, table)
end

"""
    reorder(f::Factor, vars::AbstractVector{Symbol}) -> Factor

Permute the scope (and table) of `f` into the order `vars`, which must be a
permutation of `scope(f)` ([`ScopeError`](@ref) otherwise).
"""
function reorder(f::Factor, vars::AbstractVector{Symbol})
    length(vars) == length(f.vars) ||
        throw(ScopeError(:reorder, "expected a permutation of the scope $(f.vars)",
                         collect(vars)))
    perm = _positions(f, :reorder, vars)
    perm == 1:length(perm) && return f
    return _factor(f.vars[perm], f.axes[perm], permutedims(f.table, perm))
end

# Conversion to and from kernels
# ------------------------------

"""
    Factor(k::FiniteKernel, inputs::Vector{Symbol}, output::Symbol) -> Factor

The factor `phi(inputs..., output) = k(output | inputs...)` of a mechanism.
The scope is the first-occurrence order of `(inputs..., output)`. Repeated
variable names identify input slots: the parents-first table `cpt(k)` is
restricted to their diagonal, not merely reshaped. Repeated slots must have
identical state labels. The kernel table is reused when `k` has no inputs.
Axes are renamed to the given variable names while keeping the kernel's
labels. Throws [`ShapeError`](@ref) unless `k` has one output axis and
`length(inputs)` input axes.
"""
function Factor(k::FiniteKernel, inputs::AbstractVector{Symbol}, output::Symbol)
    ndims(k.codom) == 1 ||
        throw(ShapeError(:Factor, "kernel must have exactly one output axis", 1,
                         ndims(k.codom)))
    ndims(k.dom) == length(inputs) ||
        throw(ShapeError(:Factor, "one input name per kernel input axis", ndims(k.dom),
                         length(inputs)))
    axes = FiniteAxis[FiniteAxis(v, labels(a)) for (v, a) in zip(inputs, k.dom.axes)]
    push!(axes, FiniteAxis(output, labels(k.codom.axes[1])))
    table = isempty(inputs) ? k.table : cpt(k)
    names = vcat(collect(inputs), output)
    allunique(names) && return Factor(names, axes, table)
    vars = unique(names)
    positions = Dict(v => i for (i, v) in enumerate(vars))
    unique_axes = FiniteAxis[axes[findfirst(==(v), names)] for v in vars]
    for (v, ax) in zip(names, axes)
        expected = unique_axes[positions[v]]
        labels(ax) == labels(expected) ||
            throw(ShapeError(:Factor, "repeated slots of $(repr(v)) need identical states",
                             labels(expected), labels(ax)))
    end
    diagonal = Array{eltype(table)}(undef, Tuple(length.(unique_axes)))
    for I in CartesianIndices(diagonal)
        diagonal[I] = table[ntuple(j -> I[positions[names[j]]], length(names))...]
    end
    return Factor(vars, unique_axes, diagonal)
end
Factor(k::FiniteKernel, output::Symbol) = Factor(k, Symbol[], output)

"""
    FiniteKernel(f::Factor, inputs, outputs) -> FiniteKernel

Interpret `f` as the kernel `inputs → outputs`: the scope is reordered to
`(outputs..., inputs...)` (the outputs-first internal layout) and the result
is checked for normalisation over the outputs, throwing
`FiniteKernels.KernelNormalizationError` if some input configuration does not sum
to one. `inputs` and `outputs` together must be a permutation of `scope(f)`.
"""
function FiniteKernels.FiniteKernel(f::Factor, inputs::AbstractVector{Symbol},
                                    outputs::AbstractVector{Symbol};
                                    atol::Real=BayesianNetworks.DEFAULT_ATOL)
    g = reorder(f, vcat(collect(outputs), collect(inputs)))
    no = length(outputs)
    dom = FiniteSpace(g.axes[(no + 1):end])
    codom = FiniteSpace(g.axes[1:no])
    return FiniteKernel(dom, codom, g.table; check=true, atol=atol)
end
function FiniteKernels.FiniteKernel(f::Factor, inputs::AbstractVector{Symbol},
                                    output::Symbol; kwargs...)
    return FiniteKernel(f, inputs, [output]; kwargs...)
end
