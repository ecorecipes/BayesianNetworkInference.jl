# Finite factors: unnormalised tensors over an ordered variable scope, and the
# (multiply, marginalise, condition, ...) algebra used by variable elimination.

# `ScopeError` and `ShapeError` are defined in errors.jl.

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

# Checked arithmetic (review of 2026-10-02, finding 1)
# ----------------------------------------------------
#
# Base's machine integers wrap silently when a product or a sum overflows, and its
# rationals of them throw `OverflowError`. Tables of those element types are therefore
# computed in checked arithmetic: `multiply`, `marginalize` and `normalize` give each
# product, sum and quotient its exact value, or report the first cell whose exact value the
# element type cannot hold, as a `FactorDomainError` whose backend is the operation and
# whose value is that exact value. They never wrap. The internal forms (`_product`,
# `_marginal`, `_normalize`, `_total`) return the error instead of throwing it, so that a
# posterior run can recompute in exact arithmetic instead (`_overflowed`, arithmetic.jl).
# Floating-point types, `Bool`, `BigInt` and rationals of `BigInt` are not checked: they do
# not wrap, and a float that overflows to infinity is a non-finite mass, which the trust
# test of the posterior runs already catches.
_checked(::Type) = false
_checked(::Type{<:Base.BitInteger}) = true
_checked(::Type{<:Rational{<:Base.BitInteger}}) = true

# `(x * y, overflowed)`. A rational product is formed exactly in the widened integer type,
# where two factors of the narrow one cannot overflow, and is Base's product (the same
# reduced fraction) whenever it fits.
@inline _mul(x, y) = (x * y, false)
@inline _mul(x::T, y::T) where {T<:Base.BitInteger} = Base.Checked.mul_with_overflow(x, y)
function _mul(x::Rational{T}, y::Rational{T}) where {T<:Base.BitInteger}
    W = Rational{widen(T)}
    p = convert(W, x) * convert(W, y)
    _fits(Rational{T}, p) || return zero(Rational{T}), true
    return convert(Rational{T}, p), false
end

# The exact value of an integer or rational entry, for the error that reports it.
_exact(x::Integer) = BigInt(x)
_exact(x::Rational) = Rational{BigInt}(x)

# Whether the exact value `x` is a value of the element type `T`.
_fits(::Type{T}, x::Integer) where {T<:Base.BitInteger} = typemin(T) <= x <= typemax(T)
_fits(::Type{Rational{T}}, x::Integer) where {T} = typemin(T) <= x <= typemax(T)
function _fits(::Type{Rational{T}}, x::Rational) where {T}
    return typemin(T) <= numerator(x) <= typemax(T) && denominator(x) <= typemax(T)
end

# The element type of a sum of entries of type `T` (Base's `sum` widens the small
# integers to `Int`), and an entry's value in a type in which such a sum is exact.
_sum_type(::Type{T}) where {T} = typeof(Base.add_sum(zero(T), zero(T)))
_widened(::Type{R}, x) where {R<:Base.BitInteger} = widen(convert(R, x))
_widened(::Type{<:Rational}, x) = Rational{BigInt}(x)

# The overflow report of `operation` at `index` (in the scope `vars`) whose exact value is
# `value`; an empty `index` with a nonempty scope is the total of the factor over `vars`.
function _overflow(operation::Symbol, vars::Vector{Symbol}, index::Tuple, value)
    return FactorDomainError(operation, copy(vars), index, value)
end

# The walk of `_product_into!`, prepared: the result axes with both operands' strides,
# after dropping axes of size one (they never move an offset) and merging each axis into
# the one before it when both operands step through the pair contiguously
# (`as[k + 1] == as[k] * sz[k]`, and the same for `bs`; zero strides merge too). A merged
# axis keeps the column-major order of the result's cells and each cell's offset in both
# operands, so the walk reads the same pairs in the same order with fewer, longer runs.
function _merged_axes(sz::Vector{Int}, as::Vector{Int}, bs::Vector{Int})
    msz = Int[]
    mas = Int[]
    mbs = Int[]
    for k in eachindex(sz)
        sz[k] == 1 && continue
        if !isempty(msz) && as[k] == mas[end] * msz[end] && bs[k] == mbs[end] * msz[end]
            msz[end] *= sz[k]
        else
            push!(msz, sz[k])
            push!(mas, as[k])
            push!(mbs, bs[k])
        end
    end
    return msz, mas, mbs
end

# The two innermost (merged) axes of the walk as plain loops:
# `out[o + j*n + i + 1] = a[ai + i*sa + j*ta] * b[bi + i*sb + j*tb]` for `i < n`, `j < m`,
# in that order. The layouts that the products of elimination and calibration mostly have
# -- both operands contiguous, or one of them constant, along the innermost axis -- get loops
# of their own, which vectorise. Returns whether a product overflowed (`_mul`).
@inline function _product_block!(out::Vector, o::Int, a::Vector, ai::Int, sa::Int, ta::Int,
                                 b::Vector, bi::Int, sb::Int, tb::Int, n::Int, m::Int)
    overflow = false
    if sa == 1 && sb == 1
        @inbounds for j in 0:(m - 1)
            oj, aj, bj = o + j * n, ai + j * ta, bi + j * tb
            @simd for i in 0:(n - 1)
                p, v = _mul(a[aj + i], b[bj + i])
                out[oj + 1 + i] = p
                overflow |= v
            end
        end
    elseif sa == 1 && sb == 0
        @inbounds for j in 0:(m - 1)
            oj, aj, y = o + j * n, ai + j * ta, b[bi + j * tb]
            @simd for i in 0:(n - 1)
                p, v = _mul(a[aj + i], y)
                out[oj + 1 + i] = p
                overflow |= v
            end
        end
    elseif sa == 0 && sb == 1
        @inbounds for j in 0:(m - 1)
            oj, x, bj = o + j * n, a[ai + j * ta], bi + j * tb
            @simd for i in 0:(n - 1)
                p, v = _mul(x, b[bj + i])
                out[oj + 1 + i] = p
                overflow |= v
            end
        end
    else
        @inbounds for j in 0:(m - 1)
            oj, aj, bj = o + j * n, ai + j * ta, bi + j * tb
            @simd for i in 0:(n - 1)
                p, v = _mul(a[aj + i * sa], b[bj + i * sb])
                out[oj + 1 + i] = p
                overflow |= v
            end
        end
    end
    return overflow
end

# out[i] = a[.] * b[.] over the result's joint states in column-major order, with an
# odometer over `sz` and the strides `as`, `bs` of `_result_strides` (modelled in
# `FiniteKernels.jl/proofs/FiniteKernelsProofs/Layout/Product.lean`: `bump`,
# `productLoop`, `productInto_eq`). The axes are merged first (`_merged_axes`) and the two
# innermost ones run as loops (`_product_block!`); the odometer turns the rest. Every cell
# is still the one product `a[ai] * b[bi]` at its own two offsets, written once, so the table
# is the same, bit for bit. Returns whether a product overflowed an integer or rational
# element type (`_mul`), in which case `out` is not the product.
function _product_into!(out::Vector, sz::Vector{Int}, a::Vector, as::Vector{Int},
                        b::Vector, bs::Vector{Int})
    msz, mas, mbs = _merged_axes(sz, as, bs)
    d = length(msz)
    d == 0 && return _product_block!(out, 0, a, 1, 0, 0, b, 1, 0, 0, 1, 1)
    n, sa, sb = msz[1], mas[1], mbs[1]
    m, ta, tb = d >= 2 ? (msz[2], mas[2], mbs[2]) : (1, 0, 0)
    block = n * m
    pos = zeros(Int, d)
    ai = 1
    bi = 1
    o = 0
    overflow = false
    @inbounds while true
        overflow |= _product_block!(out, o, a, ai, sa, ta, b, bi, sb, tb, n, m)
        o += block
        k = 3
        while k <= d
            pos[k] += 1
            ai += mas[k]
            bi += mbs[k]
            pos[k] < msz[k] && break
            pos[k] = 0
            ai -= mas[k] * msz[k]
            bi -= mbs[k] * msz[k]
            k += 1
        end
        k > d && return overflow
    end
end

# The first cell of the product whose exact value its element type cannot hold, found by
# walking the cells in order when `_product_into!` reported an overflow; `nothing` if none
# does (`_mul` is a pure function, so this walk's verdict is the kernel's).
function _product_overflow(vars, sz::Vector{Int}, a::Vector, as::Vector{Int}, b::Vector,
                           bs::Vector{Int})
    pos = zeros(Int, length(sz))
    ai = 1
    bi = 1
    for i in 1:prod(sz)
        if last(_mul(a[ai], b[bi]))
            index = Tuple(CartesianIndices(Tuple(sz))[i])
            return _overflow(:multiply, vars, index, _exact(a[ai]) * _exact(b[bi]))
        end
        for k in eachindex(sz)
            pos[k] += 1
            ai += as[k]
            bi += bs[k]
            pos[k] < sz[k] && break
            pos[k] = 0
            ai -= as[k] * sz[k]
            bi -= bs[k] * sz[k]
        end
    end
    return nothing
end

# An operand's flat table in the result's element type `T`. Only a checked `T` converts,
# exactly, or reporting an entry that `T` cannot hold (a negative integer multiplied by an
# unsigned one, say); otherwise the kernel promotes each pair of entries as `*` does.
_operand(::Type{T}, f::Factor{T}) where {T} = vec(f.table)
_operand(::Type{T}, f::Factor) where {T} = _checked(T) ? _converted(T, f) : vec(f.table)
function _converted(::Type{T}, f::Factor) where {T}
    t = vec(f.table)
    out = Vector{T}(undef, length(t))
    for i in eachindex(t)
        _fits(T, t[i]) || return _overflow(:multiply, f.vars,
                                           Tuple(CartesianIndices(size(f.table))[i]), t[i])
        out[i] = convert(T, t[i])
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

Integer and rational element types are multiplied in checked arithmetic: a product
that the element type cannot hold is a [`FactorDomainError`](@ref) with backend
`:multiply`, naming the cell of the result and the product's exact value, never a
wrapped integer. Use a wider type (`Float64`, `BigInt`, `Rational{BigInt}`) for such
tables.
"""
function multiply(f::Factor, g::Factor)
    p = _product(f, g)
    p isa FactorDomainError && throw(p)
    return p
end
multiply(f::Factor) = f
function multiply(f::Factor, g::Factor, h::Factor, hs::Factor...)
    return multiply(multiply(f, g), h, hs...)
end
function multiply(fs::AbstractVector{F}) where {F<:Factor}
    isempty(fs) && return unit_factor(_table_type(F))
    return reduce(multiply, fs)
end
Base.:*(f::Factor, g::Factor) = multiply(f, g)

# The element type of a vector of factors (`Float64` when the vector does not fix one), so
# that an empty product is the unit of the right type. (One method: Julia ranks
# `::Type{<:Factor}` above `::Type{Factor{T}} where T` for `Factor{Float32}`.)
_table_type(::Type{F}) where {F<:Factor} = F isa DataType ? eltype(F) : Float64

# `multiply`, returning the `FactorDomainError` of a product that overflows an integer or
# rational element type instead of throwing it.
function _product(f::Factor{S}, g::Factor{U}) where {S,U}
    vars, axes = _union_axes(f, g, :multiply)
    T = promote_type(S, U)
    a = _operand(T, f)
    a isa FactorDomainError && return a
    b = _operand(T, g)
    b isa FactorDomainError && return b
    sz = Int[length(x) for x in axes]
    table = Array{T}(undef, Tuple(sz))
    isempty(table) && return _factor(vars, axes, table)
    as = _result_strides(f, vars)
    bs = _result_strides(g, vars)
    _product_into!(vec(table), sz, a, as, b, bs) || return _factor(vars, axes, table)
    overflow = _product_overflow(vars, sz, a, as, b, bs)
    return overflow === nothing ? _factor(vars, axes, table) : overflow
end

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

Integer and rational element types are summed exactly: a sum that the element type
of the result cannot hold is a [`FactorDomainError`](@ref) with backend
`:marginalize`, naming the cell of the result and the sum's exact value, never a
wrapped integer. (As with `sum`, the small integer types sum to `Int`.)
"""
function marginalize(f::Factor, vars::AbstractVector{Symbol})
    r = _marginal(f, vars)
    r isa FactorDomainError && throw(r)
    return r
end
marginalize(f::Factor, var::Symbol) = marginalize(f, [var])

# `marginalize`, returning the `FactorDomainError` of a sum that overflows an integer or
# rational element type instead of throwing it. A checked sum is computed exactly, in a
# widened type, and converted back once it is known to fit: Base's machine-integer sum is
# exact modulo 2^n, and a rational sum is exact, so the result is Base's whenever Base's
# would not wrap or throw.
function _marginal(f::Factor{T}, vars) where {T}
    _checked(T) || return _reduce_out(sum, :marginalize, f, vars)
    isempty(vars) && return f
    dims = Tuple(_positions(f, :marginalize, vars))
    keep = setdiff(1:ndims(f), dims)
    R = _sum_type(T)
    wide = dropdims(sum(x -> _widened(R, x), f.table; dims=dims); dims=dims)
    flat = vec(wide)
    i = findfirst(x -> !_fits(R, x), flat)
    i === nothing ||
        return _overflow(:marginalize, f.vars[keep],
                         Tuple(CartesianIndices(size(wide))[i]), flat[i])
    return _factor(f.vars[keep], f.axes[keep], convert(Array{R}, wide))
end

# The sum of every entry of `f`: `sum(f.table)`, with the same check as `_marginal` for
# an integer or rational element type, reported as `operation` at the empty index.
function _total(f::Factor{T}, operation::Symbol) where {T}
    _checked(T) || return sum(f.table)
    R = _sum_type(T)
    total = sum(x -> _widened(R, x), f.table)
    _fits(R, total) || return _overflow(operation, f.vars, (), total)
    return convert(R, total)
end

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
resolve to the first label: entries are compared with `==`, so `-0.0` and `0.0`
tie as the equal numbers they are (`Base.argmax` would order them by `isless`).
This is the policy table of a decision variable in decision variable elimination,
and the least-maximizer rule its Lean model proves (`Selector.ordered`). A row
whose maximum is `NaN` has no maximizer and raises an `ArgumentError`.
"""
function argmax_table(f::Factor, var::Symbol)
    d = _position(f, :argmax_table, var)
    labs = f.axes[d].labels
    idx = dropdims(mapslices(row -> _first_maximizer(row, var), f.table; dims=d); dims=d)
    return map(i -> labs[i], idx)
end

# The first position of a row's maximum, compared with `==` so that an exact tie, signed
# zeros included, goes to the first label. `maximum` propagates `NaN`, which equals
# nothing, so a row containing `NaN` has no maximizer.
function _first_maximizer(row::AbstractVector, var::Symbol)
    best = maximum(row)
    i = findfirst(==(best), row)
    i === nothing &&
        throw(ArgumentError("argmax_table: a row of the factor over :$var contains NaN, so it has no maximum"))
    return i
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

Divide by the total mass so that the table sums to one. Throws an
`ArgumentError` naming the scope when the total is zero, as `FiniteKernels`
does for a kernel column that sums to zero. Extends `LinearAlgebra.normalize`,
as `FiniteKernels` does for kernels.

Posterior entry points never reach that error: they check the evidence mass
first and throw `BayesianNetworks.ImpossibleEvidenceError` (ADR 0012).

Integer and rational element types are summed and divided exactly: a total, or a
rational quotient, that the element type cannot hold is a [`FactorDomainError`](@ref)
with backend `:normalize` and the exact value.
"""
function normalize(f::Factor)
    p = _normalize(f)
    p isa FactorDomainError && throw(p)
    return p
end

# `normalize`, returning the `FactorDomainError` of an overflow instead of throwing it.
function _normalize(f::Factor)
    s = _total(f, :normalize)
    s isa FactorDomainError && return s
    # `normalize` cannot report a deviation from one or a tolerance, so a zero total is an
    # `ArgumentError` rather than a `KernelNormalizationError` with fabricated numbers
    # (ADR 0007 for kernels, ADR 0012 for factors).
    iszero(s) &&
        throw(ArgumentError("normalize: the factor over $(f.vars) sums to zero, so it cannot be rescaled to sum to one"))
    # `f.table ./ s` returns a *scalar* for an empty scope, where the table is a 0-d array,
    # so divide in place: an empty-scope factor arises from any maximal clique that evidence
    # fully instantiates. The log-domain path already does this.
    table = similar(f.table, typeof(one(eltype(f.table)) / s))
    return _quotients!(table, f, s)
end

# table .= f.table ./ s. A quotient of integers is a float and cannot overflow; one of
# checked rationals is formed exactly and must fit.
function _quotients!(table::Array, f::Factor, s)
    table .= f.table ./ s
    return _factor(f.vars, f.axes, table)
end
function _quotients!(table::Array{Rational{T}}, f::Factor, s) where {T<:Base.BitInteger}
    total = Rational{BigInt}(s)
    t = vec(f.table)
    out = vec(table)
    for i in eachindex(t, out)
        q = Rational{BigInt}(t[i]) / total
        _fits(Rational{T}, q) ||
            return _overflow(:normalize, f.vars, Tuple(CartesianIndices(size(table))[i]), q)
        out[i] = convert(Rational{T}, q)
    end
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
