# The two arithmetics of the exact backends (review item 8, S3).
#
# Variable elimination, junction-tree calibration and the reading of beliefs are each one
# algorithm, run either on ordinary factors (`_Linear`) or on centred log-domain factors
# (`_LogDomain`). The drivers in variable_elimination.jl and junction_tree.jl are written once,
# against the operations below; only these operations know which arithmetic they are in. A fix
# to a driver therefore reaches both arithmetics -- the duplication this replaces is how the
# empty-scope `normalize` bug (C6) reached one copy and not the other.
#
# An arithmetic's factor is a `Factor` for `_Linear` and a `_LogFactor` for `_LogDomain`;
# `_plain` gives the `Factor` that carries its scope and shape either way.

abstract type _Arithmetic end
struct _Linear <: _Arithmetic end
struct _LogDomain <: _Arithmetic end

# Log-domain factors
# ------------------

# A log-domain factor: `factor.table` holds log values centred so that their maximum is zero,
# and `log_scale` the common offset kept outside the table, so that a small positive mass is
# not lost to underflow and a large one cannot erase small differences between cells.
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

# Every entry is checked, including those that evidence will condition away: a tolerated
# negative entry has no logarithm (`FactorDomainError`, ADR 0015).
function _as_log_factor(factor::Factor)
    table = Array{Float64}(undef, size(factor))
    for index in CartesianIndices(factor.table)
        value = factor.table[index]
        isfinite(value) && value >= 0 ||
            throw(FactorDomainError(:log_domain, copy(factor.vars), Tuple(index), value))
        if iszero(value)
            table[index] = -Inf
        else
            logarithm = Float64(log(value))
            isfinite(logarithm) ||
                throw(FactorDomainError(:log_domain, copy(factor.vars), Tuple(index),
                                        value))
            table[index] = logarithm
        end
    end
    return _center_log(_factor(factor.vars, factor.axes, table), 0.0)
end

function _log_multiply(left::_LogFactor, right::_LogFactor)
    vars, axes = _union_axes(left.factor, right.factor, :log_multiply)
    shape = Tuple(length(axis) for axis in axes)
    table = Array{Float64}(undef, shape)
    table .= _broadcastable(left.factor, vars, shape) .+
             _broadcastable(right.factor, vars, shape)
    return _center_log(_factor(vars, axes, table), left.log_scale + right.log_scale)
end

function _log_sum_out(source::_LogFactor, variable::Symbol)
    factor = source.factor
    position = _position(factor, :log_marginalize, variable)
    keep = [index for index in eachindex(factor.vars) if index != position]
    shape = Tuple(size(factor)[index] for index in keep)
    table = Array{Float64}(undef, shape)
    for index in CartesianIndices(table)
        coordinates = Tuple(index)
        function full(state)
            return ntuple(i -> i == position ? state : coordinates[i < position ? i : i - 1],
                          ndims(factor))
        end
        peak = maximum(factor.table[full(state)...]
                       for state in 1:size(factor.table, position))
        if peak == -Inf
            table[index] = -Inf
        else
            total = sum(exp(factor.table[full(state)...] - peak)
                        for state in 1:size(factor.table, position))
            table[index] = peak + log(total)
        end
    end
    return _center_log(_factor(factor.vars[keep], factor.axes[keep], table),
                       source.log_scale)
end

# The log of a log-domain factor's total mass: `-Inf` for exactly zero support.
function _log_mass(factor::_LogFactor)
    total = sum(exp, factor.factor.table)
    return iszero(total) ? -Inf : factor.log_scale + log(total)
end

function _log_mass_status(log_mass)
    log_mass == -Inf && return :zero
    mass = exp(log_mass)
    return iszero(mass) ? :underflow : isinf(mass) ? :overflow : :finite
end

# The operations of an arithmetic
# -------------------------------

_plain(f::Factor) = f
_plain(f::_LogFactor) = f.factor

_factor_type(::_Linear, ::Type{T}) where {T} = Factor{T}
_factor_type(::_LogDomain, ::Type) = _LogFactor

# An input factor conditioned on the evidence. The log domain takes the logarithm of the
# whole table first, so its entry check does not depend on the evidence.
_conditioned(::_Linear, f::Factor, evidence) = condition(f, evidence)
function _conditioned(::_LogDomain, f::Factor, evidence)
    logged = _as_log_factor(f)
    return _center_log(condition(logged.factor, evidence), logged.log_scale)
end

_unit(::_Linear, ::Type{T}=Float64) where {T} = unit_factor(T)
function _unit(::_LogDomain, ::Type=Float64)
    return _LogFactor(_factor(Symbol[], FiniteAxis[], fill(0.0)), 0.0)
end

# The product of a list of factors, multiplying the smallest scopes first. The list is
# sorted in place (stably), so both arithmetics associate the product in the same order.
function _product!(::_Linear, fs::Vector{<:Factor})
    isempty(fs) && return unit_factor()
    sort!(fs; by=ndims)
    return reduce(multiply, fs)
end
function _product!(A::_LogDomain, fs::Vector{_LogFactor})
    isempty(fs) && return _unit(A)
    sort!(fs; by=f -> ndims(f.factor))
    return reduce(_log_multiply, fs)
end

_multiply(::_Linear, a::Factor, b::Factor) = multiply(a, b)
_multiply(::_LogDomain, a::_LogFactor, b::_LogFactor) = _log_multiply(a, b)

_sum_out(::_Linear, f::Factor, v::Symbol) = marginalize(f, v)
_sum_out(::_LogDomain, f::_LogFactor, v::Symbol) = _log_sum_out(f, v)

# Sum out everything in the scope of `f` that is not in `keep`.
_project(::_Linear, f::Factor, keep::Vector{Symbol}) = marginalize(f, setdiff(f.vars, keep))
function _project(A::_LogDomain, f::_LogFactor, keep::Vector{Symbol})
    result = f
    for v in setdiff(f.factor.vars, keep)
        result = _sum_out(A, result, v)
    end
    return result
end

_reorder(::_Linear, f::Factor, query) = reorder(f, query)
function _reorder(::_LogDomain, f::_LogFactor, query)
    return _LogFactor(reorder(f.factor, query), f.log_scale)
end

# A posterior read off an unnormalised factor whose global mass the caller has already
# found positive. The linear arithmetic checks the factor's own mass again as it
# normalises, so a component whose mass is zero or untrusted raises (or signals the
# fallback) rather than dividing (`_posterior_normalize`, ADR 0014); the log arithmetic's
# centred table has a positive total whenever its global mass is not `-Inf`.
_normalized(::_Linear, f::Factor, evidence) = _posterior_normalize(f, evidence)
function _normalized(::_LogDomain, f::_LogFactor, evidence)
    weights = similar(f.factor.table)
    weights .= exp.(f.factor.table)
    weights ./= sum(weights)
    return _factor(f.factor.vars, f.factor.axes, weights)
end

# Every variable's posterior: the point mass at the observed label for an observed variable,
# otherwise `marginal(v)`. Shared by the all-marginals methods of every exact backend.
function _collect_marginals(marginal, fg::FactorGraph, evidence, ::Type{R}) where {R}
    out = Dict{Symbol,Factor{R}}()
    for (v, ax) in fg.axes
        out[v] = haskey(evidence, v) ? _point_mass(R, ax, evidence[v]) : marginal(v)
    end
    return out
end

# Exact dyadic factors (ADR 0016)
# -------------------------------

# The arithmetic of the fallbacks (ADR 0014): an exact factor is an integer table times one
# power of two, `factor.table .* 2^exponent`, so products multiply integers and add
# exponents, sums add integers, and nothing is rounded until `_normalized` divides each cell
# by the total and rounds it once, to the Float64 nearest the exact posterior of the graph as
# bound. `_dyadic` and `_nearest_binary64` are BayesianNetworks' (exact_rounding.jl), shared
# with its brute-force fallback and InfluenceDiagrams' exact decision elimination.
struct _Dyadic <: _Arithmetic end

struct _DyadicFactor
    factor::Factor{BigInt}
    exponent::Int
end

_plain(f::_DyadicFactor) = f.factor
_factor_type(::_Dyadic, ::Type) = _DyadicFactor

# The exact value of every entry. The fallback runs only when the binary64 evidence mass is
# not a normal positive number, so a tolerated negative entry cannot be shown small beside
# the mass and leaves the posterior's sign to the rounding (ADR 0014). An entry that is not
# exactly a Float64 (a non-finite value, or a wider type) has no exact dyadic value here.
function _as_dyadic(f::Factor, evidence)
    entries = Array{Tuple{BigInt,Int}}(undef, size(f))
    for index in CartesianIndices(f.table)
        x = f.table[index]
        isfinite(x) && Float64(x) == x ||
            throw(FactorDomainError(:exact, copy(f.vars), Tuple(index), x))
        x < 0 &&
            throw(IndeterminatePosteriorError(Dict{Symbol,Symbol}(evidence),
                                              "the binary64 evidence mass is not a normal positive number, and a tolerated negative entry ($(x)) leaves the sign of so small a posterior to the rounding"))
        entries[index] = BayesianNetworks._dyadic(Float64(x))
    end
    exponent = minimum(last, entries; init=0)
    table = map(((n, p),) -> n << (p - exponent), entries)
    return _DyadicFactor(_factor(f.vars, f.axes, table), exponent)
end

function _conditioned(::_Dyadic, f::Factor, evidence)
    exact = _as_dyadic(f, evidence)
    return _DyadicFactor(condition(exact.factor, evidence), exact.exponent)
end

function _unit(::_Dyadic, ::Type=Float64)
    return _DyadicFactor(_factor(Symbol[], FiniteAxis[], fill(big(1))), 0)
end

function _product!(A::_Dyadic, fs::Vector{_DyadicFactor})
    isempty(fs) && return _unit(A)
    sort!(fs; by=f -> ndims(f.factor))
    return reduce((a, b) -> _multiply(A, a, b), fs)
end

function _multiply(::_Dyadic, a::_DyadicFactor, b::_DyadicFactor)
    return _DyadicFactor(multiply(a.factor, b.factor), a.exponent + b.exponent)
end

function _sum_out(::_Dyadic, f::_DyadicFactor, v::Symbol)
    return _DyadicFactor(marginalize(f.factor, v), f.exponent)
end

function _project(::_Dyadic, f::_DyadicFactor, keep::Vector{Symbol})
    return _DyadicFactor(marginalize(f.factor, setdiff(f.factor.vars, keep)), f.exponent)
end

function _reorder(::_Dyadic, f::_DyadicFactor, query)
    return _DyadicFactor(reorder(f.factor, query), f.exponent)
end

# Whether the exact mass of `f` is zero: the evidence has probability exactly zero.
_exactly_zero(f::_DyadicFactor) = iszero(sum(f.factor.table; init=big(0)))

# Each cell the Float64 nearest its exact share of the total (correct rounding). The inputs
# are nonnegative (`_as_dyadic`), so only an exact zero total is impossible evidence.
function _normalized(::_Dyadic, f::_DyadicFactor, evidence)
    total = sum(f.factor.table; init=big(0))
    iszero(total) && _impossible(evidence)
    return _factor(f.factor.vars, f.factor.axes,
                   map(t -> BayesianNetworks._nearest_binary64(t, total), f.factor.table))
end
