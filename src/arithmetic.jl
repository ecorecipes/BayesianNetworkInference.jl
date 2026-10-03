# The arithmetics of the exact backends (review item 8, S3).
#
# Variable elimination, junction-tree calibration and the reading of beliefs are each one
# algorithm, run on ordinary factors (`_Linear`), on centred log-domain factors
# (`_LogDomain`) or on exact dyadic factors (`_Dyadic`, the fallback of ADR 0016). The drivers
# in variable_elimination.jl and junction_tree.jl are written once, against the operations
# below; only these operations know which arithmetic they are in. A fix to a driver therefore
# reaches every arithmetic -- the duplication this replaces is how the empty-scope
# `normalize` bug (C6) reached one copy and not the other.
#
# An arithmetic's factor is a `Factor` for `_Linear`, a `_LogFactor` for `_LogDomain` and a
# `_DyadicFactor` for `_Dyadic`; `_plain` gives the `Factor` that carries its scope and shape.
#
# `_Linear()` computes in the factors' own element type and checks nothing: it serves
# `calibrate`, the empty query (ADR 0011: the unnormalised mass, which may legally be tiny)
# and the opt-in feasibility pass. A posterior run uses `_Linear{true}()` (`_Trusted()`),
# which applies the trust test below to every product it computes. Both compute integer and
# rational tables in checked arithmetic (factors.jl); an overflow is `FactorDomainError`
# from `_Linear()`, whose results have the graph's own type, and leaves a `_Trusted` run
# untrusted, so that the posterior is recomputed exactly (`_overflowed`).

abstract type _Arithmetic end
struct _Linear{Checked} <: _Arithmetic end
_Linear() = _Linear{false}()
const _Trusted = _Linear{true}
struct _LogDomain <: _Arithmetic end

# The trust test of a binary64 run (ADR 0014, ADR 0016, and the review of 2026-10-01), in
# the wording the BayesianNetworks and InfluenceDiagrams evaluators share: a run is
# untrusted if its final mass is not a normal positive number, or if any product it
# computed from operands that are all nonzero has magnitude below `floatmin` of its element
# type, whether that product came out subnormal or rounded all the way to 0.0. A product with
# an exactly zero operand is a structural zero and does not count, and a product with a
# nonzero operand already below `floatmin` -- a subnormal entry, or such a product -- marks
# the run too. Sums and divisions are not products: a sum of terms in the normal range is in
# it or exactly zero, unless tolerated negative entries cancel, which the final mass and the
# indeterminacy checks see.
#
# `_require_evidence_mass` (variable_elimination.jl) checks the final mass; `_check_product`
# checks every product of `_Trusted` and throws `_UnresolvedMass` at the first one below
# `floatmin`, so that the entry point recomputes exactly. The check is linear in the tables
# computed: a product is first judged from the smallest nonzero magnitudes of its two
# operands, and only when those could multiply below `floatmin` are its entries examined.
# Exact element types and `BigFloat` are not checked: exact types cannot underflow, and
# `BigFloat`'s exponent range makes the question moot.
_untrusted() = throw(_UnresolvedMass(Dict{Symbol,Symbol}()))

# An integer or rational product, sum or quotient overflowed its element type (the
# `FactorDomainError` that factors.jl's checked arithmetic returns). A posterior run
# (`_Trusted`) has an exact answer that exact arithmetic can compute, so the run is
# untrusted and recomputed exactly, as an underflowed binary64 run is (ADR 0014's reason for
# answering rather than reporting what another arithmetic can compute); any other run,
# whose result has the graph's own element type, reports the overflow.
_overflowed(::_Linear, e::FactorDomainError) = throw(e)
_overflowed(::_Trusted, ::FactorDomainError) = _untrusted()

# The smallest binary exponent among the nonzero finite entries of `table`; `typemax(Int)`
# when there is none, so that no product with it can fall below `floatmin`.
function _min_exponent(table::AbstractArray{T}) where {T<:Base.IEEEFloat}
    e = typemax(Int)
    @inbounds for x in table
        (iszero(x) || !isfinite(x)) && continue
        e = min(e, exponent(x))
    end
    return e
end

# Whether nonzero entries whose exponents are at least `ea` and `eb` can multiply below
# `floatmin(T)`, or are below it already: `|a| >= 2^ea` and `|b| >= 2^eb`, so
# `|a * b| >= 2^(ea + eb)`, and rounding never takes a product below a power of two that it
# exceeds.
function _may_underflow(::Type{T}, ea::Int, eb::Int) where {T<:Base.IEEEFloat}
    (ea == typemax(Int) || eb == typemax(Int)) && return false
    emin = exponent(floatmin(T))
    return ea < emin || eb < emin || ea + eb < emin
end

# Whether the product `z = x * y` of two nonzero operands, or one of the operands, has
# magnitude below `floatmin(T)`. A NaN or infinite value is not below it.
@inline function _lost(x::T, y::T, z) where {T<:Base.IEEEFloat}
    return !iszero(x) & !iszero(y) &
           ((abs(z) < floatmin(T)) | (abs(x) < floatmin(T)) | (abs(y) < floatmin(T)))
end

# Whether some entry of `p = a .* b` (arrays that broadcast together) is `_lost`. Belief
# propagation's factor messages use it (belief_propagation.jl).
function _product_underflowed(a::AbstractArray{T}, b::AbstractArray{T},
                              p::AbstractArray{T}) where {T<:Base.IEEEFloat}
    return any(Broadcast.instantiate(Broadcast.broadcasted(_lost, a, b, p)))
end
_product_underflowed(a::AbstractArray, b::AbstractArray, p::AbstractArray) = false

_check_product(::_Linear{false}, a::Factor, b::Factor, p::Factor) = nothing
_check_product(::_Trusted, a::Factor, b::Factor, p::Factor) = nothing
function _check_product(::_Trusted, a::Factor{T}, b::Factor{T},
                        p::Factor{T}) where {T<:Base.IEEEFloat}
    _may_underflow(T, _min_exponent(a.table), _min_exponent(b.table)) || return nothing
    # Each operand as a code laid out as `p` is (`multiply` gives both the same
    # first-occurrence scope): 0 for zero, 1 for a normal entry, 2 for one below `floatmin`.
    # Their product is 0 for a structural zero, 1 for two normal operands and at least 2
    # when a nonzero operand is below `floatmin`.
    code(x) = iszero(x) ? 0 : abs(x) < floatmin(T) ? 2 : 1
    codes(f) = _factor(f.vars, f.axes, map(code, f.table))
    both = multiply(codes(a), codes(b)).table
    any(i -> (both[i] >= 2) | ((both[i] == 1) & (abs(p.table[i]) < floatmin(T))),
        eachindex(both)) && _untrusted()
    return nothing
end

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
# sorted in place (stably), so both arithmetics associate the product in the same order. An
# empty list (a clique to which no factor is assigned) is the unit of the list's element
# type, which is the graph's.
function _product!(A::_Linear, fs::Vector{F}) where {F<:Factor}
    isempty(fs) && return _unit(A, _table_type(F))
    sort!(fs; by=ndims)
    return reduce((a, b) -> _multiply(A, a, b), fs)
end
function _product!(A::_LogDomain, fs::Vector{_LogFactor})
    isempty(fs) && return _unit(A)
    sort!(fs; by=f -> ndims(f.factor))
    return reduce(_log_multiply, fs)
end

function _multiply(A::_Linear, a::Factor, b::Factor)
    p = _product(a, b)
    p isa FactorDomainError && _overflowed(A, p)
    _check_product(A, a, b, p)
    return p
end
_multiply(::_LogDomain, a::_LogFactor, b::_LogFactor) = _log_multiply(a, b)

function _sum_out(A::_Linear, f::Factor, v::Symbol)
    r = _marginal(f, [v])
    r isa FactorDomainError && _overflowed(A, r)
    return r
end
_sum_out(::_LogDomain, f::_LogFactor, v::Symbol) = _log_sum_out(f, v)

# Sum out everything in the scope of `f` that is not in `keep`.
function _project(A::_Linear, f::Factor, keep::Vector{Symbol})
    r = _marginal(f, setdiff(f.vars, keep))
    r isa FactorDomainError && _overflowed(A, r)
    return r
end
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
_normalized(A::_Linear, f::Factor, evidence) = _posterior_normalize(f, evidence, A)
function _normalized(::_LogDomain, f::_LogFactor, evidence)
    weights = similar(f.factor.table)
    weights .= exp.(f.factor.table)
    weights ./= sum(weights)
    return _factor(f.factor.vars, f.factor.axes, weights)
end

# Support factors (the rule for tolerated entries)
# -------------------------------------------------
#
# A tolerated entry in [-atol, 0) takes part in a posterior when it lies on a configuration
# consistent with the evidence whose other entries are all nonzero (the rule BayesianNetworks
# and InfluenceDiagrams share; `_takes_part`, variable_elimination.jl). Whether one does is a
# question about supports, answered by running the shared driver on codes instead of
# numbers: bit `0x01` says that some configuration of the factor's scope has every entry
# nonzero and none negative, bit `0x02` that one has every entry nonzero and some negative.
# A product of two parts is nonzero when both are, and has a negative entry when either has;
# a sum is the union. That is the set semiring of the types {positive, negative}, so the
# driver's elimination is exact for it, and costs one pass over Boolean tables of the sizes
# the ordinary run has.
struct _Support <: _Arithmetic end

_factor_type(::_Support, ::Type) = Factor{UInt8}

_support_code(x) = iszero(x) ? 0x00 : x < 0 ? 0x02 : 0x01

function _support_product(a::UInt8, b::UInt8)
    (iszero(a) | iszero(b)) && return 0x00
    return (a & b & 0x01) | ((a | b) & 0x02)
end

function _conditioned(::_Support, f::Factor, evidence)
    g = condition(f, evidence)
    return _factor(g.vars, g.axes, map(_support_code, g.table))
end

_unit(::_Support, ::Type=Float64) = _factor(Symbol[], FiniteAxis[], fill(0x01))

function _multiply(::_Support, a::Factor{UInt8}, b::Factor{UInt8})
    vars, axes = _union_axes(a, b, :multiply)
    shape = Tuple(length(axis) for axis in axes)
    table = Array{UInt8}(undef, shape)
    table .= _support_product.(_broadcastable(a, vars, shape),
                               _broadcastable(b, vars, shape))
    return _factor(vars, axes, table)
end

function _product!(A::_Support, fs::Vector{Factor{UInt8}})
    isempty(fs) && return _unit(A)
    sort!(fs; by=ndims)
    return reduce((a, b) -> _multiply(A, a, b), fs)
end

# `reduce(|, ...; dims)` would start from a Boolean array, so the union starts from 0x00.
_union_out(t; dims) = reduce((x, y) -> x | y, t; dims=dims, init=0x00)
function _sum_out(::_Support, f::Factor{UInt8}, v::Symbol)
    return _reduce_out(_union_out, :marginalize, f, [v])
end
function _project(::_Support, f::Factor{UInt8}, keep::Vector{Symbol})
    return _reduce_out(_union_out, :marginalize, f, setdiff(f.vars, keep))
end
_reorder(::_Support, f::Factor{UInt8}, query) = reorder(f, query)

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
# power of two over one positive odd denominator, `factor.table .* 2^exponent ./
# denominator`, so products multiply integers, add exponents and multiply denominators,
# sums add integers, and nothing is rounded until `_normalized` divides each cell by the
# total, where the common scale cancels, and rounds it once, to the Float64 nearest the
# exact posterior of the graph as bound. Every entry enters at its exact value, whatever the
# graph's element type (`_exact_entry`), so the fallback is exact for integer, rational,
# Float16, Float32 and BigFloat graphs as well as Float64 ones; the denominator is one unless
# a rational entry has an odd factor in its own. `_dyadic` and `_nearest_binary64` are
# BayesianNetworks' (exact_rounding.jl), shared with its brute-force fallback and
# InfluenceDiagrams' exact decision elimination.
struct _Dyadic <: _Arithmetic end

struct _DyadicFactor
    factor::Factor{BigInt}
    exponent::Int
    denominator::BigInt
end
function _DyadicFactor(factor::Factor{BigInt}, exponent::Int)
    return _DyadicFactor(factor, exponent, big(1))
end

_plain(f::_DyadicFactor) = f.factor
_factor_type(::_Dyadic, ::Type) = _DyadicFactor

# The exact value of an entry as `(n, p, d)` with `x == n * 2^p / d` and `d` positive and
# odd, the convention of BayesianNetworks' exact fallback (`_exact_entry`, evaluation.jl):
# a Float64 is its `_dyadic` value, which is left as it is; a Float16 or Float32 converts to
# Float64 exactly; a BigFloat is its own dyadic value, at its full precision; an integer or
# a rational is itself, never a binary64 rounding of it.
_exact_entry(x::Float64) = (BayesianNetworks._dyadic(x)..., big(1))
_exact_entry(x::Union{Float16,Float32}) = _exact_entry(Float64(x))
function _exact_entry(x::BigFloat)
    n, p, s = Base.decompose(x)
    return BigInt(n) * s, Int(p), big(1)
end
_exact_entry(x::Integer) = (BigInt(x), 0, big(1))
function _exact_entry(x::Rational)
    d = BigInt(denominator(x))
    k = trailing_zeros(d)
    return BigInt(numerator(x)), -k, d >> k
end

# Whether an entry has an exact value here: a finite IEEE float, BigFloat or rational, or an
# integer. Another `Real` (or a non-finite value of a graph built with `check = false`) has
# none.
_has_exact_value(x::Union{Base.IEEEFloat,BigFloat,Rational}) = isfinite(x)
_has_exact_value(::Integer) = true
_has_exact_value(::Real) = false

# Every entry must have an exact value (`FactorDomainError(:exact, ...)` otherwise). On a
# factor graph, a tolerated negative entry anywhere, even where the evidence removes it,
# makes the posterior indeterminate (ADR 0016 decision 4): the fallback runs only when the
# binary64 run was untrusted, so the entry cannot be shown small beside the mass. Inside a
# model-level query the shared rule decides instead (`_MODEL_QUERY`, variable_elimination.jl):
# the entry keeps its exact value, and one that takes part shows in a negative exact cell
# (`_normalized`). The entries are scanned as a flat vector, which is type stable where the
# table's own rank is not.
function _check_exact_entries(f::Factor, evidence)
    model = _in_model_query()
    t = vec(f.table)
    i = findfirst(x -> !_has_exact_value(x) || (x < 0 && !model), t)
    i === nothing && return nothing
    _has_exact_value(t[i]) ||
        throw(FactorDomainError(:exact, copy(f.vars),
                                Tuple(CartesianIndices(size(f.table))[i]), t[i]))
    return throw(IndeterminatePosteriorError(Dict{Symbol,Symbol}(evidence),
                                             "the binary64 run was untrusted, and a tolerated negative entry ($(t[i])) leaves the sign of so small a posterior to the rounding"))
end

# The exact value of a checked factor: integers over the smallest power of two among its
# nonzero entries and the least common multiple of their odd denominators. A zero is zero
# at any power of two, and `_dyadic(0.0)` reports `2^-1074`, so letting zeros choose the
# power would shift every other entry left by a thousand bits and make every later product
# and sum that much wider. The common trailing zero bits of the integers are moved into the
# power as well.
function _as_dyadic(f::Factor)
    entries = map(_exact_entry, vec(f.table))
    exponent = minimum(((n, p, _),) -> iszero(n) ? typemax(Int) : p, entries;
                       init=typemax(Int))
    exponent == typemax(Int) &&
        return _DyadicFactor(_factor(f.vars, f.axes, zeros(BigInt, size(f.table))), 0)
    denominator = foldl((l, (n, _, d)) -> iszero(n) ? l : lcm(l, d), entries; init=big(1))
    table = Array{BigInt}(undef, size(f.table))
    for (i, (n, p, d)) in enumerate(entries)
        table[i] = iszero(n) ? big(0) : (n * div(denominator, d)) << (p - exponent)
    end
    shift = minimum(n -> iszero(n) ? typemax(Int) : trailing_zeros(n), table)
    if shift > 0
        table = map(n -> n >> shift, table)
        exponent += shift
    end
    return _DyadicFactor(_factor(f.vars, f.axes, table), exponent, denominator)
end

# Only the entries that survive the evidence are converted, so a tiny entry in a row the
# evidence removes does not widen the others.
function _conditioned(::_Dyadic, f::Factor, evidence)
    _check_exact_entries(f, evidence)
    return _as_dyadic(condition(f, evidence))
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
    return _DyadicFactor(multiply(a.factor, b.factor), a.exponent + b.exponent,
                         a.denominator * b.denominator)
end

function _sum_out(::_Dyadic, f::_DyadicFactor, v::Symbol)
    return _DyadicFactor(marginalize(f.factor, v), f.exponent, f.denominator)
end

function _project(::_Dyadic, f::_DyadicFactor, keep::Vector{Symbol})
    return _DyadicFactor(marginalize(f.factor, setdiff(f.factor.vars, keep)), f.exponent,
                         f.denominator)
end

function _reorder(::_Dyadic, f::_DyadicFactor, query)
    return _DyadicFactor(reorder(f.factor, query), f.exponent, f.denominator)
end

# Whether the exact mass of `f` is zero: the evidence has probability exactly zero.
_exactly_zero(f::_DyadicFactor) = iszero(sum(f.factor.table; init=big(0)))

# Each cell the Float64 nearest its exact share of the total (correct rounding). On a factor
# graph the inputs are nonnegative (`_check_exact_entries`), so only an exact zero total is
# impossible evidence. Inside a model-level query a tolerated negative entry can take part;
# the model has then checked the budget, and a negative exact cell is indeterminate, except
# in a prior, which is returned as computed.
function _normalized(::_Dyadic, f::_DyadicFactor, evidence)
    total = sum(f.factor.table; init=big(0))
    total < 0 &&
        throw(IndeterminatePosteriorError(Dict{Symbol,Symbol}(evidence),
                                          "the exact evidence mass is negative"))
    iszero(total) && _impossible(evidence)
    p = _factor(f.factor.vars, f.factor.axes,
                map(t -> BayesianNetworks._nearest_binary64(t, total), f.factor.table))
    v = minimum(p.table; init=0.0)
    v < 0 && !_model_prior() &&
        throw(IndeterminatePosteriorError(Dict{Symbol,Symbol}(evidence),
                                          "a posterior cell is negative ($(v))"))
    return p
end
