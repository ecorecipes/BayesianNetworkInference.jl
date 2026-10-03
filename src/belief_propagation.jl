# Sum-product belief propagation on the factor graph (SPEC section 20):
# exact on tree-structured graphs, loopy with convergence diagnostics otherwise.

"""
    BeliefPropagation(; damping=0.0, tol=1e-8, maxiter=200, schedule=:flooding, check_evidence=false)

Sum-product message passing on the factor graph (see
[`belief_propagation`](@ref)). Messages are damped towards their previous
value with weight `damping` (in `[0, 1)`), iteration stops when the largest
undamped message-equation residual at the returned iterate is below `tol` or after `maxiter`
sweeps, and `schedule` is `:flooding` (every message updated from the
previous sweep's messages) or `:sequential` (factor by factor, each update
seeing the latest messages). A fixed point on a feasible tree has the exact
marginals; a finite tolerance is not itself a marginal-error bound.
`check_evidence=true` first uses exact variable elimination to check global
evidence feasibility, potentially at exponential cost; it is the only exact
computation belief propagation ever runs (ADR 0011). Without that opt-in,
local zero-support errors are detected but global feasibility is not certified.
[`BPDiagnostics`](@ref) reports both convergence and whether evidence was checked.
Throws `ArgumentError` for parameters outside these ranges.
"""
struct BeliefPropagation <: InferenceBackend
    damping::Float64
    tol::Float64
    maxiter::Int
    schedule::Symbol
    check_evidence::Bool
    function BeliefPropagation(damping::Real, tol::Real, maxiter::Integer,
                               schedule::Symbol, check_evidence::Bool)
        0 <= damping < 1 ||
            throw(ArgumentError("damping must lie in [0, 1), got damping = $damping"))
        isfinite(tol) && tol > 0 ||
            throw(ArgumentError("tol must be finite and positive, got tol = $tol"))
        maxiter >= 1 ||
            throw(ArgumentError("maxiter must be at least 1, got maxiter = $maxiter"))
        schedule in (:flooding, :sequential) ||
            throw(ArgumentError("schedule must be :flooding or :sequential, got schedule = $(repr(schedule))"))
        return new(damping, tol, maxiter, schedule, check_evidence)
    end
end
function BeliefPropagation(damping::Real, tol::Real, maxiter::Integer, schedule::Symbol)
    return BeliefPropagation(damping, tol, maxiter, schedule, false)
end
function BeliefPropagation(; damping::Real=0.0, tol::Real=1e-8, maxiter::Integer=200,
                           schedule::Symbol=:flooding, check_evidence::Bool=false)
    return BeliefPropagation(damping, tol, maxiter, schedule, check_evidence)
end

"""
    BPDiagnostics(iterations, converged, max_residual, tree, evidence_checked=false)

What a run of [`belief_propagation`](@ref) did: the number of sweeps
performed, whether the undamped message-equation residual fell below the
tolerance, that residual at the returned iterate, whether the conditioned
factor graph was a tree, and whether global evidence feasibility was checked.

The field is `tree`, not `exact`, because being a tree is a property of the
graph while exactness is a property of the iterate. On a tree the fixed
point of the message equations gives exact marginals for feasible evidence.
`max_residual` is not a general bound on marginal error. In particular,
`converged` does not certify global evidence feasibility: unless
`evidence_checked` is true, that status is unknown.

`log_domain` is true when the messages were computed in the log domain. Binary64
messages are used first; when one of them would leave the normal range -- a product of
nonzero values falls below `floatmin`, or a message sums to a number that is not
normal and positive although it is not exactly zero -- the same message passing is redone
with logarithms of the messages, where nothing underflows. The marginals are still belief
propagation's: the log domain changes the arithmetic, not the algorithm, and belief
propagation never falls back to an exact backend.
"""
struct BPDiagnostics
    iterations::Int
    converged::Bool
    max_residual::Float64
    tree::Bool
    evidence_checked::Bool
    log_domain::Bool
end
function BPDiagnostics(iterations::Integer, converged::Bool, max_residual::Real, tree::Bool,
                       evidence_checked::Bool=false)
    return BPDiagnostics(iterations, converged, max_residual, tree, evidence_checked, false)
end

"""
    is_tree(fg::FactorGraph) -> Bool

Whether the bipartite factor graph (one node per variable, one per factor,
an edge for every scope membership) has no cycles, i.e. is a forest. On such
a graph undamped messages reach the exact fixed point after sufficiently many
sweeps when evidence is feasible. A compiled Bayesian network is a tree here exactly
when its DAG is a polytree. Extends `Graphs.is_tree`.
"""
Graphs.is_tree(fg::FactorGraph) = _is_forest(fg.factors)

function _is_forest(factors::AbstractVector{<:Factor})
    vars = _variables(factors)
    index = Dict{Symbol,Int}(v => i for (i, v) in enumerate(vars))
    nv = length(vars)
    g = SimpleGraph(nv + length(factors))
    for (j, f) in enumerate(factors), v in f.vars
        Graphs.add_edge!(g, index[v], nv + j)
    end
    return Graphs.ne(g) == Graphs.nv(g) - length(Graphs.connected_components(g))
end

# The arithmetics of the message equations
# ----------------------------------------
#
# The message equations below are written once, against two arithmetics, as the exact
# drivers are (arithmetic.jl): a change to the algorithm reaches both.
#
# `_LinearMessages`: messages in the float type of the graph, the potentials scaled by a
# power of two (`_linear_potentials`). Every product it forms is checked with the trust
# test of the exact backends (`_lost` and `_product_underflowed`, arithmetic.jl): a product
# of nonzero values below `floatmin`, a normalised entry below it, or a message sum that is
# not a normal positive number throws `_UnresolvedMass`, and the run is redone in the log
# domain. So when
# a message sums to exactly zero, no product before it lost a nonzero value, and the zero is
# exact. On nonnegative potentials an exact zero message proves the evidence impossible: a
# configuration of positive mass keeps every exact message positive at its own values, by
# induction over the updates, damping and normalisation included. That raises
# `ImpossibleEvidenceError` at once, with no exact inference (ADR 0011).
#
# `_LogMessages`: the logarithms of the messages and potentials. A zero is `-Inf` exactly
# and a product is a sum, so nothing underflows, and an all `-Inf` message is again an
# exact zero.
abstract type _MessageArithmetic end
struct _LinearMessages <: _MessageArithmetic
    nonnegative::Bool
end
struct _LogMessages <: _MessageArithmetic end

# Message state of one run: factor `f` has one message in each direction for
# every position `k` of its scope; `incidences[x]` lists the `(f, k)` pairs of
# variable `x`. `evidence` is carried so that a zero message or belief can report it, and
# `exponents[f]` is the smallest binary exponent of factor `f`'s nonzero entries, for the
# linear trust test.
struct _BPState{T}
    factors::Vector{Factor{T}}
    exponents::Vector{Int}
    vars::Vector{Symbol}
    axes::Vector{FiniteAxis}
    incidences::Vector{Vector{Tuple{Int,Int}}}
    to_var::Vector{Vector{Vector{T}}}      # to_var[f][k]: factor f to variable at position k
    to_factor::Vector{Vector{Vector{T}}}   # to_factor[f][k]: that variable to factor f
    evidence::Dict{Symbol,Symbol}
end

function _BPState(A::_MessageArithmetic, factors::Vector{Factor{T}},
                  axes::Dict{Symbol,FiniteAxis}, evidence::Dict{Symbol,Symbol}) where {T}
    vars = _variables(factors)
    index = Dict{Symbol,Int}(v => i for (i, v) in enumerate(vars))
    incidences = [Tuple{Int,Int}[] for _ in vars]
    for (f, fac) in enumerate(factors), (k, v) in enumerate(fac.vars)
        push!(incidences[index[v]], (f, k))
    end
    uniform(fac, k) = _uniform(A, T, size(fac.table, k))
    to_var = [[uniform(fac, k) for k in 1:ndims(fac)] for fac in factors]
    to_factor = [[uniform(fac, k) for k in 1:ndims(fac)] for fac in factors]
    exponents = [_smallest_exponent(f.table) for f in factors]
    return _BPState{T}(factors, exponents, vars, FiniteAxis[axes[v] for v in vars],
                       incidences, to_var, to_factor, evidence)
end

_smallest_exponent(table::Array{<:Base.IEEEFloat}) = _min_exponent(table)
_smallest_exponent(table::Array) = typemax(Int)

_uniform(::_LinearMessages, ::Type{T}, n::Integer) where {T} = fill(one(T) / n, n)
_uniform(::_LogMessages, ::Type{T}, n::Integer) where {T} = fill(-log(T(n)), n)

_unit_message(::_LinearMessages, ::Type{T}, n::Integer) where {T} = ones(T, n)
_unit_message(::_LogMessages, ::Type{T}, n::Integer) where {T} = zeros(T, n)

# A running product of messages is rescaled by an exact power of two once its largest entry
# drops below 2^-_RESCALE, so that a long product keeps what binary64 can hold.
const _RESCALE = 256

# m .*= other, in the arithmetic.
function _absorb!(::_LogMessages, m::Vector, other::Vector)
    m .+= other
    return m
end
function _absorb!(::_LinearMessages, m::Vector, other::Vector)
    m .*= other
    return m
end
function _absorb!(::_LinearMessages, m::Vector{T},
                  other::Vector{T}) where {T<:Base.IEEEFloat}
    peak = zero(T)
    @inbounds for i in eachindex(m, other)
        a = m[i]
        b = other[i]
        p = a * b
        _lost(a, b, p) && _untrusted()
        m[i] = p
        peak = max(peak, abs(p))
    end
    if isfinite(peak) && !iszero(peak) && exponent(peak) < -_RESCALE
        shift = -exponent(peak)
        m .= ldexp.(m, shift)
    end
    return m
end

# m_{f -> x}(x) = sum_{x_f \ x} f(x_f) prod_{y != x} m_{y -> f}(y), x at position k.
function _factor_message(A::_MessageArithmetic, st::_BPState{T}, f::Int,
                         k::Int) where {T}
    fac = st.factors[f]
    nd = ndims(fac)
    nd == 1 && return copy(vec(fac.table))
    sz = size(fac)
    safe = _products_safe(A, st, f, k)
    t = fac.table
    for j in 1:nd
        j == k && continue
        shape = ntuple(i -> i == j ? sz[j] : 1, nd)
        mj = reshape(st.to_factor[f][j], shape)
        u = _combine(A, t, mj)
        safe || !_product_underflowed(t, mj, u) || _untrusted()
        t = u
    end
    dims = Tuple(j for j in 1:nd if j != k)
    return _sum_messages(A, t, dims)
end

_combine(::_LinearMessages, t, m) = t .* m
_combine(::_LogMessages, t, m) = t .+ m

_sum_messages(::_LinearMessages, t, dims) = vec(sum(t; dims=dims))
_sum_messages(::_LogMessages, t, dims) = vec(_logsumexp(t, dims))

# Whether no product of nonzero entries in the message from factor `f` to position `k` can
# fall below `floatmin`, judged from the smallest nonzero exponents alone (`_may_underflow`,
# arithmetic.jl): each partial product's entries are at least 2^(the sum of the exponents
# so far). The log domain has nothing to check.
_products_safe(::_LogMessages, st::_BPState, f::Int, k::Int) = true
_products_safe(::_LinearMessages, st::_BPState, f::Int, k::Int) = true
function _products_safe(::_LinearMessages, st::_BPState{T}, f::Int,
                        k::Int) where {T<:Base.IEEEFloat}
    low = st.exponents[f]
    low == typemax(Int) && return true
    for (j, m) in enumerate(st.to_factor[f])
        j == k && continue
        e = _min_exponent(m)
        e == typemax(Int) && return true
        low += e
        low < exponent(floatmin(T)) && return false
    end
    return true
end

# log(sum(exp, t; dims)), with an all `-Inf` slice giving `-Inf` exactly.
function _logsumexp(t::AbstractArray{T}, dims) where {T}
    peak = maximum(t; dims=dims)
    shift = map(p -> isfinite(p) ? p : zero(T), peak)
    return shift .+ log.(sum(exp.(t .- shift); dims=dims))
end
function _logsumexp(m::AbstractVector{T}) where {T}
    peak = maximum(m)
    isfinite(peak) || return peak
    return peak + log(sum(x -> exp(x - peak), m))
end

# Normalise a message in place. A message that sums to exactly zero on nonnegative
# potentials proves the evidence impossible (see `_LinearMessages`); any other sum that is
# not a normal positive number leaves the linear run untrusted (`_require_evidence_mass`
# throws `_UnresolvedMass`), and the run is redone in the log domain.
function _normalize!(A::_LinearMessages, m::Vector, st::_BPState)
    s = sum(m)
    A.nonnegative && iszero(s) && _impossible(st.evidence)
    _require_evidence_mass(s, st.evidence)
    return _divide!(m, s)
end
function _normalize!(::_LogMessages, m::Vector, st::_BPState)
    total = _logsumexp(m)
    total == -Inf && _impossible(st.evidence)
    m .-= total
    return m
end

# m ./= s. A normalised entry is an operand of later products, so one that came out below
# `floatmin` leaves the run untrusted, as an underflowed product would.
function _divide!(m::Vector, s)
    m ./= s
    return m
end
function _divide!(m::Vector{T}, s::T) where {T<:Base.IEEEFloat}
    @inbounds for i in eachindex(m)
        q = m[i] / s
        !iszero(m[i]) & (abs(q) < floatmin(T)) && _untrusted()
        m[i] = q
    end
    return m
end

# new .= (1 - damping) .* new .+ damping .* old, in the arithmetic.
function _damp!(::_LinearMessages, new::Vector, old::Vector, damping)
    new .= (1 - damping) .* new .+ damping .* old
    return new
end
function _damp!(::_LinearMessages, new::Vector{T}, old::Vector{T},
                damping) where {T<:Base.IEEEFloat}
    c = 1 - damping
    @inbounds for i in eachindex(new, old)
        a = new[i]
        b = old[i]
        p = c * a
        q = damping * b
        x = convert(T, p + q)
        lost = (!iszero(a) & (abs(p) < floatmin(p))) |
               (!iszero(b) & (abs(q) < floatmin(q))) | issubnormal(x)
        lost && _untrusted()
        new[i] = x
    end
    return new
end
function _damp!(::_LogMessages, new::Vector{T}, old::Vector{T}, damping) where {T}
    a = T(log1p(-damping))
    b = T(log(damping))
    @inbounds for i in eachindex(new, old)
        new[i] = _logaddexp(a + new[i], b + old[i])
    end
    return new
end

function _logaddexp(x, y)
    x == -Inf && return y
    y == -Inf && return x
    return max(x, y) + log1p(exp(-abs(x - y)))
end

# The residual is measured on the probabilities in both arithmetics.
_distance(::_LinearMessages, new::Vector, old::Vector) = maximum(abs.(new .- old))
_distance(::_LogMessages, new::Vector, old::Vector) = maximum(abs.(exp.(new) .- exp.(old)))

# m_{x -> f}(x) = prod_{g in N(x) \ f} m_{g -> x}(x), for the incidence (f, k) of x.
function _variable_message(A::_MessageArithmetic, st::_BPState{T}, x::Int, f::Int,
                           k::Int) where {T}
    m = _unit_message(A, T, length(st.axes[x]))
    for (g, l) in st.incidences[x]
        (g == f && l == k) && continue
        _absorb!(A, m, st.to_var[g][l])
    end
    return _normalize!(A, m, st)
end

# Update the messages out of factor `f`; convergence is measured separately.
function _update_factor!(A::_MessageArithmetic, st::_BPState, f::Int, damping)
    fac = st.factors[f]
    residual = 0.0
    for k in 1:ndims(fac)
        new = _normalize!(A, _factor_message(A, st, f, k), st)
        old = st.to_var[f][k]
        residual = max(residual, _distance(A, new, old))
        iszero(damping) || _damp!(A, new, old, damping)
        st.to_var[f][k] = new
    end
    return residual
end

function _fixed_point_residual!(A::_MessageArithmetic, st::_BPState)
    for (x, inc) in enumerate(st.incidences), (f, k) in inc
        st.to_factor[f][k] = _variable_message(A, st, x, f, k)
    end
    residual = 0.0
    for (f, fac) in enumerate(st.factors), k in 1:ndims(fac)
        target = _normalize!(A, _factor_message(A, st, f, k), st)
        residual = max(residual, _distance(A, target, st.to_var[f][k]))
    end
    return residual
end

function _sweep!(A::_MessageArithmetic, st::_BPState, damping, schedule::Symbol)
    residual = 0.0
    if schedule === :flooding
        for (x, inc) in enumerate(st.incidences), (f, k) in inc
            st.to_factor[f][k] = _variable_message(A, st, x, f, k)
        end
        for f in eachindex(st.factors)
            residual = max(residual, _update_factor!(A, st, f, damping))
        end
    else
        index = Dict{Symbol,Int}(v => i for (i, v) in enumerate(st.vars))
        for f in eachindex(st.factors)
            for (k, v) in enumerate(st.factors[f].vars)
                st.to_factor[f][k] = _variable_message(A, st, index[v], f, k)
            end
            residual = max(residual, _update_factor!(A, st, f, damping))
        end
    end
    return residual
end

# The marginal of variable `x`: the normalised product of its incoming messages. A
# negative cell, which only tolerated entries in [-atol, 0) can produce, leaves the
# posterior indeterminate (`_posterior_normalize`).
function _belief(A::_LinearMessages, st::_BPState{T}, x::Int) where {T}
    m = _unit_message(A, T, length(st.axes[x]))
    for (g, l) in st.incidences[x]
        _absorb!(A, m, st.to_var[g][l])
    end
    A.nonnegative && iszero(sum(m)) && _impossible(st.evidence)
    return _posterior_normalize(_factor([st.vars[x]], [st.axes[x]], m), st.evidence)
end
function _belief(A::_LogMessages, st::_BPState{T}, x::Int) where {T}
    m = _unit_message(A, T, length(st.axes[x]))
    for (g, l) in st.incidences[x]
        _absorb!(A, m, st.to_var[g][l])
    end
    _normalize!(A, m, st)
    return _factor([st.vars[x]], [st.axes[x]], exp.(m))
end

# The potentials of the binary64 run: each conditioned factor in the float type `R`, scaled
# by a power of two so that its largest magnitude lies in [1, 2). Messages are normalised,
# so a constant factor does not change them, and a power of two is exact, so a graph whose
# messages stay in the normal range gets the same messages bit for bit; one whose potentials
# are tiny (say, scaled by 1e-310) no longer underflows. Returns `nothing` when some
# potential cannot be held this way -- a nonzero entry would fall below `floatmin` of `R`, as
# when one factor's entries span more than binary64's range, or an entry is not finite --
# and the log domain is used from the start.
function _linear_potentials(::Type{R}, factors::Vector{<:Factor}) where {R}
    out = Factor{R}[]
    for g in factors
        all(isfinite, g.table) || return nothing
        peak = maximum(abs, g.table)
        if iszero(peak)
            push!(out, _convert_factor(R, g))
            continue
        end
        shift = -_binary_exponent(peak)
        table = map(x -> convert(R, _scaled(x, shift)), g.table)
        _faithful(g.table, table) || return nothing
        push!(out, _factor(g.vars, g.axes, table))
    end
    return out
end

_binary_exponent(x::AbstractFloat) = exponent(x)
_binary_exponent(x::Real) = exponent(BigFloat(x))

_scaled(x::AbstractFloat, shift::Int) = ldexp(x, shift)
_scaled(x::Real, shift::Int) = ldexp(BigFloat(x), shift)

_faithful(original::Array, scaled::Array) = true
function _faithful(original::Array, scaled::Array{R}) where {R<:Base.IEEEFloat}
    return all(i -> iszero(original[i]) || abs(scaled[i]) >= floatmin(R),
               eachindex(original, scaled))
end

# The log domain needs every entry of the graph finite (`FactorDomainError`) and
# nonnegative: a tolerated negative entry has no logarithm, and binary64 messages that left
# the normal range cannot show it small beside the evidence, so the posterior is
# indeterminate (ADR 0014). Every entry is checked, including those the evidence removes,
# as for the log backends and the exact fallback.
function _check_log_entries(fg::FactorGraph, evidence)
    for f in fg.factors, index in CartesianIndices(f.table)
        x = f.table[index]
        isfinite(x) || throw(FactorDomainError(:log_domain, copy(f.vars), Tuple(index), x))
        x < 0 &&
            throw(IndeterminatePosteriorError(Dict{Symbol,Symbol}(evidence),
                                              "binary64 messages left the normal range, and the log-domain messages have no logarithm for a tolerated negative entry ($(x))"))
    end
    return nothing
end

function _log_potential(::Type{L}, g::Factor) where {L}
    return _factor(g.vars, g.axes,
                   map(x -> iszero(x) ? L(-Inf) : _log_entry(L, x), g.table))
end

_log_entry(::Type{L}, x::AbstractFloat) where {L} = log(convert(L, x))
_log_entry(::Type{L}, x::Real) where {L} = convert(L, log(BigFloat(x)))

"""
    belief_propagation(fg::FactorGraph, backend=BeliefPropagation(); evidence=Dict{Symbol,Symbol}())
        -> (marginals::Dict{Symbol,Factor}, diagnostics::BPDiagnostics)

Sum-product belief propagation on `fg` conditioned on `evidence`. Factors
are conditioned first; a factor left with an empty scope only scales the
joint, so it is dropped from the message passing after its value has been
checked. Messages use a floating-point type (integer factors are promoted), start
uniform and follow the sum-product equations of SPEC section 20 until the
largest undamped message-equation residual at the current iterate is
below `backend.tol` or `backend.maxiter` sweeps have run. The marginal of
every unobserved variable is the normalised product of its incoming
messages; an observed variable maps to the point mass at its label. On a
tree-structured graph ([`is_tree`](@ref) after conditioning) the fixed point
is the exact marginal and `diagnostics.tree` is true; otherwise the
marginals are the loopy approximation and `diagnostics.converged` says
whether the residual tolerance was reached, not a bound on marginal error.

The update equations are Pearl's belief propagation [Pearl1988](@cite) in
the factor-graph (sum-product) form of [Kschischang2001](@cite); the
behaviour of the loopy fixed point on graphs with cycles is the empirical
study of [MurphyWeissJordan1999](@cite).

Messages are computed in binary64 (the graph's float type), each potential scaled by a
power of two, which leaves the normalised messages unchanged. When a binary64 message would
leave the normal range -- a product of nonzero values falls below `floatmin`, or a sum is
not a normal positive number without being exactly zero -- the same message passing is
redone in the log domain, where nothing underflows, and `diagnostics.log_domain` is true.
The answer is belief propagation's either way: belief propagation never falls back to an
exact backend, so its cost stays that of message passing.

Detectable impossible evidence is an error: when a dropped scalar factor is
zero, or when a message or the belief of a variable is exactly zero,
`BayesianNetworks.ImpossibleEvidenceError` is thrown rather than a normalised
belief returned, as variable elimination and the junction tree do. Such a zero is exact,
since no product before it underflowed, and on nonnegative factors an exactly zero message
proves that the evidence has probability exactly zero. Local support is not a global
feasibility test on loopy graphs, and damped iterates may retain tiny positive support. Set
`backend.check_evidence=true` to run an exact VE feasibility check first
(it throws the same error for zero evidence mass);
`diagnostics.evidence_checked` records that opt-in. A tolerated entry in
`[-atol, 0)` makes a zero sum inconclusive: when binary64 messages leave the normal range
on such a graph, or a belief cell comes out negative,
`BayesianNetworks.IndeterminatePosteriorError` is thrown.

Throws [`ScopeError`](@ref) for evidence on unknown variables,
`FiniteKernels.InvalidAxisError` for a label a variable does not have, and
[`FactorDomainError`](@ref) for an entry that is not finite (in a graph built with
`check = false`) once the log domain is needed.
"""
function belief_propagation(fg::FactorGraph{T},
                            backend::BeliefPropagation=BeliefPropagation();
                            evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}()) where {T}
    _check_query(fg, Symbol[], evidence)
    ev = Dict{Symbol,Symbol}(evidence)
    backend.check_evidence && _require_feasible(fg, ev, MinFill())
    conditioned = Factor{T}[]
    for f in fg.factors
        g = condition(f, ev)
        if isempty(g.vars)
            # A factor conditioned to the empty scope is one entry of the data. Check it,
            # in its own type and not the product of all of them, which can underflow: an
            # exact zero entry makes the evidence impossible, and a negative one leaves it
            # indeterminate.
            v = g.table[]
            iszero(v) && _impossible(ev)
            v < 0 &&
                throw(IndeterminatePosteriorError(ev,
                                                  "a factor conditioned on the evidence is negative ($(v))"))
        else
            push!(conditioned, g)
        end
    end
    R = typeof(float(one(T)))
    tree = _is_forest(conditioned)
    potentials = _linear_potentials(R, conditioned)
    if potentials !== nothing
        A = _LinearMessages(all(g -> all(>=(0), g.table), potentials))
        try
            return _run_bp(A, potentials, fg, backend, ev, R, tree)
        catch e
            e isa _UnresolvedMass || rethrow()
        end
    end
    _check_log_entries(fg, ev)
    L = promote_type(R, Float64)
    logs = Factor{L}[_log_potential(L, g) for g in conditioned]
    return _run_bp(_LogMessages(), logs, fg, backend, ev, R, tree)
end

function _run_bp(A::_MessageArithmetic, potentials::Vector{<:Factor}, fg::FactorGraph,
                 backend::BeliefPropagation, ev, ::Type{R}, tree::Bool) where {R}
    st = _BPState(A, potentials, fg.axes, ev)
    iterations = 0
    residual = Inf
    converged = false
    while iterations < backend.maxiter
        _sweep!(A, st, backend.damping, backend.schedule)
        residual = _fixed_point_residual!(A, st)
        iterations += 1
        if residual < backend.tol
            converged = true
            break
        end
    end
    marginals = Dict{Symbol,Factor{R}}()
    for (x, v) in enumerate(st.vars)
        marginals[v] = _convert_factor(R, _belief(A, st, x))
    end
    for (v, l) in ev
        marginals[v] = _point_mass(R, fg.axes[v], l)
    end
    return marginals,
           BPDiagnostics(iterations, converged, residual, tree, backend.check_evidence,
                         A isa _LogMessages)
end

function _infer(backend::BeliefPropagation, fg::FactorGraph, query, evidence)
    _check_query(fg, query, evidence)
    length(query) == 1 ||
        throw(ScopeError(:infer,
                         "BeliefPropagation answers single-variable queries only; use VariableElimination or JunctionTree for a joint query",
                         collect(query)))
    marginals, diagnostics = belief_propagation(fg, backend; evidence)
    return marginals[query[1]], diagnostics
end

function _all_marginals(backend::BeliefPropagation, fg::FactorGraph, evidence)
    return belief_propagation(fg, backend; evidence)[1]
end
