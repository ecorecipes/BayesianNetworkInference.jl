# Ancestral (forward) sampling over named kernels in topological order.

"""
    AncestralSamples(vars, axes, states)

`n` joint samples: `states[i, j]` is the label of `vars[j]` in sample `i`, and
`axes[j]` carries the labels of `vars[j]`. Index with `samples[var]` to get one
column; see [`empirical_marginal`](@ref).
"""
struct AncestralSamples
    vars::Vector{Symbol}
    axes::Vector{FiniteAxis}
    states::Matrix{Symbol}
end

Base.size(s::AncestralSamples) = size(s.states)
Base.length(s::AncestralSamples) = size(s.states, 1)
function Base.getindex(s::AncestralSamples, var::Symbol)
    j = findfirst(==(var), s.vars)
    j === nothing &&
        throw(ScopeError(:getindex, "variable $(repr(var)) was not sampled", [var]))
    return view(s.states, :, j)
end
function Base.show(io::IO, s::AncestralSamples)
    return print(io, "AncestralSamples: ", length(s), " samples of ", Tuple(s.vars))
end

"""
    ancestral_sample(kernels, order::Vector{Symbol}, parents::Dict{Symbol,Vector{Symbol}}, n;
                     rng=Random.default_rng()) -> AncestralSamples

Draw `n` joint samples from the Bayesian network whose mechanism for variable
`v` is the kernel `kernels[v]` (a `Vector{Pair{Symbol,FiniteKernel}}` or a
dictionary) with inputs `parents[v]` in the kernel's input-axis order. Every
variable is sampled from its kernel given the already sampled values of its
parents, visiting the variables in `order`, which must be topological (each
parent listed before its child). Throws [`ScopeError`](@ref) if `order` is not
a permutation of the kernel names or violates the parent order, and
[`ShapeError`](@ref) if a kernel's input arity does not match `parents`.

The model bridge calls this with the kernels, parents and a topological order
read off a `BayesModel`; [`empirical_marginal`](@ref) turns the result into
factors comparable with exact posteriors.
"""
function ancestral_sample(kernels::AbstractVector{<:Pair{Symbol,<:FiniteKernel}},
                          order::AbstractVector{Symbol},
                          parents::AbstractDict{Symbol,<:AbstractVector{Symbol}},
                          n::Integer; rng::AbstractRNG=default_rng())
    return ancestral_sample(Dict{Symbol,FiniteKernel}(kernels), order, parents, n; rng)
end

function ancestral_sample(kernels::AbstractDict{Symbol,<:FiniteKernel},
                          order::AbstractVector{Symbol},
                          parents::AbstractDict{Symbol,<:AbstractVector{Symbol}},
                          n::Integer; rng::AbstractRNG=default_rng())
    n >= 0 || throw(ArgumentError("number of samples must be non-negative, got $n"))
    allunique(order) ||
        throw(ScopeError(:ancestral_sample, "sampling order repeats variables",
                         collect(order)))
    missing_vars = [v for v in keys(kernels) if !(v in order)]
    extra = [v for v in order if !haskey(kernels, v)]
    isempty(missing_vars) && isempty(extra) ||
        throw(ScopeError(:ancestral_sample,
                         "sampling order must be a permutation of the kernel names",
                         vcat(missing_vars, extra)))
    position = Dict{Symbol,Int}(v => j for (j, v) in enumerate(order))
    m = length(order)
    axes = Vector{FiniteAxis}(undef, m)
    parent_cols = Vector{Vector{Int}}(undef, m)
    tables = Vector{Array{Float64}}(undef, m)
    for (j, v) in enumerate(order)
        k = kernels[v]
        ps = get(parents, v, Symbol[])
        ndims(k.codom) == 1 ||
            throw(ShapeError(:ancestral_sample,
                             "kernel of $(repr(v)) must have one output axis", 1,
                             ndims(k.codom)))
        ndims(k.dom) == length(ps) ||
            throw(ShapeError(:ancestral_sample,
                             "kernel of $(repr(v)) must have one input axis per parent",
                             length(ps), ndims(k.dom)))
        late = [p for p in ps if !haskey(position, p) || position[p] >= j]
        isempty(late) ||
            throw(ScopeError(:ancestral_sample,
                             "parents of $(repr(v)) must be sampled before it", late))
        axes[j] = FiniteAxis(v, labels(k.codom.axes[1]))
        parent_cols[j] = Int[position[p] for p in ps]
        tables[j] = Array{Float64}(k.table)
    end
    idx = Matrix{Int}(undef, n, m)
    for i in 1:n, j in 1:m
        cols = parent_cols[j]
        col = view(tables[j], :, ntuple(l -> idx[i, cols[l]], length(cols))...)
        idx[i, j] = _sample_categorical(rng, col)
    end
    states = [axes[j].labels[idx[i, j]] for i in 1:n, j in 1:m]
    return AncestralSamples(collect(Symbol, order), axes, states)
end

function _sample_categorical(rng::AbstractRNG, p::AbstractVector{<:Real})
    u = rand(rng)
    acc = zero(eltype(p))
    for (i, pi) in enumerate(p)
        acc += pi
        u < acc && return i
    end
    return length(p)
end

"""
    empirical_marginal(samples::AncestralSamples, var::Symbol) -> Factor
    empirical_marginal(samples::AncestralSamples, vars::Vector{Symbol}) -> Factor

The relative frequencies of the labels of `var` (or of the joint labels of
`vars`) among the samples, as a normalised factor comparable to the output of
[`variable_elimination`](@ref).
"""
function empirical_marginal(s::AncestralSamples, vars::AbstractVector{Symbol})
    cols = Int[]
    for v in vars
        j = findfirst(==(v), s.vars)
        j === nothing &&
            throw(ScopeError(:empirical_marginal, "variable $(repr(v)) was not sampled",
                             [v]))
        push!(cols, j)
    end
    allunique(cols) ||
        throw(ScopeError(:empirical_marginal, "variables repeat", collect(vars)))
    axes = s.axes[cols]
    counts = zeros(Float64, Tuple(map(length, axes)))
    n = length(s)
    for i in 1:n
        ci = CartesianIndex(ntuple(l -> label_index(axes[l], s.states[i, cols[l]]),
                                   length(cols)))
        counts[ci] += 1
    end
    n == 0 && return _factor(collect(Symbol, vars), axes, counts)
    return _factor(collect(Symbol, vars), axes, counts ./ n)
end
empirical_marginal(s::AncestralSamples, var::Symbol) = empirical_marginal(s, [var])
