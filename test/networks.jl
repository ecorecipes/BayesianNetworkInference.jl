# Networks shared by the tests and vignettes: asia from its published CPTs,
# the SPEC section 45 habitat chain with random kernels, and random DAGs.
# Each builder returns (kernels, order, parents) so that the same model can be
# compiled to factors and sampled.

using FiniteKernels: random_kernel

function factor_graph_from(kernels::Vector{Pair{Symbol,FiniteKernel}},
                           parents::Dict{Symbol,Vector{Symbol}})
    factors = [Factor(k, parents[v], v) for (v, k) in kernels]
    return FactorGraph(factors; provenance=[v for (v, _) in kernels])
end

# The asia network (Lauritzen & Spiegelhalter 1988) with the bnlearn CPTs.
function asia_network()
    yn = [:yes, :no]
    ax(v) = FiniteAxis(v, yn)
    # BIF rows are (parent configuration) -> child probabilities, child fastest;
    # tables below are written in the (parents..., child) layout directly.
    k_asia = cpt(ax(:asia), [0.01, 0.99])
    k_tub = cpt(ax(:asia), ax(:tub), [0.05 0.95; 0.01 0.99])
    k_smoke = cpt(ax(:smoke), [0.5, 0.5])
    k_lung = cpt(ax(:smoke), ax(:lung), [0.1 0.9; 0.01 0.99])
    k_bronc = cpt(ax(:smoke), ax(:bronc), [0.6 0.4; 0.3 0.7])
    either = zeros(2, 2, 2)      # (lung, tub, either)
    either[1, 1, :] = [1.0, 0.0]
    either[2, 1, :] = [1.0, 0.0]
    either[1, 2, :] = [1.0, 0.0]
    either[2, 2, :] = [0.0, 1.0]
    k_either = cpt([ax(:lung), ax(:tub)], ax(:either), either)
    k_xray = cpt(ax(:either), ax(:xray), [0.98 0.02; 0.05 0.95])
    dysp = zeros(2, 2, 2)        # (bronc, either, dysp)
    dysp[1, 1, :] = [0.9, 0.1]
    dysp[2, 1, :] = [0.7, 0.3]
    dysp[1, 2, :] = [0.8, 0.2]
    dysp[2, 2, :] = [0.1, 0.9]
    k_dysp = cpt([ax(:bronc), ax(:either)], ax(:dysp), dysp)
    kernels = Pair{Symbol,FiniteKernel}[:asia => k_asia, :tub => k_tub, :smoke => k_smoke,
                                        :lung => k_lung, :bronc => k_bronc,
                                        :either => k_either, :xray => k_xray,
                                        :dysp => k_dysp]
    parents = Dict{Symbol,Vector{Symbol}}(:asia => [], :tub => [:asia], :smoke => [],
                                          :lung => [:smoke], :bronc => [:smoke],
                                          :either => [:lung, :tub], :xray => [:either],
                                          :dysp => [:bronc, :either])
    order = [:asia, :tub, :smoke, :lung, :bronc, :either, :xray, :dysp]
    return kernels, order, parents
end

# SPEC section 45: Climate -> SoilMoisture -> Vegetation -> HabitatQuality -> Occupancy,
# with Irrigation -> SoilMoisture and GrazingPressure -> Vegetation.
function habitat_network(rng::AbstractRNG)
    states = Dict(:Climate => [:dry, :wet], :Irrigation => [:none, :low, :high],
                  :SoilMoisture => [:low, :medium, :high],
                  :GrazingPressure => [:low, :high],
                  :Vegetation => [:sparse, :moderate, :dense],
                  :HabitatQuality => [:poor, :fair, :good],
                  :Occupancy => [:absent, :present])
    parents = Dict{Symbol,Vector{Symbol}}(:Climate => [], :Irrigation => [],
                                          :SoilMoisture => [:Climate, :Irrigation],
                                          :GrazingPressure => [],
                                          :Vegetation => [:SoilMoisture, :GrazingPressure],
                                          :HabitatQuality => [:Vegetation],
                                          :Occupancy => [:HabitatQuality])
    order = [:Climate, :Irrigation, :SoilMoisture, :GrazingPressure, :Vegetation,
             :HabitatQuality, :Occupancy]
    return random_network(rng, order, parents, states)
end

function random_network(rng::AbstractRNG, order::Vector{Symbol},
                        parents::Dict{Symbol,Vector{Symbol}},
                        states::Dict{Symbol,Vector{Symbol}})
    ax(v) = FiniteAxis(v, states[v])
    kernels = Pair{Symbol,FiniteKernel}[]
    for v in order
        dom = FiniteSpace(FiniteAxis[ax(p) for p in parents[v]])
        push!(kernels, v => random_kernel(rng, dom, FiniteSpace(ax(v))))
    end
    return kernels, order, parents
end

# A random DAG on `n` variables with 2-3 states each and at most `maxparents`
# parents drawn from earlier variables.
function random_dag_network(rng::AbstractRNG; n::Int=rand(rng, 3:7), maxparents::Int=3)
    order = [Symbol("X", i) for i in 1:n]
    states = Dict(v => [Symbol("s", j) for j in 1:rand(rng, 2:3)] for v in order)
    parents = Dict{Symbol,Vector{Symbol}}()
    for (i, v) in enumerate(order)
        cands = order[1:(i - 1)]
        chosen = [p for p in cands if rand(rng) < 0.4]
        length(chosen) > maxparents && (chosen = chosen[1:maxparents])
        parents[v] = chosen
    end
    return random_network(rng, order, parents, states)
end

# A moderately sized benchmark: `n` variables with 2-3 states, each with up to
# `maxparents` parents among the previous `window` variables, so that the
# treewidth stays small while the network is too large for `joint_factor`.
function benchmark_network(rng::AbstractRNG; n::Int=30, window::Int=6, maxparents::Int=3)
    order = [Symbol("B", i) for i in 1:n]
    states = Dict(v => [Symbol("s", j) for j in 1:rand(rng, 2:3)] for v in order)
    parents = Dict{Symbol,Vector{Symbol}}()
    for (i, v) in enumerate(order)
        cands = order[max(1, i - window):(i - 1)]
        k = min(rand(rng, 0:maxparents), length(cands))
        parents[v] = shuffle(rng, cands)[1:k]
    end
    return random_network(rng, order, parents, states)
end
