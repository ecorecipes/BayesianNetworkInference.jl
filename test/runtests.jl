using Test
using BayesianNetworkInference
using FiniteKernels
using Random
using Random: shuffle
using Graphs
using Logging: NullLogger, with_logger
using CliqueTrees
using BayesianNetworks
using BayesianNetworkFormats: fixture_path

include("networks.jl")

@testset "BayesianNetworkInference" begin
    include("test_factors.jl")
    include("test_orderings.jl")
    include("test_inference.jl")
    include("test_log_inference.jl")
    include("test_sampling.jl")
    include("test_junction_tree.jl")
    include("test_log_junction_tree.jl")
    include("test_belief_propagation.jl")
    include("test_model_bridge.jl")
    include("test_scores.jl")
    include("test_sensitivity.jl")
    include("test_regressions.jl")
end
