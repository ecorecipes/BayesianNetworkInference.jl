@testset "ancestral sampling" begin
    kernels, order, parents = asia_network()
    fg = factor_graph_from(kernels, parents)
    rng = MersenneTwister(7)
    n = 20_000
    s = ancestral_sample(kernels, order, parents, n; rng)
    @test size(s) == (n, 8) && length(s) == n
    @test s.vars == order
    @test all(x -> x in (:yes, :no), s[:dysp])
    @test_throws ScopeError s[:nope]
    @test sprint(show, s) ==
          "AncestralSamples: 20000 samples of (:asia, :tub, :smoke, :lung, :bronc, :either, :xray, :dysp)"
    for v in order
        exact, _ = variable_elimination(fg, [v])
        emp = empirical_marginal(s, v)
        @test scope(emp) == [v] && sum(emp.table) ≈ 1
        @test isapprox(emp, exact; atol=0.02)
    end
    joint = empirical_marginal(s, [:smoke, :dysp])
    @test isapprox(joint, variable_elimination(fg, [:smoke, :dysp])[1]; atol=0.02)
    @test_throws ScopeError empirical_marginal(s, [:smoke, :smoke])
    @test_throws ScopeError empirical_marginal(s, [:nope])
    # dictionary form and reproducibility
    s2 = ancestral_sample(Dict(kernels), order, parents, 100; rng=MersenneTwister(3))
    s3 = ancestral_sample(kernels, order, parents, 100; rng=MersenneTwister(3))
    @test s2.states == s3.states
    @test length(ancestral_sample(kernels, order, parents, 0; rng)) == 0
    # deterministic kernel is respected: either = yes whenever lung = yes
    @test all(s.states[i, 6] == :yes for i in 1:n if s.states[i, 4] == :yes)
    # validation
    @test_throws ScopeError ancestral_sample(kernels, reverse(order), parents, 10; rng)
    @test_throws ScopeError ancestral_sample(kernels, order[1:7], parents, 10; rng)
    @test_throws ScopeError ancestral_sample(kernels, vcat(order, :asia), parents, 10; rng)
    bad = Dict(parents)
    bad[:dysp] = [:bronc]
    @test_throws ShapeError ancestral_sample(kernels, order, bad, 10; rng)
    @test_throws ArgumentError ancestral_sample(kernels, order, parents, -1; rng)
    # habitat chain: joint of the two management variables
    hrng = MersenneTwister(45)
    hk, horder, hparents = habitat_network(hrng)
    hs = ancestral_sample(hk, horder, hparents, n; rng=MersenneTwister(8))
    hfg = factor_graph_from(hk, hparents)
    @test isapprox(empirical_marginal(hs, :Occupancy),
                   variable_elimination(hfg, [:Occupancy])[1]; atol=0.02)
end
