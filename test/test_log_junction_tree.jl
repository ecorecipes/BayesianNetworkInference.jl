@testset "Log-domain junction trees" begin
    a, b, c, d = (FiniteAxis(name, [:low, :high]) for name in (:A, :B, :C, :D))
    graph = FactorGraph([Factor(a, [1e-200, 1.0]), Factor(b, [1e-200, 1.0]),
                         Factor(c, [0.25, 0.75])])
    observed = Dict(:A => :low, :B => :low)
    tree = build_junction_tree(graph)
    calibrated = log_calibrate(graph, tree; evidence=observed)
    @test calibrated.tree === tree
    @test log_evidence_probability(calibrated) ≈ 2log(1e-200)
    posterior, diagnostics = infer(graph, :C; evidence=observed, backend=LogJunctionTree())
    @test posterior.table ≈ [0.25, 0.75]
    @test diagnostics isa LogJunctionTreeDiagnostics
    @test diagnostics.mass_status == :underflow
    @test !diagnostics.fallback
    marginals = all_marginals(graph; evidence=observed, backend=LogJunctionTree())
    @test marginals[:A].table == [1, 0]
    @test marginals[:B].table == [1, 0]
    @test marginals[:C] ≈ posterior
    @test all(sum(f.table) ≈ 1
              for f in clique_beliefs(graph; evidence=observed, backend=LogJunctionTree()))
    @test first(infer(graph, Symbol[]; evidence=observed, backend=LogJunctionTree())).table[] ==
          0

    impossible = FactorGraph([Factor(a, [0.0, 1.0]), Factor(c, [0.25, 0.75])])
    @test log_evidence_probability(log_calibrate(impossible; evidence=Dict(:A => :low))) ==
          -Inf
    @test_throws KernelNormalizationError infer(impossible, :C; evidence=Dict(:A => :low),
                                                backend=LogJunctionTree())
    @test_throws KernelNormalizationError all_marginals(impossible;
                                                        evidence=Dict(:A => :low),
                                                        backend=LogJunctionTree())
    @test_throws KernelNormalizationError clique_beliefs(impossible;
                                                         evidence=Dict(:A => :low),
                                                         backend=LogJunctionTree())
    for value in (-eps(), NaN, Inf)
        invalid = FactorGraph([Factor(a, [value, 1.0]), Factor(c, [0.25, 0.75])])
        @test_throws LogFactorDomainError log_calibrate(invalid; evidence=Dict(:A => :high))
    end

    rng = MersenneTwister(20260909)
    for _ in 1:20
        ab, bc = zeros(2, 2), zeros(2, 2)
        for state in 1:2
            p, q = rand(rng, 1:7) / 8, rand(rng, 1:7) / 8
            ab[state, :] = [p, 1 - p]
            bc[state, :] = [q, 1 - q]
        end
        graph = FactorGraph([Factor(a, [0.25, 0.75]), Factor([a, b], ab),
                             Factor([b, c], bc), Factor(d, [0.5, 0.5])])
        exact = FactorGraph([Factor(f.axes, Rational{BigInt}.(f.table))
                             for f in graph.factors])
        for evidence in (Dict{Symbol,Symbol}(), Dict(:B => :low), Dict(:B => :high))
            for backend in (LogJunctionTree(), LogJunctionTree(; order=MinDegree()))
                actual = all_marginals(graph; evidence, backend)
                for variable in (:A, :C, :D)
                    reference = brute_force_marginal(exact, variable; evidence)
                    @test actual[variable].table ≈ Float64.(reference.table) atol = 32eps() rtol = 128eps()
                end
            end
        end
        joint, info = @test_logs (:warn, r"falling back to log-domain") infer(graph,
                                                                              [:C, :A];
                                                                              backend=LogJunctionTree())
        @test joint.table ≈ Float64.(brute_force_marginal(exact, [:C, :A]).table) atol = 32eps() rtol = 128eps()
        @test info.fallback && info.n_messages == 0
        all_observed = Dict(:A => :low, :B => :high, :C => :low, :D => :high)
        @test all(sum(f.table) == 1
                  for f in clique_beliefs(graph; evidence=all_observed,
                                          backend=LogJunctionTree()))
    end

    scalar = Factor(FiniteAxis[], fill(2.0))
    aliases = FactorGraph([scalar, scalar, scalar])
    repeated_cliques = CompiledJunctionTree([Symbol[], Symbol[], Symbol[]],
                                            [Symbol[], Symbol[], Symbol[]],
                                            [0, 1, 1], [[2, 3], Int[], Int[]],
                                            [1], [2, 3, 1], [1, 2, 3],
                                            Dict{Symbol,Int}(), Symbol[], -1)
    @test all(belief.table[] == 8
              for belief in calibrate(aliases, repeated_cliques).beliefs)
    @test all(BayesianNetworkInference._log_mass(belief) ≈ log(8)
              for belief in log_calibrate(aliases, repeated_cliques).beliefs)
    empty = FactorGraph(Factor{Float64}[])
    @test log_evidence_probability(log_calibrate(empty)) == 0
    @test isempty(all_marginals(empty; backend=LogJunctionTree()))
    @test first(infer(empty, Symbol[]; backend=LogJunctionTree())).table[] == 1
end
