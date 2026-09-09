@testset "Centered log-domain inference" begin
    a = FiniteAxis(:A, [:rare, :usual])
    b = FiniteAxis(:B, [:rare, :usual])
    c = FiniteAxis(:C, [:low, :high])
    for p in (1e-200, ldexp(1.0, -537))
        graph = FactorGraph([Factor(a, [p, 1.0]), Factor(b, [p, 1.0]), Factor(c, [.25, .75])])
        ev = Dict(:A => :rare, :B => :rare)
        posterior, info = infer(graph, :C; evidence=ev, backend=LogVariableElimination())
        @test posterior.table ≈ [.25, .75] atol=8eps(Float64) rtol=8eps(Float64)
        @test isfinite(info.log_evidence_probability)
        @test info.log_evidence_probability ≈ 2log(p)
        @test log_evidence_probability(graph; evidence=ev) ≈ 2log(p)
        mass, mass_info = infer(graph, Symbol[]; evidence=ev, backend=LogVariableElimination())
        @test mass_info.log_evidence_probability ≈ info.log_evidence_probability
        @test mass_info.mass_status == (p == 1e-200 ? :underflow : :finite)
        @test p != 1e-200 || mass.table[] == 0
        all = all_marginals(graph; evidence=ev, backend=LogVariableElimination())
        @test all[:C] ≈ posterior
        @test all[:A].table == [1, 0]
    end

    x = FiniteAxis(:X, [:left, :middle, :right])
    intersection = FactorGraph([Factor(x, [1.0, 1e-200, 0.0]), Factor(x, [0.0, 1e-200, 1.0])])
    @test first(infer(intersection, :X; backend=LogVariableElimination())).table == [0, 1, 0]
    @test log_evidence_probability(intersection) ≈ 2log(1e-200)
    impossible = FactorGraph([Factor(a, [1.0, 0.0]), Factor(c, [.25, .75])])
    ev = Dict(:A => :usual)
    @test log_evidence_probability(impossible; evidence=ev) == -Inf
    @test_throws KernelNormalizationError infer(impossible, :C; evidence=ev, backend=LogVariableElimination())
    @test_throws KernelNormalizationError all_marginals(impossible; evidence=ev, backend=LogVariableElimination())
    @test last(infer(impossible, Symbol[]; evidence=ev, backend=LogVariableElimination())).mass_status == :zero

    for value in (-eps(), Inf, NaN)
        graph = FactorGraph([Factor(a, [value, 1.0])])
        @test_throws LogFactorDomainError infer(graph, :A; backend=LogVariableElimination())
        @test_throws LogFactorDomainError log_evidence_probability(graph; evidence=Dict(:A => :usual))
    end
    huge = FactorGraph([Factor(a, [1e300, 1e300]), Factor(a, [1e300, 1e300])])
    huge_posterior, huge_info = infer(huge, :A; backend=LogVariableElimination())
    @test huge_posterior.table == [.5, .5]
    @test huge_info.mass_status == :overflow
    @test huge_info.log_evidence_probability ≈ 2log(1e300) + log(2)
    @test isinf(first(infer(huge, Symbol[]; backend=LogVariableElimination())).table[])

    rng = MersenneTwister(214)
    for _ in 1:30
        table = rand(rng, 1:8, 2, 2, 2) ./ 8
        factors = [Factor(a, [.25, .75]), Factor(b, [.5, .5]), Factor([a, b, c], table)]
        graph = FactorGraph(factors)
        exact = FactorGraph([Factor(f.axes, Rational{BigInt}.(f.table)) for f in factors])
        for query in ([:C], [:C, :A]), evidence in (Dict{Symbol,Symbol}(), Dict(:B => :rare))
            expected = brute_force_marginal(exact, query; evidence)
            for order in (MinFill(), MinDegree(), ExactTreewidth())
                actual, _ = infer(graph, query; evidence, backend=LogVariableElimination(; order))
                @test actual.vars == query
                @test actual.table ≈ Float64.(expected.table) atol=32eps(Float64) rtol=128eps(Float64)
            end
        end
    end
    graph = FactorGraph([Factor(a, [.25, .75]), Factor([a, c], [.75 .25; .125 .875])])
    @test first(infer(graph, :C; backend=LogVariableElimination(order=UserOrder([:A])))) ≈
          first(infer(graph, :C))
    @test_throws ScopeError infer(graph, :Unknown; backend=LogVariableElimination())
    @test_throws ScopeError infer(graph, :A; evidence=Dict(:A => :rare), backend=LogVariableElimination())
    @test_throws FiniteKernels.InvalidAxisError infer(graph, :C; evidence=Dict(:A => :missing), backend=LogVariableElimination())
    empty = FactorGraph(Factor{Float64}[])
    @test log_evidence_probability(empty) == 0
    @test first(infer(empty, Symbol[]; backend=LogVariableElimination())).table[] == 1

    bn = bayesnet(:A => [:rare, :usual], :B => [:rare, :usual], :C => [:low, :high])
    model = bind_cpt(BayesModel(bn), [:A => [1e-200, 1.0], :B => [1e-200, 1.0], :C => [.25, .75]])
    observed = observe(model, [:A => :rare, :B => :rare])
    @test first(infer(observed, :C; backend=LogVariableElimination())).table ≈ [.25, .75]
    @test log_evidence_probability(observed) ≈ 2log(1e-200)
    @test log_evidence_probability(observed; evidence=:A => :usual) ≈ log(1e-200)
    @test evidence(observed) == Dict(:A => :rare, :B => :rare)
end
