@testset "posterior feasibility and numeric types" begin
    x = FiniteAxis(:X, [:yes, :no])
    y = FiniteAxis(:Y, [:yes, :no])
    fg = FactorGraph([Factor(x, [1.0, 0.0]), Factor(y, [0.3, 0.7])])
    impossible = Dict(:X => :no)
    for backend in (VariableElimination(), JunctionTree())
        @test_throws KernelNormalizationError infer(fg, :Y; evidence=impossible, backend)
        @test_throws KernelNormalizationError all_marginals(fg; evidence=impossible,
                                                            backend)
        @test infer(fg, Symbol[]; evidence=impossible, backend)[1].table[] == 0
        @test infer(fg, :Y; evidence=Dict(:X => :yes), backend)[1].table ≈ [0.3, 0.7]
        single = FactorGraph([Factor(x, [1.0, 0.0])])
        @test_throws KernelNormalizationError all_marginals(single; evidence=impossible,
                                                            backend)
        @test all_marginals(single; evidence=Dict(:X => :yes), backend)[:X].table ==
              [1.0, 0.0]
        scalar = FactorGraph([Factor(FiniteAxis[], fill(0.0))])
        @test_throws KernelNormalizationError all_marginals(scalar; backend)
        @test infer(scalar, Symbol[]; backend)[1].table[] == 0
    end
    @test_throws KernelNormalizationError clique_beliefs(fg; evidence=impossible)

    for T in (Int, Float32, Rational{Int})
        weights = FactorGraph([Factor(x, T[1, 1]), Factor(y, T[1, 3])])
        for backend in (VariableElimination(), JunctionTree(), BeliefPropagation())
            marginals = all_marginals(weights; backend)
            @test marginals[:X].table ≈ [0.5, 0.5]
            @test marginals[:Y].table ≈ [0.25, 0.75]
            @test marginals[:X] ≈ brute_force_marginal(weights, :X)
            observed = all_marginals(weights; evidence=Dict(:X => :no), backend)
            @test observed[:X].table == [0, 1]
            @test observed[:Y].table ≈ [0.25, 0.75]
        end
    end
end

@testset "BP residual and explicit feasibility contract" begin
    x = FiniteAxis(:X, [:yes, :no])
    fg = FactorGraph([Factor(x, [0.9, 0.1])])
    oracle = brute_force_marginal(fg, :X)
    for schedule in (:flooding, :sequential)
        backend = BeliefPropagation(; damping=1 - 1e-9, schedule)
        p, d = infer(fg, :X; backend)
        @test !d.converged && d.iterations == backend.maxiter
        @test d.max_residual > 0.39
        @test maximum(abs, p.table .- oracle.table) > 0.39
        @test !d.evidence_checked
        exact, checked = infer(fg, :X;
                               backend=BeliefPropagation(; schedule, check_evidence=true))
        @test exact ≈ oracle
        @test checked.converged && checked.max_residual == 0 && checked.evidence_checked
    end
    @test_throws ArgumentError BeliefPropagation(; tol=Inf)
    @test_throws ArgumentError BeliefPropagation(; tol=NaN)
    @test !BeliefPropagation(0.0, 1e-8, 200, :flooding).check_evidence
    @test !BPDiagnostics(1, false, 0.5, true).evidence_checked

    bn = bayesnet((v => [:no, :yes] for v in (:A, :B, :C, :AB, :BC, :AC))...;
                  mechanisms=[:AB => (:A, :B), :BC => (:B, :C), :AC => (:A, :C)])
    eq = zeros(2, 2, 2)
    for i in 1:2, j in 1:2
        eq[i, j, i == j ? 2 : 1] = 1.0
    end
    m = bind_cpt(BayesModel(bn),
                 [:A => [0.5, 0.5], :B => [0.5, 0.5], :C => [0.5, 0.5],
                  :AB => eq, :BC => eq, :AC => eq])
    ev = Dict(:AB => :yes, :BC => :yes, :AC => :no)
    @test infer(m, Symbol[]; evidence=ev)[1].table[] == 0
    _, unchecked = infer(m, :A; evidence=ev, backend=BeliefPropagation())
    @test unchecked.converged && !unchecked.tree && !unchecked.evidence_checked
    for schedule in (:flooding, :sequential)
        @test_throws KernelNormalizationError infer(m, :A; evidence=ev,
                                                    backend=BeliefPropagation(; schedule,
                                                                              check_evidence=true))
    end
end

@testset "repeated ordered kernel input slots" begin
    bn = bayesnet(:A => [:a0, :a1], :B => [:b0, :b1, :b2], :Y => [:y0, :y1];
                  mechanisms=[:Y => (:B, :A, :B)])
    table = zeros(3, 2, 3, 2)
    for b1 in 1:3, a in 1:2, b2 in 1:3
        p = (b1 + 2a + 3b2) / 20
        table[b1, a, b2, :] = [p, 1 - p]
    end
    m = bind_cpt(BayesModel(bn), [:A => [0.4, 0.6], :B => [0.2, 0.3, 0.5],
                                  :Y => table])
    fg = compile(m)
    f = only(f for (f, p) in zip(fg.factors, fg.provenance) if p.variable == :Y)
    @test scope(f) == [:B, :A, :Y]
    @test size(f) == (3, 2, 2)
    for b in 1:3, a in 1:2, y in 1:2
        @test f.table[b, a, y] == table[b, a, b, y]
    end
    oracle = Factor(marginal(m, :Y), :Y)
    for backend in (VariableElimination(), JunctionTree(), BeliefPropagation())
        @test infer(m, :Y; backend)[1] ≈ oracle
    end
    @test joint_factor(fg) ≈ factor_of(joint_distribution(m))
end

@testset "model normalization tolerance reaches all consumers" begin
    bn = bayesnet(:X => [:yes, :no], :Y => [:yes, :no])
    m = bind_cpt(BayesModel(bn), [:X => [0.5, 0.5000001], :Y => [0.25, 0.75]];
                 atol=1e-6)
    @test_throws BayesianNetworks.UnnormalizedKernelError compile(m)
    @test compile(m; atol=1e-6) isa FactorGraph
    expected = [0.5, 0.5000001] ./ 1.0000001
    for backend in (VariableElimination(), JunctionTree(), BeliefPropagation())
        @test infer(m, :X; backend, atol=1e-6)[1].table ≈ expected
        @test all_marginals(m; backend, atol=1e-6)[:X].table ≈ expected
    end
    @test posterior(m, :X; atol=1e-6)[:yes] ≈ expected[1]
    @test length(ancestral_sample(m, 3; rng=MersenneTwister(3), atol=1e-6)) == 3
    @test isfinite(entropy(m, :X; atol=1e-6))
    @test mutual_information(m, :X, :Y; atol=1e-6) ≈ 0 atol=1e-12
    @test length(sensitivity(m, :X; atol=1e-6)) == 1
    @test length(tornado(m, :X, :yes; atol=1e-6)) == 1
    cases = [Dict(:X => :yes, :Y => :yes), Dict(:X => :no, :Y => :no)]
    @test length(predict(m, cases, :X; atol=1e-6)) == 2
    @test isfinite(evaluate(m, cases, :X; atol=1e-6).brier)
end
