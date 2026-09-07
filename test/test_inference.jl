@testset "variable elimination" begin
    kernels, order, parents = asia_network()
    fg = factor_graph_from(kernels, parents)

    @testset "asia: known marginal and brute force" begin
        p, d = variable_elimination(fg, [:dysp])
        @test scope(p) == [:dysp]
        @test p.table[1] ≈ 0.4360 atol = 5e-5
        @test p.table[1] ≈ 0.4359706 atol = 1e-7
        @test sum(p.table) ≈ 1
        @test p ≈ brute_force_marginal(fg, [:dysp])
        @test d.order isa Vector{Symbol} && !(:dysp in d.order) && length(d.order) == 7
        @test d.treewidth == 2 && d.max_factor_size <= 8 && d.n_multiplications > 0
        @test sum(joint_factor(fg).table) ≈ 1
        @test scope(joint_factor(fg)) == variables(fg)
        # infer entry point
        q, _ = infer(fg, :dysp)
        @test q ≈ p
        @test infer(fg, [:dysp]; backend=VariableElimination(; order=MinDegree()))[1] ≈ p
        @test infer(fg, [:dysp]; backend=VariableElimination(ExactTreewidth()))[1] ≈ p
        @test variable_elimination(fg, :dysp)[1] ≈ p
        # xray, dysp joint in either order
        j1, _ = variable_elimination(fg, [:xray, :dysp])
        j2, _ = variable_elimination(fg, [:dysp, :xray])
        @test scope(j1) == [:xray, :dysp] && scope(j2) == [:dysp, :xray]
        @test j1 ≈ j2 && j1 ≈ brute_force_marginal(fg, [:xray, :dysp])
        # published posteriors (bnlearn / Netica): P(lung = yes | smoke = yes) = 0.1
        @test variable_elimination(fg, [:lung]; evidence=Dict(:smoke => :yes))[1].table[1] ≈
              0.1
        # P(dysp | asia = yes, xray = yes) vs brute force
        ev = Dict(:asia => :yes, :xray => :yes)
        pe, de = variable_elimination(fg, [:dysp]; evidence=ev)
        @test pe ≈ brute_force_marginal(fg, [:dysp]; evidence=ev)
        @test !(:asia in de.order) && !(:xray in de.order)
        # empty query gives P(evidence)
        pev, _ = variable_elimination(fg, Symbol[]; evidence=ev)
        @test scope(pev) == Symbol[]
        @test pev.table[] ≈ brute_force_marginal(fg, Symbol[]; evidence=ev).table[]
        @test 0 < pev.table[] < 1
        @test variable_elimination(fg, Symbol[])[1].table[] ≈ 1
        # every strategy agrees
        for s in (MinFill(), MinDegree(), ExactTreewidth(), AMDOrder(),
                  UserOrder([:asia, :tub, :smoke, :lung, :bronc, :either, :xray]))
            @test variable_elimination(fg, [:dysp]; evidence=ev, order=s)[1] ≈ pe
        end
        # a UserOrder listing evidence variables is filtered
        full = UserOrder([:asia, :tub, :smoke, :lung, :bronc, :either, :xray])
        @test variable_elimination(fg, [:dysp]; evidence=ev, order=full)[1] ≈ pe
    end

    @testset "errors" begin
        @test_throws ScopeError variable_elimination(fg, [:nope])
        @test_throws ScopeError variable_elimination(fg, [:dysp, :dysp])
        @test_throws ScopeError variable_elimination(fg, [:dysp];
                                                     evidence=Dict(:nope => :yes))
        @test_throws ScopeError variable_elimination(fg, [:dysp];
                                                     evidence=Dict(:dysp => :yes))
        bad_label = Dict(:asia => :maybe)
        @test_throws FiniteKernels.InvalidAxisError variable_elimination(fg, [:dysp];
                                                                         evidence=bad_label)
        @test_throws ScopeError brute_force_marginal(fg, [:nope])
    end

    @testset "habitat chain (SPEC section 45) with random kernels" begin
        rng = MersenneTwister(45)
        hk, horder, hparents = habitat_network(rng)
        hfg = factor_graph_from(hk, hparents)
        @test sum(joint_factor(hfg).table) ≈ 1
        for q in ([:Occupancy], [:HabitatQuality, :Occupancy], [:Climate, :Vegetation])
            @test variable_elimination(hfg, q)[1] ≈ brute_force_marginal(hfg, q)
        end
        ev = Dict(:GrazingPressure => :low, :Climate => :wet)
        for q in ([:Occupancy], [:SoilMoisture, :Vegetation])
            @test variable_elimination(hfg, q; evidence=ev)[1] ≈
                  brute_force_marginal(hfg, q; evidence=ev)
        end
        @test treewidth(hfg, ExactTreewidth()) == 2
    end

    @testset "random DAGs (property tests)" begin
        rng = MersenneTwister(2026)
        for case in 1:20
            rk, rorder, rparents = random_dag_network(rng)
            rfg = factor_graph_from(rk, rparents)
            @test sum(joint_factor(rfg).table) ≈ 1
            nq = rand(rng, 1:min(2, length(rorder) - 1))
            shuffled = shuffle(rng, rorder)
            q = shuffled[1:nq]
            rest = shuffled[(nq + 1):end]
            ne = rand(rng, 0:min(2, length(rest)))
            ev = Dict{Symbol,Symbol}(v => rand(rng, rfg.axes[v].labels) for v in rest[1:ne])
            oracle = brute_force_marginal(rfg, q; evidence=ev)
            for s in (MinFill(), MinDegree(), ExactTreewidth(), AMDOrder())
                p, d = variable_elimination(rfg, q; evidence=ev, order=s)
                @test scope(p) == q
                @test p ≈ oracle
                @test d.treewidth >= 0
            end
        end
    end
end
