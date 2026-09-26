# Sum-product belief propagation: exact on trees (against VE), loopy on asia
# with convergence diagnostics.

@testset "belief propagation" begin
    kernels, order, parents = asia_network()
    fg = factor_graph_from(kernels, parents)

    @testset "backend construction" begin
        bp = BeliefPropagation()
        @test bp.damping == 0 && bp.tol == 1e-8 && bp.maxiter == 200 &&
              bp.schedule == :flooding
        @test BeliefPropagation(; damping=0.5, schedule=:sequential).schedule == :sequential
        @test_throws ArgumentError BeliefPropagation(; damping=1.0)
        @test_throws ArgumentError BeliefPropagation(; damping=-0.1)
        @test_throws ArgumentError BeliefPropagation(; tol=0.0)
        @test_throws ArgumentError BeliefPropagation(; maxiter=0)
        @test_throws ArgumentError BeliefPropagation(; schedule=:random)
    end

    @testset "is_tree" begin
        ax(v) = FiniteAxis(v, [:a, :b])
        chain = FactorGraph([Factor([ax(Symbol(:V, i)), ax(Symbol(:V, i + 1))], rand(2, 2))
                             for i in 1:5])
        @test is_tree(chain)
        @test !is_tree(fg)                                   # asia has loops
        rng = MersenneTwister(45)
        hk, horder, hparents = habitat_network(rng)
        hfg = factor_graph_from(hk, hparents)
        @test is_tree(hfg)                                   # a polytree
        loop = FactorGraph([Factor([ax(:A), ax(:B)], rand(2, 2)),
                            Factor([ax(:B), ax(:C)], rand(2, 2)),
                            Factor([ax(:C), ax(:A)], rand(2, 2))])
        @test !is_tree(loop)
        @test is_tree(FactorGraph(Factor{Float64}[]))
    end

    @testset "exact on chains and polytrees" begin
        ax(v) = FiniteAxis(v, [:a, :b, :c])
        rng = MersenneTwister(7)
        chain = FactorGraph(vcat([Factor(ax(:V1), rand(rng, 3))],
                                 [Factor([ax(Symbol(:V, i)), ax(Symbol(:V, i + 1))],
                                         rand(rng, 3, 3))
                                  for i in 1:7]))
        for schedule in (:flooding, :sequential), damping in (0.0, 0.3)
            # damped messages approach the fixed point geometrically, so the
            # marginal error is of the order of the tolerance: tighten it
            bp = BeliefPropagation(; schedule, damping, tol=(damping == 0 ? 1e-8 : 1e-13))
            ms, d = belief_propagation(chain, bp)
            @test d.tree && d.converged && d.max_residual < bp.tol
            damping == 0 && schedule == :flooding && @test d.max_residual == 0
            for v in variables(chain)
                @test isapprox(ms[v], variable_elimination(chain, [v])[1]; atol=1e-10)
            end
        end
        ev = Dict(:V3 => :b, :V8 => :a)
        ms, d = belief_propagation(chain; evidence=ev)
        @test d.tree && d.converged
        for v in variables(chain)
            exact = haskey(ev, v) ? nothing :
                    variable_elimination(chain, [v]; evidence=ev)[1]
            exact === nothing || @test isapprox(ms[v], exact; atol=1e-10)
        end
        @test ms[:V3].table == [0.0, 1.0, 0.0]
        # polytree: the habitat chain, through infer and all_marginals
        hk, horder, hparents = habitat_network(MersenneTwister(45))
        hfg = factor_graph_from(hk, hparents)
        p, d = infer(hfg, :Occupancy; backend=BeliefPropagation())
        @test d isa BPDiagnostics && d.tree && d.converged
        @test isapprox(p, variable_elimination(hfg, [:Occupancy])[1]; atol=1e-10)
        hev = Dict(:GrazingPressure => :low, :Climate => :wet)
        bpm = all_marginals(hfg; evidence=hev, backend=BeliefPropagation())
        vem = all_marginals(hfg; evidence=hev, backend=VariableElimination())
        for v in variables(hfg)
            @test isapprox(bpm[v], vem[v]; atol=1e-10)
        end
        # random polytrees: chains built from random DAGs restricted to one parent
        rng = MersenneTwister(99)
        for case in 1:10
            n = rand(rng, 3:7)
            order_ = [Symbol("T", i) for i in 1:n]
            states = Dict(v => [Symbol("s", j) for j in 1:rand(rng, 2:3)] for v in order_)
            ps = Dict{Symbol,Vector{Symbol}}(v => (i == 1 ? Symbol[] :
                                                   [order_[rand(rng, 1:(i - 1))]])
                                             for (i, v) in enumerate(order_))
            tk, to, tp = random_network(rng, order_, ps, states)
            tfg = factor_graph_from(tk, tp)
            @test is_tree(tfg)
            ev = Dict(order_[end] => rand(rng, states[order_[end]]))
            ms, d = belief_propagation(tfg; evidence=ev)
            @test d.tree && d.converged
            for v in order_[1:(end - 1)]
                @test isapprox(ms[v], variable_elimination(tfg, [v]; evidence=ev)[1];
                               atol=1e-10)
            end
        end
    end

    @testset "loopy on asia" begin
        ms, d = belief_propagation(fg)
        @test !d.tree && d.converged && d.iterations < 200
        ve = all_marginals(fg; backend=VariableElimination())
        for v in variables(fg)
            @test isapprox(ms[v], ve[v]; atol=5e-3)
        end
        @test isapprox(ms[:dysp].table[1], 0.4359706; atol=5e-3)
        ev = Dict(:asia => :yes, :xray => :yes)
        p, de = infer(fg, :dysp; evidence=ev, backend=BeliefPropagation(; damping=0.2))
        @test de.converged
        @test isapprox(p, variable_elimination(fg, [:dysp]; evidence=ev)[1]; atol=5e-2)
        seq, ds = belief_propagation(fg, BeliefPropagation(; schedule=:sequential))
        @test ds.converged && isapprox(seq[:dysp], ms[:dysp]; atol=1e-6)
        # evidence on `either` cuts every loop: exact again
        cut = Dict(:either => :no)
        msc, dc = belief_propagation(fg; evidence=cut)
        @test dc.tree && dc.converged
        for v in setdiff(variables(fg), keys(cut))
            @test isapprox(msc[v], variable_elimination(fg, [v]; evidence=cut)[1];
                           atol=1e-10)
        end
        # an iteration cap is reported, not raised
        _, dcap = belief_propagation(fg, BeliefPropagation(; maxiter=1))
        @test dcap.iterations == 1 && !dcap.converged && dcap.max_residual > 1e-8
    end

    # Impossible evidence must be reported identically by the three backends
    # (review item 6a): belief propagation used to drop the empty-scope
    # conditioned factor and return a normalised belief with converged = true.
    @testset "impossible evidence agrees across backends" begin
        a = FiniteAxis(:A, [:a1, :a2])
        b = FiniteAxis(:B, [:b1, :b2, :b3])
        zero_prior = FactorGraph([Factor(a, [0.0, 1.0]),
                                  Factor([a, b], [1/3 1/3 1/3; 0.5 0.25 0.25])])
        ev = Dict(:A => :a1)
        # the empty-scope route: conditioning the prior leaves the scalar 0
        @test_throws ImpossibleEvidenceError belief_propagation(zero_prior; evidence=ev)
        for backend in (VariableElimination(), JunctionTree(), BeliefPropagation())
            @test_throws ImpossibleEvidenceError infer(zero_prior, :B; evidence=ev,
                                                       backend=backend)
            @test_throws ImpossibleEvidenceError all_marginals(zero_prior; evidence=ev,
                                                               backend=backend)
        end
        # the non-empty-scope route: `either` is deterministic in asia, so
        # lung = yes with either = no has probability zero and the conditioned
        # factor keeps the scope (tub,) with an all-zero table
        det = Dict(:lung => :yes, :either => :no)
        @test brute_force_marginal(fg, Symbol[]; evidence=det).table[] == 0
        for backend in (VariableElimination(), JunctionTree(), BeliefPropagation())
            @test_throws ImpossibleEvidenceError infer(fg, :dysp; evidence=det,
                                                       backend=backend)
        end
        # possible evidence on the same variables still works everywhere
        ok = Dict(:lung => :yes, :either => :yes)
        pv = infer(fg, :dysp; evidence=ok)[1]
        for backend in (JunctionTree(), BeliefPropagation())
            @test isapprox(infer(fg, :dysp; evidence=ok, backend=backend)[1], pv; atol=5e-3)
        end
    end

    @testset "errors" begin
        @test_throws ScopeError infer(fg, [:dysp, :xray]; backend=BeliefPropagation())
        @test_throws ScopeError infer(fg, Symbol[]; backend=BeliefPropagation())
        @test_throws ScopeError infer(fg, [:nope]; backend=BeliefPropagation())
        @test_throws ScopeError infer(fg, :dysp; evidence=Dict(:dysp => :yes),
                                      backend=BeliefPropagation())
        @test_throws ScopeError belief_propagation(fg; evidence=Dict(:nope => :yes))
    end

    @testset "model level" begin
        m = reference_habitat_model()
        @test is_tree(compile(m))
        ms = all_marginals(m; backend=BeliefPropagation())
        @test isapprox(ms[:Occupancy], Factor(marginal(m, :Occupancy), :Occupancy);
                       atol=1e-10)
        p, d = infer(m, :Occupancy; evidence=Dict(:Vegetation => :dense),
                     backend=BeliefPropagation())
        @test d.tree
        @test p.table ≈ [0.3325, 0.6675] atol = 1e-4
    end
end
