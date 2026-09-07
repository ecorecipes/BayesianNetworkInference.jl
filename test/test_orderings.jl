@testset "factor graphs and orderings" begin
    kernels, order, parents = asia_network()
    fg = factor_graph_from(kernels, parents)

    @testset "factor graph" begin
        @test length(fg) == 8
        @test variables(fg) == [:asia, :tub, :smoke, :lung, :bronc, :either, :xray, :dysp]
        @test fg.provenance == order
        @test fg.axes[:dysp] == FiniteAxis(:dysp, [:yes, :no])
        @test sprint(show, fg) == "FactorGraph{Float64} with 8 factors over 8 variables"
        g, vars, index = interaction_graph(fg)
        @test Graphs.nv(g) == 8 && vars == variables(fg)
        @test all(index[v] == i for (i, v) in enumerate(vars))
        # moral edges: lung-tub (co-parents of either) and bronc-either (of dysp)
        @test Graphs.has_edge(g, index[:lung], index[:tub])
        @test Graphs.has_edge(g, index[:bronc], index[:either])
        @test !Graphs.has_edge(g, index[:asia], index[:smoke])
        @test Graphs.ne(g) == 10
        a = FiniteAxis(:A, [:x, :y])
        a2 = FiniteAxis(:A, [:u, :v])
        @test_throws ShapeError FactorGraph([Factor(a, [1.0, 2.0]), Factor(a2, [1.0, 2.0])])
        @test_throws ShapeError FactorGraph([Factor(a, [1.0, 2.0])]; provenance=[1, 2])
        mixed = FactorGraph([Factor(a, [1, 2]), Factor(a, [0.5, 0.5])])
        @test mixed isa FactorGraph{Float64}
        empty = FactorGraph(Factor{Float64}[])
        @test variables(empty) == Symbol[] && Graphs.nv(interaction_graph(empty)[1]) == 0
    end

    strategies = [MinFill(), MinDegree(), ExactTreewidth(), AMDOrder()]

    @testset "each strategy returns a permutation of the non-kept variables" begin
        allvars = variables(fg)
        for s in strategies
            o = elimination_order(fg, s)
            @test sort(o) == sort(allvars)
            for keep in ([:dysp], [:asia, :xray], [:lung, :bronc, :smoke])
                ok = elimination_order(fg, s; keep=keep)
                @test sort(ok) == sort(setdiff(allvars, keep))
                @test isempty(intersect(ok, keep))
            end
        end
        @test elimination_order(fg) == elimination_order(fg, MinFill())
        @test elimination_order(FactorGraph(Factor{Float64}[]), MinFill()) == Symbol[]
        @test_throws ScopeError elimination_order(fg, MinFill(); keep=[:nope])
        @test_throws ScopeError elimination_order(fg, MinFill(); keep=[:dysp, :dysp])
    end

    @testset "CompositeRotations puts kept vertices last" begin
        g, vars, index = interaction_graph(fg)
        for s in strategies, keep in ([:dysp], [:asia, :xray], [:lung, :bronc, :smoke])
            alg = CliqueTrees.CompositeRotations([index[v] for v in keep],
                                                 BayesianNetworkInference._algorithm(s))
            o, _ = CliqueTrees.permutation(g; alg=alg)
            @test Set(vars[o[(end - length(keep) + 1):end]]) == Set(keep)
        end
    end

    @testset "UserOrder" begin
        o = [:asia, :tub, :smoke, :lung, :bronc, :either, :xray]
        @test elimination_order(fg, UserOrder(o); keep=[:dysp]) == o
        @test elimination_order(fg, UserOrder(vcat(o, :dysp))) == vcat(o, :dysp)
        @test_throws ScopeError UserOrder([:asia, :asia])
        @test_throws ScopeError elimination_order(fg, UserOrder(o))              # omits dysp
        @test_throws ScopeError elimination_order(fg, UserOrder(vcat(o, :dysp));
                                                  keep=[:dysp])
        @test_throws ScopeError elimination_order(fg, UserOrder(vcat(o, :zzz));
                                                  keep=[:dysp])
    end

    @testset "treewidth" begin
        # chain of five binary variables
        ax(v) = FiniteAxis(v, [:a, :b])
        chain = FactorGraph([Factor(ax(:V1), [0.5, 0.5]);
                             [Factor([ax(Symbol("V", i - 1)), ax(Symbol("V", i))],
                                     fill(0.5, 2, 2))
                              for i in 2:5]])
        for s in strategies
            @test treewidth(chain, s) == 1
        end
        @test treewidth(chain) == 1
        @test treewidth(chain, UserOrder([:V1, :V2, :V3, :V4, :V5])) == 1
        @test treewidth(chain, UserOrder([:V3, :V1, :V5, :V2, :V4])) == 2
        # 3x3 grid of pairwise factors: treewidth 3
        gv(i, j) = Symbol("G", i, j)
        grid = Factor{Float64}[]
        for i in 1:3, j in 1:3
            i < 3 && push!(grid, Factor([ax(gv(i, j)), ax(gv(i + 1, j))], fill(0.5, 2, 2)))
            j < 3 && push!(grid, Factor([ax(gv(i, j)), ax(gv(i, j + 1))], fill(0.5, 2, 2)))
        end
        gfg = FactorGraph(grid)
        tw = treewidth(gfg, ExactTreewidth())
        @test tw == 3
        for s in strategies
            @test treewidth(gfg, s) >= tw
        end
        @test treewidth(fg, ExactTreewidth()) == 2
        @test treewidth(fg, MinFill()) == 2
        @test treewidth(FactorGraph(Factor{Float64}[])) == -1
        # isolated variables and several components (BT alone fails on these)
        iso = FactorGraph([Factor(ax(:I1), [0.5, 0.5]), Factor(ax(:I2), [0.5, 0.5]),
                           Factor([ax(:V1), ax(:V2)], fill(0.5, 2, 2)),
                           Factor([ax(:V2), ax(:V3)], fill(0.5, 2, 2)),
                           Factor([ax(:W1), ax(:W2)], fill(0.5, 2, 2))])
        for s in strategies
            @test treewidth(iso, s) == 1
            o = elimination_order(iso, s; keep=[:V2, :I1])
            @test sort(o) == [:I2, :V1, :V3, :W1, :W2]
        end
        @test treewidth(FactorGraph([Factor(ax(:I1), [0.5, 0.5])]), ExactTreewidth()) == 0
        @test treewidth(FactorGraph([Factor(ax(:I1), [0.5, 0.5])]), MinFill()) == 0
    end
end
