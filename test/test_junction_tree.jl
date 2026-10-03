# Junction-tree inference: the clique tree from CliqueTrees, Shafer-Shenoy
# calibration, single-clique queries and `all_marginals`, all checked
# against variable elimination and the brute-force oracle.

# Every non-evidence variable's JT marginal equals the VE marginal.
function check_all_marginals(fg, ev; backend=JunctionTree(), atol=1e-12)
    jt_m = all_marginals(fg; evidence=ev, backend)
    ve_m = all_marginals(fg; evidence=ev, backend=VariableElimination())
    @test Set(keys(jt_m)) == Set(keys(ve_m)) == Set(variables(fg))
    for v in variables(fg)
        @test scope(jt_m[v]) == [v]
        @test isapprox(jt_m[v], ve_m[v]; atol)
        if haskey(ev, v)
            @test jt_m[v].table[FiniteKernels.label_index(fg.axes[v], ev[v])] == 1
            @test sum(jt_m[v].table) == 1
        end
    end
    return jt_m
end

@testset "junction tree" begin
    kernels, order, parents = asia_network()
    fg = factor_graph_from(kernels, parents)

    @testset "structure on asia" begin
        jt = build_junction_tree(fg)
        @test jt isa CompiledJunctionTree
        @test length(jt) == 6 && jt.treewidth == 2
        @test build_junction_tree(fg) === jt                       # cached
        @test build_junction_tree(fg; order=MinDegree()) !== jt
        # the cache is keyed on the identity of `fg.factors`, not on their
        # content, so an independently built but equal graph gets its own
        # tree (and no full-table comparison is paid on a cache hit)
        k2, _, p2 = asia_network()
        fg2 = factor_graph_from(k2, p2)
        @test fg2.factors == fg.factors && fg2.factors !== fg.factors
        jt2 = build_junction_tree(fg2)
        @test jt2 !== jt && build_junction_tree(fg2) === jt2
        @test sort(sort.(jt2.cliques)) == sort(sort.(jt.cliques))
        allvars = Set(variables(fg))
        @test union(Set.(jt.cliques)...) == allvars
        @test sort(jt.order) == sort(variables(fg))
        # a tree: one root, every non-root's separator lies in it and its parent
        @test length(jt.roots) == 1 && jt.parent[jt.roots[1]] == 0
        for i in 1:length(jt)
            p = jt.parent[i]
            if p == 0
                @test isempty(jt.separators[i]) && i in jt.roots
            else
                @test i in jt.children[p]
                @test Set(jt.separators[i]) ==
                      intersect(Set(jt.cliques[i]), Set(jt.cliques[p]))
                @test !isempty(jt.separators[i])
            end
            @test length(jt.cliques[i]) - 1 <= jt.treewidth
        end
        @test sort(jt.postorder) == 1:length(jt)
        @test all(findfirst(==(i), jt.postorder) > findfirst(==(c), jt.postorder)
                  for i in 1:length(jt) for c in jt.children[i])
        # running intersection: the cliques containing a variable are connected
        for v in variables(fg)
            with_v = [i for i in 1:length(jt) if v in jt.cliques[i]]
            crossings = count(i -> jt.parent[i] != 0 && jt.parent[i] in with_v, with_v)
            @test crossings == length(with_v) - 1
            @test v in jt.cliques[jt.home[v]]
        end
        # every factor is assigned to a clique containing its scope
        for (f, c) in zip(fg.factors, jt.assignment)
            @test issubset(scope(f), jt.cliques[c])
        end
        # The maximal cliques of the MinFill triangulation of asia. The
        # triangulation is deterministic, so this is an equality, not a
        # disjunction with the width: asia has treewidth 2 and the MinFill
        # order produces these six cliques and these five separators.
        @test treewidth(fg, MinFill()) == treewidth(fg, ExactTreewidth()) == 2
        @test sort(sort.(jt.cliques)) ==
              [[:asia, :tub], [:bronc, :dysp, :either], [:bronc, :either, :smoke],
               [:either, :lung, :smoke], [:either, :lung, :tub], [:either, :xray]]
        @test sort(sort.(jt.separators)) ==
              [Symbol[], [:bronc, :either], [:either], [:either, :lung],
               [:either, :smoke], [:tub]]
        @test occursin("6 cliques (treewidth 2)", sprint(show, jt))
        @test occursin("-->", sprint(show, MIME("text/plain"), jt))
        for s in (MinDegree(), AMDOrder(), ExactTreewidth(),
                  UserOrder([:asia, :tub, :smoke, :lung, :bronc, :either, :xray, :dysp]))
            j = build_junction_tree(fg; order=s)
            @test union(Set.(j.cliques)...) == allvars
            @test j.treewidth >= 2
            s isa ExactTreewidth && @test j.treewidth == 2
        end
    end

    @testset "calibration and queries on asia" begin
        cal = calibrate(fg)
        @test cal isa CalibratedJunctionTree{Float64}
        @test cal.evidence_probability ≈ 1
        @test cal.n_messages == 10
        @test occursin("P(evidence)", sprint(show, cal))
        for (i, b) in enumerate(cal.beliefs)
            @test Set(scope(b)) == Set(cal.tree.cliques[i])
            @test sum(b.table) ≈ 1
            @test b ≈ brute_force_marginal(fg, scope(b))
        end
        @test all(sum(b.table) ≈ 1 for b in clique_beliefs(fg))
        p, d = infer(fg, :dysp; backend=JunctionTree())
        @test p.table[1] ≈ 0.4359706 atol = 1e-7
        @test p ≈ variable_elimination(fg, [:dysp])[1]
        @test d isa JunctionTreeDiagnostics
        @test d.n_cliques == 6 && d.treewidth == 2 && d.max_clique_size == 8 &&
              d.n_messages == 10 && :dysp in cal.tree.cliques[d.clique]
        # a joint query inside one clique, in both orders
        j = infer(fg, [:bronc, :either]; backend=JunctionTree())[1]
        @test j ≈ brute_force_marginal(fg, [:bronc, :either])
        @test scope(infer(fg, [:either, :bronc]; backend=JunctionTree())[1]) ==
              [:either, :bronc]
        # evidence, including a deterministic node and P(evidence)
        ev = Dict(:asia => :yes, :xray => :yes)
        pe, de = infer(fg, [:dysp]; evidence=ev, backend=JunctionTree())
        @test pe ≈ brute_force_marginal(fg, [:dysp]; evidence=ev)
        @test de.max_clique_size == 8      # three binary variables remain in the largest cliques
        pev = infer(fg, Symbol[]; evidence=ev, backend=JunctionTree())[1]
        @test scope(pev) == Symbol[] &&
              pev.table[] ≈ brute_force_marginal(fg, Symbol[]; evidence=ev).table[]
        @test infer(fg, :lung; evidence=Dict(:smoke => :yes), backend=JunctionTree())[1].table[1] ≈
              0.1
        ev2 = Dict(:either => :yes)
        @test infer(fg, :tub; evidence=ev2, backend=JunctionTree())[1] ≈
              brute_force_marginal(fg, [:tub]; evidence=ev2)
        for s in (MinDegree(), AMDOrder(), ExactTreewidth())
            @test infer(fg, :dysp; evidence=ev, backend=JunctionTree(; order=s))[1] ≈ pe
        end
        # A query that spans cliques falls back to VE with a warning. The
        # warning is issued on every such query (no `maxlog`), so it is
        # testable, and the diagnostics type does not change: `fallback`
        # records what happened.
        q = [:asia, :smoke]
        pq, dq = @test_logs (:warn, r"does not lie in one clique") infer(fg, q;
                                                                         backend=JunctionTree())
        @test pq ≈ brute_force_marginal(fg, q)
        @test dq isa JunctionTreeDiagnostics && dq.fallback
        @test dq.clique == 0 && dq.n_messages == 0 && dq.n_cliques == 6
        @test dq.treewidth == 2 && dq.max_clique_size > 0
        # a second fallback query warns again
        @test_logs (:warn, r"does not lie in one clique") infer(fg, q;
                                                                backend=JunctionTree())
        # a single-clique query neither warns nor reports a fallback
        _, dq1 = @test_logs infer(fg, [:bronc, :dysp]; backend=JunctionTree())
        @test !dq1.fallback && dq1.clique != 0
        # errors
        @test_throws ScopeError infer(fg, [:nope]; backend=JunctionTree())
        @test_throws ScopeError infer(fg, [:dysp]; evidence=Dict(:dysp => :yes),
                                      backend=JunctionTree())
        @test_throws ScopeError calibrate(fg; evidence=Dict(:nope => :yes))
        @test_throws FiniteKernels.InvalidAxisError calibrate(fg;
                                                              evidence=Dict(:asia => :maybe))
        other = FactorGraph(fg.factors[1:2])
        @test_throws ShapeError calibrate(other, build_junction_tree(fg))
        # impossible evidence cannot be normalised
        impossible = Dict(:lung => :yes, :either => :no)
        @test_throws ImpossibleEvidenceError infer(fg, :dysp; evidence=impossible,
                                                   backend=JunctionTree())
        @test infer(fg, Symbol[]; evidence=impossible, backend=JunctionTree())[1].table[] ==
              0
    end

    @testset "all_marginals on asia" begin
        m = check_all_marginals(fg, Dict{Symbol,Symbol}())
        @test m[:dysp].table[1] ≈ 0.4359706 atol = 1e-7
        check_all_marginals(fg, Dict(:asia => :yes, :xray => :yes))
        check_all_marginals(fg, Dict(:either => :yes, :smoke => :no, :dysp => :yes))
        bp = all_marginals(fg; backend=VariableElimination(; order=MinDegree()))
        @test bp[:dysp] ≈ m[:dysp]
        @test_throws ScopeError all_marginals(fg; evidence=Dict(:nope => :yes))
    end

    @testset "habitat chain (SPEC section 45)" begin
        rng = MersenneTwister(45)
        hk, horder, hparents = habitat_network(rng)
        hfg = factor_graph_from(hk, hparents)
        jt = build_junction_tree(hfg)
        @test jt.treewidth == 2
        for q in ([:Occupancy], [:HabitatQuality, :Occupancy],
                  [:Climate, :Irrigation, :SoilMoisture])
            @test infer(hfg, q; backend=JunctionTree())[1] ≈ brute_force_marginal(hfg, q)
        end
        ev = Dict(:GrazingPressure => :low, :Climate => :wet)
        @test infer(hfg, [:SoilMoisture, :Vegetation]; evidence=ev, backend=JunctionTree())[1] ≈
              brute_force_marginal(hfg, [:SoilMoisture, :Vegetation]; evidence=ev)
        check_all_marginals(hfg, ev)
        for (i, b) in enumerate(clique_beliefs(hfg; evidence=ev))
            @test b ≈ brute_force_marginal(hfg, scope(b); evidence=ev)
        end
    end

    @testset "random DAGs (property tests)" begin
        rng = MersenneTwister(1911)
        for case in 1:20
            rk, rorder, rparents = random_dag_network(rng)
            rfg = factor_graph_from(rk, rparents)
            jt = build_junction_tree(rfg)
            @test jt.treewidth == treewidth(rfg, MinFill())
            @test all(issubset(scope(f), jt.cliques[c])
                      for (f, c) in zip(rfg.factors, jt.assignment))
            shuffled = shuffle(rng, rorder)
            ne = rand(rng, 0:min(2, length(rorder) - 1))
            ev = Dict{Symbol,Symbol}(v => rand(rng, rfg.axes[v].labels)
                                     for v in shuffled[1:ne])
            check_all_marginals(rfg, ev)
            rest = shuffled[(ne + 1):end]
            q = rest[1:min(2, length(rest))]
            oracle = brute_force_marginal(rfg, q; evidence=ev)
            for s in (MinFill(), MinDegree(), ExactTreewidth())
                # a query spanning cliques falls back to VE and warns; the
                # warning is checked above, here it would only be noise
                p, d = with_logger(NullLogger()) do
                    return infer(rfg, q; evidence=ev, backend=JunctionTree(; order=s))
                end
                @test scope(p) == q && p ≈ oracle
                @test d isa JunctionTreeDiagnostics
            end
            @test infer(rfg, Symbol[]; evidence=ev, backend=JunctionTree())[1].table[] ≈
                  brute_force_marginal(rfg, Symbol[]; evidence=ev).table[]
        end
    end

    @testset "disconnected graphs and components created by evidence" begin
        ax(v) = FiniteAxis(v, [:a, :b])
        # two independent chains: a forest with two roots
        chain(u, v, w) = [Factor(ax(u), [0.3, 0.7]),
                          Factor([ax(u), ax(v)], [0.9 0.1; 0.2 0.8]),
                          Factor([ax(v), ax(w)], [0.6 0.4; 0.5 0.5])]
        forest = FactorGraph(vcat(chain(:A, :B, :C), chain(:D, :E, :F)))
        jt = build_junction_tree(forest)
        @test length(jt.roots) == 2 && jt.treewidth == 1
        check_all_marginals(forest, Dict{Symbol,Symbol}())
        check_all_marginals(forest, Dict(:B => :a, :F => :b))
        pev = infer(forest, Symbol[]; evidence=Dict(:B => :a, :F => :b),
                    backend=JunctionTree())[1]
        @test pev.table[] ≈
              brute_force_marginal(forest, Symbol[]; evidence=Dict(:B => :a, :F => :b)).table[]
        # evidence on the middle of asia cuts the tree: still exact
        check_all_marginals(fg, Dict(:either => :no))
        check_all_marginals(fg, Dict(:smoke => :yes, :either => :no))
        # a graph with only scalar factors and one with a single variable
        scalar = FactorGraph([Factor(Symbol[], FiniteAxis[], fill(0.5))])
        @test isempty(build_junction_tree(scalar).cliques) &&
              build_junction_tree(scalar).treewidth == -1
        @test infer(scalar, Symbol[]; backend=JunctionTree())[1].table[] ≈ 0.5
        @test isempty(all_marginals(scalar))
        single = FactorGraph([Factor(ax(:Z), [0.25, 0.75])])
        @test build_junction_tree(single).cliques == [[:Z]]
        @test all_marginals(single)[:Z].table ≈ [0.25, 0.75]
    end

    @testset "benchmark network: all marginals versus VE" begin
        rng = MersenneTwister(30)
        bk, border, bparents = benchmark_network(rng)
        bfg = factor_graph_from(bk, bparents)
        @test length(bfg) == 30
        jt = build_junction_tree(bfg)
        @test jt.treewidth <= 8
        check_all_marginals(bfg, Dict{Symbol,Symbol}(); atol=1e-10)
        ev = Dict(:B5 => :s1, :B17 => :s2, :B30 => :s1)
        check_all_marginals(bfg, ev; atol=1e-10)
        @test infer(bfg, Symbol[]; evidence=ev, backend=JunctionTree())[1].table[] ≈
              variable_elimination(bfg, Symbol[]; evidence=ev)[1].table[]
    end

    @testset "model level" begin
        m = reference_habitat_model()
        ms = all_marginals(m)
        for x in [:Climate, :Vegetation, :Occupancy]
            @test ms[x] ≈ Factor(marginal(m, x), x)
            @test infer(m, x; backend=JunctionTree())[1] ≈ ms[x]
        end
        ev = Dict(:Vegetation => :dense)
        mo = observe(m, :Vegetation => :dense)
        ms_ev = all_marginals(m; evidence=ev)
        @test ms_ev[:Occupancy].table ≈ [0.3325, 0.6675] atol = 1e-4
        @test all_marginals(mo)[:Occupancy] ≈ ms_ev[:Occupancy]
        @test ms_ev[:Vegetation].table == [0.0, 0.0, 1.0]
        @test all_marginals(mo; evidence=[:Climate => :dry])[:Occupancy] ≈
              infer(mo, :Occupancy; evidence=Dict(:Climate => :dry))[1]
        @test all_marginals(m; backend=VariableElimination())[:Occupancy] ≈ ms[:Occupancy]
        a = read_bayesnet(fixture_path("bif/asia.bif"))
        @test all_marginals(a)[:dysp].table[1] ≈ 0.4359706 atol = 1e-7
        @test infer(a, :dysp; backend=JunctionTree())[1] ≈ infer(a, :dysp)[1]
    end

    @testset "water benchmark (ECOLOGICAL_BN_SLOW)" begin
        gz = normpath(joinpath(@__DIR__, "..", "..", "EcologicalBayesianNetworks.jl",
                               "models",
                               "water", "water.net.gz"))
        if get(ENV, "ECOLOGICAL_BN_SLOW", "false") == "true" && isfile(gz) &&
           Sys.which("gunzip") !== nothing
            net = tempname() * ".net"
            run(pipeline(`gunzip -c $gz`; stdout=net))
            w = read_bayesnet(net; renormalize=true)
            wfg = compile(w)
            @test length(wfg) == 32
            jt = build_junction_tree(wfg)
            @info "water: $(length(jt)) cliques, treewidth $(jt.treewidth)"
            check_all_marginals(wfg, Dict{Symbol,Symbol}(); atol=1e-9)
            rm(net; force=true)
        else
            @test_skip "water benchmark skipped (set ECOLOGICAL_BN_SLOW=true with the sibling zoo checked out)"
        end
    end
end

@testset "clique_beliefs survives a fully instantiated clique" begin
    # Evidence that instantiates every variable of a maximal clique leaves an empty-scope
    # factor, whose table is a 0-d array. `f.table ./ s` returns a *scalar* there, so
    # `normalize` raised a `MethodError` from inside `clique_beliefs` on ordinary evidence.
    # [:either, :xray] is a maximal MinFill clique of asia. The log-domain path always
    # divided in place and so was unaffected.
    kernels, _, parents = asia_network()
    fg = factor_graph_from(kernels, parents)
    beliefs = clique_beliefs(fg; evidence=Dict(:either => :yes, :xray => :yes))
    @test length(beliefs) == 6
    @test all(b -> isapprox(sum(b.table), 1.0; atol=1e-10), values(beliefs))

    empty_scope = marginalize(Factor(FiniteAxis(:a, [:x, :y]), [0.25, 0.75]), [:a])
    @test isempty(scope(empty_scope))
    @test normalize(empty_scope).table == fill(1.0)
    @test normalize(empty_scope).table isa AbstractArray{Float64,0}
end

# Review of 2026-10-01, finding 4: a query that no clique contains is answered by variable
# elimination. That fallback rejected every `UserOrder` (the tree's order lists the query
# variables, which variable elimination keeps), and its exact fallback was dropped from the
# diagnostics.
@testset "queries spanning cliques" begin
    a = FiniteAxis(:A, [:x, :y])
    b = FiniteAxis(:B, [:x, :y])
    c = FiniteAxis(:C, [:x, :y])
    fg = FactorGraph([Factor([a, b], [0.2 0.8; 0.6 0.4]),
                      Factor([b, c], [0.3 0.7; 0.9 0.1])])
    want = first(infer(fg, [:A, :C]))
    for order in (UserOrder([:A, :B, :C]), UserOrder([:C, :B, :A]), MinFill())
        p, d = with_logger(NullLogger()) do
            return infer(fg, [:A, :C]; backend=JunctionTree(; order))
        end
        @test d.fallback && !d.exact_fallback
        @test p ≈ want
    end
    # Every product is 1e-400, which binary64 rounds to zero: the fallback is exact, and the
    # diagnostics say so.
    tiny = FactorGraph([Factor([a, b], fill(1e-200, 2, 2)),
                        Factor([b, c], fill(1e-200, 2, 2))])
    q, dq = with_logger(NullLogger()) do
        return infer(tiny, [:A, :C]; backend=JunctionTree())
    end
    @test dq.fallback && dq.exact_fallback
    @test q.table == fill(0.25, 2, 2)
end

@testset "junction-tree validation (Lean Good and checkAssignment)" begin
    check = BayesianNetworkInference._check_junction_tree
    check_compiled(jt, fg) = check(jt.cliques, jt.separators, jt.parent, jt.children,
                                   jt.roots, jt.postorder, jt.assignment,
                                   [scope(f) for f in fg.factors], variables(fg))
    # The ScopeError a corrupted tree raises, or nothing.
    function tree_error(t)
        try
            check(t.cliques, t.separators, t.parent, t.children, t.roots, t.postorder,
                  t.assignment, t.scopes, t.vars)
        catch e
            return e
        end
        return nothing
    end
    rejects(t, fragment) = (e = tree_error(t);
                            e isa ScopeError && e.operation == :build_junction_tree &&
                                occursin(fragment, e.msg))

    # A hand-built forest: [:A, :B] under [:B, :C], and [:D] on its own; the last
    # factor has an empty scope and sits in the scalar constant (clique 0).
    valid = (cliques=[[:A, :B], [:B, :C], [:D]], separators=[[:B], Symbol[], Symbol[]],
             parent=[2, 0, 0], children=[Int[], [1], Int[]], roots=[2, 3],
             postorder=[1, 2, 3], assignment=[1, 2, 3, 0],
             scopes=[[:A, :B], [:B, :C], [:D], Symbol[]], vars=[:A, :B, :C, :D])
    @test tree_error(valid) === nothing
    @test tree_error(merge(valid, (assignment=[1, 2, 3, 2],))) === nothing
    @test tree_error((cliques=Vector{Symbol}[], separators=Vector{Symbol}[],
                      parent=Int[], children=Vector{Int}[], roots=Int[], postorder=Int[],
                      assignment=[0], scopes=[Symbol[]], vars=Symbol[])) === nothing

    @test rejects(merge(valid, (separators=[Symbol[], Symbol[], Symbol[]],)),
                  "separator of clique 1 is not its intersection")
    @test rejects(merge(valid, (separators=[[:A, :B], Symbol[], Symbol[]],)),
                  "separator of clique 1 is not its intersection")
    @test rejects(merge(valid, (separators=[[:B], Symbol[], [:D]],)),
                  "separator of clique 3")
    # :A is in clique 1 and in the separate root clique 3
    @test rejects(merge(valid, (cliques=[[:A, :B], [:B, :C], [:D, :A]],)),
                  "cliques containing a variable are not connected")
    @test rejects(merge(valid, (assignment=[2, 2, 3, 0],)),
                  "factor 1 is assigned to clique 2, which does not hold its scope")
    @test rejects(merge(valid, (assignment=[0, 2, 3, 0],)),
                  "no clique contains the scope of a factor")
    @test rejects(merge(valid, (assignment=[1, 2, 3, 4],)), "empty-scope factor 4")
    @test rejects(merge(valid, (assignment=[1, 2, 3],)), "exactly one clique per factor")
    @test rejects(merge(valid, (parent=[2, 1, 0], children=[[2], [1], Int[]], roots=[3])),
                  "parent links contain a cycle")
    @test rejects(merge(valid, (parent=[1, 0, 0],)), "which is not another clique")
    @test rejects(merge(valid, (children=[Int[], Int[], [1]],)),
                  "children of clique 3 disagree with the parent links")
    @test rejects(merge(valid, (children=[Int[], Int[], Int[]],)),
                  "clique 1 is missing from the children of its parent 2")
    @test rejects(merge(valid, (children=[Int[], [1, 1], Int[]],)),
                  "disagree with the parent links")
    @test rejects(merge(valid, (roots=[2],)), "clique 3 has no parent but is not listed")
    @test rejects(merge(valid, (roots=[1, 2, 3],)), "root 1 is not a distinct clique")
    @test rejects(merge(valid, (postorder=[2, 1, 3],)),
                  "postorder visits clique 2 before its child 1")
    @test rejects(merge(valid, (postorder=[1, 1, 3],)), "every clique exactly once")
    @test rejects(merge(valid, (vars=[:A, :B, :C, :D, :E],)),
                  "a variable of the interaction graph is in no clique")
    @test rejects(merge(valid, (cliques=[[:A, :B], [:B, :C], [:D, :D]],)),
                  "clique 3 repeats a variable")
    @test rejects(merge(valid, (cliques=[[:A, :B], [:B, :C], [:D, :E]],)),
                  "outside the interaction graph")
    @test rejects(merge(valid, (parent=[2, 0],)), "differ in length")
    e = tree_error(merge(valid, (vars=[:A, :B, :C, :D, :E],)))
    @test e.vars == [:E]

    # Every tree CliqueTrees builds on the fixtures passes (build_junction_tree has
    # already run the check; this calls it again directly).
    kernels, _, parents = asia_network()
    fg = factor_graph_from(kernels, parents)
    for s in (MinFill(), MinDegree(), AMDOrder(), ExactTreewidth(),
              UserOrder([:asia, :tub, :smoke, :lung, :bronc, :either, :xray, :dysp]))
        @test check_compiled(build_junction_tree(fg; order=s), fg) === nothing
    end
    hk, _, hparents = habitat_network(MersenneTwister(45))
    hfg = factor_graph_from(hk, hparents)
    @test check_compiled(build_junction_tree(hfg), hfg) === nothing
    rng = MersenneTwister(2027)
    for _ in 1:20
        rk, _, rparents = random_dag_network(rng)
        rfg = factor_graph_from(rk, rparents)
        for s in (MinFill(), MinDegree(), ExactTreewidth())
            @test check_compiled(build_junction_tree(rfg; order=s), rfg) === nothing
        end
    end
    bk, _, bparents = benchmark_network(MersenneTwister(30))
    bfg = factor_graph_from(bk, bparents)
    @test check_compiled(build_junction_tree(bfg), bfg) === nothing
    # a forest (two components and an isolated variable) with a scalar factor
    ax(v) = FiniteAxis(v, [:a, :b])
    forest = FactorGraph([Factor([ax(:A), ax(:B)], [0.9 0.1; 0.2 0.8]),
                          Factor([ax(:B), ax(:C)], [0.6 0.4; 0.5 0.5]),
                          Factor([ax(:D), ax(:E)], [0.3 0.7; 0.1 0.9]),
                          Factor(ax(:F), [0.25, 0.75]),
                          Factor(Symbol[], FiniteAxis[], fill(0.5))])
    fjt = build_junction_tree(forest)
    @test length(fjt.roots) == 3 && check_compiled(fjt, forest) === nothing
    # the empty factor graph and a graph of scalar factors only
    empty_fg = FactorGraph(Factor{Float64}[])
    @test isempty(build_junction_tree(empty_fg).cliques)
    @test check_compiled(build_junction_tree(empty_fg), empty_fg) === nothing
    scalar = FactorGraph([Factor(Symbol[], FiniteAxis[], fill(0.5))])
    @test build_junction_tree(scalar).assignment == [0]
    @test check_compiled(build_junction_tree(scalar), scalar) === nothing
end

# Review of 2026-10-02, finding 3: the product of an empty factor list was the Float64 unit,
# so calibrating a graph of another element type crashed ("Cannot convert Factor{Float64}
# to Factor{Float32}") whenever a clique got no factor -- as cliques 10, 19 and 20 of a 5 x 5
# pairwise grid do under MinFill. Int, Rational{Int} and BigFloat graphs failed the same way.
@testset "a clique without factors has the graph's element type" begin
    function grid(T)
        ax = Dict((i, j) => FiniteAxis(Symbol("V", i, "_", j), [:s0, :s1])
                  for i in 1:5, j in 1:5)
        fs = Factor{T}[]
        for i in 1:5, j in 1:5
            i < 5 && push!(fs, Factor([ax[(i, j)], ax[(i + 1, j)]], T[1 1; 1 2]))
            j < 5 && push!(fs, Factor([ax[(i, j)], ax[(i, j + 1)]], T[2 1; 1 1]))
        end
        return FactorGraph(fs)
    end
    reference = all_marginals(grid(Float64))
    jt = build_junction_tree(grid(Float64))
    @test any(c -> !(c in jt.assignment), 1:length(jt))
    for T in (Float32, Int, Rational{Int}, BigFloat)
        g = grid(T)
        ms = all_marginals(g)
        @test valtype(ms) == Factor{BayesianNetworkInference._division_type(T)}
        @test all(v -> isapprox(Float64.(ms[v].table), reference[v].table; rtol=1e-5),
                  keys(reference))
        @test length(clique_beliefs(g)) == length(jt)
        @test isapprox(Float64.(first(infer(g, :V3_3; backend=JunctionTree())).table),
                       reference[:V3_3].table; rtol=1e-5)
    end
    # The exact element types are exact: the junction tree on Rational{Int} against
    # variable elimination on Rational{BigInt}.
    exact = grid(Rational{BigInt})
    rational = all_marginals(grid(Rational{Int}))
    @test all(v -> rational[v].table == first(infer(exact, v)).table, [:V1_1, :V3_3, :V5_2])
end
