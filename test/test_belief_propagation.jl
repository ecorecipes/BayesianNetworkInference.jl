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

    # Review of 2026-10-01, finding 1: belief propagation used to replace any run with a zero
    # or subnormal message by an exact junction-tree calibration of the whole graph, which is
    # exponential in treewidth on exactly the loopy graphs belief propagation is for.
    @testset "no exact fallback" begin
        ax(v) = FiniteAxis(v, [:a, :b])
        # An n x n grid with soft couplings and two hard equalities in the first row: X_1_1 =
        # a and X_1_3 = b is impossible, and only message passing can see it.
        function grid(n)
            v(i, j) = Symbol("X_", i, "_", j)
            fs = Factor{Float64}[Factor(ax(v(i, j)), [0.5, 0.5]) for i in 1:n for j in 1:n]
            soft = [0.9 0.1; 0.1 0.9]
            hard = [1.0 0.0; 0.0 1.0]
            for i in 1:n, j in 1:n
                j < n && push!(fs,
                               Factor([ax(v(i, j)), ax(v(i, j + 1))],
                                      (i == 1 && j <= 2) ? hard : soft))
                i < n && push!(fs, Factor([ax(v(i, j)), ax(v(i + 1, j))], soft))
            end
            return FactorGraph(fs), Dict(v(1, 1) => :a, v(1, 3) => :b)
        end

        @testset "an exactly zero message raises at once" begin
            # The exact fallback took 1.4 GiB at 12 x 12 and could not finish from 20 x 20;
            # message passing needs a few MiB here.
            g20, ev = grid(20)
            call() =
                try
                    infer(g20, :X_2_2; evidence=ev, backend=BeliefPropagation())
                catch e
                    e
                end
            call()
            e = call()
            @test e isa ImpossibleEvidenceError && e.evidence == ev
            @test (@allocated call()) < 64 * 2^20
            @test_throws ImpossibleEvidenceError infer(g20, :X_2_2; evidence=ev,
                                                       backend=BeliefPropagation(;
                                                                                 schedule=:sequential))
            # Damped iterates keep a little of every earlier message, so they never reach the
            # exact zero; that is the documented limit of local support, not a fallback.
            _, damped = infer(g20, :X_2_2; evidence=ev,
                              backend=BeliefPropagation(; damping=0.5, maxiter=5))
            @test !damped.log_domain
            # Feasible evidence on the same grid is answered in binary64.
            _, d = infer(g20, :X_2_2; evidence=Dict(:X_1_1 => :a, :X_1_3 => :a),
                         backend=BeliefPropagation())
            @test d.converged && !d.log_domain
        end

        @testset "a tiny potential keeps belief propagation's answer" begin
            fs = [Factor(ax(:A), [0.6, 0.4]), Factor([ax(:A), ax(:B)], [0.7 0.3; 0.2 0.8]),
                  Factor([ax(:A), ax(:C)], [0.4 0.6; 0.9 0.1]),
                  Factor([ax(:B), ax(:D)], [0.8 0.2; 0.3 0.7]),
                  Factor([ax(:C), ax(:D)], [0.6 0.4; 0.1 0.9])]
            p, d = infer(FactorGraph(fs), :D; backend=BeliefPropagation())
            @test !d.tree && d.converged
            exact = first(infer(FactorGraph(fs), :D)).table
            @test maximum(abs.(p.table .- exact)) > 1e-3     # loopy: not the exact answer
            # Scaled by 1e-310 the entries are subnormal: the old code saw subnormal message
            # sums and answered with the exact junction tree instead.
            tiny = copy(fs)
            tiny[4] = Factor([ax(:B), ax(:D)], fs[4].table .* 1e-310)
            q, dq = infer(FactorGraph(tiny), :D; backend=BeliefPropagation())
            @test !dq.log_domain && dq.converged
            @test isapprox(q.table, p.table; atol=1e-12)  # subnormals keep ~44 bits
            # A power of two leaves every message bit for bit.
            scaled = copy(fs)
            scaled[4] = Factor([ax(:B), ax(:D)], fs[4].table .* 2.0^-1000)
            @test first(infer(FactorGraph(scaled), :D; backend=BeliefPropagation())).table ==
                  p.table
        end

        @testset "a rational graph reports impossible evidence" begin
            # It used to reach the exact fallback, which cannot take 1//3.
            a = FiniteAxis(:A, [:a1, :a2])
            b = FiniteAxis(:B, [:b1, :b2])
            c = FiniteAxis(:C, [:c1, :c2])
            eq = [1//1 0//1; 0//1 1//1]
            qfg = FactorGraph([Factor(a, [1 // 3, 2 // 3]), Factor([a, b], eq),
                               Factor([b, c], eq)])
            e = try
                infer(qfg, :B; evidence=Dict(:A => :a1, :C => :c2),
                      backend=BeliefPropagation())
            catch e
                e
            end
            @test e isa ImpossibleEvidenceError
            p, _ = infer(qfg, :B; evidence=Dict(:A => :a1), backend=BeliefPropagation())
            @test p.table == [1.0, 0.0]
        end

        @testset "underflowing messages are redone in the log domain" begin
            # Each rare state has probability 1e-200, and the factor allows only the
            # configuration with both: its message multiplies 1e-200 by 1e-200, which
            # binary64 rounds to zero. That zero is not exact, so it is not impossible
            # evidence; the same message passing in the log domain answers.
            a = FiniteAxis(:A, [:usual, :rare])
            b = FiniteAxis(:B, [:usual, :rare])
            c = FiniteAxis(:C, [:c0, :c1])
            both = zeros(2, 2, 2)
            both[2, 2, :] = [0.3, 0.7]
            ufg = FactorGraph([Factor(a, [1.0, 1e-200]), Factor(b, [1.0, 1e-200]),
                               Factor([a, b, c], both)])
            p, d = infer(ufg, :C; backend=BeliefPropagation())
            @test d.log_domain && d.tree && d.converged
            @test isapprox(p.table, [0.3, 0.7]; atol=1e-12)
            @test isapprox(p.table, first(infer(ufg, :C)).table; atol=1e-12)
            # A tolerated negative entry has no logarithm, and so small a mass cannot show
            # it small: the posterior is indeterminate.
            signed = copy(both)
            signed[1, 1, 1] = -1e-9
            sfg = FactorGraph([Factor(a, [1.0, 1e-200]), Factor(b, [1.0, 1e-200]),
                               Factor([a, b, c], signed)])
            @test_throws IndeterminatePosteriorError infer(sfg, :C;
                                                           backend=BeliefPropagation())
        end

        @testset "the log domain runs the same message passing" begin
            # The message equations are written once; on loopy asia both arithmetics reach
            # the same fixed point.
            asia = factor_graph_from(kernels, parents)
            linear, dl = belief_propagation(asia)
            BNI = BayesianNetworkInference
            logs = [BNI._log_potential(Float64, f) for f in asia.factors]
            logbp, dlog = BNI._run_bp(BNI._LogMessages(), logs, asia, BeliefPropagation(),
                                      Dict{Symbol,Symbol}(), Float64, false)
            @test dlog.log_domain && dlog.converged && !dl.log_domain
            for v in variables(asia)
                @test isapprox(logbp[v], linear[v]; atol=1e-12)
            end
        end
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
