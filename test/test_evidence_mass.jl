# Evidence mass (ADR 0014). ImpossibleEvidenceError means probability exactly zero. A binary64
# mass that is not a normal positive number -- zero, subnormal or non-finite -- is recomputed in
# the log domain and answered, and the diagnostics say so; tolerated negative entries that
# leave the posterior's sign to the rounding raise IndeterminatePosteriorError.
function detection_chain(n)
    v = [Symbol("D", i) for i in 1:n]
    bn = bayesnet([x => [:no, :yes] for x in v]...;
                  mechanisms=vcat([v[1] => ()], [v[i] => (v[i - 1],) for i in 2:n]))
    m = bind_cpt(BayesModel(bn), v[1] => [0.999, 0.001])
    for i in 2:n
        m = bind_cpt(m, v[i] => [0.999 0.001; 0.9 0.1])
    end
    return m, v
end

function caught_mass_error(f)
    try
        f()
    catch e
        return e
    end
    return nothing
end

@testset "evidence mass (ADR 0014)" begin
    @testset "rare evidence is answered on every exact backend" begin
        # The closed form: conditioning on the previous site leaves [0.9, 0.1].
        for (n, fallback) in ((300, false), (320, true), (340, true))
            m, v = detection_chain(n)
            ev = Dict(x => :yes for x in v[1:(n - 1)])
            for backend in (VariableElimination(), JunctionTree())
                p, d = infer(m, v[n]; evidence=ev, backend)
                @test isapprox(p.table, [0.9, 0.1]; rtol=1e-10)
                @test d.log_fallback == fallback
            end
            @test isapprox(first(infer(m, v[n]; evidence=ev,
                                       backend=BeliefPropagation())).table,
                           [0.9, 0.1]; rtol=1e-10)
        end
        m, v = detection_chain(340)
        ev = Dict(x => :yes for x in v[1:339])
        fg = compile(m)
        @test isapprox(all_marginals(m; evidence=ev)[v[340]].table, [0.9, 0.1]; rtol=1e-10)
        @test isapprox(all_marginals(m; evidence=ev,
                                     backend=VariableElimination())[v[340]].table,
                       [0.9, 0.1]; rtol=1e-10)
        beliefs = clique_beliefs(fg; evidence=ev)
        @test all(b -> isapprox(sum(b.table), 1.0), beliefs)
        @test isapprox(first(infer(m, v[340]; evidence=ev,
                                   backend=BeliefPropagation(; check_evidence=true))).table,
                       [0.9, 0.1]; rtol=1e-10)
        # The trace records binary64 execution and cannot fall back.
        underflow = try
            trace_variable_elimination(fg, v[340]; evidence=ev)
        catch e
            e
        end
        @test underflow isa TraceLimitError && underflow.limit === :evidence_underflow
    end

    @testset "exact zero is impossible on every backend" begin
        bn = bayesnet(:A => [:no, :yes], :B => [:no, :yes], :C => [:lo, :hi];
                      mechanisms=[:A => (), :B => (:A,), :C => (:A,)])
        dm = bind_cpt(BayesModel(bn),
                      [:A => [0.5, 0.5], :B => [1.0 0.0; 0.0 1.0],
                       :C => [0.5 0.5; 0.2 0.8]])
        ev = Dict(:A => :yes, :B => :no)
        for backend in (VariableElimination(), JunctionTree(), BeliefPropagation(),
                        BeliefPropagation(; check_evidence=true), LogVariableElimination(),
                        LogJunctionTree())
            e = caught_mass_error(() -> infer(dm, :C; evidence=ev, backend))
            @test e isa ImpossibleEvidenceError && e.evidence == ev
        end
        @test_throws ImpossibleEvidenceError trace_variable_elimination(compile(dm), :C;
                                                                        evidence=ev)
    end

    @testset "belief propagation checks conditioned entries one by one" begin
        # Each conditioned scalar is 1e-200, so their product underflows, but none is zero.
        a = FiniteAxis(:A, [:a0, :a1])
        b = FiniteAxis(:B, [:b0, :b1])
        c = FiniteAxis(:C, [:c0, :c1])
        fg = FactorGraph([Factor(a, [1e-200, 1.0]), Factor(b, [1e-200, 1.0]),
                          Factor(c, [0.25, 0.75])])
        p, d = infer(fg, :C; evidence=Dict(:A => :a0, :B => :b0),
                     backend=BeliefPropagation())
        @test p.table ≈ [0.25, 0.75]
        @test !d.exact_fallback
    end

    @testset "tolerated negatives leave the posterior indeterminate" begin
        bn = bayesnet(:X => [:x0, :x1], :Y => [:y0, :y1];
                      mechanisms=[:X => (), :Y => (:X,)])
        mn = bind_cpt(BayesModel(bn), [:X => [0.9, 0.1], :Y => [0.5 0.5; 1.0+5e-7 -5e-7]];
                      atol=1e-6)
        for backend in (VariableElimination(), JunctionTree())
            e = caught_mass_error(() -> infer(mn, :X; evidence=Dict(:Y => :y1), backend,
                                              atol=1e-6))
            @test e isa IndeterminatePosteriorError && occursin("negative", e.detail)
        end
        @test all(>=(0),
                  first(infer(mn, :X; evidence=Dict(:Y => :y0), atol=1e-6)).table)
        # A positive, normal mass within the tolerance budget of zero, at the model level.
        mb = bind_cpt(BayesModel(bn),
                      [:X => [1.0 - 1e-6, 1e-6],
                       :Y => [1.0+1e-7 -1e-7; 0.5 0.5]]; atol=1e-6)
        e = caught_mass_error(() -> infer(mb, :X; evidence=Dict(:Y => :y1), atol=1e-6))
        @test e isa IndeterminatePosteriorError && occursin("budget", e.detail)
        # The empty query still returns the unnormalised mass (ADR 0011).
        @test first(infer(mb, Symbol[]; evidence=Dict(:Y => :y1), atol=1e-6)).table[] > 0
    end
end
