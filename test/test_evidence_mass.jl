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

@testset "the fallback is correctly rounded (ADR 0016)" begin
    # Two independent observations of probability 1e-200: the binary64 mass is zero, and
    # every exact backend falls back to exact arithmetic. The reference is the exact
    # posterior of the graph as bound, rounded once.
    nearest = BayesianNetworks._nearest_binary64
    exact(x) = Rational{BigInt}(x)
    a = FiniteAxis(:A, [:rare, :usual])
    b = FiniteAxis(:B, [:rare, :usual])
    c = FiniteAxis(:C, [:c0, :c1])
    fg = FactorGraph([Factor(a, [1e-200, 1.0]), Factor(b, [1e-200, 1.0]),
                      Factor([a, c], [0.1 0.9; 0.5 0.5])])
    ev = Dict(:A => :rare, :B => :rare)
    w = [exact(1e-200) * exact(1e-200) * exact(0.1),
         exact(1e-200) * exact(1e-200) * exact(0.9)]
    want = [nearest(w[1] / sum(w)), nearest(w[2] / sum(w))]
    for backend in (VariableElimination(), JunctionTree())
        p, d = infer(fg, :C; evidence=ev, backend)
        @test p.table == want
        @test d.exact_fallback
    end
    # Belief propagation normalises every message, so on this tree nothing underflows and
    # it needs no fallback.
    @test first(infer(fg, :C; evidence=ev, backend=BeliefPropagation())).table == want
    @test brute_force_marginal(fg, :C; evidence=ev).table == want
    @test all_marginals(fg; evidence=ev)[:C].table == want
    @test all_marginals(fg; evidence=ev, backend=VariableElimination())[:C].table == want
    beliefs = clique_beliefs(fg; evidence=ev)
    @test any(bf -> bf.vars == [:C] && bf.table == want, beliefs)
    # A posterior that depends on the rare numbers themselves.
    x = FiniteAxis(:X, [:x0, :x1])
    y = FiniteAxis(:Y, [:n, :y])
    fx = FactorGraph([Factor(x, [0.3, 0.7]),
                      Factor([x, y], [1-1e-200 1e-200; 1-3e-200 3e-200]),
                      Factor(b, [1e-200, 1.0])])
    wx = [exact(0.3) * exact(1e-200), exact(0.7) * exact(3e-200)]
    @test first(infer(fx, :X; evidence=Dict(:Y => :y, :B => :rare))).table ==
          [nearest(wx[1] / sum(wx)), nearest(wx[2] / sum(wx))]
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
                @test d.exact_fallback == fallback
                # Float64(0.9) + Float64(0.1) is 1 + 2^-55; the exact posterior rounds back
                # to the row itself, which the exact fallback returns bit for bit.
                fallback && @test p.table == [0.9, 0.1]
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
        @test !d.log_domain
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

# Review of 2026-10-01, finding 2. The trust test looked only at the final binary64 mass, so
# a posterior built from underflowed products was returned whenever the mass came out
# normal. The rule, shared with BayesianNetworks and InfluenceDiagrams: a binary64 run is
# untrusted if its final mass is not a normal positive number, or if any product it computed
# from operands that are all nonzero has magnitude below `floatmin` of its element type,
# whether that product came out subnormal or rounded all the way to 0.0.
@testset "the trust test covers every product" begin
    nearest = BayesianNetworks._nearest_binary64
    exact(x) = Rational{BigInt}(x)
    rounded(w) = [nearest(x / sum(w)) for x in w]
    a = FiniteAxis(:A, [:a0, :a1])
    b = FiniteAxis(:B, [:b0, :b1])
    c = FiniteAxis(:C, [:c0, :c1])

    @testset "a subnormal product under a normal mass" begin
        fg = FactorGraph([Factor([a, c], [0.3e-300 0.0; 0.7e-300 0.0]),
                          Factor(c, [1e-21, 1.0]), Factor(b, [1e300, 1e300])])
        want = rounded([exact(0.3e-300), exact(0.7e-300)])
        for backend in (VariableElimination(), JunctionTree())
            p, d = infer(fg, :A; backend)
            @test p.table == want     # was [0.30049261083743845, 0.6995073891625615]
            @test d.exact_fallback
        end
        @test brute_force_marginal(fg, :A).table == want
        @test all_marginals(fg)[:A].table == want
        @test all_marginals(fg; backend=VariableElimination())[:A].table == want
        # Belief propagation scales its potentials, so nothing underflows there.
        p, d = infer(fg, :A; backend=BeliefPropagation())
        @test isapprox(p.table, want; atol=1e-15) && !d.log_domain
    end

    @testset "a product that rounds straight to zero" begin
        # 1e-300 * 1e-30 is 0.0 in binary64, with no subnormal on the way.
        @test 1e-300 * 1e-30 == 0.0
        fg = FactorGraph([Factor([a, c], [1e-300 0.0; 0.0 1e-300]),
                          Factor(c, [1e-30, 1.0]), Factor(b, [1e300, 1e300])])
        want = rounded([exact(1e-300) * exact(1e-30), exact(1e-300)])
        @test want[1] > 0
        for backend in (VariableElimination(), JunctionTree())
            p, d = infer(fg, :A; backend)
            @test p.table == want     # was [0.0, 1.0]
            @test d.exact_fallback
        end
        @test brute_force_marginal(fg, :A).table == want
        # The empty query still returns the binary64 mass unchecked (ADR 0011).
        @test first(infer(fg, Symbol[])).table[] == 2.0
    end

    @testset "a compiled chain" begin
        bn = bayesnet(:A => [:a0, :a1], :B => [:b0, :b1], :C => [:c0, :c1];
                      mechanisms=[:A => (), :B => (:A,), :C => (:B,)])
        for small in (1e-60, 1e-70)
            m = bind_cpt(BayesModel(bn),
                         [:A => [1.0, small], :B => [1e-255 1.0; 1.0000000001e-255 1.0],
                          :C => [0.5 0.5; 0.5 0.5]]; atol=1e-6)
            ev = Dict(:B => :b0)
            want = rounded([exact(1e-255), exact(small) * exact(1.0000000001e-255)])
            for backend in (VariableElimination(), JunctionTree())
                p, d = infer(m, :A; evidence=ev, backend, atol=1e-6)
                @test p.table == want  # was 9.999999984816837e-61, and 0.0 for 1e-70
                @test d.exact_fallback
            end
        end
    end

    @testset "the threshold is floatmin of the element type" begin
        # 1f-30 * 1f-20 is below floatmin(Float32) though far above floatmin(Float64).
        fg = FactorGraph([Factor([a, c], Float32[1.0f-30 0; 0 1.0f-30]),
                          Factor(c, Float32[1.0f-20, 1]),
                          Factor(b, Float32[1.0f30, 1.0f30])])
        p, d = infer(fg, :A)
        @test d.exact_fallback
        @test p.table == rounded([exact(1.0f-30) * exact(1.0f-20), exact(1.0f-30)])
    end

    @testset "an operand already below floatmin marks the run" begin
        # 1e-310 is subnormal; times 1e10 the product is normal, but its operand was not.
        fg = FactorGraph([Factor(a, [1e-310, 1.0]), Factor(a, [1e10, 1.0])])
        p, d = infer(fg, :A)
        @test d.exact_fallback
        @test p.table == rounded([exact(1e-310) * exact(1e10), exact(1.0)])
    end

    @testset "a structural zero does not count" begin
        # The smallest entries, 1e-300 and 1e-30, could multiply below floatmin, so every
        # product is examined; but each meets an exact zero, so the run stays in binary64.
        fg = FactorGraph([Factor([a, c], [1e-300 0.0; 0.0 1.0]),
                          Factor(c, [0.0, 1e-30]), Factor(b, [1e300, 1e300])])
        p, d = infer(fg, :A)
        @test !d.exact_fallback
        @test p.table == [0.0, 1.0]
    end
end

# Review of 2026-10-01, finding 5: `_dyadic(0.0)` is `0 * 2^-1074`, and letting a zero entry
# choose a factor's power of two widened every other integer by about a thousand bits.
@testset "a zero does not widen exact factors" begin
    D = BayesianNetworkInference._Dyadic()
    a = FiniteAxis(:A, [:a0, :a1, :a2])
    b = FiniteAxis(:B, [:b0, :b1])
    none = Dict{Symbol,Symbol}()
    f = BayesianNetworkInference._conditioned(D, Factor(a, [0.0, 0.5, 0.25]), none)
    @test f.factor.table == [0, 2, 1] && f.exponent == -2
    # Only the entries that survive the evidence choose the power...
    g = BayesianNetworkInference._conditioned(D,
                                              Factor([a, b],
                                                     [0.5 1e-300; 0.25 0.5; 0.0 1.0]),
                                              Dict(:B => :b0))
    @test g.factor.table == [2, 1, 0] && g.exponent == -2
    # ...but every entry is still checked (ADR 0016 decision 4).
    signed = Factor([a, b], [0.5 -1e-9; 0.25 0.5; 0.0 1.0])
    @test_throws IndeterminatePosteriorError BayesianNetworkInference._conditioned(D,
                                                                                   signed,
                                                                                   Dict(:B => :b0))
    # A table of decimal probabilities with a zero keeps integers of about 53 bits (55
    # here, as 0.1 and 0.9 lie in different binades), not the thousand a zero gave them.
    h = BayesianNetworkInference._conditioned(D, Factor(a, [0.0, 0.1, 0.9]), none)
    @test maximum(n -> ndigits(n; base=2), h.factor.table) <= 64
end

# Review of 2026-10-01, finding 3: only `infer` and `all_marginals` applied the tolerance
# budget of ADR 0014, so the other model-level entry points answered evidence that they
# reject. Every model-level entry point now applies the rule BayesianNetworks' `marginal`
# and InfluenceDiagrams share: a tolerated entry in [-atol, 0) takes part when it lies on a
# configuration consistent with the evidence whose other entries are all nonzero, and then
# the posterior is indeterminate if the evidence mass is not larger than the budget
# (1 + atol)^n - 1 or a cell of the queried posterior is negative; a prior is exempt.
@testset "every model-level entry point applies the tolerance rule" begin
    bn = bayesnet(:X => [:x0, :x1], :W => [:w0, :w1], :Y => [:y0, :y1];
                  mechanisms=[:X => (), :W => (), :Y => (:X, :W)])
    # P(Y = y1) is 1e-7, within the budget (1 + 1e-6)^3 - 1 of zero, and setting the
    # tolerated -1e-7 to zero moves P(X | Y = y1) from [0.5, 0.5] to [0.6, 0.4].
    function model(entry)
        y = zeros(2, 2, 2)
        y[1, 1, :] = [1 - entry, entry]
        y[1, 2, :] = [1 - 3e-7, 3e-7]
        y[2, 1, :] = [1 - 1e-7, 1e-7]
        y[2, 2, :] = [1 - 1e-7, 1e-7]
        return bind_cpt(BayesModel(bn), [:X => [0.5, 0.5], :W => [0.5, 0.5], :Y => y];
                        atol=1e-6)
    end
    m = model(-1e-7)
    ev = Dict(:Y => :y1)
    cases = [Dict(:X => :x0, :Y => :y1), Dict(:X => :x1, :Y => :y1)]
    @test caught_mass_error(() -> marginal(m, :X; evidence=ev, atol=1e-6)) isa
          IndeterminatePosteriorError
    calls = ["infer" => () -> infer(m, :X; evidence=ev, atol=1e-6),
             "posterior" => () -> posterior(m, :X; evidence=ev, atol=1e-6),
             "all_marginals" => () -> all_marginals(m; evidence=ev, atol=1e-6),
             "predict" => () -> predict(m, cases, :X; atol=1e-6),
             "baseline" => () -> baseline(observe(m, :Y => :y1), cases, :X; atol=1e-6),
             "evaluate" => () -> evaluate(m, cases, :X; atol=1e-6),
             "entropy" => () -> entropy(m, :X; evidence=ev, atol=1e-6),
             "mutual_information" => () -> mutual_information(m, :X, :W; evidence=ev,
                                                              atol=1e-6),
             "sensitivity" => () -> sensitivity(m, :X; evidence=ev, atol=1e-6),
             "tornado" => () -> tornado(m, :X, :x0; evidence=ev, atol=1e-6),
             # a finding can bring the mass within the budget when the base evidence is not
             "tornado, finding" => () -> tornado(m, :X, :x0; variables=[:Y], atol=1e-6),
             "infer, JunctionTree" => () -> infer(m, :X; evidence=ev, atol=1e-6,
                                                  backend=JunctionTree()),
             "infer, checked BP" => () -> infer(m, :X; evidence=ev, atol=1e-6,
                                                backend=BeliefPropagation(;
                                                                          check_evidence=true))]
    for (name, call) in calls
        @testset "$name" begin
            e = caught_mass_error(call)
            @test e isa IndeterminatePosteriorError && occursin("budget", e.detail)
        end
    end
    # With the tolerated entry set to zero the same evidence is answered, and a model with no
    # negative entry skips the rule whole.
    m0 = model(0.0)
    @test predict(m0, cases, :X; atol=1e-6).probabilities ≈ [0.6 0.4; 0.6 0.4]
    @test BayesianNetworkInference._tolerance_budget(compile(m0; atol=1e-6), 1e-6) ===
          nothing
    # Belief propagation computes no evidence mass and certifies evidence only on request
    # (ADR 0011): without `check_evidence` it answers, approximately, at its own cost.
    p, d = infer(m, :X; evidence=ev, atol=1e-6, backend=BeliefPropagation())
    @test p.table ≈ [0.5, 0.5] && !d.evidence_checked
    # The empty query is exempt (ADR 0011).
    @test isapprox(first(infer(m, Symbol[]; evidence=ev, atol=1e-6)).table[], 1e-7;
                   rtol=1e-6)
end

@testset "a tolerated entry counts only where it takes part" begin
    bn = bayesnet(:X => [:x0, :x1], :Y => [:y0, :y1], :W => [:w0, :w1];
                  mechanisms=[:X => (), :Y => (:X,), :W => (:Y,)])
    m = bind_cpt(BayesModel(bn),
                 [:X => [0.5, 0.5], :Y => [1+1e-7 -1e-7; 1-2e-7 2e-7],
                  :W => [0.3 0.7; 0.6 0.4]]; atol=1e-6)
    # Y = y1 has mass 1e-7 under X = x1, within the budget, but the negative entry lies in
    # the row X = x0 that the evidence rules out: it takes no part, and the posterior is
    # answered, as `marginal` answers it.
    off = Dict(:X => :x1, :Y => :y1)
    @test first(infer(m, :W; evidence=off, atol=1e-6)).table ≈ [0.6, 0.4]
    @test marginal(m, :W; evidence=off, atol=1e-6).table ≈ [0.6, 0.4]
    # Under X = x0 it takes part, and the mass -5e-8 is within the budget.
    on = Dict(:X => :x0, :Y => :y1)
    for f in (() -> infer(m, :W; evidence=on, atol=1e-6),
              () -> marginal(m, :W; evidence=on, atol=1e-6))
        @test caught_mass_error(f) isa IndeterminatePosteriorError
    end
    # A prior is the model's own law, tolerance included: its cells are returned as
    # computed, as `marginal` returns them; under evidence on which the entry takes part, a
    # negative cell is indeterminate.
    bp = bind_cpt(BayesModel(bn),
                  [:X => [1.0, 0.0], :Y => [1+1e-9 -1e-9; 0.5 0.5],
                   :W => [0.3 0.7; 0.6 0.4]]; atol=1e-6)
    prior = first(infer(bp, :Y; atol=1e-6)).table
    @test prior[2] < 0 && marginal(bp, :Y; atol=1e-6).table[2] < 0
    @test caught_mass_error(() -> infer(bp, :Y; evidence=Dict(:W => :w0), atol=1e-6)) isa
          IndeterminatePosteriorError
    # The exact fallback decides by the same rule: a chain of 340 rare detections underflows
    # binary64, and a negative entry that the evidence rules out no longer makes the exact
    # posterior indeterminate.
    n = 340
    v = [Symbol("D", i) for i in 1:n]
    chain = bayesnet(:Z => [:z0, :z1], [x => [:no, :yes] for x in v]...;
                     mechanisms=vcat([:Z => ()], [v[1] => ()],
                                     [v[i] => (v[i - 1],) for i in 2:n]))
    mc = bind_cpt(BayesModel(chain), [:Z => [1 + 1e-9, -1e-9], v[1] => [0.999, 0.001]])
    for i in 2:n
        mc = bind_cpt(mc, v[i] => [0.999 0.001; 0.9 0.1])
    end
    rare = merge(Dict(x => :yes for x in v[1:(n - 1)]), Dict(:Z => :z0))
    p, d = infer(mc, v[n]; evidence=rare)
    @test p.table == [0.9, 0.1] && d.exact_fallback
    # On a factor graph, which has no tolerance, the exact fallback still rejects any
    # negative entry (ADR 0016 decision 4).
    @test caught_mass_error(() -> infer(compile(mc), v[n]; evidence=rare)) isa
          IndeterminatePosteriorError
end

# The verdicts of the model-level entry points against BayesianNetworks' `marginal` on small
# random models with tolerated entries on and off the evidence: rare, deterministic and
# zero-holding rows, so that every verdict occurs.
@testset "model-level verdicts agree with marginal" begin
    function random_row(rng, k)
        kind = rand(rng, 1:4)
        if kind == 1
            w = rand(rng, k) .+ 0.05
            return w ./ sum(w)
        elseif kind == 2
            eps = rand(rng, [1e-7, 3e-7, -1e-7, 0.0, -5e-7])
            row = fill((1 - eps) / (k - 1), k)
            row[end] = eps
            return row
        elseif kind == 3
            row = zeros(k)
            j = rand(rng, 1:k)
            row[j] = 1.0
            if rand(rng) < 0.5
                row[mod1(j + 1, k)] = -5e-7
                row[j] = 1 + 5e-7
            end
            return row
        else
            w = rand(rng, k)
            w[rand(rng, 1:k)] = 0.0
            return w ./ sum(w)
        end
    end
    function random_model(rng)
        n = rand(rng, 3:4)
        names = [Symbol("V", i) for i in 1:n]
        k = Dict(v => rand(rng, 2:3) for v in names)
        parents = Dict{Symbol,Vector{Symbol}}()
        for (i, v) in enumerate(names)
            np = i == 1 ? 0 : rand(rng, 0:min(2, i - 1))
            parents[v] = sort(shuffle(rng, names[1:(i - 1)])[1:np])
        end
        label(v, j) = Symbol(lowercase(string(v)), "_", j)
        bn = bayesnet((v => [label(v, j) for j in 1:k[v]] for v in names)...;
                      mechanisms=[v => Tuple(parents[v]) for v in names])
        cpts = Pair{Symbol,Array{Float64}}[]
        for v in names
            pdims = [k[p] for p in parents[v]]
            table = zeros(pdims..., k[v])
            for I in CartesianIndices(Tuple(pdims))
                table[Tuple(I)..., :] = random_row(rng, k[v])
            end
            push!(cpts, v => table)
        end
        return bind_cpt(BayesModel(bn), cpts; atol=1e-6), names, k, label
    end
    function verdict(f)
        try
            return (:ok, f())
        catch e
            e isa IndeterminatePosteriorError && return (:indeterminate, nothing)
            e isa ImpossibleEvidenceError && return (:impossible, nothing)
            rethrow()
        end
    end
    rng = MersenneTwister(20261002)
    seen = Dict{Symbol,Int}()
    takes = Dict{Symbol,Int}(:on => 0, :off => 0, :prior => 0)
    for _ in 1:200
        m, names, k, label = random_model(rng)
        q = rand(rng, names)
        others = [v for v in names if v != q]
        ev = Dict{Symbol,Symbol}(v => label(v, rand(rng, 1:k[v]))
                                 for v in shuffle(rng, others)[1:rand(rng,
                                                                      0:min(2,
                                                                            length(others)))])
        fg = compile(m; atol=1e-6)
        if BayesianNetworkInference._tolerance_budget(fg, 1e-6) !== nothing
            key = isempty(ev) ? :prior :
                  BayesianNetworkInference._takes_part(fg, ev, MinFill()) ? :on : :off
            takes[key] += 1
        end
        want = verdict(() -> marginal(m, [q]; evidence=ev, atol=1e-6).table)
        seen[want[1]] = get(seen, want[1], 0) + 1
        free = [v for v in others if !haskey(ev, v)]
        joint = isempty(free) ? nothing : rand(rng, free)
        calls = Any[(want, () -> first(infer(m, q; evidence=ev, atol=1e-6)).table),
                    (want,
                     () -> first(infer(m, q; evidence=ev, atol=1e-6,
                                       backend=JunctionTree())).table),
                    (want, () -> (entropy(m, q; evidence=ev, atol=1e-6); nothing)),
                    (want,
                     () -> vec(predict(m, [merge(ev, Dict(q => label(q, 1)))], q;
                                       evidence_vars=collect(keys(ev)),
                                       atol=1e-6).probabilities))]
        if joint !== nothing
            push!(calls,
                  (verdict(() -> marginal(m, [q, joint]; evidence=ev, atol=1e-6).table),
                   () -> (mutual_information(m, q, joint; evidence=ev, atol=1e-6); nothing)))
        end
        for (expected, call) in calls
            got = verdict(call)
            @test got[1] == expected[1]
            if got[1] == expected[1] == :ok && got[2] !== nothing
                @test isapprox(got[2], expected[2]; atol=isempty(ev) ? 1e-5 : 1e-9)
            end
        end
    end
    # Every verdict occurs, and negative entries both take part and do not.
    @test all(get(seen, v, 0) > 10 for v in (:ok, :indeterminate, :impossible))
    @test all(>(10), values(takes))
end

# Review of 2026-10-02, finding 1: integer graphs wrapped silently. The kernel multiplied
# `Factor{Int}` tables unchecked, so `infer` on 2^32-weighted factors returned
# [0.9999999995343387, 4.656612870908988e-10], counts in the millions wrapped negative into
# a misleading `IndeterminatePosteriorError`, and `Rational{Int}` raised `OverflowError`.
# A posterior run now treats an overflow as it treats an untrusted binary64 run: the exact
# fallback answers, correctly rounded; only the empty query and `calibrate`, whose results
# are tables of the graph's own type, report the overflow.
@testset "integer and rational graphs that overflow are answered exactly" begin
    nearest = BayesianNetworks._nearest_binary64
    rounded(w) = [nearest(Rational{BigInt}(x) / sum(w)) for x in w]
    x = FiniteAxis(:X, [:x1, :x2])
    y = FiniteAxis(:Y, [:y1, :y2])
    fg = FactorGraph([Factor(x, [2^32, 1]), Factor([x, y], [2^32 1; 1 1])])
    want = rounded([big(2)^32 * (big(2)^32 + 1), big(2)])
    @test want == [1.0, 1.084202172233069e-19]
    for backend in (VariableElimination(), JunctionTree())
        p, d = infer(fg, :X; backend)
        @test p.table == want && d.exact_fallback
    end
    @test brute_force_marginal(fg, :X).table == want
    for backend in (VariableElimination(), JunctionTree())
        @test all_marginals(fg; backend)[:X].table == want
    end
    total = big(2)^64 + big(2)^32 + 2
    joint = reorder(only(clique_beliefs(fg)), [:X, :Y])
    @test joint.table == [nearest(big(2)^64 // total) nearest(big(2)^32 // total);
                          nearest(1 // total) nearest(1 // total)]
    # Belief propagation computes in Float64; its opt-in check decides exactly.
    @test isapprox(first(infer(fg, :X; backend=BeliefPropagation(; check_evidence=true))).table,
                   want; rtol=1e-12)
    # The empty query and `calibrate` return the graph's own type, which cannot hold the mass.
    @test_throws FactorDomainError infer(fg, Symbol[])
    @test_throws FactorDomainError calibrate(fg)
    @test_throws FactorDomainError joint_factor(fg)
    # Counts in the millions: P(X) is [39, 13] * 10^18, which wrapped negative.
    counts = FactorGraph([Factor(x, [3_000_000, 1_000_000]),
                          Factor([x, y], [3_000_000 1_000_000; 2_000_000 5_000_000]),
                          Factor(y, [4_000_000, 1_000_000])])
    for backend in (VariableElimination(), JunctionTree())
        p, d = infer(counts, :X; backend)
        @test p.table == [0.75, 0.25] && d.exact_fallback
    end
    # Small counts never overflow and keep the integer run.
    small = FactorGraph([Factor(x, [3, 1]), Factor([x, y], [3 1; 2 5])])
    p, d = infer(small, :X; evidence=Dict(:Y => :y2))
    @test p.table == [3 / 8, 5 / 8] && !d.exact_fallback
    # An entry above 2^53 has no exact Float64 value; the fallback takes it exactly.
    wide = FactorGraph([Factor(x, [2^53 + 1, 1]), Factor([x, y], [2^32 1; 1 1])])
    @test first(infer(wide, :X)).table ==
          rounded([(big(2)^53 + 1) * (big(2)^32 + 1), big(2)])
    # Rationals of Int raised OverflowError; the fallback takes them exactly.
    r = FactorGraph([Factor(x, [1 // 2^40, 1 // 3]),
                     Factor([x, y], [1//2^40 1//1; 1//5 1//1])])
    p, d = infer(r, :X; evidence=Dict(:Y => :y1))
    @test d.exact_fallback && p isa Factor{Float64}
    @test p.table == rounded([(big(1) // big(2)^40)^2, big(1) // 15])
end

# Review of 2026-10-02, with BayesianNetworks' exact fallback, which now reads every entry
# exactly (`_exact_entry`): this package's fallback took every entry through Float64, so a
# Rational or BigFloat graph whose run was untrusted raised `FactorDomainError(:exact)` (or,
# with evidence of probability exactly zero, raised it instead of
# `ImpossibleEvidenceError`). Every entry now enters at its exact value: integers and
# rationals themselves, IEEE floats and BigFloats their dyadic values.
@testset "the exact fallback takes every element type exactly" begin
    nearest = BayesianNetworks._nearest_binary64
    exact(v) = Rational{BigInt}(v)
    rounded(w) = [nearest(exact(v) / sum(exact, w)) for v in w]
    a = FiniteAxis(:A, [:a0, :a1])
    b = FiniteAxis(:B, [:b0, :b1])
    # BigFloat: a power of two two thirds of the way down BigFloat's exponent range, which
    # depends on the platform (MPFR's exponent is a C `long`, 32 bits on Windows), so its
    # square underflows the range: the BigFloat mass is zero and the run is untrusted.
    t = BigFloat(2)^(2 * (exponent(nextfloat(zero(BigFloat))) ÷ 3))
    @test t > 0 && iszero(t * t)
    tiny = FactorGraph([Factor(a, [t, 2t]), Factor(b, [t, t])])
    for backend in (VariableElimination(), JunctionTree())
        p, d = infer(tiny, :A; evidence=Dict(:B => :b0), backend)
        @test p.table == [nearest(big(1) // 3), nearest(big(2) // 3)] && d.exact_fallback
    end
    # BigFloat entries that are not Float64 values, under evidence of probability exactly
    # zero: impossible, decided exactly.
    tenths = FactorGraph([Factor(a, [big"0.1", big"0.9"]),
                          Factor([a, b], BigFloat[1 0; 1 0])])
    @test Float64(big"0.1") != big"0.1"
    zero_ev = Dict(:B => :b1)
    for call in (() -> infer(tenths, :A; evidence=zero_ev),
                 () -> infer(tenths, :A; evidence=zero_ev, backend=JunctionTree()),
                 () -> all_marginals(tenths; evidence=zero_ev),
                 () -> all_marginals(tenths; evidence=zero_ev,
                                     backend=VariableElimination()),
                 () -> clique_beliefs(tenths; evidence=zero_ev),
                 () -> infer(tenths, :A; evidence=zero_ev,
                             backend=BeliefPropagation(; check_evidence=true)))
        @test caught_mass_error(call) isa ImpossibleEvidenceError
    end
    # Rational: entries that are not dyadic, in a run that overflows Rational{Int}.
    third = FactorGraph([Factor(a, [1 // 3, 2 // 7]), Factor(b, [1 // 2^62, 1 // 11]),
                         Factor([a, b], [1//2^62 1//1; 3//2^61 1//1])])
    p, d = infer(third, :A; evidence=Dict(:B => :b0))
    @test d.exact_fallback
    @test p.table == rounded([big(1) // 3 * (big(1) // big(2)^62)^2,
                              big(2) // 7 * big(1) // big(2)^62 * big(3) // big(2)^61])
    # Float32 and Float16: their own values, whose exact rationals differ from the decimals
    # (Float32(0.1) is 13421773 * 2^-27), in runs whose products leave the normal range.
    for (T, small) in ((Float32, 1.0f-30), (Float16, Float16(1.0e-3)))
        w = T[0.1, 0.3]
        g = FactorGraph([Factor(a, w), Factor([a, b], T[small 1; small 1]),
                         Factor(b, T[small, 1])])
        p, d = infer(g, :A; evidence=Dict(:B => :b0))
        @test d.exact_fallback
        @test p.table ==
              rounded([exact(w[1]) * exact(small)^2, exact(w[2]) * exact(small)^2])
    end
end

# Review of 2026-10-02, finding 4: `all_marginals` with variable elimination collected into
# a dictionary of the graph's division type, but the exact fallback returns
# `Factor{Float64}`, so it crashed exactly when the fallback ran (Float16 evidence of mass
# 3e-5, below floatmin(Float16); Float32 of mass 1e-40). Every marginal now comes from the
# same arithmetic, as `infer` returns it.
@testset "all_marginals on a fallback has infer's element type" begin
    a = FiniteAxis(:A, [:a0, :a1])
    b = FiniteAxis(:B, [:b0, :b1])
    c = FiniteAxis(:C, [:c0, :c1])
    half = FactorGraph([Factor(a, Float16[0.5, 0.5]),
                        Factor([a, b], Float16[0.006 0.994; 0.004 0.996]),
                        Factor([b, c], Float16[0.005 0.995; 0.5 0.5])])
    ev = Dict(:B => :b0, :C => :c0)
    single = Factor{Float32}[Factor(a, Float32[1.0f-20, 2.0f-20]),
                             Factor([a, b], Float32[1.0f-20 1; 3.0f-20 1]),
                             Factor(c, Float32[0.5, 0.5])]
    for (g, e) in ((half, ev), (FactorGraph(single), Dict(:B => :b0)))
        p, d = infer(g, :A; evidence=e)
        @test d.exact_fallback && p isa Factor{Float64}
        for backend in (VariableElimination(), JunctionTree())
            ms = all_marginals(g; evidence=e, backend)
            @test valtype(ms) == Factor{Float64}
            @test ms[:A].table == p.table
            @test all(v -> ms[v].table == [1.0, 0.0], keys(e))
        end
    end
    # Without the fallback the marginals keep the graph's own type.
    ordinary = FactorGraph([Factor(a, Float32[0.25, 0.75]),
                            Factor([a, b], Float32[0.5 0.5; 0.1 0.9])])
    for backend in (VariableElimination(), JunctionTree())
        @test valtype(all_marginals(ordinary; backend)) == Factor{Float32}
    end
end

# Review of 2026-10-02: a tolerated negative cell in a prior, at the default atol. The model-level `infer` returns it as `marginal` does (a prior is the model's
# own law, tolerance included); a factor graph has no tolerance and no prior, and a negative
# posterior cell there is indeterminate (ADR 0014 decision 4, ADR 0016 decision 4).
@testset "a tolerated negative prior cell at the default atol" begin
    m = bind_cpt(BayesModel(bayesnet(:W => [:w0, :w1])), :W => [1 + 5e-9, -5e-9])
    @test marginal(m, :W).table == [1 + 5e-9, -5e-9]
    for backend in (VariableElimination(), JunctionTree(), BeliefPropagation())
        @test first(infer(m, :W; backend)).table == marginal(m, :W).table
    end
    @test all_marginals(m)[:W].table == marginal(m, :W).table
    @test caught_mass_error(() -> infer(compile(m), :W)) isa IndeterminatePosteriorError
end
