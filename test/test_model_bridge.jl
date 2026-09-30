# The bridge to BayesianNetworks.jl: compile, model-level infer / posterior /
# ancestral_sample, checked against the brute-force `marginal` of that package.

# The state `I -> vars` returned by `BayesianNetworks.marginal` as a factor.
factor_of(k::FiniteKernel) = Factor(collect(FiniteAxis, factors(k.codom)), k.table)

# A random closed model built with the BayesianNetworks constructors: 3-7
# variables with 2-3 states, parents drawn from earlier variables, random CPTs.
function random_bayes_model(rng::AbstractRNG; n::Int=rand(rng, 3:7), maxparents::Int=3)
    names_ = [Symbol("X", i) for i in 1:n]
    vars = [v => [Symbol("s", j) for j in 1:rand(rng, 2:3)] for v in names_]
    mechs = Pair{Symbol,Vector{Symbol}}[]
    for (i, v) in enumerate(names_)
        chosen = [p for p in names_[1:(i - 1)] if rand(rng) < 0.4]
        length(chosen) > maxparents && (chosen = chosen[1:maxparents])
        isempty(chosen) || push!(mechs, v => chosen)
    end
    m = BayesModel(bayesnet(vars...; mechanisms=mechs))
    for v in names_
        m = bind_kernel(m, v => random_kernel(rng, parent_space(m, v), space(m, v)))
    end
    return m
end

@testset "model bridge (BayesianNetworks.jl)" begin
    m = reference_habitat_model()
    bn = syntax(m)
    topo = [:Climate, :Irrigation, :SoilMoisture, :GrazingPressure, :Vegetation,
            :HabitatQuality, :Occupancy]
    strategies = (MinFill(), MinDegree(), AMDOrder(), ExactTreewidth())

    @testset "compile" begin
        fg = compile(m)
        @test fg isa FactorGraph{Float64}
        @test length(fg) == 7
        @test variables(fg) == topo
        @test compile(m, FactorGraphBackend()) isa FactorGraph{Float64}
        for (f, prov) in zip(fg.factors, fg.provenance)
            x = prov.variable
            ps = Symbol[variable_name(bn, p) for p in parents(bn, x)]
            @test scope(f) == vcat(ps, x)
            @test f.table == cpt(kernel(m, x))
            @test prov.mechanism == mechanism_name(bn, prov.id) == Symbol(x, "_mechanism")
            @test target(bn, prov.id) == variable_id(bn, x)
            @test FiniteKernel(f, ps, x) ≈ kernel(m, x)
        end
        @test fg.axes[:SoilMoisture].labels == [:low, :medium, :high]
        @test scope(fg.factors[5]) == [:SoilMoisture, :GrazingPressure, :Vegetation]
        # the product of the factors is the joint distribution of the model
        J = joint_distribution(m)
        @test joint_factor(fg) ≈ factor_of(J)
        @test sum(joint_factor(fg).table) ≈ 1
        # the interaction graph is the moral graph: SoilMoisture and GrazingPressure married
        g, vars, index = interaction_graph(fg)
        @test Graphs.has_edge(g, index[:SoilMoisture], index[:GrazingPressure])
        @test Graphs.has_edge(g, index[:Climate], index[:Irrigation])
        @test !Graphs.has_edge(g, index[:Climate], index[:Occupancy])
        @test treewidth(fg, ExactTreewidth()) == 2
        # a model read from a file compiles to the same factor graph
        for f in ("dne/habitat_reference.dne", "xdsl/habitat_reference.xdsl",
                  "bif/habitat_reference.bif")
            mf = read_bayesnet(fixture_path(f))
            fgf = compile(mf)
            @test Set(variables(fgf)) == Set(topo)
            @test joint_factor(fgf) ≈ joint_factor(fg)
        end
    end

    @testset "compile errors" begin
        unbound = BayesModel(reference_habitat_bn())
        @test_throws CompileError compile(unbound)
        e = try
            compile(unbound)
        catch err
            err
        end
        @test e isa CompileError
        @test Set(e.variables) == Set(topo)
        @test occursin("without a kernel", sprint(showerror, e))
        @test occursin(":Occupancy", sprint(showerror, e))
        partial = bind_cpt(unbound, :Climate => [0.3, 0.5, 0.2])
        e2 = try
            compile(partial)
        catch err
            err
        end
        @test e2 isa CompileError && !(:Climate in e2.variables) &&
              :Occupancy in e2.variables
        open_model = BayesModel(bayesnet(:A => [:a, :b], :B => [:x, :y];
                                         mechanisms=[:B => [:A]], closed=false))
        e3 = try
            compile(open_model)
        catch err
            err
        end
        @test e3 isa CompileError && e3.variables == [:A]
        @test occursin("open", sprint(showerror, e3))
        @test sprint(showerror, CompileError("msg", [:X])) ==
              "CompileError: msg (variables [:X])"
        @test_throws CompileError infer(unbound, :Occupancy)
        @test_throws CompileError posterior(unbound, :Occupancy)
        @test_throws CompileError ancestral_sample(unbound, 3)
    end

    @testset "infer matches marginal on the reference model" begin
        for x in topo, s in strategies
            p, d = infer(m, x; backend=VariableElimination(s))
            @test scope(p) == [x]
            @test p ≈ factor_of(marginal(m, x))
            @test maximum(abs, p.table .- marginal(m, x).table) < 1e-10
            @test !(x in d.order) && length(d.order) == 6
        end
        @test infer(m, :Occupancy)[1].table ≈ [0.5238, 0.4762] atol = 1e-4
        @test infer(m, :Occupancy)[1] ≈ infer(m, [:Occupancy])[1]
        @test infer(m, [:Occupancy])[1] == infer(compile(m), [:Occupancy])[1]
        # joint queries, including a user-supplied order
        q = [:Occupancy, :Climate]
        exact_q = factor_of(marginal(m, q))
        for s in strategies
            pq, _ = infer(m, q; backend=VariableElimination(s))
            @test scope(pq) == q && pq ≈ exact_q
        end
        rest = [v for v in topo if !(v in q)]
        @test infer(m, q; backend=VariableElimination(UserOrder(rest)))[1] ≈ exact_q
        @test infer(m, reverse(q))[1] ≈ exact_q
        # evidence given explicitly and recorded with observe
        ev = Dict(:Vegetation => :dense, :Irrigation => :high)
        mo = observe(m, [:Vegetation => :dense, :Irrigation => :high])
        for x in topo
            haskey(ev, x) && continue
            exact = factor_of(marginal(m, x; evidence=ev))
            for s in strategies
                @test infer(m, x; evidence=ev, backend=VariableElimination(s))[1] ≈ exact
                @test infer(mo, x; backend=VariableElimination(s))[1] ≈ exact
            end
            @test infer(mo, x)[1] ≈ factor_of(marginal(mo, x))
        end
        @test infer(mo, :Occupancy)[1].table ≈ marginal(mo, :Occupancy).table
        @test infer(observe(m, :Vegetation => :dense), :Occupancy)[1].table ≈
              [0.3325, 0.6675] atol = 1e-4
        # explicit evidence merges with, and wins over, the recorded evidence
        merged = Dict(:Vegetation => :sparse, :Irrigation => :high)
        @test infer(mo, :Occupancy; evidence=Dict(:Vegetation => :sparse))[1] ≈
              factor_of(marginal(m, :Occupancy; evidence=merged))
        three = Dict(:Vegetation => :dense, :Irrigation => :high, :Climate => :dry)
        exact3 = factor_of(marginal(m, :Occupancy; evidence=three))
        @test infer(mo, :Occupancy; evidence=[:Climate => :dry])[1] ≈ exact3
        @test infer(mo, :Occupancy; evidence=(:Climate => :dry))[1] ≈ exact3
        @test infer(unobserve(mo), :Occupancy)[1] ≈ infer(m, :Occupancy)[1]
        # the model is untouched by inference
        @test m == reference_habitat_model()
        @test evidence(mo) == ev
        # errors
        @test_throws ScopeError infer(mo, :Vegetation)
        # A name the model lacks is BayesianNetworks' label error (ADR 0015).
        @test_throws BayesianNetworks.UnknownVariableError infer(m, :Nope)
        @test_throws ScopeError infer(m, [:Occupancy, :Occupancy])
        @test_throws BayesianNetworks.UnknownVariableError infer(m, :Occupancy;
                                                                 evidence=Dict(:Nope => :x))
        @test_throws BayesianNetworks.UnknownStateError infer(m, :Occupancy;
                                                              evidence=Dict(:Climate => :arid))
        # posterior as a dictionary
        pd = posterior(m, :Occupancy; evidence=ev)
        @test pd isa Dict{Symbol,Float64}
        @test Set(keys(pd)) == Set([:absent, :present])
        @test pd[:present] ≈ marginal(m, :Occupancy; evidence=ev).table[2]
        @test sum(values(pd)) ≈ 1
        @test posterior(mo, :Occupancy) == pd
        @test posterior(m, :Climate) == Dict(:dry => 0.3, :normal => 0.5, :wet => 0.2)
        @test posterior(m, :Occupancy; backend=VariableElimination(MinDegree()))[:present] ≈
              0.4762 atol = 1e-4
    end

    @testset "asia from the BIF fixture" begin
        a = read_bayesnet(fixture_path("bif/asia.bif"))
        fg = compile(a)
        @test length(fg) == 8
        @test [p.variable for p in fg.provenance] == variables(fg)
        p, d = infer(a, :dysp)
        @test scope(p) == [:dysp] && axis(p, :dysp).labels == [:yes, :no]
        @test p.table[1] ≈ 0.4360 atol = 5e-5
        @test p.table[1] ≈ 0.4359706 atol = 1e-7
        @test p ≈ factor_of(marginal(a, :dysp))
        @test d.treewidth == 2 && d.max_factor_size <= 8
        # the hand-built asia of networks.jl agrees with the file
        kernels, order, parents_ = asia_network()
        @test infer(factor_graph_from(kernels, parents_), :dysp)[1] ≈ p
        # the classic posterior P(lung | asia = yes, dysp = yes)
        ev = Dict(:asia => :yes, :dysp => :yes)
        pl = posterior(a, :lung; evidence=ev)
        exact = marginal(a, :lung; evidence=ev).table
        @test pl[:yes] ≈ exact[1] && pl[:no] ≈ exact[2]
        @test pl[:yes] ≈ 0.0995 atol = 1e-4
        for s in strategies, x in (:lung, :tub, :bronc, :xray, :either, :smoke)
            @test infer(a, x; evidence=ev, backend=VariableElimination(s))[1] ≈
                  factor_of(marginal(a, x; evidence=ev))
        end
        @test infer(observe(a, :asia => :yes), :tub)[1].table ≈ [0.05, 0.95]
        @test infer(a, :lung; evidence=Dict(:smoke => :yes))[1].table[1] ≈ 0.1
        @test infer(a, [:xray, :dysp]; evidence=Dict(:either => :yes))[1] ≈
              factor_of(marginal(a, [:xray, :dysp]; evidence=Dict(:either => :yes)))
        # the other asia files give the same answer
        for f in ("net/asia.net", "dsc/asia.dsc", "uai/asia.uai")
            @test infer(read_bayesnet(fixture_path(f)), :dysp)[1] ≈ p
        end
    end

    @testset "interventions" begin
        d = do_intervention(m, :GrazingPressure => :low)
        fgd = compile(d)
        @test length(fgd) == 7
        i = findfirst(p -> p.variable == :GrazingPressure, fgd.provenance)
        @test fgd.provenance[i].mechanism == hard_intervention_name(:GrazingPressure, :low)
        @test scope(fgd.factors[i]) == [:GrazingPressure]
        @test fgd.factors[i].table == [1.0, 0.0]
        for x in topo
            x == :GrazingPressure && continue
            for s in strategies
                @test infer(d, x; backend=VariableElimination(s))[1] ≈
                      factor_of(marginal(d, x))
            end
        end
        @test posterior(d, :GrazingPressure) == Dict(:low => 1.0, :high => 0.0)
        @test infer(d, :Occupancy)[1].table ≈ marginal(d, :Occupancy).table
        # GrazingPressure is a root, so forcing it and seeing it agree downstream
        @test infer(d, :Occupancy)[1] ≈
              infer(m, :Occupancy; evidence=Dict(:GrazingPressure => :low))[1]
        @test !isapprox(infer(d, :Occupancy)[1], infer(m, :Occupancy)[1]; atol=1e-6)
        # seeing dense vegetation says something about the climate; forcing it does not
        prior = infer(m, :Climate)[1]
        seen = infer(observe(m, :Vegetation => :dense), :Climate)[1]
        forced = infer(do_intervention(m, :Vegetation => :dense), :Climate)[1]
        @test forced ≈ prior
        @test !isapprox(seen, prior; atol=1e-6)
        @test seen ≈ factor_of(marginal(observe(m, :Vegetation => :dense), :Climate))
        @test forced ≈
              factor_of(marginal(do_intervention(m, :Vegetation => :dense), :Climate))
        @test posterior(observe(m, :Vegetation => :dense), :Climate)[:wet] > 0.2
        @test posterior(do_intervention(m, :Vegetation => :dense), :Climate)[:wet] ≈ 0.2
        # intervention combined with evidence, and a soft intervention
        de = observe(d, :Climate => :dry)
        @test infer(de, :Occupancy)[1] ≈ factor_of(marginal(de, :Occupancy))
        k = cpt(FiniteAxis(:GrazingPressure, [:low, :high]), [0.9, 0.1])
        soft = soft_intervention(m, :GrazingPressure => k)
        @test infer(soft, :Occupancy)[1] ≈ factor_of(marginal(soft, :Occupancy))
        @test !isapprox(infer(soft, :Occupancy)[1], infer(m, :Occupancy)[1]; atol=1e-6)
        # a model read from a file behaves the same under do
        mf = read_bayesnet(fixture_path("dne/habitat_reference.dne"))
        @test infer(do_intervention(mf, :GrazingPressure => :low), :Occupancy)[1] ≈
              infer(d, :Occupancy)[1]
        mx = read_bayesnet(fixture_path("xdsl/habitat_reference.xdsl"))
        @test infer(do_intervention(mx, :GrazingPressure => :low), :Occupancy)[1] ≈
              infer(d, :Occupancy)[1]
    end

    @testset "random models (property tests)" begin
        rng = MersenneTwister(2026)
        for case in 1:15
            rm = random_bayes_model(rng)
            names_ = variable_names(syntax(rm))
            fg = compile(rm)
            @test length(fg) == length(names_)
            @test sum(joint_factor(fg).table) ≈ 1
            @test joint_factor(fg) ≈ factor_of(joint_distribution(rm))
            nq = rand(rng, 1:min(2, length(names_) - 1))
            shuffled = shuffle(rng, names_)
            q = shuffled[1:nq]
            rest = shuffled[(nq + 1):end]
            ne = rand(rng, 0:min(2, length(rest)))
            ev = Dict{Symbol,Symbol}(v => rand(rng, states(syntax(rm), v))
                                     for v in rest[1:ne])
            oracle = factor_of(marginal(rm, q; evidence=ev))
            for s in strategies
                p, d = infer(rm, q; evidence=ev, backend=VariableElimination(s))
                @test scope(p) == q
                @test p ≈ oracle
                @test d.treewidth >= 0
            end
            @test infer(observe(rm, collect(ev)), q)[1] ≈ oracle
            # an intervention on a random variable
            x = rand(rng, names_)
            s = rand(rng, states(syntax(rm), x))
            done = do_intervention(rm, x => s)
            qd = [v for v in q if v != x]
            evd = Dict(k => v for (k, v) in ev if k != x)
            if !isempty(qd)
                @test infer(done, qd; evidence=evd)[1] ≈
                      factor_of(marginal(done, qd; evidence=evd))
            end
        end
    end

    @testset "sampling" begin
        n = 20_000
        s = ancestral_sample(m, n; rng=MersenneTwister(7))
        @test s isa AncestralSamples
        @test s.vars == topo && length(s) == n && size(s) == (n, 7)
        for x in topo
            emp = empirical_marginal(s, x)
            @test scope(emp) == [x] && sum(emp.table) ≈ 1
            @test isapprox(emp, infer(m, x)[1]; atol=0.02)
        end
        @test isapprox(empirical_marginal(s, [:Vegetation, :Occupancy]),
                       infer(m, [:Vegetation, :Occupancy])[1]; atol=0.02)
        @test ancestral_sample(m, 50; rng=MersenneTwister(3)).states ==
              ancestral_sample(m, 50; rng=MersenneTwister(3)).states
        @test length(ancestral_sample(m, 0)) == 0
        # the output of BayesianNetworks.sample converts and compares the same way
        bs = sample(m, n; rng=MersenneTwister(8))
        conv = AncestralSamples(m, bs)
        @test conv.vars == topo && size(conv) == (n, 7)
        @test conv[:Occupancy] == [b[:Occupancy] for b in bs]
        @test isapprox(empirical_marginal(m, bs, :Occupancy), infer(m, :Occupancy)[1];
                       atol=0.02)
        @test isapprox(empirical_marginal(m, bs, [:Climate, :SoilMoisture]),
                       infer(m, [:Climate, :SoilMoisture])[1]; atol=0.02)
        @test empirical_marginal(m, bs, :Occupancy) == empirical_marginal(conv, :Occupancy)
        # BayesianNetworks' own dictionary form is untouched
        freq = empirical_marginal(bs, :Occupancy)
        @test freq isa Dict{Symbol,Float64}
        @test freq[:present] ≈ empirical_marginal(m, bs, :Occupancy).table[2]
        # samples of an intervened model respect the intervention
        d = do_intervention(m, :GrazingPressure => :low)
        sd = ancestral_sample(d, 2000; rng=MersenneTwister(9))
        @test all(==(:low), sd[:GrazingPressure])
        @test isapprox(empirical_marginal(sd, :Occupancy), infer(d, :Occupancy)[1];
                       atol=0.03)
        # evidence is ignored by sampling
        @test ancestral_sample(observe(m, :Climate => :dry), 20;
                               rng=MersenneTwister(1)).states ==
              ancestral_sample(m, 20; rng=MersenneTwister(1)).states
        # conversion errors
        @test_throws ScopeError AncestralSamples(m, [Dict(:Climate => :dry)])
        bad = copy(bs[1])
        bad[:Climate] = :arid
        @test_throws FiniteKernels.InvalidAxisError AncestralSamples(m, [bad])
        @test length(AncestralSamples(m, Dict{Symbol,Symbol}[])) == 0
        # asia: deterministic `either` is respected
        a = read_bayesnet(fixture_path("bif/asia.bif"))
        sa = ancestral_sample(a, 5000; rng=MersenneTwister(11))
        @test all(sa[:either][i] == :yes for i in 1:5000 if sa[:lung][i] == :yes)
        @test isapprox(empirical_marginal(sa, :dysp), infer(a, :dysp)[1]; atol=0.03)
    end
end
