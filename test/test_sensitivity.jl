# Information-theoretic sensitivity: entropy, mutual information (zero for
# d-separated variables, equal to the average entropy reduction), the variable
# ranking and the findings tornado.

@testset "sensitivity" begin
    m = reference_habitat_model()
    fg = compile(m)

    @testset "entropy" begin
        # a uniform binary variable has one bit of entropy, a point mass none
        a = FiniteAxis(:A, [:a1, :a2])
        b = FiniteAxis(:B, [:b1, :b2])
        unif = FactorGraph([Factor(a, [0.5, 0.5])])
        @test entropy(unif, :A) ≈ 1.0
        @test entropy(unif, :A; base=exp(1)) ≈ log(2)
        point = FactorGraph([Factor(a, [1.0, 0.0])])
        @test entropy(point, :A) == 0.0
        # the joint entropy of two independent variables adds up
        indep = FactorGraph([Factor(a, [0.5, 0.5]), Factor(b, [0.25, 0.75])])
        @test entropy(indep, [:A, :B]) ≈ entropy(indep, :A) + entropy(indep, :B)
        # the model-level method matches the posterior it is computed from
        p = posterior(m, :Occupancy)
        @test entropy(m, :Occupancy) ≈
              -sum(q * log2(q) for q in values(p))
        @test entropy(m, :Occupancy) ≈ 0.9983706038378161
        # conditioning on a parent reduces it here
        @test entropy(m, :Occupancy; evidence=Dict(:HabitatQuality => :good)) <
              entropy(m, :Occupancy)
        @test entropy(m, :Occupancy; backend=JunctionTree()) ≈ entropy(m, :Occupancy)
        @test_throws ArgumentError entropy(m, :Occupancy; base=1)
    end

    @testset "mutual information" begin
        # d-separated variables carry no information about each other:
        # Climate, Irrigation and GrazingPressure are the roots of the
        # reference model and share no ancestor
        for (x, y) in ((:Climate, :Irrigation), (:Climate, :GrazingPressure),
                       (:Irrigation, :GrazingPressure))
            @test mutual_information(m, x, y) ≈ 0 atol = 1e-12
        end
        # ... and conditioning on a non-collider keeps them d-separated
        @test mutual_information(m, :Climate, :GrazingPressure;
                                 evidence=Dict(:Irrigation => :low)) ≈ 0 atol = 1e-12
        # a chain: I(target; X) is the expected reduction of H(target)
        for x in (:HabitatQuality, :Vegetation, :SoilMoisture, :Climate)
            px = posterior(m, x)
            expected = sum(q * entropy(m, :Occupancy; evidence=Dict(x => l))
                           for (l, q) in px)
            @test mutual_information(m, :Occupancy, x) ≈
                  entropy(m, :Occupancy) - expected
        end
        # symmetry, non-negativity and the backend agreeing
        @test mutual_information(m, :Occupancy, :Vegetation) ≈
              mutual_information(m, :Vegetation, :Occupancy)
        @test mutual_information(m, :Occupancy, :Vegetation) > 0
        @test mutual_information(m, :Occupancy, :Vegetation; backend=JunctionTree()) ≈
              mutual_information(m, :Occupancy, :Vegetation)
        # a deterministic copy carries the full entropy of its source
        a = FiniteAxis(:A, [:a1, :a2])
        b = FiniteAxis(:B, [:b1, :b2])
        copy_fg = FactorGraph([Factor(a, [0.5, 0.5]), Factor([a, b], [1.0 0.0; 0.0 1.0])])
        @test mutual_information(copy_fg, :A, :B) ≈ 1.0
        @test mutual_information(copy_fg, :A, :B; base=exp(1)) ≈ log(2)
        # errors
        @test_throws ScopeError mutual_information(m, :Occupancy, :Occupancy)
        @test_throws BayesianNetworks.UnknownVariableError mutual_information(m, :Occupancy,
                                                                              :Nope)
        @test_throws ScopeError mutual_information(m, :Occupancy, :Vegetation;
                                                   evidence=Dict(:Occupancy => :present))
        @test_throws ArgumentError mutual_information(m, :Occupancy, :Vegetation; base=0.5)
    end

    @testset "ranking" begin
        s = sensitivity(m, :Occupancy)
        @test length(s) == 6                       # every variable but the target
        @test !any(r -> r.variable == :Occupancy, s)
        @test issorted([-r.mutual_information for r in s])
        @test first(s).variable == :HabitatQuality
        @test s[2].variable == :Vegetation
        h = entropy(m, :Occupancy)
        # `sensitivity` builds each row by calling `mutual_information` and dividing by the
        # entropy, so comparing a row against those functions restates the implementation.
        # Compare against the full joint instead, summed by hand from `joint_factor`, which
        # is the oracle variable elimination itself is tested against.
        fgm = compile(m)
        full = joint_factor(fgm)
        function mi_from_joint(a::Symbol, b::Symbol; base=2)
            pab = reorder(marginalize(full, setdiff(scope(full), [a, b])), [a, b])
            pa = marginalize(pab, [b]).table
            pb = marginalize(pab, [a]).table
            total = 0.0
            for i in axes(pab.table, 1), j in axes(pab.table, 2)
                p = pab.table[i, j]
                p > 0 && (total += p * log(p / (pa[i] * pb[j])))
            end
            return total / log(base)
        end
        for r in s
            @test isapprox(r.mutual_information, mi_from_joint(:Occupancy, r.variable);
                           atol=1e-10)
            @test isapprox(r.entropy_reduction, r.mutual_information / h; atol=1e-12)
            @test 0 <= r.entropy_reduction <= 1
        end
        # Mutual information is symmetric and vanishes exactly for independent variables:
        # two properties of the quantity, not of how it is computed here.
        @test isapprox(mutual_information(m, :Occupancy, :Climate),
                       mutual_information(m, :Climate, :Occupancy); atol=1e-12)
        @test isapprox(mutual_information(m, :Climate, :GrazingPressure), 0; atol=1e-12)
        # evidence removes a variable from the ranking and can d-separate others
        se = sensitivity(m, :Occupancy; evidence=Dict(:HabitatQuality => :good))
        @test !any(r -> r.variable == :HabitatQuality, se)
        # HabitatQuality is the whole Markov blanket of Occupancy here
        @test all(r -> isapprox(r.mutual_information, 0; atol=1e-12), se)
        # a restricted list keeps only the named variables
        s2 = sensitivity(m, :Occupancy; variables=[:Vegetation, :Climate])
        @test [r.variable for r in s2] == [:Vegetation, :Climate]
        @test sensitivity(fg, :Occupancy) == s
        @test_throws BayesianNetworks.UnknownVariableError sensitivity(m, :Nope)
    end

    @testset "tornado" begin
        t = tornado(m, :Occupancy, :present)
        @test length(t) == 6
        @test issorted([-r.range for r in t])
        @test first(t).variable == :HabitatQuality
        for r in t
            @test r.low <= r.high && r.range ≈ r.high - r.low
            @test r.low ≈ posterior(m, :Occupancy;
                                    evidence=Dict(r.variable => r.low_state))[:present]
            @test r.high ≈ posterior(m, :Occupancy;
                                     evidence=Dict(r.variable => r.high_state))[:present]
        end
        # P(Occupancy = present | HabitatQuality) is 0.2 or 0.75 by construction
        hq = first(t)
        @test hq.low ≈ 0.2 && hq.high ≈ 0.75
        @test hq.low_state == :poor && hq.high_state == :good
        # impossible findings are skipped rather than raised
        a = FiniteAxis(:A, [:a1, :a2])
        b = FiniteAxis(:B, [:b1, :b2])
        zero_fg = FactorGraph([Factor(a, [0.0, 1.0]), Factor([a, b], [0.5 0.5; 0.2 0.8])])
        tz = tornado(zero_fg, :B, :b1)
        @test length(tz) == 1 && tz[1].variable == :A
        @test tz[1].low ≈ 0.2 && tz[1].high ≈ 0.2 && tz[1].low_state == :a2
        @test_throws BayesianNetworks.UnknownVariableError tornado(m, :Nope, :present)
        @test_throws BayesianNetworks.UnknownStateError tornado(m, :Occupancy, :nope)
    end
end
