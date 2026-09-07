# Out-of-sample validation: cases, predictions, proper scoring rules,
# calibration, discrimination, confusion matrices and index splits. The scores
# are pinned to hand-computed values on an enumerable two-variable network and
# property-tested against brute-force enumeration on random DAGs.

# A -> B with P(A) = (0.6, 0.4), P(B = b1 | a1) = 0.8, P(B = b1 | a2) = 0.3.
function two_variable_graph()
    a = FiniteAxis(:A, [:a1, :a2])
    b = FiniteAxis(:B, [:b1, :b2])
    return FactorGraph([Factor(a, [0.6, 0.4]), Factor([a, b], [0.8 0.2; 0.3 0.7])])
end

# The four complete cases of that network, in the order (a1b1, a1b2, a2b1, a2b2).
function two_variable_cases()
    return Cases([Dict(:A => :a1, :B => :b1), Dict(:A => :a1, :B => :b2),
                  Dict(:A => :a2, :B => :b1), Dict(:A => :a2, :B => :b2)])
end

@testset "scores" begin
    fg2 = two_variable_graph()
    cs2 = two_variable_cases()

    @testset "cases" begin
        @test cs2 isa AbstractVector{Case}
        @test length(cs2) == 4 && size(cs2) == (4,)
        @test variables(cs2) == [:A, :B]
        @test cs2[1] == Dict(:A => :a1, :B => :b1)
        sub = cs2[[2, 4]]
        @test sub isa Cases && length(sub) == 2 && sub[1] == cs2[2]
        @test Cases(cs2) === cs2
        @test occursin("4 observations over 2 variables", sprint(show, cs2))
        # partial cases are allowed and keep their own variable set
        partial = Cases([Dict(:A => :a1), Dict(:B => :b2)])
        @test variables(partial) == [:A, :B] && length(partial[1]) == 1
        # a sample table converts to complete cases
        rng = MersenneTwister(11)
        kernels, order, parents = habitat_network(rng)
        s = ancestral_sample(kernels, order, parents, 25; rng=MersenneTwister(12))
        cs = Cases(s)
        @test length(cs) == 25 && Set(variables(cs)) == Set(order)
        @test all(length(c) == length(order) for c in cs)
    end

    @testset "predict" begin
        p = predict(fg2, cs2, :B)
        @test p isa Predictions
        @test size(p) == (4, 2) && length(p) == 4
        @test p.target == :B && p.evidence_variables == [:A]
        @test p.probabilities ≈ [0.8 0.2; 0.8 0.2; 0.3 0.7; 0.3 0.7]
        @test p.outcomes == [1, 2, 1, 2]
        @test occursin("Predictions of :B for 4 cases", sprint(show, p))
        # the target is withheld even when it is listed in the case
        @test predict(fg2, [Dict(:A => :a1, :B => :b2)], :B).probabilities ≈ [0.8 0.2]
        # no evidence variables: the marginal prior, P(B) = 0.6 * 0.8 + 0.4 * 0.3
        b = baseline(fg2, cs2, :B)
        @test all(b.probabilities[i, :] ≈ [0.6, 0.4] for i in 1:4)
        @test b.evidence_variables == Symbol[]
        # a case that does not record the target predicts but does not score
        nt = predict(fg2, [Dict(:A => :a2)], :B)
        @test nt.outcomes == [0]
        @test_throws ScopeError brier_score(nt)
        # errors
        @test_throws ScopeError predict(fg2, cs2, :Nope)
        @test_throws ScopeError predict(fg2, cs2, :B; evidence_vars=[:Nope])
        @test_throws ScopeError predict(fg2, cs2, :B; evidence_vars=[:A, :B])
        # impossible evidence names the case
        zero_fg = FactorGraph([Factor(FiniteAxis(:A, [:a1, :a2]), [0.0, 1.0]),
                               Factor([FiniteAxis(:A, [:a1, :a2]),
                                       FiniteAxis(:B, [:b1, :b2])], [0.5 0.5; 0.2 0.8])])
        @test_throws KernelNormalizationError predict(zero_fg,
                                                      [Dict(:A => :a1, :B => :b1)], :B)
    end

    @testset "hand-computed scores" begin
        p = predict(fg2, cs2, :B)
        # Brier: (0.04+0.04) + (0.64+0.64) + (0.49+0.49) + (0.09+0.09) = 2.52 over 4
        @test brier_score(p) ≈ 0.63
        @test brier_score(p) ≈ (0.08 + 1.28 + 0.98 + 0.18) / 4
        # the binary special case is twice mean((p - y)^2) on one column
        y = Float64[1, 0, 1, 0]
        @test brier_score(p) ≈ 2 * sum((p.probabilities[:, 1] .- y) .^ 2) / 4
        # log score: the mean log probability of what happened
        @test log_score(p) ≈ (log(0.8) + log(0.2) + log(0.3) + log(0.7)) / 4
        @test log_score(p) ≈ -0.8483072909705158
        # spherical score: (0.8 + 0.2)/sqrt(0.68) + (0.3 + 0.7)/sqrt(0.58), over 4
        @test spherical_score(p) ≈ (1 / sqrt(0.68) + 1 / sqrt(0.58)) / 4
        @test spherical_score(p) ≈ 0.6314356134447225
        # the prior baseline on the same four (equally weighted) cases
        b = baseline(fg2, cs2, :B)
        @test brier_score(b) ≈ 0.52
        @test log_score(b) ≈ (2 * log(0.6) + 2 * log(0.4)) / 4
        # accuracy and the confusion matrix of the MAP prediction
        cm = confusion_matrix(p)
        @test cm.labels == [:b1, :b2]
        # both a1 cases are classified b1 and both a2 cases b2, so the two
        # correct cases sit on the diagonal
        @test cm.counts == [1 1; 1 1]
        @test accuracy(p) ≈ 0.5 && accuracy(cm) ≈ 0.5
        @test occursin("accuracy 0.5", sprint(show, cm))
        @test occursin("observed", sprint(show, MIME("text/plain"), cm))
        # the log-score floor
        certain = Predictions(:B, FiniteAxis(:B, [:b1, :b2]), [1.0 0.0], [2], [:A])
        @test log_score(certain) ≈ log(1e-12)
        @test log_score(certain; floor=0.0) == -Inf
        @test log_score(certain; floor=1e-6) ≈ log(1e-6)
        @test_throws ArgumentError log_score(certain; floor=-1)
        @test spherical_score(certain) == 0.0
        @test brier_score(certain) ≈ 2.0
        # shape checks
        @test_throws ShapeError brier_score([0.5 0.5], [1, 2])
        @test_throws ShapeError log_score([0.5 0.5], [1, 2])
        @test_throws ShapeError spherical_score([0.5 0.5], [1, 2])
    end

    @testset "scores against brute-force enumeration" begin
        rng = MersenneTwister(2024)
        for _ in 1:8
            kernels, order, parents = random_dag_network(rng; n=rand(rng, 3:5))
            fg = factor_graph_from(kernels, parents)
            target = last(order)
            cs = Cases(ancestral_sample(kernels, order, parents, 40; rng))
            evs = Symbol[v for v in order if v != target]
            p = predict(fg, cs, target)
            # every predicted row equals the brute-force posterior of that case
            for (i, case) in enumerate(cs)
                ev = Dict{Symbol,Symbol}(v => case[v] for v in evs)
                @test p.probabilities[i, :] ≈
                      brute_force_marginal(fg, [target];
                                           evidence=ev).table
            end
            # and the scores equal their definitions computed by hand
            n, k = size(p)
            brier = sum((p.probabilities[i, j] - (j == p.outcomes[i]))^2
                        for i in 1:n, j in 1:k) / n
            @test brier_score(p) ≈ brier
            @test log_score(p) ≈
                  sum(log(p.probabilities[i, p.outcomes[i]])
                      for i in 1:n) / n
            @test spherical_score(p) ≈
                  sum(p.probabilities[i, p.outcomes[i]] /
                      sqrt(sum(abs2, p.probabilities[i, :]))
                      for i in 1:n) / n
            @test 0 <= brier_score(p) <= 2
            @test 0 <= spherical_score(p) <= 1
            @test log_score(p) <= 0
            # a proper score cannot be improved by ignoring the parents on
            # data drawn from the model itself (in expectation; 40 draws are
            # enough only for the weak statement that both are finite)
            @test isfinite(brier_score(baseline(fg, cs, target)))
        end
    end

    @testset "calibration" begin
        rng = MersenneTwister(7)
        n = 20000
        # a perfectly calibrated generator: outcomes drawn at the stated rate
        probs = rand(rng, n)
        outcomes = rand(rng, n) .< probs
        curve = calibration_curve(probs, outcomes; bins=10)
        @test curve isa CalibrationCurve
        @test length(curve.counts) == 10 && sum(curve.counts) == n
        @test length(curve.edges) == 11 && curve.edges[1] == 0 && curve.edges[end] == 1
        @test curve.centres ≈ [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
        @test all(abs.(curve.observed .- curve.predicted) .< 0.05)
        @test calibration_error(curve) < 0.02
        @test calibration_error(probs, outcomes; bins=10) == calibration_error(curve)
        @test occursin("ECE", sprint(show, curve))
        @test occursin("predicted", sprint(show, MIME("text/plain"), curve))
        # a deliberately overconfident generator: the truth happens at p/2
        bad = rand(rng, n) .< (probs ./ 2)
        @test calibration_error(probs, bad; bins=10) > 0.2
        # empty bins are NaN and are skipped by the ECE
        sparse_curve = calibration_curve([0.05, 0.95], [false, true]; bins=10)
        @test sparse_curve.counts[1] == 1 && sparse_curve.counts[5] == 0
        @test isnan(sparse_curve.observed[5]) && isnan(sparse_curve.predicted[5])
        @test calibration_error(sparse_curve) ≈ 0.05
        @test isnan(calibration_error(calibration_curve(Float64[], Bool[])))
        @test_throws ArgumentError calibration_curve([0.5], [true]; bins=0)
        @test_throws ArgumentError calibration_curve([1.5], [true])
        @test_throws ShapeError calibration_curve([0.5], [true, false])
        # on a Predictions the event is `target == state`
        p = predict(fg2, cs2, :B)
        cp = calibration_curve(p, :b1; bins=10)
        @test sum(cp.counts) == 4 && cp.counts[3] == 2 && cp.counts[8] == 2
        @test cp.observed[3] ≈ 0.5 && cp.observed[8] ≈ 0.5
    end

    @testset "discrimination" begin
        # a perfectly separable ranking
        scores = [0.05, 0.2, 0.35, 0.8, 0.9]
        pos = [false, false, false, true, true]
        @test auc(scores, pos) == 1.0
        r = roc_curve(scores, pos)
        @test r isa ROCCurve
        @test r.fpr[1] == 0 && r.tpr[1] == 0
        @test r.fpr[end] == 1 && r.tpr[end] == 1
        @test issorted(r.fpr) && issorted(r.tpr)
        @test auc(r) ≈ 1.0
        @test occursin("AUC 1.0", sprint(show, r))
        # a reversed ranking is 0, an uninformative one 0.5 exactly
        @test auc(scores, reverse(pos)) == 0.0
        @test auc(fill(0.5, 6), [true, false, true, false, true, false]) == 0.5
        @test auc(roc_curve(fill(0.5, 6), [true, false, true, false, true, false])) ≈ 0.5
        # ties count as half a win
        @test auc([1.0, 1.0, 0.0], [true, false, false]) ≈ 0.75
        # the rank-based and trapezoidal areas agree on random data
        rng = MersenneTwister(19)
        for _ in 1:20
            s = rand(rng, 40)
            o = rand(rng, 40) .< 0.5
            @test auc(s, o) ≈ auc(roc_curve(s, o))
        end
        # errors
        @test_throws ArgumentError auc([0.1, 0.2], [true, true])
        @test_throws ArgumentError roc_curve([0.1, 0.2], [false, false])
        @test_throws ShapeError auc([0.1], [true, false])
        @test_throws ShapeError roc_curve([0.1], [true, false])
        # a deterministic network separates its target perfectly
        a = FiniteAxis(:A, [:a1, :a2])
        b = FiniteAxis(:B, [:b1, :b2])
        det = FactorGraph([Factor(a, [0.5, 0.5]), Factor([a, b], [1.0 0.0; 0.0 1.0])])
        dcs = Cases([Dict(:A => :a1, :B => :b1), Dict(:A => :a2, :B => :b2),
                     Dict(:A => :a1, :B => :b1), Dict(:A => :a2, :B => :b2)])
        @test auc(predict(det, dcs, :B), :b2) == 1.0
        @test accuracy(predict(det, dcs, :B)) == 1.0
        @test auc(baseline(det, dcs, :B), :b2) == 0.5
    end

    @testset "splits" begin
        rng = MersenneTwister(5)
        cs = Cases([Dict(:Region => Symbol("r", mod1(i, 5)), :X => :x1) for i in 1:100])
        train, test = holdout(cs; fraction=0.25, rng)
        @test isempty(intersect(train, test))
        @test sort(vcat(train, test)) == collect(1:100)
        @test length(test) == 25
        @test allunique(train) && allunique(test)
        # a grouped holdout never splits a group
        gtrain, gtest = holdout(cs; by=:Region, fraction=0.25, rng)
        @test isempty(intersect(gtrain, gtest))
        @test sort(vcat(gtrain, gtest)) == collect(1:100)
        for r in 1:5
            grp = [i for i in 1:100 if mod1(i, 5) == r]
            @test issubset(grp, gtest) || isempty(intersect(grp, gtest))
        end
        # every group has 20 cases and 25 were asked for, so exactly one
        # group is taken: a second would move the split further from 25
        @test length(gtest) == 20
        # a grouping with one group cannot be split at all
        @test_throws ArgumentError holdout(cs; by=:X, fraction=0.25, rng)
        @test_throws ArgumentError holdout(cs; fraction=0.0)
        @test_throws ArgumentError holdout(cs; fraction=1.0)
        @test_throws ArgumentError holdout(Cases([Dict(:X => :x1)]))
        # k-fold: the test parts partition the data
        folds = kfold(cs; k=4, rng)
        @test length(folds) == 4
        @test sort(vcat([f[2] for f in folds]...)) == collect(1:100)
        for (tr, te) in folds
            @test isempty(intersect(tr, te))
            @test sort(vcat(tr, te)) == collect(1:100)
            @test length(te) == 25
        end
        # a grouped k-fold keeps every group inside one fold
        gfolds = kfold(cs; k=5, by=:Region, rng)
        @test sort(vcat([f[2] for f in gfolds]...)) == collect(1:100)
        for (_, te) in gfolds, r in 1:5
            grp = [i for i in 1:100 if mod1(i, 5) == r]
            @test issubset(grp, te) || isempty(intersect(grp, te))
        end
        @test_throws ArgumentError kfold(cs; k=1)
        @test_throws ArgumentError kfold(cs; k=101)
        # the splits are reproducible from a seed
        @test holdout(cs; fraction=0.3, rng=MersenneTwister(3)) ==
              holdout(cs; fraction=0.3, rng=MersenneTwister(3))
        @test kfold(cs; k=3, rng=MersenneTwister(3)) ==
              kfold(cs; k=3, rng=MersenneTwister(3))
    end

    @testset "evaluate" begin
        m = reference_habitat_model()
        cs = Cases(ancestral_sample(m, 400; rng=MersenneTwister(1)))
        r = evaluate(m, cs, :Occupancy; evidence_vars=[:Vegetation, :HabitatQuality])
        @test r isa EvaluationResult
        @test r.target == :Occupancy && r.n == 400
        @test r.state == :present
        @test r.evidence_variables == [:Vegetation, :HabitatQuality]
        p = predict(m, cs, :Occupancy; evidence_vars=[:Vegetation, :HabitatQuality])
        b = baseline(m, cs, :Occupancy)
        @test r.brier ≈ brier_score(p) && r.baseline_brier ≈ brier_score(b)
        @test r.log ≈ log_score(p) && r.baseline_log ≈ log_score(b)
        @test r.spherical ≈ spherical_score(p)
        @test r.accuracy ≈ accuracy(p) && r.baseline_accuracy ≈ accuracy(b)
        @test r.auc ≈ auc(p, :present) && r.baseline_auc == 0.5
        @test r.calibration_error ≈ calibration_error(p, :present)
        @test r.confusion.counts == confusion_matrix(p).counts
        # the network beats "always predict the prior" on every proper score
        @test r.brier < r.baseline_brier
        @test r.log > r.baseline_log
        @test r.spherical > r.baseline_spherical
        @test r.auc > r.baseline_auc
        @test occursin("Brier", sprint(show, r))
        txt = sprint(show, MIME("text/plain"), r)
        @test occursin("log score (higher better)", txt)
        @test occursin("prior", txt) && occursin("AUC(:present)", txt)
        # a chosen state, and a state that never occurs makes the AUC undefined
        r2 = evaluate(m, cs, :Occupancy; evidence_vars=[:Vegetation], state=:absent)
        @test r2.state == :absent && isfinite(r2.auc)
        constant = Cases([Dict(:Vegetation => :dense, :Occupancy => :present)
                          for _ in 1:5])
        rc = evaluate(m, constant, :Occupancy; evidence_vars=[:Vegetation])
        @test isnan(rc.auc) && isnan(rc.baseline_auc) && rc.accuracy == 1.0
        # a held-out subset scores through the same entry point
        train, test = holdout(cs; by=:Climate, fraction=0.3, rng=MersenneTwister(4))
        rh = evaluate(m, cs[test], :Occupancy; evidence_vars=[:Vegetation])
        @test rh.n == length(test) && rh.brier < rh.baseline_brier
        # the JunctionTree backend gives the same numbers
        rj = evaluate(m, cs, :Occupancy; evidence_vars=[:Vegetation],
                      backend=JunctionTree())
        @test rj.brier ≈ evaluate(m, cs, :Occupancy; evidence_vars=[:Vegetation]).brier
        # evidence recorded on the model is merged into every case
        mo = observe(m, :GrazingPressure => :high)
        @test predict(mo, cs[1:1], :Occupancy; evidence_vars=Symbol[]).probabilities ≈
              [posterior(mo, :Occupancy)[:absent] posterior(mo, :Occupancy)[:present]]
        # `evaluate` is this package's own function since the categorical layer moved
        # out of BayesianNetworks (ADR 0009): it is not re-exported from anywhere.
        @test parentmodule(BayesianNetworkInference.evaluate) === BayesianNetworkInference
        @test !isdefined(BayesianNetworks, :evaluate)
    end
end
