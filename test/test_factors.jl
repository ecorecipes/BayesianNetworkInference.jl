@testset "factors" begin
    a = FiniteAxis(:A, [:a0, :a1])
    b = FiniteAxis(:B, [:b0, :b1, :b2])
    c = FiniteAxis(:C, [:c0, :c1])
    rng = MersenneTwister(1)
    f = Factor([a, b], rand(rng, 2, 3))
    g = Factor([b, c], rand(rng, 3, 2))
    h = Factor([c, a], rand(rng, 2, 2))

    @testset "construction and invariants" begin
        @test scope(f) == [:A, :B]
        @test size(f) == (2, 3) && ndims(f) == 2 && length(f) == 6
        @test axis(f, :B) == b
        @test_throws ScopeError axis(f, :Z)
        @test_throws ShapeError Factor([a, b], rand(3, 2))
        @test_throws ShapeError Factor([a], rand(2, 3))
        @test_throws ShapeError Factor([:A, :B], [a], rand(2))
        @test_throws ScopeError Factor([a, a], rand(2, 2))
        @test_throws ScopeError Factor([:A, :Q], [a, b], rand(2, 3))
        u = unit_factor()
        @test scope(u) == Symbol[] && size(u) == () && u.table[] == 1.0
        @test eltype(unit_factor(Float32)) == Float32
        @test Factor(a, [1, 2]) isa Factor{Int}
        @test sprint(show, f) == "Factor{Float64} over (:A, :B) with size (2, 3)"
    end

    @testset "multiply: commutative modulo axis order, associative, unit" begin
        fg = multiply(f, g)
        gf = multiply(g, f)
        @test scope(fg) == [:A, :B, :C]
        @test scope(gf) == [:B, :C, :A]
        @test fg ≈ gf
        @test fg == reorder(gf, scope(fg))
        @test reorder(gf, [:A, :B, :C]).table == fg.table
        @test multiply(multiply(f, g), h) ≈ multiply(f, multiply(g, h))
        @test multiply(f, unit_factor()) == f
        @test multiply(unit_factor(), f) == f
        @test f * g == multiply(f, g)
        @test multiply([f, g, h]) ≈ multiply(f, g, h)
        @test multiply(Factor[]) == unit_factor()
        # entries by hand
        for i in 1:2, j in 1:3, k in 1:2
            @test fg.table[i, j, k] ≈ f.table[i, j] * g.table[j, k]
        end
        # element type promotion
        @test eltype(multiply(Factor(a, [1, 2]), f)) == Float64
        # disagreeing states
        a2 = FiniteAxis(:A, [:x, :y])
        @test_throws ShapeError multiply(f, Factor(a2, [1.0, 2.0]))
    end

    @testset "marginalize and maximize vs brute force" begin
        fgh = multiply(f, g, h)
        m = marginalize(fgh, [:B])
        @test scope(m) == [:A, :C]
        @test m.table ≈ dropdims(sum(fgh.table; dims=2); dims=2)
        @test marginalize(fgh, :B) == m
        @test marginalize(fgh, [:C, :A]).table ≈ vec(sum(fgh.table; dims=(1, 3)))
        all_out = marginalize(fgh, [:A, :B, :C])
        @test scope(all_out) == Symbol[] && all_out.table[] ≈ sum(fgh.table)
        @test marginalize(fgh, Symbol[]) === fgh
        @test_throws ScopeError marginalize(fgh, [:Z])
        @test_throws ScopeError marginalize(fgh, [:A, :A])
        # marginalize(multiply(f, g), B) == sum_B f g, entrywise
        mm = marginalize(multiply(f, g), :B)
        for i in 1:2, k in 1:2
            @test mm.table[i, k] ≈ sum(f.table[i, j] * g.table[j, k] for j in 1:3)
        end
        mx = maximize(fgh, [:B])
        @test mx.table ≈ dropdims(maximum(fgh.table; dims=2); dims=2)
        @test maximize(fgh, :C) == maximize(fgh, [:C])
        am = argmax_table(fgh, :B)
        @test size(am) == (2, 2)
        for i in 1:2, k in 1:2
            @test am[i, k] == b.labels[argmax(fgh.table[i, :, k])]
        end
        @test argmax_table(f, :A) isa Vector{Symbol}
        @test argmax_table(Factor(a, [0.2, 0.8]), :A)[] == :a1
    end

    @testset "condition" begin
        fg = multiply(f, g)
        ev = Dict(:B => :b1)
        cf = condition(fg, ev)
        @test scope(cf) == [:A, :C]
        @test cf.table == fg.table[:, 2, :]
        @test condition(fg, Dict(:Z => :z)) === fg
        c2 = condition(fg, Dict(:A => :a1, :C => :c0))
        @test scope(c2) == [:B] && c2.table == fg.table[2, :, 1]
        c3 = condition(f, Dict(:A => :a0, :B => :b2))
        @test scope(c3) == Symbol[] && c3.table[] == f.table[1, 3]
        @test_throws FiniteKernels.InvalidAxisError condition(f, Dict(:A => :nope))
    end

    @testset "normalize and reorder" begin
        n = normalize(f)
        @test sum(n.table) ≈ 1 && n.table ≈ f.table ./ sum(f.table)
        @test_throws KernelNormalizationError normalize(Factor(a, [0.0, 0.0]))
        r = reorder(f, [:B, :A])
        @test scope(r) == [:B, :A] && r.table == permutedims(f.table)
        @test reorder(f, [:A, :B]) === f
        @test_throws ScopeError reorder(f, [:A])
        @test_throws ScopeError reorder(f, [:A, :C])
        @test_throws ScopeError reorder(f, [:A, :A])
        @test isapprox(f, r) && f == r
        @test !(f == g)
        @test !isapprox(f, Factor([a, b], f.table .+ 1e-3))
        @test isapprox(f, Factor([a, b], f.table .+ 1e-3); atol=1e-2)
    end

    @testset "kernel <-> factor round trip" begin
        # the same table as before, written without `|>` so that the yas
        # formatter leaves the expression alone
        rows = permutedims(reshape([0.9, 0.1, 0.2, 0.8, 0.5, 0.5, 0.3, 0.7, 0.6, 0.4, 0.1,
                                    0.9], 2, 3, 2), (2, 3, 1))
        k = cpt([a, b], c, permutedims(reshape(rows, 3, 2, 2), (2, 1, 3)))
        fk = Factor(k, [:A, :B], :C)
        @test scope(fk) == [:A, :B, :C]
        @test fk.table == cpt(k)
        @test axis(fk, :C).labels == [:c0, :c1]
        # renaming axes keeps labels
        fr = Factor(k, [:P, :Q], :R)
        @test scope(fr) == [:P, :Q, :R] && axis(fr, :P).labels == [:a0, :a1]
        # sums to one over the child
        @test all(sum(fk.table; dims=3) .≈ 1)
        # no-input kernels reuse the table
        s = cpt(a, [0.3, 0.7])
        fs = Factor(s, :A)
        @test fs.table === s.table
        @test Factor(s, Symbol[], :A) == fs
        # round trip
        k2 = FiniteKernel(fk, [:A, :B], [:C])
        @test k2 ≈ k
        @test FiniteKernel(fk, [:A, :B], :C) ≈ k
        # reordered factor still converts
        @test FiniteKernel(reorder(fk, [:C, :B, :A]), [:A, :B], :C) ≈ k
        # multiple outputs: joint over (B, C) given A
        j = multiply(Factor(cpt(a, b, [0.2 0.3 0.5; 0.6 0.2 0.2]), [:A], :B), fk)
        kj = FiniteKernel(j, [:A], [:B, :C])
        @test size(kj.table) == (3, 2, 2)
        @test FiniteKernels.is_normalized(kj)
        # normalisation failure
        @test_throws KernelNormalizationError FiniteKernel(f, [:A], [:B])
        @test_throws ScopeError FiniteKernel(fk, [:A], [:C])
        # shape checks
        @test_throws ShapeError Factor(k, [:A], :C)
        @test_throws ShapeError Factor(kj, [:A], :B)
    end
end

@testset "multiply matches an independent reference on random scopes" begin
    # `multiply` walks the result linearly with precomputed strides rather than
    # broadcasting reshaped views, because `f.table` has no statically known rank. The
    # reference below indexes through CartesianIndices instead, so it shares no code with
    # the implementation and would catch a stride or axis-order error.
    function reference_multiply(f, g)
        vars = copy(f.vars)
        axes = copy(f.axes)
        for (v, a) in zip(g.vars, g.axes)
            v in vars || (push!(vars, v); push!(axes, a))
        end
        sz = Tuple(length(a) for a in axes)
        out = zeros(Float64, sz)
        for ci in CartesianIndices(sz)
            fi = CartesianIndex(Tuple(ci[findfirst(==(v), vars)] for v in f.vars))
            gi = CartesianIndex(Tuple(ci[findfirst(==(v), vars)] for v in g.vars))
            out[ci] = f.table[fi] * g.table[gi]
        end
        return vars, out
    end

    rng = MersenneTwister(3)
    names = [:A, :B, :C, :D]
    compared = 0
    for _ in 1:400
        card = Dict(n => rand(rng, 2:4) for n in names)
        fv = sort(randsubseq(rng, names, 0.6))
        gv = sort(randsubseq(rng, names, 0.6))
        (isempty(fv) || isempty(gv)) && continue
        mk(vs) = [FiniteAxis(v, [Symbol(v, i) for i in 1:card[v]]) for v in vs]
        fa, ga = mk(fv), mk(gv)
        f = Factor(fa, rand(rng, Tuple(length(a) for a in fa)...))
        g = Factor(ga, rand(rng, Tuple(length(a) for a in ga)...))
        want_vars, want_table = reference_multiply(f, g)
        got = multiply(f, g)
        @test got.vars == want_vars
        @test got.table ≈ want_table
        compared += 1
    end
    @test compared >= 300
end

@testset "the stride kernel and _broadcastable agree" begin
    # `multiply` walks the result with explicit strides; the log-domain path and
    # InfluenceDiagrams' valuation layer still align factors with `_broadcastable`. Two ways
    # of computing the same alignment is exactly the shape of divergence that let
    # `normalize`'s empty-scope bug survive in the log path only, so check they agree rather
    # than trusting the comment in CLAUDE.md that says to keep them in step.
    rng = MersenneTwister(21)
    names = [:A, :B, :C, :D]
    compared = 0
    for _ in 1:300
        card = Dict(n => rand(rng, 2:4) for n in names)
        fv = sort(randsubseq(rng, names, 0.6))
        gv = sort(randsubseq(rng, names, 0.6))
        (isempty(fv) || isempty(gv)) && continue
        mk(vs) = [FiniteAxis(v, [Symbol(v, i) for i in 1:card[v]]) for v in vs]
        fa, ga = mk(fv), mk(gv)
        f = Factor(fa, rand(rng, Tuple(length(a) for a in fa)...))
        g = Factor(ga, rand(rng, Tuple(length(a) for a in ga)...))

        vars, axes = BayesianNetworkInference._union_axes(f, g, :test)
        sz = Tuple(length(a) for a in axes)
        broadcast_table = BayesianNetworkInference._broadcastable(f, vars, sz) .*
                          BayesianNetworkInference._broadcastable(g, vars, sz)
        @test multiply(f, g).table ≈ broadcast_table
        compared += 1
    end
    @test compared >= 200
end
