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
        # Ties resolve to the first label, signed zeros included: -0.0 == 0.0, although
        # `Base.argmax` orders them by `isless` (LEAN-JULIA-DISCREPANCIES-2026-09-30, 1).
        @test argmax_table(Factor(a, [-0.0, 0.0]), :A)[] == first(a.labels)
        @test argmax_table(Factor(a, [0.0, -0.0]), :A)[] == first(a.labels)
        @test argmax_table(Factor(a, [0.5, 0.5]), :A)[] == first(a.labels)
        @test argmax_table(Factor([a, b], [-0.0 1.0 0.0; 0.0 1.0 -0.0]), :A) ==
              fill(first(a.labels), 3)
        # A row containing NaN has no maximizer.
        @test_throws ArgumentError argmax_table(Factor(a, [NaN, 1.0]), :A)
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
        # a zero total is an ArgumentError naming the scope, as for a kernel column
        # (ADR 0012); posterior code raises ImpossibleEvidenceError before reaching it
        @test_throws ArgumentError normalize(Factor(a, [0.0, 0.0]))
        zero_total = try
            normalize(Factor(a, [0.0, 0.0]))
        catch e
            e
        end
        @test occursin("[:A]", sprint(showerror, zero_total))
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

# Review of 2026-10-02, finding 5: the stride kernel merges axes and runs its two innermost
# ones as loops, with paths of their own for contiguous and constant operands. Each cell is
# still the one product of its two entries, so the result must equal an independent
# reference bit for bit -- on permuted scopes, axes of size one, scalars and mixed element
# types, which are the layouts the special paths separate.
@testset "multiply equals the reference bit for bit on every layout" begin
    function reference_multiply(f, g)
        vars = copy(f.vars)
        axes = copy(f.axes)
        for (v, a) in zip(g.vars, g.axes)
            v in vars || (push!(vars, v); push!(axes, a))
        end
        sz = Tuple(length(a) for a in axes)
        out = Array{promote_type(eltype(f), eltype(g))}(undef, sz)
        for ci in CartesianIndices(sz)
            fi = CartesianIndex(Tuple(ci[findfirst(==(v), vars)] for v in f.vars))
            gi = CartesianIndex(Tuple(ci[findfirst(==(v), vars)] for v in g.vars))
            out[ci] = f.table[fi] * g.table[gi]
        end
        return vars, out
    end
    rng = MersenneTwister(20261002)
    names = [:A, :B, :C, :D, :E, :F]
    layouts = Dict(:contiguous => 0, :constant => 0, :strided => 0)
    for trial in 1:600
        card = Dict(n => rand(rng, [1, 1, 2, 3, 4, 7]) for n in names)
        fv = shuffle(rng, randsubseq(rng, names, 0.5))
        gv = shuffle(rng, randsubseq(rng, names, 0.5))
        mk(vs) = FiniteAxis[FiniteAxis(v, [Symbol(v, i) for i in 1:card[v]]) for v in vs]
        S, U = rand(rng,
                    [(Float64, Float64), (Float64, Float64), (Float32, Float64),
                     (Int, Float64), (Bool, Float64), (Float32, Float32)])
        function draw(T, n)
            return T === Bool ? rand(rng, Bool, n) :
                   T === Int ? rand(rng, 0:9, n) :
                   T.(rand(rng, n) .* 10.0 .^ rand(rng, -3:3, n))
        end
        fa, ga = mk(fv), mk(gv)
        fsz, gsz = Tuple(length(a) for a in fa), Tuple(length(a) for a in ga)
        f = Factor(fv, fa, reshape(draw(S, prod(fsz; init=1)), fsz))
        g = Factor(gv, ga, reshape(draw(U, prod(gsz; init=1)), gsz))
        want_vars, want = reference_multiply(f, g)
        got = multiply(f, g)
        @test got.vars == want_vars
        @test eltype(got) == eltype(want)
        @test isequal(got.table, want)
        # Which inner path the walk takes, so that every one is known to be exercised.
        BNI = BayesianNetworkInference
        vars, axes = BNI._union_axes(f, g, :test)
        sz = Int[length(a) for a in axes]
        as, bs = BNI._result_strides(f, vars), BNI._result_strides(g, vars)
        msz, mas, mbs = BNI._merged_axes(sz, as, bs)
        if !isempty(msz)
            key = mas[1] == mbs[1] == 1 ? :contiguous :
                  (mas[1] == 0 || mbs[1] == 0) ? :constant : :strided
            layouts[key] += 1
        end
    end
    @test all(>(50), values(layouts))
end

# Review of 2026-10-02, finding 3: an empty product was the Float64 unit whatever the
# factors' type.
@testset "an empty product has the element type of its list" begin
    @test multiply(Factor{Float32}[]) isa Factor{Float32}
    @test multiply(Factor{Rational{Int}}[]).table[] === 1 // 1
    @test multiply(Factor[]) == unit_factor()
end

# Review of 2026-10-02, finding 1: the kernel multiplied in Int arithmetic unchecked, so
# integer tables wrapped silently (2^32 * 2^32 == 0), and `Rational{Int}` raised Base's
# `OverflowError`. Integer and rational tables are now computed exactly or the cell is
# reported, as a `FactorDomainError` naming the factor, the cell and the exact value.
@testset "integer and rational tables never wrap" begin
    x = FiniteAxis(:X, [:x1, :x2])
    y = FiniteAxis(:Y, [:y1, :y2])
    caught(f) =
        try
            f()
            nothing
        catch e
            e
        end
    e = caught(() -> multiply(Factor(x, [2^32, 1]), Factor([x, y], [2^32 1; 1 1])))
    @test e isa FactorDomainError && e.backend === :multiply
    @test e.vars == [:X, :Y] && e.index == (1, 1) && e.value == big(2)^64
    @test occursin("overflows", sprint(showerror, e)) &&
          occursin("18446744073709551616", sprint(showerror, e))
    # Products that fit are Base's, and an operand that is exact zero gives zero.
    @test multiply(Factor(x, [2^31, 0]), Factor([x, y], [2^31 1; 7 1])).table ==
          [2^62 2^31; 0 0]
    # Sums: the cell of the result, and the total.
    big_counts = Factor([x, y], [typemax(Int) 0; 1 1])
    e = caught(() -> marginalize(big_counts, :X))
    @test e isa FactorDomainError && e.backend === :marginalize && e.vars == [:Y] &&
          e.index == (1,) && e.value == big(typemax(Int)) + 1
    @test marginalize(big_counts, :Y).table == [typemax(Int), 2]
    e = caught(() -> normalize(big_counts))
    @test e isa FactorDomainError && e.backend === :normalize && e.index == () &&
          e.vars == [:X, :Y]
    @test occursin("the total of the factor over [:X, :Y]", sprint(showerror, e))
    # The small integer types sum to Int, as `sum` does, and do not wrap at their own width.
    @test marginalize(Factor(x, Int8[100, 100]), :X).table[] === 200
    # Mixed signs: a negative integer times an unsigned one has no unsigned value.
    e = caught(() -> multiply(Factor(x, UInt[2, 1]), Factor(x, [-1, 1])))
    @test e isa FactorDomainError && e.backend === :multiply && e.value == -1
    # Rationals: Base raised OverflowError; now the cell is reported, and a product or sum
    # whose exact value fits is Base's.
    r = Factor(x, [1 // 2^40, 1 // 1])
    e = caught(() -> multiply(r, Factor([x, y], [1//2^40 1//1; 1//1 1//1])))
    @test e isa FactorDomainError && e.value == big(1) // big(2)^80
    @test multiply(r, Factor(x, [2^39 // 1, 3 // 1])).table == [1 // 2, 3 // 1]
    @test marginalize(Factor(x, [1 // 3, 1 // 6]), :X).table[] == 1 // 2
    e = caught(() -> marginalize(Factor(x, [1 // typemax(Int), 1 // (typemax(Int) - 1)]),
                                 :X))
    @test e isa FactorDomainError && e.backend === :marginalize
    @test normalize(Factor(x, [1 // 3, 2 // 3])).table == [1 // 3, 2 // 3]
    # BigInt and Rational{BigInt} tables are exact and unchecked.
    @test multiply(Factor(x, big.([2^62, 1])), Factor(x, big.([2^62, 1]))).table ==
          [big(2)^124, 1]
end
