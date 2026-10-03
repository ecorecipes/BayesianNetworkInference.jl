# The exception hierarchy of ADR 0013. Every exception type this package defines subtypes
# `InferenceError`, which subtypes BayesianNetworks' `BayesNetError`. `ScopeError` is a
# frozen name: the constructor-schedules checker compares its bare name, so it keeps its
# module and its unqualified printing. The factor API raises FiniteKernels' errors, so every
# exception type FiniteKernels exports is re-exported, as the same binding, and so are
# BayesianNetworks' root, its `AnyBayesNetError` union and `ImpossibleEvidenceError`, and
# BayesianNetworkFormats' root but none of its concrete types.
using FiniteKernels: FiniteKernels
using BayesianNetworkFormats: BayesianNetworkFormats

# Concrete and abstract exception types whose binding `M` owns. Only data types are
# considered, because `parentmodule` throws on a `Union` such as `AnyBayesNetError`.
function owned_exception_types(M)
    types = Type[]
    for n in names(M; all=true)
        isdefined(M, n) || continue
        T = getglobal(M, n)
        T isa DataType && T <: Exception && parentmodule(T) === M && push!(types, T)
    end
    return types
end

function exported_exception_names(M)
    return filter(names(M)) do n
        T = getglobal(M, n)
        return T isa Type && T <: Exception
    end
end

function caught(f)
    try
        f()
    catch e
        return e
    end
    return nothing
end

@testset "errors" begin
    @testset "hierarchy" begin
        owned = owned_exception_types(BayesianNetworkInference)
        # The search is not vacuous: it finds the root and the concrete types.
        @test issubset([InferenceError, ScopeError, ShapeError, CompileError,
                        FactorDomainError, FactorEntryError, TraceLimitError], owned)
        for T in owned
            @test T <: InferenceError
            @test parentmodule(T) === BayesianNetworkInference
            # Exported names print bare (the frozen-name rule); an internal type, such as the
            # `_UnresolvedMass` signal, is never shown to a caller.
            Base.isexported(BayesianNetworkInference, nameof(T)) &&
                @test string(T) == string(nameof(T))
        end
        @test isabstracttype(InferenceError)
        @test supertype(InferenceError) === BayesNetError
        @test InferenceError <: AnyBayesNetError
        # A frozen name: the constructor-schedules checker compares it as a bare string.
        @test parentmodule(ScopeError) === BayesianNetworkInference
        @test string(ScopeError) == "ScopeError"
    end

    @testset "re-exports" begin
        exported = names(BayesianNetworkInference)
        @test issubset([:InferenceError, :ScopeError, :ShapeError, :CompileError,
                        :FactorDomainError, :FactorEntryError, :TraceLimitError], exported)
        @test !isdefined(BayesianNetworkInference, :LogFactorDomainError)
        # Drift: every exception type FiniteKernels exports, the root included.
        fk = exported_exception_names(FiniteKernels)
        @test issubset([:FiniteKernelsError, :InvalidAxisError, :KernelShapeError,
                        :KernelEntryError, :KernelNormalizationError, :SpaceMismatchError],
                       fk)
        for n in fk
            @test n in exported
            @test getglobal(BayesianNetworkInference, n) === getglobal(FiniteKernels, n)
        end
        # BayesianNetworks' root, union and zero-mass error, as its own bindings.
        for n in (:BayesNetError, :AnyBayesNetError, :ImpossibleEvidenceError)
            @test n in exported
            @test getglobal(BayesianNetworkInference, n) === getglobal(BayesianNetworks, n)
        end
        # Of BayesianNetworkFormats, only its root, as its own binding; none of its
        # concrete types: the conformance adapters load this package with `using`, and the
        # inspect adapter records Formats' types under their qualified names.
        fmt = exported_exception_names(BayesianNetworkFormats)
        @test issubset([:BayesianNetworkFormatsError, :ParseError, :ValidationError], fmt)
        @test :BayesianNetworkFormatsError in exported
        @test BayesianNetworkInference.BayesianNetworkFormatsError ===
              BayesianNetworkFormats.BayesianNetworkFormatsError
        for n in fmt
            n === :BayesianNetworkFormatsError && continue
            @test n ∉ exported
        end
        # What the adapters see: they load BayesianNetworks and this package with `using`,
        # so the frozen names print bare.
        adapter = Module(:AdapterLike)
        Core.eval(adapter, :(using BayesianNetworks, BayesianNetworkInference))
        shown(T) = sprint(show, T; context=:module => adapter)
        @test shown(ScopeError) == "ScopeError"
        @test shown(ImpossibleEvidenceError) == "ImpossibleEvidenceError"
        @test shown(InferenceError) == "InferenceError"
        @test shown(InvalidAxisError) == "InvalidAxisError"
        @test isdefined(adapter, :ScopeError) && adapter.ScopeError === ScopeError
    end

    @testset "errors of every layer are caught by their roots" begin
        a = FiniteAxis(:A, [:a0, :a1])
        c = FiniteAxis(:C, [:c0, :c1])
        f = Factor(a, [0.3, 0.7])
        fg = FactorGraph([f, Factor([a, c], [0.9 0.1; 0.2 0.8])])
        # This package's own types.
        own = [caught(() -> infer(fg, :Z)),
               caught(() -> Factor([a], rand(3))),
               caught(() -> compile(BayesModel(bayesnet(:A => [:a, :b], :B => [:x, :y];
                                                        mechanisms=[:B => [:A]],
                                                        closed=false)))),
               caught(() -> infer(FactorGraph([Factor(a, [-eps(), 1.0])]), :A;
                                  backend=LogVariableElimination())),
               caught(() -> FactorGraph([Factor(a, [NaN, 1.0])])),
               caught(() -> trace_variable_elimination(fg, [:C]; max_entries=1))]
        @test map(typeof, own) == [ScopeError, ShapeError, CompileError,
                                   FactorDomainError, FactorEntryError, TraceLimitError]
        for e in own
            @test e isa InferenceError
            @test e isa BayesNetError
            @test e isa AnyBayesNetError
        end
        @test_throws InferenceError infer(fg, :Z)
        # BayesianNetworks' zero-mass error passes through: a BayesNetError, but not an
        # InferenceError.
        impossible = FactorGraph([Factor(a, [1.0, 0.0]), Factor(c, [0.5, 0.5])])
        e = caught(() -> infer(impossible, :C; evidence=Dict(:A => :a1)))
        @test e isa ImpossibleEvidenceError
        @test e isa BayesNetError && !(e isa InferenceError)
        # FiniteKernels' errors pass through: an unknown label, and a factor that is not a
        # kernel. They are outside BayesNetError, inside AnyBayesNetError.
        for e in (caught(() -> condition(f, Dict(:A => :nope))),
                  caught(() -> FiniteKernel(Factor(a, [0.3, 0.3]), Symbol[], [:A])))
            @test e isa FiniteKernelsError
            @test e isa AnyBayesNetError
            @test !(e isa BayesNetError)
        end
        @test caught(() -> condition(f, Dict(:A => :nope))) isa InvalidAxisError
        # Normalising a zero total is an invalid argument, outside every root.
        e = caught(() -> normalize(Factor(a, [0.0, 0.0])))
        @test e isa ArgumentError
        @test !(e isa AnyBayesNetError)
    end

    @testset "structural equality from BayesNetError" begin
        # Placing the types under BayesNetError gives them its structural `==` and
        # `hash`, which compare fields with `isequal`.
        @test ScopeError(:infer, "m", [:A]) == ScopeError(:infer, "m", [:A])
        @test hash(ScopeError(:infer, "m", [:A])) == hash(ScopeError(:infer, "m", [:A]))
        @test ScopeError(:infer, "m", [:A]) != ScopeError(:infer, "m", [:B])
        @test ShapeError(:f, "m", NaN, 1) == ShapeError(:f, "m", NaN, 1)
        @test CompileError("m", [:X]) == CompileError("m", [:X])
        @test FactorDomainError(:log_domain, [:A], (1,), -0.0) !=
              FactorDomainError(:log_domain, [:A], (1,), 0.0)
        @test TraceLimitError(:t, :cells, "m", [:A]) ==
              TraceLimitError(:t, :cells, "m", [:A])
        @test length(Set([CompileError("m", [:X]), CompileError("m", [:X])])) == 1
    end
end

@testset "FactorGraph entry check (ADR 0013)" begin
    a = FiniteAxis(:A, [:a0, :a1])
    e = try
        FactorGraph([Factor(a, [1.5, -0.5])])
    catch err
        err
    end
    @test e isa FactorEntryError && e isa InferenceError
    @test e.vars == [:A] && e.value == -0.5 && Tuple(e.index) == (2,)
    # Before the check, VE returned [1.5, -0.5] as a "posterior".
    @test FactorGraph([Factor(a, [1.5, -0.5])]; check=false) isa FactorGraph
    @test FactorGraph([Factor(a, [1.0 + 1e-9, -1e-9])]) isa FactorGraph
    @test_throws FactorEntryError FactorGraph([Factor(a, [1.0 + 1e-6, -1e-6])])
    @test FactorGraph([Factor(a, [1.0 + 1e-6, -1e-6])]; atol=1e-5) isa FactorGraph
    # compile uses the unchecked constructor: a model validated at atol = 1e-6 with a
    # -5e-7 entry still compiles and runs.
    bn = bayesnet(:X => [:x0, :x1], :Y => [:y0, :y1]; mechanisms=[:Y => [:X]])
    m = bind_cpt(BayesModel(bn), :X => [0.5, 0.5])
    m = bind_cpt(m, :Y => [1.0+5e-7 -5e-7; 0.4 0.6]; atol=1e-6)
    p, _ = infer(m, :Y; atol=1e-6)
    @test isapprox(sum(p.table), 1.0; atol=1e-9)
end

@testset "model-level label errors (ADR 0015)" begin
    bn = bayesnet(:A => [:a, :b], :B => [:x, :y]; mechanisms=[:A => (), :B => (:A,)])
    m = bind_cpt(BayesModel(bn), [:A => [0.5, 0.5], :B => [0.9 0.1; 0.2 0.8]])
    fg = compile(m)
    V, S = BayesianNetworks.UnknownVariableError, BayesianNetworks.UnknownStateError
    # The model methods raise what BayesianNetworks.marginal raises.
    @test_throws V marginal(m, :Z)
    @test_throws S marginal(m, :B; evidence=Dict(:A => :zz))
    for backend in (VariableElimination(), JunctionTree(), BeliefPropagation(),
                    LogVariableElimination(), LogJunctionTree())
        @test_throws V infer(m, :Z; backend)
        @test_throws V infer(m, :B; evidence=Dict(:Z => :a), backend)
        @test_throws S infer(m, :B; evidence=Dict(:A => :zz), backend)
    end
    @test_throws V posterior(m, :Z)
    @test_throws S all_marginals(m; evidence=Dict(:A => :zz))
    @test_throws V all_marginals(m; evidence=Dict(:Z => :a))
    @test_throws S log_evidence_probability(m; evidence=Dict(:A => :zz))
    @test_throws V trace_variable_elimination(m, [:Z])
    @test_throws V entropy(m, :Z)
    @test_throws V mutual_information(m, :A, :Z)
    @test_throws V sensitivity(m, :Z)
    @test_throws V sensitivity(m, :B; variables=[:Z])
    @test_throws V tornado(m, :Z, :x)
    @test_throws S tornado(m, :B, :zz)
    @test_throws V predict(m, [Dict(:A => :a)], :Z)
    @test_throws V predict(m, [Dict(:A => :a)], :B; evidence_vars=[:Z])
    @test_throws S predict(m, [Dict(:A => :zz)], :B)
    @test_throws S evaluate(m, [Dict(:A => :a, :B => :zz)], :B)
    @test_throws S evaluate(m, [Dict(:A => :a, :B => :x)], :B; state=:zz)
    # A case may record variables and states that are never entered.
    @test predict(m, [Dict(:A => :a, :Other => :q)], :B) isa Predictions
    @test predict(m, [Dict(:A => :zz)], :B; evidence_vars=Symbol[]) isa Predictions
    # The factor-graph methods keep the factor-level types.
    @test_throws ScopeError infer(fg, :Z)
    @test_throws InvalidAxisError infer(fg, :B; evidence=Dict(:A => :zz))
    @test_throws ScopeError tornado(fg, :Z, :x)
    @test_throws InvalidAxisError tornado(fg, :B, :zz)
    @test_throws ScopeError predict(fg, [Dict(:A => :a)], :Z)
end

# Review of 2026-10-02, finding 6: the entry check iterated `CartesianIndices` of a table
# whose rank is not part of its type, dispatching on every entry (about 150 times slower
# than it needs to be, and allocating per entry). It now scans the entries as a vector.
@testset "the FactorGraph entry check does not allocate per entry" begin
    axes4 = [FiniteAxis(Symbol(:V, i), [Symbol(:s, k) for k in 1:10]) for i in 1:4]
    f = Factor(axes4, rand(MersenneTwister(6), 10, 10, 10, 10))
    check(f) = BayesianNetworkInference._check_factor_entries(f, 1e-8)
    check(f)
    @test (@allocated check(f)) < 1_000
    # The offending entry is still reported with its cell.
    bad = copy(f.table)
    bad[3, 1, 4, 2] = -1.0
    e = try
        FactorGraph([Factor(axes4, bad)])
    catch err
        err
    end
    @test e isa FactorEntryError && Tuple(e.index) == (3, 1, 4, 2) && e.value == -1.0
end
