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
        # The search is not vacuous: it finds the root and the four concrete types.
        @test issubset([InferenceError, ScopeError, ShapeError, CompileError,
                        LogFactorDomainError], owned)
        for T in owned
            @test T <: InferenceError
            @test parentmodule(T) === BayesianNetworkInference
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
                        :LogFactorDomainError], exported)
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
        # This package's own four types.
        own = [caught(() -> infer(fg, :Z)),
               caught(() -> Factor([a], rand(3))),
               caught(() -> compile(BayesModel(bayesnet(:A => [:a, :b], :B => [:x, :y];
                                                        mechanisms=[:B => [:A]],
                                                        closed=false)))),
               caught(() -> infer(FactorGraph([Factor(a, [-eps(), 1.0])]), :A;
                                  backend=LogVariableElimination()))]
        @test map(typeof, own) == [ScopeError, ShapeError, CompileError,
                                   LogFactorDomainError]
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
        @test LogFactorDomainError([:A], (1,), -0.0) !=
              LogFactorDomainError([:A], (1,), 0.0)
        @test length(Set([CompileError("m", [:X]), CompileError("m", [:X])])) == 1
    end
end
