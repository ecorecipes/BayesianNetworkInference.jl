@testset "Actual VE execution capture" begin
    a, b = FiniteAxis(:A, [:low, :high]), FiniteAxis(:B, [:off, :on])
    graph = FactorGraph([Factor(a, [0.25, 0.75]), Factor([a, b], [0.75 0.25; 0.125 0.875])])
    before = deepcopy(graph.factors)
    actual, diagnostics, trace = trace_variable_elimination(graph, :B)
    ordinary, old_diagnostics = variable_elimination(graph, :B)
    @test actual == ordinary
    @test diagnostics.order == old_diagnostics.order
    @test trace["layout"] == "first-axis-fastest"
    @test trace["inputs"][2]["values"] ==
          [string(reinterpret(UInt64, value); base=16, pad=16)
           for value in [0.75, 0.125, 0.25, 0.875]]
    @test trace["steps"][1]["inputs"] == [0, 1]
    @test trace["steps"][1]["product"]["values"] ==
          [string(reinterpret(UInt64, value); base=16, pad=16)
           for value in [0.1875, 0.09375, 0.0625, 0.65625]]
    @test graph.factors == before
    @test trace["result"]["values"] isa Vector{String}
    @test_throws ScopeError trace_variable_elimination(graph, :B; max_entries=1)
    @test_throws ScopeError trace_variable_elimination(graph, :B; evidence=Dict(:B => :off))
    # An entry below -atol is rejected when the graph is built; one within the tolerance
    # builds, and the trace's own non-negativity guard still rejects it.
    @test_throws FactorEntryError FactorGraph([Factor(a, [-0.1, 1.1])])
    @test_throws ScopeError trace_variable_elimination(FactorGraph([Factor(a,
                                                                           [-0.1, 1.1])];
                                                                   check=false), :A)
    @test_throws ScopeError trace_variable_elimination(FactorGraph([Factor(a,
                                                                           [-1e-12,
                                                                            1.0 + 1e-12])]),
                                                       :A)
    @test_throws ScopeError trace_variable_elimination(FactorGraph([Factor(a,
                                                                           [1 // 4, 3 // 4])]),
                                                       :A)
    _, _, observed = trace_variable_elimination(graph, :B; evidence=Dict(:A => :high))
    @test observed["conditioned"][1]["scope"] == String[]
    @test observed["conditioned"][1]["values"] == ["3fe8000000000000"]
    scalar, _, empty = trace_variable_elimination(graph, Symbol[];
                                                  evidence=Dict(:A => :low))
    @test scalar.table[] == 0.25
    @test empty["query"] == String[]
    @test empty["result"]["values"] == ["3fd0000000000000"]
end
