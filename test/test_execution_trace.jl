@testset "Actual VE execution capture" begin
    a, b = FiniteAxis(:A, [:low, :high]), FiniteAxis(:B, [:off, :on])
    graph = FactorGraph([Factor(a, [.25, .75]), Factor([a, b], [.75 .25; .125 .875])])
    before = deepcopy(graph.factors)
    actual, diagnostics, trace = trace_variable_elimination(graph, :B)
    ordinary, old_diagnostics = variable_elimination(graph, :B)
    @test actual == ordinary
    @test diagnostics.order == old_diagnostics.order
    @test trace["layout"] == "first-axis-fastest"
    @test trace["inputs"][2]["values"] == [string(reinterpret(UInt64, value); base=16, pad=16)
                                          for value in [.75, .125, .25, .875]]
    @test trace["steps"][1]["inputs"] == [0, 1]
    @test trace["steps"][1]["product"]["values"] == [string(reinterpret(UInt64, value); base=16, pad=16)
                                                    for value in [.1875, .09375, .0625, .65625]]
    @test graph.factors == before
    @test trace["result"]["values"] isa Vector{String}
    @test_throws ScopeError trace_variable_elimination(graph, :B; max_entries=1)
    @test_throws ScopeError trace_variable_elimination(graph, :B; evidence=Dict(:B => :off))
    @test_throws ScopeError trace_variable_elimination(FactorGraph([Factor(a, [-.1, 1.1])]), :A)
    @test_throws ScopeError trace_variable_elimination(FactorGraph([Factor(a, [1//4, 3//4])]), :A)
    _, _, observed = trace_variable_elimination(graph, :B; evidence=Dict(:A => :high))
    @test observed["conditioned"][1]["scope"] == String[]
    @test observed["conditioned"][1]["values"] == ["3fe8000000000000"]
    scalar, _, empty = trace_variable_elimination(graph, Symbol[]; evidence=Dict(:A => :low))
    @test scalar.table[] == .25
    @test empty["query"] == String[]
    @test empty["result"]["values"] == ["3fd0000000000000"]
end
