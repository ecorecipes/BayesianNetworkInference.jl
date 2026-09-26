# Every exported name this package owns has a docstring (CLAUDE.md, "Style"). A name
# re-exported from another package is that package's to document, so it is skipped.
# `Base.which` names the module that owns a binding; `parentmodule` would throw on
# constants and `Union`s.
function owned_undocumented(M; allow=Symbol[])
    return filter(n -> Base.which(M, n) === M && n ∉ allow, Base.Docs.undocumented_names(M))
end

@testset "docstrings on every exported name" begin
    @test isempty(owned_undocumented(BayesianNetworkInference))
end
