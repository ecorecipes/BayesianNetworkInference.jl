# ADR 0006: BayesianNetworks' test suite depends on BayesianNetworkInference

Date: 2026-09-07. Status: accepted.

## Context

`BayesianNetworks.jl/test/test_dynamic.jl` checks `unroll` on a long horizon. The package's own reference semantics
(`joint_distribution`, `marginal`) enumerate the joint, so they cannot reach horizon 8 of the vegetation/herbivore
template (16 variables); the check that "marginals of a prefix do not depend on the horizon" needs variable
elimination, which lives one layer up in `BayesianNetworkInference.jl`. The test therefore imports
`BayesianNetworkInference.infer` and compares it against brute force on horizon 4.

That inverts the layering of ADR 0001: Inference depends on BayesianNetworks, so the test edge closes a cycle.
`BayesianNetworkInference` is declared in `[extras]`, listed in `[targets] test`, and given a `[sources]` path in
`BayesianNetworks.jl/Project.toml`, but `BayesianNetworks.jl/.github/workflows/CI.yml` checked out only
MarkovCategories and BayesianNetworkFormats, so `Pkg.test` failed on a fresh CI clone. SPEC §60 rule 23 requires
deviations to be recorded; this record is that deviation.

The alternative considered was to drop the extra and have `test/test_dynamic.jl` skip the two cross-checks when the
package is unavailable (`Base.find_package`, or a `try import`). That keeps the layering clean at the cost of a check
that then runs nowhere: BayesianNetworks' own CI would never see it, and no other package's suite covers `unroll`.

## Decision

Keep the test-only dependency and make CI provide it. `BayesianNetworks.jl/.github/workflows/CI.yml` now checks out
`ecorecipes/BayesianNetworkInference.jl` alongside MarkovCategories and BayesianNetworkFormats in both the `test` and
the `docs` jobs, so the `[sources]` path `../BayesianNetworkInference.jl` resolves exactly as it does in a local
side-by-side workspace.

The cycle is resolvable. `Pkg.resolve` was run on 2026-09-07 (Julia 1.12.7) in an environment built from copies of
the four packages with every `Manifest.toml` removed, with `[sources]` paths for all four including both directions
of the cycle; it produced a 131-package manifest with all four resolved to their paths and no error or warning. A
`[sources]` entry whose path does not exist is ignored when the package is not in the dependency graph, so the docs
environment (whose `docs/Project.toml` does not list Inference) resolves with or without the sibling; this was
checked the same way.

Consequences of the cycle are contained to development-time layout: `[extras]`/`[targets]` test dependencies are not
recorded in registry metadata, so the cycle does not exist for a consumer who installs either package, and neither
package's `[deps]` changes.

## Consequences

- Both repositories must be present to run either suite, which is already true of every other sibling relationship in
  this workspace (`[sources]` paths, ADR 0001).
- BayesianNetworks' CI is coupled to `ecorecipes/BayesianNetworkInference.jl@main`: a change that breaks `infer` reds
  BayesianNetworks' CI as well as its own. That is the intended signal — the two implementations are supposed to
  agree — but it means the two repositories cannot be released from unrelated states.
- `Pkg.test()` in a lone clone of `BayesianNetworks.jl` fails until the sibling is cloned next to it. `CONTRIBUTING.md`
  already instructs contributors to clone the ecosystem side by side.
- If the cycle ever becomes an obstacle (for example at registration, or if Pkg tightens path-source resolution), the
  fallback is the alternative above: guard the `variable elimination on a longer horizon` testset with a
  `Base.find_package("BayesianNetworkInference") === nothing` skip and drop the extra.
