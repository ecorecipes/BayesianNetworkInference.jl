# ADR 0029: FiniteKernels depends on ResilienceCore

Date: 2026-11-17. Status: accepted.

## Context

An endokernel `X → X` is a Markov chain. Its resilience is the rate at which a distribution
returns to the stationary one: the spectral gap `-log |λ₂|`, the stationary distribution and
the mixing time. These metrics already live in
[ResilienceCore.jl](https://github.com/ecorecipes/ResilienceCore.jl), the ecorecipes package
that dispatches resilience metrics on what a matrix means. Its `MarkovKernel` stores a
row-stochastic matrix, while this package stores kernels outputs first (ADR 0002), so the
two meet through a transpose that is easy to get wrong by hand.

Until now this package depended on `LinearAlgebra` and `Random` only (ADR 0001, CLAUDE.md).
The options were a ResilienceCore extension, an extension in this package with ResilienceCore
as a weak dependency, or a hard dependency. The owner chose the hard dependency
(2026-11-17), in line with the other ecorecipes model packages that implement ResilienceCore's
interface themselves.

## Decision

`FiniteKernels` depends on `ResilienceCore`, with `[sources]` URLs for it and for
`StructuredPopulationCore`, its dependency. Pkg does not follow a URL dependency's own
`[sources]`, so `StructuredPopulationCore` is listed directly. `src/resilience.jl` extends
ResilienceCore's functions and defines no new names:

- `linearise(k)`: the `MarkovKernel` of an endokernel, `P = kernel_matrix(k)'`, rows in
  `joint_states` order, with negative entries smaller than `atol` set to zero and rows
  rescaled to sum to exactly one;
- `stationary_distribution(k)`: a labelled state `I → X`, not a bare vector;
- `resilience(k)` and `mixing_time(k)`, forwarded to the `MarkovKernel`.

A kernel whose domain and codomain differ raises `SpaceMismatchError`, as it has no dynamics.

ResilienceCore also exports `state`, the abundance vector of its community interface. To
avoid an ambiguous name when both packages are loaded, this package's `state` methods now
extend `ResilienceCore.state`. The signatures do not overlap, so behaviour is unchanged and
`FiniteKernels.state` keeps working for downstream packages. `Base.which(FiniteKernels,
:state)` is now `ResilienceCore`.

## Consequences

The no-Catlab, no-GATlab invariant is unaffected: ResilienceCore depends on
`LinearAlgebra`, `Random` and `StructuredPopulationCore` only. The dependency tree grows from
two standard libraries to eleven packages. `Project.toml` now has a `[sources]` section, so
installing the package needs Julia 1.11 or later (already required by ADR 0004). Downstream
packages that load both this package and ResilienceCore see one `state` function.
