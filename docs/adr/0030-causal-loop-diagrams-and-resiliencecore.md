# ADR 0030: CausalLoopDiagrams depends on ResilienceCore for qualitative stability

Date: 2026-11-17. Status: accepted.

## Context

A causal loop diagram read as the signed digraph of a dynamical system says something about
stability before any magnitude is known. Qualitative (loop) analysis asks how often a random
system with those signs is stable, and which way each variable responds to a sustained press
(Dambacher & Ramos-Jiliberto 2007; Melbourne-Thomas et al. 2012). Sign patterns, quasi-sign
stability and press-sign determinacy live in
[ResilienceCore.jl](https://github.com/ecorecipes/ResilienceCore.jl), next to the other
resilience metrics of the ecorecipes packages. The alternative at the time was
EcologicalResilience.jl, which is heavy (CSV, DataFrames, an ODE solver); it has since been
retired (2026-11-18). ADR 0029 made the same choice for FiniteKernels.

## Decision

`CausalLoopDiagrams` depends on `ResilienceCore`, with `[sources]` URLs for it and for
`StructuredPopulationCore`. It extends ResilienceCore's functions rather than defining its own:
`SignPattern(cld; self_regulation = :all_negative, collapse = true)` turns a diagram into a
`ResilienceCore.SignPattern`, and `quasi_sign_stability` and `press_sign_determinacy` then
apply unchanged.

- A link `A -> B` becomes the entry `[B, A]` (row: the affected factor; column: the acting
  factor). Parallel links are summed in the rig first, and lags are collapsed with
  `collapse_lags`.
- `INDETERMINATE` becomes a link of unknown sign, drawn with either sign in each sample;
  `NO_EFFECT` becomes no link. The two stay distinct, as the four-polarity invariant
  (ADR 0026) requires.

## Consequences

The no-Catlab, no-GATlab invariant holds (ADR 0009): ResilienceCore depends on `LinearAlgebra`,
`Random` and `StructuredPopulationCore` only. Installing the package needs Julia 1.11 or later
for `[sources]`. Tutorial 02 (qualitative stability) shows the analysis on diagrams.
