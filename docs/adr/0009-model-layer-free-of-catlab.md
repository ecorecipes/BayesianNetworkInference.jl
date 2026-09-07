# ADR 0009: Separate the model layer from the category-theory layer

Date: 2026-09-07. Status: accepted. Supersedes the package split of ADR 0001.

## Context

The `ecorecipes` organisation keeps a clean separation between packages that implement models and the
package that adds category theory over them. `CategoricalPopulationDynamics.jl` depends on the model
packages and holds all of the Catlab machinery; `ProjectionModels.jl` (module `StructuredPopulationCore`)
is a small shared base; and `IntegralProjectionModels.jl` goes further still, keeping Catlab as a
`[weakdeps]` behind `IntegralProjectionModelsCatlabExt`, so the model works without category theory and
gains it when asked.

The six-package split of ADR 0001 did not follow that convention. It applied the idea in exactly one
place, making `BayesianNetworkFormats.jl` a Catlab-free leaf, and welded the two layers together
everywhere else. Measured before this record was written:

| Package | Source lines | Lines in files with no categorical reference |
|---|---|---|
| `BayesianNetworkInference.jl` | 3278 | 3278 (100%) |
| `InfluenceDiagrams.jl` | 4210 | ~3280 (78%) |
| `BayesianNetworks.jl` | 6735 | ~3565 (53%) |
| `MarkovCategories.jl` | 1107 | ~750 (68%) |

The clearest symptom: `BayesianNetworkInference.jl` contained no reference to Catlab, ACSets or GATlab
across 3278 lines of factor algebra, variable elimination, junction trees, belief propagation and
scoring, yet could not be loaded without Catlab, because `MarkovCategories.jl` needed
`Catlab.Theories.ThMonoidalCategoryWithDiagonals` and everything depended on it transitively.

The root cause was a design decision, not an accident of file placement. Making the attributed C-set the
sole representation of a network (SPEC §8) left no Catlab-free model layer to factor out. The
organisation's model packages have native representations first, and the C-set appears only in the
categorical layer.

## Decision

Split along one mechanical rule: **a module that needs Catlab lives in a categorical package; a module
for which ACSets and GATlab suffice lives in the model layer.** The result is eight packages:

```
FiniteKernels.jl                 finite spaces and kernels, cpt, compose/tensor/copy/discard
  ├── MarkovCategories.jl        ThMarkovCategory, FreeMarkovCategory, the FinStoch instance
  └── BayesianNetworks.jl        schemas, semantics, evaluation, interventions, dynamics,
       │                         validation, model cards, formats bridge, CatColab export
       ├── BayesianNetworkInference.jl    factors, variable elimination, junction trees, belief
       │                                  propagation, scoring and sensitivity
       ├── InfluenceDiagrams.jl           decisions, information, utilities, policies, decision
       │                                  variable elimination, value of information
       └── CategoricalBayesianNetworks.jl open networks as structured cospans, composition,
                                          wiring diagrams, free Markov-category semantics
```

with `BayesianNetworkFormats.jl` a leaf below the model layer and `EcologicalBayesianNetworks.jl` the
application package on top. `FiniteKernels.jl` is the analogue of `ProjectionModels.jl`;
`CategoricalBayesianNetworks.jl` is the analogue of `CategoricalPopulationDynamics.jl`.

Catlab appears in the dependency graph of exactly two packages, `MarkovCategories.jl` and
`CategoricalBayesianNetworks.jl`. `FiniteKernels.jl` resolves to eight manifest entries and depends only
on `LinearAlgebra` and `Random`.

Three consequences of the rule were not obvious in advance and are recorded because they constrain future
work:

1. `@present Sch(FreeSchema)` needs Catlab, because `FreeSchema` lives in `Catlab.Theories` even though
   `@present` comes from GATlab. The schemas are therefore built as ACSets `BasicSchema` values in the
   same generator order. The emitted schema JSON is byte-identical, which the Lean project's
   `emit_schema --check` verifies.
2. Visualisation would otherwise have become a Catlab-only feature. `BayesianNetworks.jl` now carries a
   small `Graphviz` submodule that emits DOT directly and renders through `Graphviz_jll`, mirroring
   Catlab's API so the categorical package and the influence-diagram package extend the same
   `to_graphviz` generic.
3. Graph derivation moved from Catlab's `Graph`/`SymmetricGraph` to Graphs.jl, which the inference
   package already depended on.

## Consequences

`BayesianNetworks.jl`, `BayesianNetworkInference.jl` and `InfluenceDiagrams.jl` are free of Catlab. A user
who wants to read a network from a file, bind conditional probability tables, run variable elimination and
perform an intervention no longer pays for the categorical stack; a user who wants to compose open
networks adds one package.

Costs accepted: eight repositories rather than six; `evaluate` is now owned by
`BayesianNetworkInference.jl` and clashes with the Markov-category functor of the same name if both are
loaded, which the user resolves by qualifying; and `canonicalize`/`is_isomorphic` use a local exact search
over name-tied variable orderings instead of Catlab's homomorphism search.

One known mismatch is left open deliberately. The Lean project in `BayesianNetworks.jl/proofs/` proves the
open-network closure theorem and Proposition 3, results about the layer that now lives in
`CategoricalBayesianNetworks.jl`, while also emitting the schema JSON that `BayesianNetworks.jl` tests
against. It genuinely serves both packages, and moving a Lake project that shares a packages directory is
a separate decision. It stays where it is, noted in both packages' guidance files.
