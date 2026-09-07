# ADR 0001: Six focused packages instead of the SPEC's two

Date: 2026-09-06. Status: accepted.

## Context

`docs/SPEC.md` §3 proposes `CategoricalDecisionModels.jl` and `ProbabilisticDecisionInference.jl`. The user wants an
ecosystem under the `ecorecipes` org in which each package has its own quarto vignettes, and the ecosystem must also
read interchange formats and ship a model zoo. The SPEC places finite kernels in the inference package, but structural
validation (kernel shapes, normalisation) needs kernels *below* the syntax layer.

## Decision

Six repositories, developed as sibling directories of one workspace and published as `ecorecipes/<Name>.jl`
(arrows = depends on):

```
EcologicalBayesianNetworks → InfluenceDiagrams → BayesianNetworkInference → BayesianNetworks → MarkovCategories
                                                                             BayesianNetworks → BayesianNetworkFormats
```

`MarkovCategories.jl` is the base (finite stochastic kernels + GATlab theory). `BayesianNetworkFormats.jl` is a
Catlab-free leaf (see ADR 0003). `BayesianNetworks.jl` owns the reference semantics (`joint_distribution`, `sample`) so
Props 1-4 are testable without an inference dependency. Decision variable elimination lives in `InfluenceDiagrams.jl`.
Names follow the org convention: descriptive nouns, no prefix, module name equals repo name. None of the six names
exist in the General registry (checked 2026-09-06).

## Consequences

More repositories and CI sibling checkouts (0, 0, 2, 3, 4, 5 upstream siblings respectively), offset by lighter
dependency sets per package and vignettes grouped by audience. SPEC §3 and §63 are superseded by this record.
