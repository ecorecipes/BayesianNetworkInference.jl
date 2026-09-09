# BayesianNetworkInference.jl

[![Build Status](https://github.com/ecorecipes/BayesianNetworkInference.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ecorecipes/BayesianNetworkInference.jl/actions/workflows/CI.yml)
[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://ecorecipes.github.io/BayesianNetworkInference.jl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Exact and approximate inference for compositional Bayesian networks: factor algebra, variable elimination with CliqueTrees.jl orderings, sampling, junction trees and belief propagation.

Part of the ecorecipes compositional Bayesian-network ecosystem:
`FiniteKernels.jl` → `BayesianNetworks.jl` → `BayesianNetworkInference.jl` →
`InfluenceDiagrams.jl`,
with `BayesianNetworkFormats.jl` (file formats) and `EcologicalBayesianNetworks.jl` (model zoo).

## Features

- `Factor`: unnormalised tensors over an ordered variable scope with state metadata from `FiniteKernels.jl`,
  and the algebra `multiply`, `marginalize`, `maximize`, `argmax_table`, `condition`, `normalize`, `reorder`.
  Factors and `FiniteKernel`s are distinct types with explicit conversions `Factor(k, inputs, output)` and
  `FiniteKernel(f, inputs, outputs)`: a kernel carries an input/output partition and a normalisation invariant,
  a factor is a symmetric tensor under pointwise product and summation.
- `FactorGraph`: a bag of factors with its interaction (moral) graph as a `Graphs.SimpleGraph` and per-factor
  provenance.
- The bridge to `BayesianNetworks.jl`: `compile(model; atol=DEFAULT_ATOL)` builds one factor per mechanism
  (first-occurrence parent/child scope, diagonal extraction for repeated input slots,
  provenance `(variable, mechanism, id)`), and `infer(model, query; evidence, backend, atol)`,
  `posterior(model, var)` and `ancestral_sample(model, n)` work directly on a `BayesModel`, including models read
  from Netica, GeNIe, HUGIN, BIF, DSC and UAI files and models rewritten by `observe`, `do_intervention` and
  `soft_intervention`. Every model-level answer is tested against the brute-force `marginal` of `BayesianNetworks.jl`.
- Elimination orderings from [CliqueTrees.jl](https://github.com/AlgebraicJulia/CliqueTrees.jl): `MinFill`
  (default), `MinDegree`, `AMDOrder`, `ExactTreewidth` (Bouchitte-Todinca), `UserOrder`; query variables are forced
  last with `CompositeRotations`; `treewidth` reports the induced width.
- `variable_elimination` / `infer` with evidence, returning the posterior factor and `InferenceDiagnostics`
  (order used, largest intermediate factor, number of products, induced width); `joint_factor` and
  `brute_force_marginal` as the oracle every optimised path is tested against.
- Opt-in `LogVariableElimination` uses centered log-domain factors and log-sum-exp.
  `log_evidence_probability` retains tiny positive evidence masses that underflow in
  ordinary arithmetic; `LogInferenceDiagnostics.mass_status` distinguishes that from
  truly impossible evidence. The default backend is unchanged.
- `JunctionTree` backend: the clique tree of the interaction graph from CliqueTrees.jl
  (`cliquetree(graph; alg, snd=Maximal())`, cached per factor-graph identity), calibrated by Shafer-Shenoy
  message passing; `all_marginals(fg; evidence)` (also on a `BayesModel`) and `clique_beliefs` from one
  calibration, `infer` for any query inside a clique (variable elimination otherwise, reported by
  `JunctionTreeDiagnostics.fallback`), `JunctionTreeDiagnostics`.
- `BeliefPropagation` backend: sum-product with damping, iteration limits and flooding/sequential
  schedules. `BPDiagnostics` reports the undamped fixed-point residual at the returned iterate;
  it is not a general marginal-error bound. Fixed points on feasible trees are exact, whereas
  loopy beliefs are approximations. Local zero support raises `KernelNormalizationError`,
  but does not detect every globally impossible event. Opt into
  `BeliefPropagation(check_evidence=true)` for an exact VE feasibility pass;
  `diagnostics.evidence_checked` distinguishes that from unknown feasibility.
- Exact VE/JT posterior entry points check global mass, including disconnected components and
  all-observed shortcuts. An empty `infer` query still returns unnormalized evidence mass,
  which may legitimately be zero. Integer factors promote to division-compatible posterior types.
- `ancestral_sample` over named kernels in topological order and `empirical_marginal` for Monte Carlo cross-checks.
- Out-of-sample validation: `Cases` datasets (a vector of `Dict{Symbol,Symbol}` observations, complete or
  partial), `predict` for a withheld target, the proper scoring rules `brier_score`, `log_score` and
  `spherical_score` (Brier 1950; Gneiting and Raftery 2007), `calibration_curve` / `calibration_error` (ECE),
  `roc_curve` / `auc`, `confusion_matrix` / `accuracy`, a marginal-prior `baseline`, `holdout` and `kfold`
  index splits (with `by` for grouped spatial or temporal holdout), and `evaluate` for the lot in one table.
- Sensitivity analysis: `entropy`, `mutual_information`, `sensitivity` (Marcot 2012's entropy-reduction
  ranking of every variable against a target) and `tornado` (the range one finding could move an answer).
- Typed exceptions (`ScopeError`, `ShapeError`, `CompileError`, plus `KernelNormalizationError` / `InvalidAxisError`
  from `FiniteKernels.jl`) carrying the offending variable names.

## Exact finite-model proof scope

The sibling `BayesianNetworks.jl/proofs/` project now proves posterior
normalization and evidence clamping, scoped bucket elimination, actual cached
Shafer-Shenoy collect/distribute passes, and d-separation soundness for the
moralized ancestral graph. Junction-tree hypotheses are structural running
intersection and complete factor/variable coverage; messages are computed, not
assumed correct. Arbitrary branching and disconnected forests are covered.

These are exact finite-model theorems, not verification of this package's
Float64 arrays, CliqueTrees construction or iterative BP. In a forest, Julia
stores component-local raw beliefs and a separate global mass; the formal
virtual-root encoding includes outside-component scalar factors in every
belief. Empty `infer` queries return unnormalized mass rather than a normalized
empty-query posterior. Conditional numerical bounds require an explicit
positive evidence-mass floor and a sufficiently small error budget.

`BayesianNetworks.proof_certificate(m)` exports ordered raw records and exact
bound scalars for a separate Lean data checker, before `compile` converts
factors to Float64. See the
[certificate guide](https://ecorecipes.github.io/BayesianNetworks.jl/certificates/).
No proof-assistant dependency is added to runtime inference.

## Installation

The ecosystem packages are not registered. Install this package and its ecosystem
dependencies by URL, in dependency order:

```julia
using Pkg
Pkg.add(url="https://github.com/ecorecipes/FiniteKernels.jl")
Pkg.add(url="https://github.com/ecorecipes/BayesianNetworkFormats.jl")
Pkg.add(url="https://github.com/ecorecipes/BayesianNetworks.jl")
Pkg.add(url="https://github.com/ecorecipes/BayesianNetworkInference.jl")
```

Requires Julia ≥ 1.12.

## Quick Start

Model level, on a `BayesianNetworks.jl` model (here the SPEC section 45 reference habitat network, read from
its Netica file; `reference_habitat_model()` builds the same model in code):

```julia
using BayesianNetworks, BayesianNetworkInference

m = read_bayesnet(fixture_path("dne/habitat_reference.dne"))
fg = compile(m)                                       # FactorGraph{Float64} with 7 factors over 7 variables
fg.provenance[1]                                      # (variable = :Climate, mechanism = :Climate_mechanism, id = 1)

post, diag = infer(m, :Occupancy)                     # P(Occupancy); diag.order, diag.treewidth
post.table                                            # [0.5238, 0.4762]
post ≈ Factor(marginal(m, :Occupancy), :Occupancy)    # true: BayesianNetworks' brute-force oracle agrees

posterior(m, :Occupancy; evidence=Dict(:Vegetation => :dense))   # Dict(:absent => 0.3325, :present => 0.6675)
posterior(observe(m, :Vegetation => :dense), :Occupancy)         # same: recorded evidence is merged in
posterior(do_intervention(m, :GrazingPressure => :low), :Occupancy)   # P(Occupancy | do(GrazingPressure = low))

s = ancestral_sample(m, 20_000)                       # AncestralSamples in topological order
isapprox(empirical_marginal(s, :Occupancy), post; atol=0.02)     # true
```

For rare evidence, select the log-domain backend explicitly:

```julia
post, diag = infer(m, :Occupancy; backend=LogVariableElimination())
log_mass = log_evidence_probability(m; evidence=:Vegetation => :dense)
```

The backend returns Float64 posterior cells. A posterior cell that is itself
unrepresentably small can still underflow; this is not a universal error bound.
Empty queries retain the ordinary unnormalized-mass API, with the log mass in
the diagnostics. Negative or nonfinite factor entries raise `LogFactorDomainError`.

Factor level, building the factors by hand:

```julia
using BayesianNetworkInference, FiniteKernels

rain = FiniteAxis(:Rain, [:yes, :no])
grass = FiniteAxis(:Grass, [:wet, :dry])

f_rain = Factor(cpt(rain, [0.2, 0.8]), :Rain)                       # scope (Rain,)
f_grass = Factor(cpt(rain, grass, [0.9 0.1; 0.2 0.8]), [:Rain], :Grass)   # scope (Rain, Grass)
fg = FactorGraph([f_rain, f_grass])

posterior, diag = infer(fg, [:Rain]; evidence=Dict(:Grass => :wet))
posterior.table                                       # P(Rain | Grass = wet) = [0.529, 0.471]
diag.order, diag.treewidth                            # (Symbol[], 0)

posterior ≈ brute_force_marginal(fg, [:Rain]; evidence=Dict(:Grass => :wet))   # true
elimination_order(fg, MinFill(); keep=[:Rain])        # [:Grass]

# Factor algebra
joint = multiply(f_rain, f_grass)                     # P(Rain, Grass), scope (Rain, Grass)
marginalize(joint, :Rain).table                       # P(Grass) = [0.34, 0.66]
FiniteKernel(f_grass, [:Rain], :Grass) ≈ cpt(rain, grass, [0.9 0.1; 0.2 0.8])   # true
```

## Vignettes

Rendered vignettes live in [`vignettes/`](vignettes/) and are published in the
[documentation](https://ecorecipes.github.io/BayesianNetworkInference.jl/).

## References

The algorithms this package implements, with the sources they come from. The same
entries, with the rest of the ecosystem's bibliography, are in
[`vignettes/references.bib`](vignettes/references.bib) and on the
[References page](https://ecorecipes.github.io/BayesianNetworkInference.jl/references/)
of the documentation.

- Dechter, R. (1999). Bucket elimination: a unifying framework for reasoning.
  *Artificial Intelligence* 113(1-2), 41-85.
  doi:[10.1016/S0004-3702(99)00059-4](https://doi.org/10.1016/S0004-3702(99)00059-4)
  -- variable elimination.
- Zhang, N. L. and Poole, D. (1994). A simple approach to Bayesian network computations.
  *Proceedings of the Tenth Canadian Conference on Artificial Intelligence*, 171-178
  -- eliminating one variable at a time by touching only the factors that mention it.
- Koller, D. and Friedman, N. (2009). *Probabilistic Graphical Models: Principles and
  Techniques*. MIT Press -- the factor presentation of variable elimination (ch. 9) and
  of clique trees (ch. 10).
- Shafer, G. R. and Shenoy, P. P. (1990). Probability propagation. *Annals of
  Mathematics and Artificial Intelligence* 2, 327-351.
  doi:[10.1007/BF01531015](https://doi.org/10.1007/BF01531015) -- the junction-tree
  calibration used by the `JunctionTree` backend.
- Lauritzen, S. L. and Spiegelhalter, D. J. (1988). Local computations with probabilities
  on graphical structures and their application to expert systems. *JRSS B* 50(2),
  157-224.
  doi:[10.1111/j.2517-6161.1988.tb01721.x](https://doi.org/10.1111/j.2517-6161.1988.tb01721.x)
  -- the division-based architecture, and the *asia* network used throughout.
- Jensen, F. V., Lauritzen, S. L. and Olesen, K. G. (1990). Bayesian updating in causal
  probabilistic networks by local computations. *Computational Statistics Quarterly* 4,
  269-282 -- the Hugin refinement of that architecture.
- Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems*. Morgan Kaufmann
  -- belief propagation.
- Kschischang, F. R., Frey, B. J. and Loeliger, H.-A. (2001). Factor graphs and the
  sum-product algorithm. *IEEE Transactions on Information Theory* 47(2), 498-519.
  doi:[10.1109/18.910572](https://doi.org/10.1109/18.910572) -- the factor-graph form
  implemented by `BeliefPropagation`.
- Murphy, K. P., Weiss, Y. and Jordan, M. I. (1999). Loopy belief propagation for
  approximate inference: an empirical study. *UAI 1999*, 467-475 -- what to expect of
  the loopy fixed point.
- Brier, G. W. (1950). Verification of forecasts expressed in terms of probability.
  *Monthly Weather Review* 78(1), 1-3.
  doi:[10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2](https://doi.org/10.1175/1520-0493(1950)078%3C0001:VOFEIT%3E2.0.CO;2)
  -- `brier_score`.
- Gneiting, T. and Raftery, A. E. (2007). Strictly proper scoring rules, prediction, and
  estimation. *JASA* 102(477), 359-378.
  doi:[10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)
  -- `log_score` and `spherical_score`, and why propriety is the property that matters.
- Marcot, B. G. (2012). Metrics for evaluating performance and uncertainty of Bayesian
  network models. *Ecological Modelling* 230, 50-62.
  doi:[10.1016/j.ecolmodel.2012.01.013](https://doi.org/10.1016/j.ecolmodel.2012.01.013)
  -- the entropy-reduction metric computed by `sensitivity`.
- Chen, S. H. and Pollino, C. A. (2012). Good practice in Bayesian network modelling.
  *Environmental Modelling & Software* 37, 134-145.
  doi:[10.1016/j.envsoft.2012.03.016](https://doi.org/10.1016/j.envsoft.2012.03.016)
  -- the life cycle in which validation and sensitivity are separate steps.
- Samuelson, R. and Fairbanks, J. `CliqueTrees.jl`: tree decompositions and elimination
  orderings in Julia. <https://github.com/AlgebraicJulia/CliqueTrees.jl> -- the
  fill-reducing orderings and clique trees behind every `EliminationStrategy`.
