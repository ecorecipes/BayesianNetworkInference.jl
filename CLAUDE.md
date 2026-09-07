# BayesianNetworkInference.jl

Exact and approximate inference for compositional Bayesian networks: factor algebra, variable elimination with CliqueTrees.jl orderings, sampling, junction trees and belief propagation.

## Place in the ecosystem

Dependency order (arrows = depends on):
EcologicalBayesianNetworks → InfluenceDiagrams → BayesianNetworkInference → BayesianNetworks →
FiniteKernels, and BayesianNetworks → BayesianNetworkFormats.
This package depends on: BayesianNetworks, FiniteKernels and Graphs (BayesianNetworkFormats
only as a test/docs/vignette dependency, for `fixture_path`). It has no Catlab in its
dependency graph since ADR 0009: `MarkovCategories` was dropped when `evaluate` became this
package's own function. Sibling packages are expected at `../<Name>.jl` (see `[sources]` in Project.toml,
`vignettes/Project.toml` and `docs/Project.toml`; the CI jobs check all three siblings out).

## Invariants that must not be broken

- Structural syntax (ACSets) and numerical semantics (kernels, utilities) stay separate; CPT arrays are never ACSet attributes.
- Parent / input order is explicit (`input_position`) and total. Never rely on part-id order.
- Axis conventions: user-facing CPTs are `(parents..., child)` normalised over the last axis; FinStoch kernels internally are outputs-first. Convert with the documented `permutedims`, never by hand.
- Observation (`observe`) and intervention (`do_intervention`) are different operations and stay different.
- Every optimised path is checked against a slower oracle (`joint_distribution`, exhaustive policy search) on small models.
- `Factor` and `FiniteKernel` are distinct types with explicit conversions (`Factor(k, inputs, output)`,
  `FiniteKernel(f, inputs, outputs)`). A kernel carries an input/output partition and a normalisation invariant;
  a factor is an unnormalised symmetric tensor under (×, Σ). Never treat one as the other implicitly.
- `Factor(k, inputs, output)` has scope `(inputs..., output)` and table `cpt(k)` (parents-first); the asia test
  `P(dysp = yes) = 0.4360` pins this against `BayesianNetworkFormats`' independent value.
- `variable_elimination` is checked against `brute_force_marginal` (product of all factors) on asia, the SPEC section 45
  habitat chain and random DAGs; ancestral sampling is checked against it within Monte Carlo tolerance;
  `JunctionTree` and tree-exact `BeliefPropagation` are checked against VE per variable (1e-10) and loopy BP on
  asia within a loose tolerance with `converged` asserted.
- `compile(::BayesModel)` emits one factor per mechanism, `Factor(kernel(m, X), parents, X)` with parents in
  `input_position` order, in topological order of the targets; `infer(m, ...)` on a model is checked against
  `BayesianNetworks.marginal` (the joint-table oracle) on the reference model, asia, intervened models and random models.
- `infer(m, query; evidence)` merges the explicit evidence into the evidence recorded by `observe` (explicit wins);
  interventions are mechanism rewrites and compile unchanged (point masses via `BayesianNetworks.kernel`).
- Names shared with BayesianNetworks.jl (`variables`, `axis`, `empirical_marginal`), CliqueTrees.jl (`treewidth`)
  and Graphs.jl (`is_tree`) are extended with `import ...: ...`, never shadowed, so `using BayesianNetworks, BayesianNetworkInference` stays
  unambiguous. Never redefine `BayesianNetworks.sample(::BayesModel, n)`; the sampler here is `ancestral_sample(m, n)`.

## Layout

- `src/factors.jl`: `Factor`, `ScopeError`, `ShapeError`, `unit_factor`, `scope`, `axis`, `multiply`, `marginalize`,
  `maximize`, `argmax_table`, `condition`, `normalize` (extends `LinearAlgebra.normalize`), `reorder`, kernel conversions.
- `src/factor_graph.jl`: `FactorGraph` (factors, axes, provenance), `variables`, `interaction_graph` (Graphs.SimpleGraph
  plus variable/vertex maps; already the moral graph).
- `src/orderings.jl`: `EliminationStrategy` types `MinFill` (CliqueTrees `MF`), `MinDegree` (`MMD`), `AMDOrder` (`AMD`,
  needs `import AMD`), `ExactTreewidth` (`BT`, needs `import TreeWidthSolver`; run per connected component because BT
  fails on isolated vertices), `UserOrder`; `elimination_order(fg, strategy; keep)` via
  `permutation(graph; alg=CompositeRotations(keep_idx, alg))`, `treewidth(fg, strategy)`.
- `src/variable_elimination.jl`: `VariableElimination` backend, `InferenceDiagnostics`, `variable_elimination`,
  `infer`, oracle `joint_factor` / `brute_force_marginal`.
- `src/junction_tree.jl`: `JunctionTree(; order)` backend, `CompiledJunctionTree` from
  `CliqueTrees.cliquetree(graph; alg, snd=Maximal())` (`residual`/`separator`, `parentindex`/`childindices`/
  `rootindices`; cliques are numbered in the permuted order, `label[v]` maps back), cached in a `Dict` keyed by
  `objectid(fg.factors)` with a `WeakRef` checked by `===` (identity, not content); `calibrate` (Shafer-Shenoy: collect then distribute, no division, so zero entries are
  safe), `clique_beliefs`, `all_marginals` (JT in one pass; VE method loops per variable; evidence variables
  are point masses), `infer` for one-clique queries with a VE fallback that warns and sets
  `JunctionTreeDiagnostics.fallback` (the return type never changes), `JunctionTreeDiagnostics`.
- `src/belief_propagation.jl`: `BeliefPropagation(; damping, tol, maxiter, schedule)`, `BPDiagnostics`,
  `belief_propagation(fg, backend; evidence)` (sum-product, SPEC section 20; `:flooding` or `:sequential`),
  `is_tree` (extends `Graphs.is_tree`; forest check of the bipartite factor graph; `BPDiagnostics.tree` reports it per run after conditioning),
  `infer` for single-variable queries only (`ScopeError` otherwise).
- `src/sampling.jl`: `AncestralSamples`, `ancestral_sample(kernels, order, parents, n; rng)`, `empirical_marginal`.
- `src/compile.jl`: `FactorGraphBackend`, `CompileError` (open model / missing kernels, naming the variables),
  `compile(m::BayesModel)`, internal `_model_kernels(m)` shared with sampling.
- `src/model_inference.jl`: `infer(m::BayesModel, query; evidence, backend)`, `posterior(m, var)` (state => probability),
  `all_marginals(m; evidence, backend)`,
  `ancestral_sample(m, n; rng)`, `AncestralSamples(m, samples)` (converts `BayesianNetworks.sample` output),
  `empirical_marginal(m, samples, vars)`.
- `src/scores.jl`: `Case` (= `Dict{Symbol,Symbol}`) and `Cases` (an `AbstractVector{Case}`, also built from
  `AncestralSamples`), `Predictions`, `predict(m, cases, target; evidence_vars, backend)` (compiles once,
  withholds the target, merges the model's recorded evidence), `baseline` (the marginal prior),
  the proper scoring rules `brier_score` (multi-category, negatively oriented), `log_score` (positively
  oriented, `floor` keyword) and `spherical_score`, `calibration_curve` / `CalibrationCurve` /
  `calibration_error` (ECE against the mean predicted probability per bin), `roc_curve` / `ROCCurve` / `auc`
  (rank-based with mid-ranks, and the trapezoidal method on a curve), `confusion_matrix` / `ConfusionMatrix` /
  `accuracy` (maximum a posteriori), `holdout` / `kfold` (index splits, `by` groups whole spatial or temporal
  units; they never refit CPTs), and `evaluate` / `EvaluationResult` with a two-column `show`.
  `evaluate` is this package's own function. It used to extend `MarkovCategories.evaluate`,
  which `BayesianNetworks` re-exported; since ADR 0009 the categorical layer is
  `CategoricalBayesianNetworks.jl` and the two functions are unrelated, so
  `using BayesianNetworkInference, CategoricalBayesianNetworks` clashes on the name and the
  user qualifies whichever they mean. This is deliberate: extending a Markov-category
  functor with a scoring method was always a pun on the word.
- `src/sensitivity.jl`: `entropy(m, x; evidence, base)`, `mutual_information(m, x, y; evidence)` from the joint
  posterior, `sensitivity(m, target)` (Marcot 2012's entropy-reduction ranking) and `tornado(m, target, state)`
  (sensitivity to findings, not to parameters).
- `test/networks.jl`: asia from its published CPTs, SPEC section 45 habitat chain, random DAG generator (seeded);
  `test/test_{factors,orderings,inference,sampling}.jl`; `test/test_model_bridge.jl` (reference model, asia from
  `fixture_path("bif/asia.bif")`, interventions, 15 seeded random `BayesModel`s, sampling);
  `test/test_junction_tree.jl` (JT versus VE and brute force on asia, the habitat chain, random DAGs, forests,
  the 30-variable `benchmark_network`, and the bnlearn `water` model when `ECOLOGICAL_BN_SLOW=true` and the
  sibling zoo is checked out); `test/test_belief_propagation.jl` (exact on chains and polytrees, loopy on asia, and impossible
  evidence raising `KernelNormalizationError` in all three backends); `test/test_scores.jl`
  (hand-computed Brier / log / spherical scores on an enumerable two-variable network, scores against
  brute-force enumeration on random DAGs, calibrated and miscalibrated synthetic generators, AUC 1.0 and
  0.5, split invariants); `test/test_sensitivity.jl` (zero mutual information for d-separated variables,
  the entropy-reduction identity on a chain).
- `vignettes/`: 01 factors and VE (ends with the model-level Demo 1: `.dne` -> `infer` -> `do` -> `.xdsl`),
  02 orderings, 03 sampling and Monte Carlo checks, 04 junction trees and belief propagation,
  05 validation and scoring (simulate cases, grouped holdout, calibration, prior baseline, mutual information).
- Later milestones: `ext/` adapters for external backends.

## Commands

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'   # the test suite
julia --project=docs docs/make.jl                                 # build docs locally
cd vignettes && quarto render                                     # render vignettes to html/gfm/pdf (julia engine; PDF needs lualatex + ../fonts/JuliaMono)
julia scripts/sync_vignettes.jl [--check]                         # copy vignettes into docs/src/tutorials
```

## Files not to edit by hand

- `docs/src/tutorials/` is generated by `scripts/sync_vignettes.jl`.
- `vignettes/*/*.md`, `*.html`, `*.pdf` and `*_files/` are quarto output; edit the `.qmd`.
- `proofs/schemas/*.json` (where present) is emitted by the Lean project; edit the Lean source.

## Style

JuliaFormatter `yas`; docstrings on every exported name; typed exceptions with variable names in the message;
no emojis in code or docs.
