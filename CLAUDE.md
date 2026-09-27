# BayesianNetworkInference.jl

Exact and approximate inference for compositional Bayesian networks: factor algebra, variable elimination with CliqueTrees.jl orderings, sampling, junction trees and belief propagation.

## Place in the ecosystem

Dependency order (arrows = depends on):
EcologicalBayesianNetworks → InfluenceDiagrams → BayesianNetworkInference → BayesianNetworks →
FiniteKernels, and BayesianNetworks → BayesianNetworkFormats.
This package depends on: BayesianNetworks, FiniteKernels, Graphs, CliqueTrees, AMD and
TreeWidthSolver (all hard dependencies -- AMD and TreeWidthSolver back the `AMDOrder` and
`ExactTreewidth` strategies and are `import`ed, not optional), plus SparseArrays, LinearAlgebra
and Random from the standard library. BayesianNetworkFormats is only a
test/docs/vignette dependency, for `fixture_path`. It has no Catlab in its
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
- `Factor(k, inputs, output)` has first-occurrence scope order and the diagonal of `cpt(k)` when input
  slots repeat. It never drops a repeated name without identifying its table indices. The asia test
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
- Every model-facing inference, sampling, scoring and sensitivity entry point forwards
  `atol=BayesianNetworks.DEFAULT_ATOL` to compilation/semantic validation.
- `_union_axes` and `_broadcastable` in `factors.jl` are also consumed by InfluenceDiagrams'
  valuation arithmetic. Keep their axis-agreement and first-occurrence ordering contracts stable.
- Exact posterior entry points reject globally zero mass before returning component marginals or
  observed point masses. Empty inference queries retain their unnormalized mass convention.
  The rejection is `BayesianNetworks.ImpossibleEvidenceError` (re-exported here, never redefined),
  raised through `_require_evidence_mass` / `_posterior_normalize` in `variable_elimination.jl`
  with the test `mass <= 0`, so NaN is not impossibility (ADR 0012). `normalize(::Factor)` on a
  zero total is an `ArgumentError`; `KernelNormalizationError` means only kernel columns.
- Every exception type this package defines lives in `src/errors.jl` and subtypes `InferenceError <:
  BayesianNetworks.BayesNetError` (ADR 0013). `ScopeError` is a frozen name: the pinned
  constructor-schedules checker (`NativeOrderRecords.v`) compares the recorded `string(typeof(e))` with
  `"ScopeError"`, so it keeps its name, its defining module (this one) and its unqualified printing
  under `using BayesianNetworks, BayesianNetworkInference`. Never move it, rename it or export a second
  binding. Re-exports are the owners' own bindings: every exception type FiniteKernels exports, and
  BayesianNetworks' `BayesNetError`, `AnyBayesNetError` and `ImpossibleEvidenceError`
  (`test/test_errors.jl` checks for drift). Of BayesianNetworkFormats only the root
  `BayesianNetworkFormatsError` is re-exported, never its concrete types: the conformance adapters
  load this package with `using`, and the inspect adapter records Formats' types under their
  qualified names.
- BP convergence uses the undamped residual at the returned iterate, not the damped step and not a
  marginal-error bound. `check_evidence=true` opts into VE feasibility; `evidence_checked=false`
  leaves global feasibility unknown. Integer inputs promote to division-compatible types.
- Names shared with BayesianNetworks.jl (`variables`, `axis`, `empirical_marginal`), CliqueTrees.jl (`treewidth`)
  and Graphs.jl (`is_tree`) are extended with `import ...: ...`, never shadowed, so `using BayesianNetworks, BayesianNetworkInference` stays
  unambiguous. Never redefine `BayesianNetworks.sample(::BayesModel, n)`; the sampler here is `ancestral_sample(m, n)`.

## Layout

- `src/errors.jl` (included first): the root `InferenceError <: BayesNetError` and the four exception types
  with their `showerror` methods: `ScopeError` and `ShapeError` (factor scopes and table shapes, and the
  argument checks of the entry points built on them), `CompileError` (open model / missing kernels, naming
  the variables) and `LogFactorDomainError` (a factor entry the log domain cannot take).
- `src/factors.jl`: `Factor`, `unit_factor`, `scope`, `axis`, `multiply`, `marginalize`,
  `maximize`, `argmax_table`, `condition`, `normalize` (extends `LinearAlgebra.normalize`), `reorder`, kernel conversions.
- `src/factor_graph.jl`: `FactorGraph` (factors, axes, provenance), `variables`, `interaction_graph` (Graphs.SimpleGraph
  plus variable/vertex maps; already the moral graph).
- `src/orderings.jl`: `EliminationStrategy` types `MinFill` (CliqueTrees `MF`), `MinDegree` (`MMD`), `AMDOrder` (`AMD`,
  needs `import AMD`), `ExactTreewidth` (`BT`, needs `import TreeWidthSolver`; run per connected component because BT
  fails on isolated vertices), `UserOrder`; `elimination_order(fg, strategy; keep)` via
  `permutation(graph; alg=CompositeRotations(keep_idx, alg))`, `treewidth(fg, strategy)`.
- `src/variable_elimination.jl`: `VariableElimination` backend, `InferenceDiagnostics`, `variable_elimination`,
  and the elimination-order cache: `graph`/`vars`/`index` exist only to produce `elim`, so the order is what
  is cached, keyed on `objectid(fg.factors)` plus the evidence's *keys*, the strategy and the query (the
  conditioned scopes depend on which variables are observed, not on their values, so `predict` shares one
  entry across every case). Same `WeakRef`/`===` discipline as the junction-tree cache, and the same rule:
  do not mutate `fg.factors` after a query.
  `infer`, oracle `joint_factor` / `brute_force_marginal`.
- `src/log_variable_elimination.jl`: `LogVariableElimination` backend,
  `LogInferenceDiagnostics`, `log_variable_elimination`,
  `log_evidence_probability`. The same elimination algorithm as `variable_elimination.jl`
  carried out in the log domain, for models whose joint underflows Float64. Keep the two in
  step: they are written out twice, and a fix applied to one and not the other is how
  `normalize`'s empty-scope bug came about.
- `src/log_junction_tree.jl`: `LogJunctionTree`, `LogCalibratedJunctionTree`,
  `LogJunctionTreeDiagnostics`, `log_calibrate`. The log-domain counterpart of
  `junction_tree.jl`, with the same caveat.
- `src/execution_trace.jl`: `trace_variable_elimination`, which records each product and
  marginalisation of a run as data for the external conformance checker. The trace's
  `inputs` are in factor-graph order, which is *not* the association order of the recorded
  product (`_product!` sorts `touching` by `ndims` first); the format does not record that
  order, and ADR 0011 notes that reassociation alone changes the answer.
- `src/junction_tree.jl`: `JunctionTree(; order)` backend, `CompiledJunctionTree` from
  `CliqueTrees.cliquetree(graph; alg, snd=Maximal())` (`residual`/`separator`, `parentindex`/`childindices`/
  `rootindices`; cliques are numbered in the permuted order, `label[v]` maps back), cached in a `Dict` keyed by
  `objectid(fg.factors)` with a `WeakRef` checked by `===` (identity, not content); `calibrate` (Shafer-Shenoy: collect then distribute, no division, so zero entries are
  safe), `clique_beliefs`, `all_marginals` (JT in one pass; VE method loops per variable; evidence variables
  are point masses), `infer` for one-clique queries with a VE fallback that warns and sets
  `JunctionTreeDiagnostics.fallback` (the return type never changes), `JunctionTreeDiagnostics`.
- `src/belief_propagation.jl`: `BeliefPropagation(; damping, tol, maxiter, schedule, check_evidence)`, `BPDiagnostics`,
  `belief_propagation(fg, backend; evidence)` (sum-product, SPEC section 20; `:flooding` or `:sequential`),
  `is_tree` (extends `Graphs.is_tree`; forest check of the bipartite factor graph; `BPDiagnostics.tree` reports it per run after conditioning),
  `infer` for single-variable queries only (`ScopeError` otherwise).
- `src/sampling.jl`: `AncestralSamples`, `ancestral_sample(kernels, order, parents, n; rng)`, `empirical_marginal`.
- `src/compile.jl`: `FactorGraphBackend`, `compile(m::BayesModel)` (`CompileError` for an open model or
  missing kernels), internal `_model_kernels(m)` shared with sampling.
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
  evidence raising `ImpossibleEvidenceError` in all three backends); `test/test_scores.jl`
  (hand-computed Brier / log / spherical scores on an enumerable two-variable network, scores against
  brute-force enumeration on random DAGs, calibrated and miscalibrated synthetic generators, AUC 1.0 and
  0.5, split invariants); `test/test_sensitivity.jl` (zero mutual information for d-separated variables,
  the entropy-reduction identity on a chain); `test/test_log_inference.jl` and
  `test/test_log_junction_tree.jl` (the log-domain backends against their linear counterparts,
  and on models whose joint underflows Float64); `test/test_execution_trace.jl`;
  `test/test_regressions.jl` (the ADR 0011 suite: the obstructions that composition and
  reassociation are known to have, pinned so they cannot be quietly "fixed"; and ADR 0012's
  matrix: one impossible model gives `ImpossibleEvidenceError` from every entry point);
  `test/test_errors.jl` (every owned exception type is an `InferenceError`, keeps this module and prints
  bare, `ScopeError` among them; the drift test that every FiniteKernels exception type and BN's root,
  union and zero-mass error and Formats' root are re-exported as the same bindings and no concrete Formats exception type is; real
  errors of each layer against the roots; the structural `==`/`hash` inherited from `BayesNetError`);
  `test/test_docstrings.jl` (every exported name this package owns has a docstring).
- `vignettes/`: 01 factors and VE (ends with the model-level Demo 1: `.dne` -> `infer` -> `do` -> `.xdsl`),
  02 orderings, 03 sampling and Monte Carlo checks, 04 junction trees and belief propagation,
  05 validation and scoring (simulate cases, grouped holdout, calibration, prior baseline, mutual information),
  06 inference in the log domain (a 340-site chain whose evidence mass underflows: the default backend
  drifts at 320 sites and raises at 340, `LogVariableElimination` returns the closed-form posterior
  throughout, and `log_evidence_probability` separates rare from impossible).
- Later milestones: `ext/` adapters for external backends.

## Formal correspondence

The exact finite-model inference proofs live in the sibling
`BayesianNetworks.jl/proofs/`, not in a runtime dependency. They include
partial-assignment posteriors and explicit evidence clamping, bucket VE,
cached collect/distribute on structurally valid junction trees, and
moralized-ancestral d-separation soundness. Full factor/variable coverage and
running intersection are premises; precomputed correct messages are not.
Arbitrary branching is encoded by proved grafting.

The formal virtual-root forest has global beliefs; `cal.beliefs` here are
component-local and require outside scalar masses for raw-array comparison.
Do not equate the formal normalized empty-query posterior with `infer(fg, [])`,
which returns unnormalized mass. Concrete CliqueTrees/array/IEEE and iterative
BP refinement remain separate. The numerical posterior bound has an explicit
positive evidence-mass floor and error-budget premise.
`BayesianNetworks.proof_certificate` exports exact bound data before this
package's Float64 factor conversion; it does not certify that conversion.

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

JuliaFormatter `yas`; docstrings on every exported name, which `test/test_docstrings.jl` enforces; the docs
build is strict (no `warnonly`), so a docstring left out of the manual or a broken `@ref` fails it; typed
exceptions with variable names in the message, following ADR 0013: they live in `src/errors.jl` and subtype
the nearest root (`FiniteKernelsError`, `BayesianNetworkFormatsError` or `BayesNetError`), invalid arguments
and keywords raise `ArgumentError`, typed errors from a lower package pass through unchanged and documented,
and another package's type is named as a code span, never with `@ref`; no emojis in code or docs.
