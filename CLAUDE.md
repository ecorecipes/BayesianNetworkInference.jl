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
- `multiply`'s result (axis order from `_union_axes`, column-major table, each cell the one
  product `a[ai] * b[bi]` at the offsets `_result_strides` gives) is modelled in
  `FiniteKernels.jl/proofs/FiniteKernelsProofs/Layout/Product.lean` (`productInto_eq`). An
  optimisation of the kernel may change how the walk is organised (merged axes, inner loops),
  never which cell gets which product; `test_factors.jl` compares it bit for bit with an
  independent reference on every layout.
- Integer and rational tables never wrap (review of 2026-10-02): `multiply`, `marginalize` and
  `normalize` compute machine-integer and `Rational{<:BitInteger}` tables in checked
  arithmetic (`_checked`, `_mul`, `_marginal`, `_total`, `_normalize` in factors.jl), and an
  overflow is `FactorDomainError` with the operation as backend, the cell and the exact value.
  A posterior run treats it as an untrusted run (`_overflowed`) and the exact fallback answers.
  A new operation on factor tables must keep both.
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
- Label errors follow the layer (ADR 0015). Every method on a `BayesModel` (`infer`, `posterior`,
  `all_marginals`, `log_evidence_probability`, `trace_variable_elimination`, `entropy`,
  `mutual_information`, `sensitivity`, `tornado`, `predict`, `baseline`, `evaluate`) checks names with
  `_check_labels` / `_check_case_labels` after `compile` and raises BN's `UnknownVariableError` /
  `UnknownStateError`, as `BayesianNetworks.marginal` does. The `FactorGraph` methods keep `ScopeError` and
  FiniteKernels' `InvalidAxisError`. A new model-level entry point must call the check.
- `ScopeError` means a scope or argument-scope problem only. Trace budgets and profile limits are
  `TraceLimitError`; an entry a backend's arithmetic cannot take is `FactorDomainError` (ADR 0015).
- BP convergence uses the undamped residual at the returned iterate, not the damped step and not a
  marginal-error bound. `check_evidence=true` opts into VE feasibility; `evidence_checked=false`
  leaves global feasibility unknown. Integer inputs promote to division-compatible types.
- BP never runs an exact computation except that opt-in (ADR 0011; the review of 2026-10-01 found
  the ADR 0014/0016 exact junction-tree fallback exponential in treewidth on loopy grids). An
  exactly zero message or belief is certified exact by the trust test (no product before it
  underflowed) and, on nonnegative potentials, raises `ImpossibleEvidenceError` at once; any other
  untrusted binary64 message redoes the same message passing in the log domain
  (`BPDiagnostics.log_domain`). The answer is always BP's own; never add an exact fallback to it.
- Tolerated entries in `[-atol, 0)` follow the rule BayesianNetworks' `marginal` and InfluenceDiagrams
  share: one takes part when it lies on a configuration consistent with the evidence whose other
  entries are all nonzero; then the posterior is indeterminate if the evidence mass is not above the
  budget `(1 + atol)^n - 1` or a queried cell is negative; a prior is exempt; the binary64 and exact
  runs decide alike. Every model-level entry point asks for each posterior through
  `_under_tolerance` (model_inference.jl), once per evidence set (each case of `predict`/`evaluate`,
  each finding of `tornado`), so it raises exactly when `marginal` does (a randomized test checks
  this). `_tolerance_budget` is `nothing` for a model without a negative entry and the rule is
  skipped whole; otherwise `_takes_part` is one support elimination (`_Support`, Boolean tables,
  same schedule) and the mass one more elimination only when an entry takes part. With
  `BeliefPropagation` the budget runs only under `check_evidence = true`. Inside `_MODEL_QUERY` a
  prior's cells are returned as computed and the exact fallback takes negative entries at their
  exact value; on a factor graph a negative cell is indeterminate and the exact fallback rejects
  any negative entry (ADR 0016 decision 4). A new model-level entry point must use the helper.
- Names shared with BayesianNetworks.jl (`variables`, `axis`, `empirical_marginal`), CliqueTrees.jl (`treewidth`)
  and Graphs.jl (`is_tree`) are extended with `import ...: ...`, never shadowed, so `using BayesianNetworks, BayesianNetworkInference` stays
  unambiguous. Never redefine `BayesianNetworks.sample(::BayesModel, n)`; the sampler here is `ancestral_sample(m, n)`.

## Layout

- `src/errors.jl` (included first): the root `InferenceError <: BayesNetError` and the exception types
  with their `showerror` methods: `ScopeError` and `ShapeError` (factor scopes and table shapes, and the
  argument checks of the entry points built on them; nothing else, ADR 0015), `CompileError` (open model /
  missing kernels, naming the variables), `FactorEntryError` (a factor-graph entry that is not finite or is
  below `-atol`), `FactorDomainError` (a valid entry a backend's arithmetic cannot take: `:log_domain`,
  `:trace_variable_elimination`, InfluenceDiagrams' `:stable_decision_elimination`) and `TraceLimitError`
  (a run an execution trace cannot record: cell budgets, the Float64-only profile, an underflowed mass;
  InfluenceDiagrams' trace raises it too).
- `src/factors.jl`: `Factor`, `unit_factor`, `scope`, `axis`, `multiply`, `marginalize`,
  `maximize`, `argmax_table`, `condition`, `normalize` (extends `LinearAlgebra.normalize`), `reorder`, kernel conversions.
  The product kernel `_product_into!` walks the result with the strides of `_result_strides`:
  `_merged_axes` merges axes that both operands step through contiguously, `_product_block!` runs
  the two innermost as loops with paths for contiguous and constant operands, and an odometer
  turns the rest (review of 2026-10-02: variable elimination and the junction tree on barley,
  mildew and water ran 1.8-3.6x faster than with the plain odometer, and 1.1-2.3x faster than with
  the broadcast it replaced). The checked forms `_product`, `_marginal`, `_total` and
  `_normalize` return an overflow's `FactorDomainError` instead of throwing it.
- `src/factor_graph.jl`: `FactorGraph` (factors, axes, provenance), `variables`, `interaction_graph` (Graphs.SimpleGraph
  plus variable/vertex maps; already the moral graph). The keyword constructor checks entries (finite, `>= -atol`,
  else `FactorEntryError`; `check = false` opts out) once per graph, scanning each table as a vector (a
  `CartesianIndices` loop over a table of unknown rank dispatched per entry); the inner `FactorGraph{T}(factors, provenance)`
  never checks and is what `compile` and InfluenceDiagrams' signed ordering graph use. Entries in `[-atol, 0)` are
  valid here and still rejected by the log backends (`FactorDomainError`), a backend-domain restriction.
- `src/orderings.jl`: `EliminationStrategy` types `MinFill` (CliqueTrees `MF`), `MinDegree` (`MMD`), `AMDOrder` (`AMD`,
  needs `import AMD`), `ExactTreewidth` (`BT`, needs `import TreeWidthSolver`; run per connected component because BT
  fails on isolated vertices), `UserOrder`; `elimination_order(fg, strategy; keep)` via
  `permutation(graph; alg=CompositeRotations(keep_idx, alg))`, `treewidth(fg, strategy)`.
- `src/arithmetic.jl` (review item 8, S3): the arithmetics the exact drivers are written against,
  `_Linear` (a `Factor`), `_LogDomain` (a `_LogFactor`: centred log table plus log scale) and `_Dyadic`
  (a `_DyadicFactor`: integer table times a power of two, the exact fallback, ADR 0016). `_Linear()`
  checks nothing (`calibrate`, empty queries); `_Trusted()` (`_Linear{true}`) is every posterior
  run's, and `_check_product` applies the trust test to each product: a product of nonzero
  operands below `floatmin(T)`, subnormal or rounded to 0.0, throws `_UnresolvedMass` (judged first
  from the operands' smallest exponents, so the cost is linear in the tables computed). Sums and
  divisions are not checked. An integer or rational overflow (factors.jl's checked arithmetic)
  is `FactorDomainError` from `_Linear()` and an untrusted run under `_Trusted` (`_overflowed`).
  A `_DyadicFactor` is an integer table times a power of two over one odd denominator, and
  `_exact_entry` gives every entry its exact value whatever the element type (BayesianNetworks'
  `(n, p, d)` convention; `_dyadic(::Float64)` itself is BayesianNetworks' and unchanged), so the
  fallback is exact on integer, rational, Float16, Float32 and BigFloat graphs.
  `_as_dyadic` lets only nonzero entries choose the power of two
  (`_dyadic(0.0)` is `0 * 2^-1074`) and converts only the entries the evidence keeps, after
  `_check_exact_entries` has checked them all. Empty products are the unit of the list's own
  element type (`_table_type`), never Float64 by default. With the
  operations `_conditioned`, `_product!`, `_multiply`, `_sum_out`, `_project`, `_reorder`, `_unit` and
  `_normalized`, the log primitives (`_as_log_factor`, `_log_multiply`, `_log_sum_out`, `_log_mass`,
  `_log_mass_status`) and `_collect_marginals` (point masses for observed variables). The drivers are
  written once: `_eliminate` (bucket elimination, variable_elimination.jl) serves `variable_elimination`
  and `log_variable_elimination`; `_calibrate` and `_belief_marginal` (junction_tree.jl) serve `calibrate`,
  `log_calibrate` and every belief read-off. A change to an algorithm goes in the driver; a change to an
  arithmetic goes in its operations. Never write a second copy of a driver for one arithmetic -- that is
  how C6 (the empty-scope `normalize`) reached one copy and not the other.
- `src/variable_elimination.jl`: `VariableElimination` backend, `InferenceDiagnostics`, `variable_elimination`,
  the shared driver `_eliminate`, and the elimination-order cache (shared by both arithmetics, whose
  conditioned scopes are the same): `graph`/`vars`/`index` exist only to produce `elim`, so the order is what
  is cached, keyed on `objectid(fg.factors)` plus the evidence's *keys*, the strategy and the query (the
  conditioned scopes depend on which variables are observed, not on their values, so `predict` shares one
  entry across every case). Same `WeakRef`/`===` discipline as the junction-tree cache, and the same rule:
  do not mutate `fg.factors` after a query. Like that cache it is pruned of collected graphs (amortised, when
  it has doubled) and every access holds a lock (`_ORDER_LOCK`; the ordering itself is computed outside it),
  and a caller always gets a copy of the cached order, never the cached vector (review of 2026-10-02).
  `infer`, oracle `joint_factor` / `brute_force_marginal`.
- `src/log_variable_elimination.jl`: `LogVariableElimination` backend,
  `LogInferenceDiagnostics`, `log_variable_elimination`, `log_evidence_probability`: the shared
  `_eliminate` run in `_LogDomain`, plus the log mass and its status.
- `src/log_junction_tree.jl`: `LogJunctionTree`, `LogCalibratedJunctionTree`,
  `LogJunctionTreeDiagnostics`, `log_calibrate`: the shared `_calibrate` run in `_LogDomain`.
- `src/execution_trace.jl`: `trace_variable_elimination`, which records each product and
  marginalisation of a run as data for the external conformance checker. The trace's
  `inputs` are in factor-graph order, which is *not* the association order of the recorded
  product (`_product!` sorts `touching` stably by `ndims` first, in both arithmetics); the format does not record that
  order, and ADR 0011 notes that reassociation alone changes the answer.
- `src/junction_tree.jl`: `JunctionTree(; order)` backend, `CompiledJunctionTree` from
  `CliqueTrees.cliquetree(graph; alg, snd=Maximal())` (`residual`/`separator`, `parentindex`/`childindices`/
  `rootindices`; cliques are numbered in the permuted order, `label[v]` maps back), cached in a `Dict` keyed by
  `objectid(fg.factors)` with a `WeakRef` checked by `===` (identity, not content). Before it is returned,
  `_check_junction_tree` checks CliqueTrees' output and the factor assignment against the hypotheses of the Lean
  `calibrate_correct` (`Good`, `checkAssignment`, via `graft_good`/`forest_good`): a rooted forest whose
  children, roots and postorder agree with `parent`, separators equal to clique ∩ parent clique, running
  intersection (one top clique per variable), every variable covered and none repeated, every factor scope in its
  clique. A failure is `ScopeError(:build_junction_tree, ...)` naming the condition. It is a runtime check, once per
  compiled (cached) tree, not a proof that CliqueTrees is correct; every tree is built through
  `_build_junction_tree`, so keep any new tree source going through it. `calibrate` (Shafer-Shenoy: collect then distribute, no division, so zero entries are
  safe), `clique_beliefs`, `all_marginals` (JT in one pass; VE method loops per variable; evidence variables
  are point masses), `infer` for one-clique queries with a VE fallback that warns and sets
  `JunctionTreeDiagnostics.fallback` (the return type never changes), `JunctionTreeDiagnostics`.
- `src/belief_propagation.jl`: `BeliefPropagation(; damping, tol, maxiter, schedule, check_evidence)`, `BPDiagnostics`
  (`log_domain`, which replaced `exact_fallback`),
  `belief_propagation(fg, backend; evidence)` (sum-product, SPEC section 20; `:flooding` or `:sequential`).
  The message equations are written once against two message arithmetics: `_LinearMessages`
  (binary64, potentials scaled by a power of two by `_linear_potentials`, which leaves normalised
  messages bit for bit unchanged; every product, damping's included, and every normalised entry
  checked against `floatmin`) and `_LogMessages` (the rerun, after `_check_log_entries`: a non-finite
  entry is `FactorDomainError(:log_domain)`, a tolerated negative one `IndeterminatePosteriorError`),
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
  `test/test_{factors,orderings,inference,sampling}.jl` (with the review of 2026-10-02: `multiply` bit for bit
  against a reference on every layout and element type, integer and rational overflow at the factor level,
  typed empty products, and the elimination-order cache: copies, pruning, and concurrent queries in a
  `julia -t 4` subprocess); `test/test_model_bridge.jl` (reference model, asia from
  `fixture_path("bif/asia.bif")`, interventions, 15 seeded random `BayesModel`s, sampling);
  `test/test_junction_tree.jl` (JT versus VE and brute force on asia, the habitat chain, random DAGs, forests,
  the 30-variable `benchmark_network`, the bnlearn `water` model when `ECOLOGICAL_BN_SLOW=true` and the
  sibling zoo is checked out, and a 5 x 5 grid with factorless cliques in four element types); `test/test_belief_propagation.jl` (exact on chains and polytrees, loopy on asia, impossible
  evidence raising `ImpossibleEvidenceError` in all three backends, and no exact fallback: an impossible 20 x 20 grid
  under an allocation bound, tiny and rational potentials, the log-domain rerun); `test/test_evidence_mass.jl`
  (ADR 0014/0016 against `Rational{BigInt}` references, the trust test on products, zero-free dyadic
  factors, the tolerance budget at every model-level entry point, integer and rational graphs that overflow
  answered exactly, the exact fallback on Rational, BigFloat, Float32 and Float16 graphs, and
  `all_marginals` on a fallback); `test/test_scores.jl`
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
  06 inference in the log domain (a 340-site chain whose evidence mass underflows: the default backends fall
  back to the log domain and return the closed-form posterior, `log_evidence_probability` separates rare
  from impossible, and tolerated negative entries give `IndeterminatePosteriorError`; ADR 0014).
- Evidence mass (ADR 0014, ADR 0016, and the trust test of the review of 2026-10-01: a binary64 run is
  untrusted if its final mass is not a normal positive number, or if any product it computed from
  operands that are all nonzero has magnitude below `floatmin` of its element type, subnormal or
  rounded to 0.0; an integer or rational run is untrusted when a product, sum or quotient overflows its
  element type): `_require_evidence_mass` never decides impossibility for a binary64
  mass that is not a normal positive number; it, `_check_product` for a product and `_overflowed` for an
  overflow throw the internal `_UnresolvedMass` signal, which every
  public entry point but BP resolves with `_resolving_mass` by recomputing in exact dyadic arithmetic (`_Dyadic`,
  arithmetic.jl: the shared drivers on integer tables times a power of two over an odd denominator, every
  entry at its exact value) and rounding each posterior cell once with BayesianNetworks'
  `_nearest_binary64`, so the fallback returns `Factor{Float64}` whatever the element type; variable
  elimination's `all_marginals` recomputes every marginal exactly as soon as one run is untrusted, as the
  junction tree does, so its dictionary never mixes the two. Only that exact computation (`_exactly_zero`,
  `_exactly_impossible`) raises `ImpossibleEvidenceError`; a tolerated negative entry met by it is
  `IndeterminatePosteriorError`. The diagnostics flag is `exact_fallback` (it was `log_fallback`). A new
  entry point that normalises a posterior must go through `_resolving_mass`, or the signal escapes. The
  explicit log backends are not a fallback and are not correctly rounded.
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
and keywords raise `ArgumentError`, typed errors from a lower package pass through unchanged and documented, content read from a file, document or manifest is checked before it is converted and raises the package's typed error (ADR 0015: never catch the `MethodError` or `InexactError` of an unchecked conversion),
and another package's type is named as a code span, never with `@ref`; no emojis in code or docs.
