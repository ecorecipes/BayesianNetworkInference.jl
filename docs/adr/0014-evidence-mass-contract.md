# ADR 0014: The evidence-mass contract: exact zero, fallback, and indeterminate posteriors

Date: 2026-09-29. Status: accepted.

## Context

ADR 0012 gave zero evidence mass one exception, `BayesianNetworks.ImpossibleEvidenceError`. Its decision 5 recorded
that the exception did not yet mean exact zero. The paths that form the evidence mass as one binary64 number also
raised it for two other cases:

- a positive mass that underflowed;
- a non-positive mass produced by tolerated negative entries.

It deferred the separation to this record. The other cases it named were subnormal masses, which are not zero but have
lost their significand, and NaN or infinite masses from an overflowed raw product.

The certifier already assumes exact zero. The frozen Rocq domain certifier accepts an `impossible_evidence` rejection
only when every legal policy gives the evidence exact zero mass.

Probes on a detection chain whose closed-form posterior is `[0.9, 0.1]` at every length showed three regimes in the
default backends:

- the correct answer at 300 sites;
- a silently wrong one at 320 sites, `[0.901, 0.099]`, because the mass was subnormal;
- `ImpossibleEvidenceError` from 340 sites, because the mass underflowed to zero.

On the influence-diagram side, the default decision elimination reported rare evidence as impossible, and
`stable = true` was the only way out.

Tolerated entries in `[-atol, 0)` (ADR 0007) are the second problem. ADR 0011 keeps them unclamped and not
renormalised (`0011:55-56`). They can make a computed evidence mass zero or negative, or a posterior cell negative,
once the evidence is so improbable that the tolerance is no longer small beside it. ADR 0011 defines the multiplicative
budget `(1 + atol)^n - 1` but did not say what a posterior should do when the evidence mass falls within it.

The item-15 plan proposed a sibling type for a mass that is positive but not representable, `UnrepresentableEvidenceError`
under an abstract `EvidenceMassError`. Every case that type would have reported has a correct answer that another
arithmetic can compute. Reporting such a case as an error hands the caller a problem the library can solve.

## Decision

1. **Zero means exactly zero.** Every posterior entry point, in every package, raises `ImpossibleEvidenceError` only
   when the evidence has probability exactly zero under the model. Its docstring and message say "probability exactly
   zero".
2. **A binary64 mass is trusted only if it is finite and at least `floatmin(Float64)`.** A mass that is zero,
   subnormal, infinite or NaN decides nothing. The entry point recomputes its answer on the failure path only, so an
   ordinary query costs nothing extra:
   - `BayesianNetworks.marginal` and `conditional` recompute from the log-domain joint table (a streaming
     log-sum-exp). `conditional` recomputes every column when any denominator is untrusted, and its `on_zero` applies
     only to exact zeros.
   - `BayesianNetworkInference`'s variable elimination, brute force, junction tree, `clique_beliefs` and
     `all_marginals` rerun in `LogVariableElimination` or `LogJunctionTree`, and set `log_fallback = true` in their
     diagnostics.
   - Belief propagation resolves a zero or untrusted message or belief with the exact log-domain evidence mass. It
     raises `ImpossibleEvidenceError` if that mass is `-Inf`. Otherwise the log-domain junction tree answers, and
     `BPDiagnostics.exact_fallback` is true.
   - `InfluenceDiagrams`' default `DecisionVariableElimination` reruns the same bucket schedule in `Rational{BigInt}`
     arithmetic. The value-table `ExhaustivePolicySearch` falls back to scoring each strategy through
     `expected_utility` when a product is untrusted. Both record `exact_fallback = true` in the diagnostics.
     `expected_utility` inherits the log-domain fallback of `BayesianNetworks.marginal`.

   Inside a package, the binary64 run signals an untrusted mass with an internal type (`_UnresolvedMass` in Inference,
   `_UnresolvedDecisionMass` in InfluenceDiagrams). Each public entry point resolves the signal, and it never reaches
   a caller.
3. **Only the fallback decides impossibility.** A log mass of `-Inf`, or an exact rational zero, raises
   `ImpossibleEvidenceError`. An exact rational type decides directly: a positive mass passes and a zero raises.
4. **`IndeterminatePosteriorError`.** `BayesianNetworks` defines `IndeterminatePosteriorError(evidence, detail) <:
   BayesNetError`, exported and re-exported with the other exception types (ADR 0013). A posterior is indeterminate
   when the model has entries in `[-atol, 0)` and any of the following holds:
   - the evidence mass is not larger than ADR 0011's budget `(1 + atol)^n - 1`, with `n` the number of mechanisms
     (`BayesianNetworks._joint_atol`);
   - a posterior cell comes out negative;
   - the log-domain fallback meets a negative entry, which has no logarithm;
   - an exact-arithmetic fallback is needed, since exact arithmetic does not accept tolerated entries.

   Nothing is clamped or renormalised, so ADR 0011 `:55-56` stands. The error names the evidence and the condition
   that fired. Removing the negative entries, or binding at a tolerance that rejects them, resolves it. A model with
   no negative entry never raises it.
5. **Empty queries keep ADR 0011's meaning.** They return the unnormalised mass, which may legally be zero or tiny,
   and are exempt from the budget check.
6. **The execution trace records binary64 execution** (ADR 0011), so it cannot fall back silently.
   `trace_variable_elimination` raises `ImpossibleEvidenceError` when the log mass is `-Inf`. Otherwise it raises
   `ScopeError(:trace_variable_elimination, ...)`, which names `LogVariableElimination` for that posterior.
   `trace_decision_elimination` already runs in exact arithmetic and is unchanged.
7. **Log backends.** `LogVariableElimination`, `LogJunctionTree` and `log_evidence_probability` are unchanged. The log
   backends still reject a negative entry with `LogFactorDomainError`, a backend-domain restriction. Called directly,
   they are not a fallback, so they do not translate that error.

## Consequences

- `ImpossibleEvidenceError` now means the same thing everywhere, and the certifier's reading of it holds for every
  path. Callers that caught it to detect rare evidence get the answer instead.
- **Behaviour changes, all in the direction of correct answers:**
  - the 320-site chain returns `[0.9, 0.1]` rather than a silently perturbed posterior;
  - 340 and more sites return `[0.9, 0.1]` rather than raising;
  - an overflowed raw product with a zero factor returns the exact posterior rather than NaN;
  - rare evidence in an influence diagram is solved by the default backends.
- **Cost.** The fallback runs only when the binary64 mass is untrusted. It is exponential in the same width as the
  ordinary run: treewidth for the elimination backends, and the joint table for `marginal`. Exact rational DVE costs
  more per cell than Float64. The diagnostics flags make the fallback visible.
- The per-scalar checks in belief propagation test each conditioned scalar. They do not test the product, which can
  underflow.
- **New error type.** `IndeterminatePosteriorError` is a breaking addition only for code that relied on a negative or
  NaN posterior being returned. No conformance fixture has tolerated negative entries.
- The frozen names `ImpossibleEvidenceError`, `IrregularDiagramError` and `ScopeError` keep their bindings and their
  unqualified printing.
- **Conformance.** The Julia adapter's `evidence-underflow` and `subnormal-posterior` stress queries now return the
  exact posterior. `stress.py` used to classify them as expected precision limits (`evidence_underflow` and
  `subnormal_product_loss`). It now classifies them as `strict_agreement`. Nothing pins those counts, and no protocol
  or version bump is needed.
- **The `stable = true` options remain.** They are no longer needed for rare evidence. They are still the paths that
  avoid cancellation among large signed utilities.
- **Separate fix.** While testing the 340-site cases, the brute-force state counts in `BayesianNetworks`,
  `CategoricalBayesianNetworks` and `InfluenceDiagrams` were found to wrap past `2^127` in `Int128`. That bypassed the
  `ModelTooLargeError` guard, so the counts are now `BigInt`.
- **Relation to earlier records:**
  - this record amends ADR 0011 `:40-42` (the non-zero-mass check now means exact zero, and the check is decided by a
    fallback) and `:51-57` (the budget now decides indeterminate posteriors);
  - it replaces ADR 0012 decision 5;
  - it supersedes the item-15 plan's `UnrepresentableEvidenceError`, which is not added.
