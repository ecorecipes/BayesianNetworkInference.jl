# ADR 0012: One exception for evidence with zero mass

Date: 2026-09-26. Status: accepted.

## Context

Item 15 of the 2026-09-24 review asked for a single normalisation-error name across the ecosystem. Checking that
request against the code found a narrower and more serious problem: the packages report one condition, evidence of
probability zero, under two unrelated types.

- `BayesianNetworks` (`marginal`, and `conditional` with `on_zero=:error`) and every `InfluenceDiagrams` solver raise
  `BayesianNetworks.ImpossibleEvidenceError` when the evidence has zero mass.
- `BayesianNetworkInference` raises `FiniteKernels.KernelNormalizationError` for the same condition, at five sites:
  the evidence-mass helper in `variable_elimination.jl` (which the junction tree also uses), `normalize(::Factor)` in
  `factors.jl`, the message and conditioned-scalar checks in `belief_propagation.jl`, and `predict`'s re-wrap in
  `scores.jl`. Each fabricates a `max_deviation` of `1.0` and an `atol` of `0.0`. Every function built on these
  backends inherits the error, and `tornado` catches it as an "impossible finding".
- ADR 0007 removed exactly that fabrication from the kernel `normalize`, which `FiniteKernels` now owns: a zero column
  raises an `ArgumentError` naming the input index.
- ADR 0011 requires every exact posterior entry point to check for non-zero global mass, but names no type for the
  failure.
- The `observe` and `do_intervention` docstrings in `BayesianNetworks` promise that contradictory evidence surfaces
  "at the next query as an `ImpossibleEvidenceError`". A query through `BayesianNetworkInference.infer` breaks that
  promise.
- Both of Inference's linear-domain checks test `iszero`, so a negative computed mass, which tolerated negative
  entries can produce, passes them unreported. `BayesianNetworks` (`total > 0`) and `InfluenceDiagrams` (`pe > 0`)
  already reject it.
- The frozen Rocq domain certifier (`Rejections.v`) accepts an `impossible_evidence` rejection only when the recorded
  exception type is the bare name `ImpossibleEvidenceError`, and only when every legal policy gives the evidence exact
  zero mass.

## Decision

1. **One type.** Every posterior entry point of every package raises `BayesianNetworks.ImpossibleEvidenceError` when
   the computed evidence mass is zero or negative. In `BayesianNetworkInference` this covers:
   - variable elimination, brute force and the junction tree;
   - belief propagation: a zero message, belief or conditioned scalar, and the `check_evidence=true` feasibility pass;
   - both log backends, when `mass_status == :zero`;
   - everything built on these, such as `posterior`, `predict`, `sensitivity` and `tornado`.

   A factor graph whose total mass is zero, with no evidence at all, raises it with empty evidence, and its message
   says that the model or factor graph has zero total mass.
2. **Empty queries keep ADR 0011's meaning.** They return the unnormalised mass, which may legally be zero, and never
   raise.
3. **`KernelNormalizationError` has one meaning:** kernel columns do not sum to one within `atol`. `normalize(::Factor)`
   on a zero total raises an `ArgumentError` naming the scope, as `FiniteKernels.normalize` does for a zero column.
   Posterior code checks the mass before it normalises, so it never reaches that error.
4. **Skipping impossible cases.** Code that deliberately skips impossible cases catches exactly
   `ImpossibleEvidenceError`. `tornado` first checks the base evidence, and raises if it is impossible instead of
   returning no rows.
5. **What "zero" covers until a later ADR.** Some paths also report two other cases as this error: a positive mass
   that underflowed, and a non-positive mass produced by tolerated negative entries.
   - These are the paths that form the global binary64 mass: variable elimination, the junction tree, brute force,
     belief propagation with `check_evidence` or a zero conditioned scalar, `BayesianNetworks.marginal`, and the
     default decision-elimination, exhaustive and `expected_utility` paths of `InfluenceDiagrams`.
   - The log backends, plain belief propagation and the stable decision paths do not.
   - A NaN or infinite mass (an overflowed raw product) is not reported as impossible, and it keeps its current
     result. The test is `mass <= 0`, which is false for NaN.
   - The error's documentation says "zero computed probability" and names the ways to tell the cases apart:
     `LogVariableElimination`, `LogJunctionTree`, `stable=true` and `log_evidence_probability`.
   - The exact-arithmetic paths (`stable=true` decision elimination and `trace_decision_elimination`) raise it only
     for an exact zero, which is the only kind the certifier sees.

   A later ADR will separate exact zero from a mass that is positive but not representable (underflow, overflow and
   NaN), and will amend this decision.
6. **One binding.** `BayesianNetworkInference` and `InfluenceDiagrams` re-export this binding, each in the commit
   that adopts this ADR there. No package defines another binding with this name, so it stays unambiguous and prints
   unqualified, as the certifier requires.

## Consequences

- **Breaking changes in `BayesianNetworkInference`:**
  - code that catches `KernelNormalizationError` for impossible evidence must catch `ImpossibleEvidenceError`, and
    17 assertions in the package's own tests switch to the new type;
  - `normalize(::Factor)` on a zero total raises `ArgumentError`;
  - `predict`'s error carries the failing case's merged evidence, not the case's index;
  - `tornado` raises when the base evidence is impossible.
- `BayesianNetworks` changes only the error's docstring and message, and `InfluenceDiagrams` only adds the re-export.
- The conformance adapters accept both types while the change lands, and only the new one after it.
- **No conformance protocol or version bump is needed:**
  - `conformance/trace_suite.py` regenerates its fixtures and traces on every run;
  - `conformance/schema/decision-result.schema.json` types `error_type` as a free string;
  - the frozen checkers read only the `ImpossibleEvidenceError`, `IrregularDiagramError` and `ScopeError` names;
  - outside the sealed bundles, the workspace documents name `KernelNormalizationError` only in the dated review and
    planning records, SPEC §54 and ADR 0007;
  - dated and sealed bundles are historical records and are not re-verified.
- The shared elimination driver that item 8 of the 2026-09-24 review extracts (finding S3) must carry this check for
  both arithmetics.
- This ADR amends ADR 0011 by naming the type of its non-zero-mass check, and extends ADR 0007's `normalize`
  consequence from kernels to factors.
