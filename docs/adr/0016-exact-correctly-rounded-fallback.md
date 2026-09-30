# ADR 0016: The evidence-mass fallback is exact and correctly rounded

Date: 2026-09-30. Status: accepted.

## Context

ADR 0014 made `ImpossibleEvidenceError` mean probability exactly zero. When a binary64 evidence mass is not a
normal positive number, the posterior entry points recompute on the failure path:

- `BayesianNetworks.marginal` and `conditional` recomputed by a streaming log-sum-exp over the joint;
- `BayesianNetworkInference`'s backends recomputed in `LogVariableElimination` or `LogJunctionTree`;
- `InfluenceDiagrams`' decision elimination recomputed in exact rational arithmetic, rounding its value once.

The log domain decides impossibility correctly: a log mass of `-Inf` is an exact zero. Its posterior cells are only
accurate to a few units in the last place, because every logarithm and exponential rounds. The Phase B conformance
run showed this on the `evidence-underflow` stress fixture. The Julia adapter returned `[0.25, 0.7499999999999999]`
where the exact posterior is `[0.25, 0.75]`. `stress.py` therefore still classified the query as a precision limit
rather than as agreement.

Every finite Float64 is exactly an integer times a power of two. A product of bound probabilities is exact in
integer arithmetic, and so is a sum once its terms share a power of two. The fallback can therefore be exact at the
cost of `BigInt` arithmetic, on a path that runs only when the ordinary mass is untrusted.

## Decision

1. **The fallback is exact.** When a posterior entry point's binary64 evidence mass is not a normal positive number,
   the entry point recomputes the posterior in exact arithmetic on the values as bound. It then rounds each cell
   once to the nearest Float64, ties to even. The returned posterior is the correctly rounded exact posterior of the
   model or factor graph as bound. This does not describe the model its author meant: an entry written `0.1` is
   bound as the nearest Float64.
2. **One definition of the rounding.** `BayesianNetworks/src/exact_rounding.jl` defines `_dyadic` (a Float64 as an
   exact integer and exponent), `_rational_exponent` and `_nearest_binary64`. The last two were moved verbatim from
   `InfluenceDiagrams/src/exact_arithmetic.jl`, which now imports them, so `InfluenceDiagrams._nearest_binary64` is
   the same function.
3. **Where it applies:**
   - **`BayesianNetworks`:** `marginal` and `conditional` enumerate the joint exactly. Every product is scaled to a
     common power of two, and the sums are `BigInt`s. Each cell is rounded once; in `conditional` that includes
     each underflowed column.
   - **`BayesianNetworkInference`:** a third arithmetic, `_Dyadic` (`_DyadicFactor`: a `Factor{BigInt}` times one
     power of two), runs the shared drivers of review item 8 unchanged. `_eliminate` gives the elimination,
     `_calibrate` the junction tree and `_belief_marginal` every belief read-off. `_normalized` rounds each cell
     once. The fallback covers:
     - variable elimination;
     - brute force;
     - the junction-tree query, `clique_beliefs` and `all_marginals`;
     - the variable-elimination `all_marginals` shortcut;
     - belief propagation's feasibility checks and its fallback;
     - the execution trace's decision between impossible evidence and a trace limit.
   - **`InfluenceDiagrams`:** unchanged. Its decision-elimination fallback was already exact and correctly rounded,
     and its exhaustive search inherits `BayesianNetworks.marginal`'s fallback.
4. **Impossibility and tolerated entries are decided exactly.** An exact total of zero is `ImpossibleEvidenceError`.
   A tolerated negative entry is `IndeterminatePosteriorError`. For `BayesianNetworks`, that means one on a
   configuration consistent with the evidence; for `BayesianNetworkInference`, one in any input factor. The
   fallback runs only when the binary64 mass is below the normal range or not finite. The tolerance cannot then be
   shown small beside the mass, as ADR 0014 decision 4 already required. This replaces the log domain's
   "negative entry has no logarithm" rule. An entry that is not exactly a Float64 is `FactorDomainError` with
   backend `:exact`: a non-finite value in a graph built with `check = false`, or a wider type.
5. **The diagnostics flag is `exact_fallback`.** `InferenceDiagnostics.log_fallback` and
   `JunctionTreeDiagnostics.log_fallback` are renamed `exact_fallback`. That is the name `BPDiagnostics` and
   `InfluenceDiagrams` already used. The packages are unregistered, so there is no alias.
6. **The log backends are unchanged.** `LogVariableElimination`, `LogJunctionTree`, `log_calibrate` and
   `log_evidence_probability` remain opt-in numerical backends. They are fast on wide models and not correctly
   rounded. None of them is a fallback any more.

## Consequences

- A posterior returned through the fallback is bit-for-bit the correctly rounded exact posterior. The tests check
  this against independently computed `Rational{BigInt}` references in `BayesianNetworks` and
  `BayesianNetworkInference`. The `evidence-underflow` stress query should now agree strictly; the conformance run
  of this change records the outcome.
- **Cost.** The fallback does `BigInt` arithmetic. Integers grow by about 53 bits for each factor multiplied into a
  product, and the exponents of the dyadic form add rather than enlarge the integers. The fallback runs only when
  the binary64 mass is untrusted. It is exponential in the same width as the ordinary run.
- **Conformance provenance.** `conformance/machine_conformance.py` now records the hash of
  `BayesianNetworks/src/exact_rounding.jl` as its rounding source. `freeze_refinement_data.py` snapshots that file
  beside `exact_arithmetic.jl`. Neither script is pinned by an audit, and no frozen bundle hashes the live files.
- **Relation to earlier records:**
  - this record amends ADR 0014 decisions 2 (the fallback arithmetic), 4 (how tolerated entries are decided on
    the fallback), 5 (the diagnostics name) and 7 (the log backends are no longer fallbacks);
  - it amends ADR 0015 decision 1 by adding the `:exact` backend of `FactorDomainError`.
