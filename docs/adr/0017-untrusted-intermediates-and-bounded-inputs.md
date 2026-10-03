# ADR 0017: Untrusted intermediates, one rule for tolerated entries, and bounded inputs

Date: 2026-10-02. Status: accepted.

## Context

A code review of the eight packages on 2026-10-02 covered the commits since the 2026-09-24 review. Several of the
defects it confirmed came from the evidence-mass contract of ADR 0014 and ADR 0016:

- **The trust test looked only at the final binary64 mass.** A posterior whose intermediate products had underflowed
  was returned silently. Variable elimination gave `[0.3005, 0.6995]` where the exact posterior is `[0.3, 0.7]`, and a
  chain gave `0.0` for a cell of `1e-70`.
- **Belief propagation's fallback was exponential.** Its exact fallback calibrated the whole loopy graph in `BigInt`,
  at a cost exponential in treewidth. On a 20×20 grid, impossible evidence used to raise in milliseconds; now it no
  longer finished.
- **The tolerated-entry rule differed between packages and paths.**
  - `marginal` checked the whole conditioned joint, so an unrelated negative root made a determined posterior
    indeterminate.
  - The binary64 DVE guard used its constancy tolerance as the budget and had no negative-cell check. It returned an
    expected utility of 3.58 where no utility exceeds 3.0, while every other backend raised.
  - Six model-level Inference functions skipped the check altogether.

Others concerned input handling:

- A declared count was used as an allocation size: a 19-byte UAI file asked for 330 GiB.
- A JSON text that names a file was read as that file.
- JSON3 and ACSets read repeated JSON keys differently.
- Unbounded nesting overflowed the stack.
- A rejected write truncated its target.
- DVE certificates followed Julia's string hash for their key order, and that order changed in Julia 1.13.

## Decision

1. **Untrusted intermediates (amends ADR 0014 decision 2 and ADR 0016 decision 1).** BayesianNetworks,
   BayesianNetworkInference and InfluenceDiagrams word the rule identically:

   > A binary64 run is untrusted if its final mass is not a normal positive number, or if any product it computed
   > from operands that are all nonzero has magnitude below `floatmin` of its element type, whether that product came
   > out subnormal or rounded all the way to 0.0. A product with an exactly zero operand is a structural zero and does
   > not count.

   - **The response.** An untrusted run takes the exact fallback of ADR 0016. The exception is belief propagation
     (decision 3).
   - **Where it applies:**
     - `marginal` and `conditional`;
     - variable elimination and brute force;
     - the junction tree, including its mass product;
     - DVE, through `_checked_product`.
   - **Each run judges its own products**, so two backends can differ on whether a borderline run is trusted. They
     raise in the same cases, and their answers differ only in the last bits.
   - **Integer and rational graphs.** An overflowing product or sum likewise makes the run untrusted.
   - **The trace.** `trace_variable_elimination` reports an underflowing product as
     `TraceLimitError(:evidence_underflow)`.
2. **One rule for tolerated entries (amends ADR 0014 decision 4 and ADR 0016 decision 4).** The rule uses
   BayesianNetworks' wording, from `marginal`'s docstring.
   - **When an entry takes part.** A tolerated entry in `[-atol, 0)` takes part in a posterior when it lies on a
     configuration consistent with the evidence whose other entries are all nonzero.
   - **When the posterior is indeterminate.** If an entry takes part, `IndeterminatePosteriorError` is raised when
     either holds:
     - the evidence mass is within the budget `(1 + atol)^n - 1`. Here `n` counts mechanisms, those of the
       instantiated network for an influence diagram, and `atol` is the normalisation tolerance;
     - a cell of the queried posterior is negative.
   - **What does not count.** An entry the evidence rules out, and negative cells of the joint that the query sums out.
   - **Priors.** A model's prior marginal (empty evidence) is the model's own law and is returned as computed. A plain
     factor graph has no prior exemption.
   - **Scope.** `conditional` applies the rule per column. For an influence diagram, "consistent with the evidence"
     ranges over all action values.
   - **Both runs.** The binary64 and exact runs decide by the same rule, the exact run on exact values. Below the
     normal range the rule reduces to ADR 0016's.
   - **Which exact runs accept tolerated entries.**
     - The exact fallbacks of `marginal`, of the model-level queries and of DVE accept them and decide by the rule.
     - `stable = true` DVE keeps `FactorDomainError` (ADR 0015).
     - The log backends keep their domain restriction.
     - The exact fallback on a plain factor graph still rejects negative entries.
   - **Cost.** Deciding "takes part" costs no more than the backend itself. `marginal` decides it during its
     enumeration. Inference and InfluenceDiagrams use a Boolean support elimination over their own schedule. A model
     with no tolerated entry skips the check entirely.
3. **Belief propagation reruns in the log domain (amends ADR 0014 decision 2 and ADR 0016 decisions 3 and 5).**
   ADR 0011 governs: "Approximate BP does not gain an unconditional exact-inference cost."
   - **Scaling.** Binary64 BP scales each potential by a power of two, which leaves its messages bit-identical, and
     applies the trust rule.
   - **Impossible evidence.** A message that sums to exactly zero, in a trusted run on nonnegative factors, proves the
     evidence impossible and raises `ImpossibleEvidenceError`.
   - **Other untrusted runs.** Any other untrusted run is redone in the log domain. The answer is still BP's, and
     `BPDiagnostics.log_domain` records the rerun; it replaces `exact_fallback`.
   - **Exact computation.** BP runs one only under the `check_evidence` opt-in, which is also where it checks the
     tolerance budget.
4. **Structural zeros decide directly (amends ADR 0014 decision 3).** A trusted run in which every contributing product
   is a structural zero has an exact zero. `conditional` sends such a column to `on_zero` without an exact rerun, and BP
   raises `ImpossibleEvidenceError` (decision 3).
5. **Exact entries (amends ADR 0016 decision 4).**
   - **Reading entries.** The exact fallbacks read every supported element type at its exact value. Integers and
     rationals are read as themselves; BigFloat, Float32 and Float16 entries at their dyadic values.
   - **Zeros.** `_dyadic(::Float64)` is unchanged, because it is proved in Lean. Zeros are normalised where they are
     used instead, so they no longer drag a factor's exponent down to `2^-1074`.
   - **`FactorDomainError`.** `FactorDomainError(:exact)` now means only a non-finite entry or an unsupported type.
     `FactorDomainError` gains the backends `:multiply`, `:marginalize`, `:normalize` and `:calibrate`, for integer
     overflow outside a posterior run.
6. **Bounded inputs (extends ADR 0015).**
   - **No declared count is an allocation size.** Readers grow lists as items are read and compute table sizes as
     checked products. They take two keyword limits, `max_states = 65_536` (the states of one variable) and
     `max_table_cells = 2^27` (the cells of one table). `read_network` and the zoo's `[parser_options]` pass both
     through. Exceeding either is a `ParseError` that names the keyword.
   - **Nesting.** Nesting deeper than 512 levels is rejected before any recursive parse: a `ParseError` for JSON,
     Netica and HUGIN, and a `FormatError` for ACSet JSON.
   - **JSON input.** JSON is parsed from its bytes, never from a string that JSON3 may read as a path. Only the
     parser's own error becomes `ParseError`; interrupts propagate.
   - **ACSet JSON.** A repeated key in any object is rejected, and an integer column accepts only an integer literal.
     Lean's trusted `Lean.Json.parse` keeps the last copy of a repeated key and reads `1e0` with exponent 0, so the
     Lean pipeline from text accepts what Julia rejects (`docs/LEAN-JULIA-DISCREPANCIES-2026-09-30.md`, section 18).
   - **Writes.** `write_network`, `write_ir_json` and `write_uai_evidence` serialise first and replace the target
     atomically, so a rejected or failed write never truncates an existing file.
7. **Unset references (extends ADR 0013 decision 3).** An unset `Ref` attribute is structurally valid, and
   `validation_errors` accepts it. When an operation needs its value:
   - an in-memory operation raises `MissingAttributeError(part, id, attr)`;
   - a certificate exporter raises its coded error, such as `OpenCertificateError(:unset_reference)`, because export
     errors carry a code and a path.

   `BayesNetError` equality now compares kernel fields structurally, as ADR 0013 decision 7 requires of every field.
8. **Deterministic certificate bytes.** A DVE certificate is an `OrderedDict` in its schema's property order, with
   nested objects in schema order too, so its bytes no longer depend on Julia's string hash. The version-1 fixtures
   were recaptured once, on 2026-10-02. `trace_decision_elimination` still writes `Dict` JSON; nothing pins its bytes.

## Consequences

- **Tolerated entries.** Every backend now answers when the evidence rules out a tolerated negative entry.
  InfluenceDiagrams' indeterminate example (vignette 06) now places its entry on the evidence.
- **More exact fallbacks.** More queries take the exact fallback. Where products underflowed, the cells are now
  correctly rounded. The extra check costs a few percent: variable elimination on `water` went from 23.8 ms to
  26.2 ms.
- **Belief propagation.** BP raises on impossible evidence in milliseconds again: 5 ms and 3.3 MiB on a 20×20 grid. BP
  never returns the junction tree's answer.
- **The input limits.** A file is rejected only above 65,536 states for one variable or 2^27 cells in one table. The
  largest committed or offline zoo model has 100 states and 280,000 cells, and the deepest nesting is 7 levels.
- **The diagnostics rename.** `BPDiagnostics.exact_fallback` is renamed `log_domain`. The packages are unregistered,
  so there is no alias.
- **Certificates.** Certificate bytes changed once.
- **Duplicated code.** Inference's exact-entry conversion repeats BayesianNetworks' `_exact_entry`. The one definition
  of ADR 0016 decision 2 could absorb it.
- **Lean.** None of these guards is modelled in Lean. The exact DVE algorithm, the label order and the `sum_out` keep
  rule are unchanged. A version-2 solution from a run that accepted a tolerated entry lies outside the nonnegativity
  premise of `certificate_tables`.
