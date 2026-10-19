# ADR 0019: Reward certificates for open influence diagrams

Date: 2026-10-17. Status: accepted (2026-10-17).

## Context

`CategoricalInfluenceDiagrams.jl` interprets an open influence diagram `f : X → Y` under a fixed
deterministic strategy as a reward kernel `(K_f, W_f)` (ADR 0018, decision 3). Its Lean project
proves facts about the same semantics: that the interpretation is functorial on
representatives, that the contextual identity holds, and that isolated optimisation fails. ADR
0018 left the Julia-to-Lean refinement open. Nothing ties a diagram that Julia built, and the
numbers Julia computed for it, to those theorems.

The ecosystem has two models of how to do this.

- `CategoricalBayesianNetworks.jl` exports `OpenNet.RawCertificate/v1`, a structural record of
  an open Bayesian network. Its Lean reader (`CertificateJSON.lean`) decodes the record, and
  `Network.check` decides whether it is a valid open network (`read_sound`, `checkedNet`). The
  record carries no numbers.
- `InfluenceDiagrams.jl` exports `ecorecipes.dve-certificate`, the model data of a closed
  diagram. Cells are binary64 words (`{"f64": word}`), optionally with exact rational
  companions checked against them by nearest-even rounding. Version 2 also records Julia's
  solution, and its Lean checker compares that solution with an exact run on the certificate's
  numbers.

An open influence diagram needs the first's structure, extended with decisions, utilities and
observability flags. It also needs the second's numbers, extended with a strategy and Julia's
reward kernel.

## Decision

1. **A new format, `OpenInfluence.RewardCertificate/v1`.**
   - It records one open influence diagram as Julia holds it, which may be a composite. It also
     records the diagram's observability flags, one deterministic strategy, the diagram's
     numerical tables, and the reward kernel `(K, W)` that Julia's `interpret` computed for that
     strategy.
   - `export_reward_certificate` in `CategoricalInfluenceDiagrams.jl` produces it as
     JSON-compatible data, in schema key order.
   - A Lean executable in the package's proof project, `lake exe check_certificate`, checks it.
2. **The network is CategoricalBayesianNetworks' raw network.**
   - The structural part is the body of `OpenNet.RawCertificate/v1`: dense 0-based variable
     indices, ranks, positioned state rows, generators with tagged references and positioned
     input rows, and attributed boundary rows.
   - Decisions are generators whose `kernelRef` is `policy(decision name)` and whose input rows
     are the decision's information inputs. This is the convention `instantiate` already uses.
     A chance mechanism may not carry a policy reference.
   - The Lean side reuses `Network.check` and its validity proof unchanged.
3. **What the format adds to that network.**
   - **Flags:** an observability flag for each boundary row.
   - **Chance tables:** one per chance generator.
   - **Utilities:** each with a name, positioned scope rows and a table.
   - **Strategy:** one policy per distinct decision name, keyed by name, because equally named
     decisions share a policy (ADR 0018, Amendment item 4).
   - **Result:** Julia's `K` and `W`.
   - **Cell layout:** every table is a flat list of cells in lexicographic order of its axes,
     with the last axis varying fastest and states in `state_position` order. The axes are:
     - chance tables: parents in input-position order, then the child;
     - utilities: scope in position order;
     - policies: information inputs in position order, each cell holding a 1-based action
       position;
     - `K` and `W`: input rows, then output rows.
   - **Cell values:** values are InfluenceDiagrams' certificate values, so the Lean decoder and
     `Value.toRat` are reused.
4. **What acceptance means.** The checker accepts a certificate when all of these hold:
   - the network is valid;
   - every decision reads only internal or observed-input variables (rule 3 of ADR 0018,
     information locality, with the recorded flags);
   - every table has the right number of cells;
   - every chance cell is nonnegative;
   - every policy cell is in range;
   - every chance row sums to exactly one over the rationals;
   - the recorded `K` and `W` are within a tolerance `τ` of the exact reward kernel of the
     certificate's numbers.

   It reports the largest error it found. Acceptance is proved to mean that the decoded diagram
   is a valid `IDNet`, that it satisfies information locality, and that the recorded tables are
   within `τ` of `interpret σ f` from `Interpretation.lean`. That is the definition the
   functoriality and contextual-identity theorems are stated for.

   Binary64 rows seldom sum to exactly one. When the rows are not exactly normalised, the
   checker reports the largest deviation and makes no semantic claim. The `rational_exact`
   mode, with explicit companions, is the way to certify such a model.
5. **The trust boundary.** These steps are trusted, not proved:
   - JSON parsing;
   - the Julia exporter;
   - Catlab's colimits;
   - Float64 arithmetic in Julia.

   A certificate certifies the diagram it records, not the operations that built it. It omits
   decision precedence and no-forgetting, because the fixed-strategy semantics reads neither.
   Contextual values and contextual optimisation are not covered in version 1.

## Consequences

- Julia's interpretation of a concrete diagram can now be checked against the definition the
  Lean theorems are about, cell by cell, with the error stated.
- **Coupling to the two parent projects.** The Lean project uses CategoricalBayesianNetworks'
  raw-certificate types and InfluenceDiagrams' value decoder. A change to either format has to
  be made in step here.
- **The fixtures are a contract.** Fixtures written by the Julia exporter are committed beside
  the Lean project. The Lean workflow checks them, and the Julia tests compare the exporter
  with them byte for byte.
- **The operation itself stays unproved.** Lean proves composition functorial for its own
  syntax. A certificate of a Julia composite shows that Julia's numbers match Lean's semantics
  for the composite as recorded. It does not show that Julia's composite is the Lean composite
  of the parts. That gap, CategoricalBayesianNetworks' "unproved adapter lemma", stays open.

## Amendment (2026-10-19): the cost of the exact check

The checker no longer sums over every generator state for each recorded cell. For each input
state it accumulates the whole row of `K` and `W` in one pass over the generator states, into
an array indexed by the output state read (`Certificate/Fast.lean`). The fast errors are
proved equal to the cell-by-cell errors (`kErrorFast_eq`, `wErrorFast_eq`), so acceptance and
every soundness theorem are unchanged. The cost falls from `|X| · |Y| · |Z|` weight
evaluations to `|X| · |Z|`. On a benchmark with 8 input, 64 output and 1024 generator states,
the checker's run time fell from 3.5 s to 1.3 s of CPU, the rest being parsing and the other
stages.
