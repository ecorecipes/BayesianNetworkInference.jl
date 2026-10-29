# ADR 0028: Mathlib for the composition theorem, and for nothing else

Date: 2026-10-29. Status: accepted (2026-10-29). Amends ADR 0026.

## Context

ADR 0026 gave `CausalLoopDiagrams.jl` a Mathlib-free `proofs/` project, the only one in the
ecosystem, and the reasoning was good: `Polarity` has four constructors, so every rig law it
satisfies is a finite case split closed by `rfl`, and taking a 7 GB dependency in order to
`decide` on four constructors would be absurd. ADR 0027 added `Order.lean` on the same terms,
six more theorems, all still axiom-free.

The same ADR 0026 left one thing unproved and said so in `Roadmap.lean`: that **monotone
kernels compose**. This is not a marginal lemma. `polarity(kernel, parent)` *reads* a polarity
off a conditional probability table by testing stochastic monotonicity, rather than asserting
one, and that reading is a labelling in Baez's sense only if the label of a path is determined
by the labels of its links. Everything downstream assumes it: `pathPolarity` is a product over
a list, `loop_polarity` multiplies entries of a matrix around a cycle, and ADR 0027's motif
matching uses `mul_subset_mul` to carry a motif along a path. Julia checked the identity
numerically against `compose_kernel` on all four sign combinations; Lean could not state it.

The obstruction was known and recorded. The theorem quantifies over real-valued kernels, and
its proof is the equivalence between dominance of upper tails and monotonicity of expectations
of increasing functions -- an Abel-summation argument. There are no real numbers in Lean core,
no `Finset.sum_range_by_parts`, and no `linarith`. `Roadmap.lean` therefore said, in as many
words, that closing the gap means taking the Mathlib dependency, and that it was the one gap
worth closing.

The choice was between leaving the package's central claim unproved in order to keep a build
property, or taking the dependency in the narrowest way that can be checked.

## Decision

1. **Take Mathlib, for one library.** `proofs/lakefile.toml` gains `require mathlib` pinned to
   `v4.30.0`, matching `lean-toolchain` and the nine sibling projects, and the shared
   `packagesDir = "../../.lake-packages"` of ADR 0005. A second `lean_lib`, `Theory`, globs
   `CausalLoopDiagramsProofs.Theory.*`, and `defaultTargets` is unchanged: `lake build` still
   builds the four Mathlib-free modules and nothing else.

2. **The confinement is checked, not asserted.** Nothing in `CausalLoopDiagramsProofs` imports
   `Theory`, and `make mathlib-free` greps every core module for an import of either `Mathlib`
   or `Theory` and fails if it finds one. CI runs that check as its **first** step, before
   anything is built, so the separation cannot rot quietly into a project that happens to still
   compile.

3. **Two audits, because the two libraries have different axiom claims.** `Audit.lean` keeps
   importing only the core and keeps reporting no axioms for the sixteen rig and order laws and
   `propext` for the three list theorems. `AuditTheory.lean` reports the new theorems, which
   inherit Mathlib's `propext`, `Classical.choice` and `Quot.sound`. That is the price of the
   real numbers; it is paid by every other proofs/ project here, and recording it in a separate
   file keeps the stronger claim legible rather than averaging the two together.

4. **The model is the Julia test, not a general probability theory.** A distribution on `n`
   states is a function `ℕ → ℝ` with mass one on `Finset.range n`; `tail n p k` is the upper
   tail, the quantity the Julia loop accumulates as it scans states from the top down; and
   `Dominates` is the comparison `polarity` makes between adjacent parent levels. Parent levels
   range over all of `ℕ`, since a matrix with finitely many rows extends by repeating its last
   row and that preserves monotonicity. `Theory/Dominance.lean` proves the analytic step and
   `Theory/Composition.lean` the composition theorem: `reads_comp` on all four sign
   combinations, and `polarity_comp` stating the conclusion in the rig of `Basic.lean`.

5. **`noEffect` is a theorem and `indeterminate` is not.** A constant link on either side makes
   the composite constant (`constant_comp_of_constant_left`, `..._right`), which is exactly
   `noEffect` being absorbing for `mul`. `indeterminate` needs nothing: it is the greatest
   element of the subset order, so it admits every behaviour and asserts none, and ADR 0027's
   `mul_subset_mul` already makes coarsening to it sound -- without Mathlib.

## Consequences

- **The package's central claim is now proved.** That `polarity(kernel, parent)` reads a
  labelling and does not merely assert one is what distinguishes this package from a drawing
  tool, and it is no longer resting on a numerical test over four hand-built tables.
  `Direction.polarity_mul`, the bridge from the analytic statement to `Polarity.mul`, is itself
  **axiom-free**: it is four cases closed by `rfl`, so the join between the two halves of the
  project costs nothing.

- **ADR 0026's Mathlib-free decision is amended, not withdrawn.** It still holds for the rig,
  the order and the loops, which is where its reasoning applied; `lake build` still takes
  seconds and `make audit` still prints "does not depend on any axioms" sixteen times. What
  changes is that the project is no longer *wholly* Mathlib-free, and the claim made for it has
  to be made per library from now on. The closing line of ADR 0027 -- that the composition
  theorem remains the one open gap -- is superseded by this record.

- **CI gets slower and more fragile, in the way the siblings already are.** The Lean workflow
  now redirects `packagesDir` in both `lakefile.toml` and `lake-manifest.json` and sets
  `use-mathlib-cache: 'true'`, so it fetches prebuilt oleans rather than building Mathlib. A
  cold local build spends minutes reading ~5 GB of oleans before elaborating anything; that is
  I/O and not elaboration, and a second build of the same files takes seconds. Worth knowing
  before diagnosing a hang: a `lean` process at 1% CPU with 7 minutes of wall clock is reading
  the dependency, not looping in a tactic.

- **Two things the theorem does not cover**, now the honest residue in `Roadmap.lean` in place
  of the gap it replaces. The Lean statement is for kernels with **one parent**, which is what a
  link of a causal loop diagram is, while `polarity(kernel, parent)` holds the other parents
  fixed and demands monotonicity for *every* configuration of them; that quantifier is not
  modelled. And the Lean statement is about **exact real arithmetic**, while the Julia test
  allows `atol` slack per comparison, so nothing proved here says that composing two
  within-tolerance tables stays within any particular tolerance. The numerical test remains what
  stands behind the floating-point case, and it is now a complement rather than the only
  evidence.

- **No Julia code changes.** The `polarity` docstring gains a pointer to the theorem, and
  `CLAUDE.md` records which library is which. The rig tables in `src/polarity.jl` and
  `Basic.lean` are untouched, so ADR 0026's requirement that the two be changed together is not
  engaged.
