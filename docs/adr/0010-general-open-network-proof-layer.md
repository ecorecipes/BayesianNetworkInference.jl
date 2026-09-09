# ADR 0010: Give general open-network syntax its own categorical proof layer

Date: 2026-09-08. Status: accepted.

## Context

ADR 0009 left the existing finite-model and schema proofs in
`BayesianNetworks.jl/proofs/`, while identifying a future general categorical
layer as work for `CategoricalBayesianNetworks.jl`. The old finite representation
uses subsets of apex variables as feet. It cannot express a copied output
occurrence, and its own-variable composition formula excludes pass-through
interfaces.

Those limitations are representation issues, not missing matrix identities.
Proving a category of stochastic kernels does not prove a category of open
network syntax. Nor should structural equality be replaced silently by equality
of numerical interpretations: a discarded internal mechanism remains part of
the syntax even when its stochastic interpretation contributes unit mass.

## Decision

Add `CategoricalBayesianNetworks.jl/proofs/` as a separate Lean project. It uses
Lean, Mathlib and mdgen 4.30.0 and shares the workspace dependency checkout.
The existing finite core and schema emitters remain where they are.

The categorical project is parameterised by an arbitrary typed mechanism
signature. An interface is a finite typed set. A network has a finite set of
generators, one generated variable per generator, an ordered parent list that
may repeat a variable, and an arbitrary typed output map. The canonical
variable representation is inputs plus generators. Input injectivity,
exogeneity and generator uniqueness therefore hold by construction; existence
of a finite ranking expresses acyclicity.

Arrow equality is precisely structural isomorphism: renumbering generators
while preserving labels, ordered incidences and boundary maps. Category,
tensor and copy/discard equations are proved by concrete isomorphisms, not
inserted into the quotient relation. The independent general-legged
presentation permits arbitrary variable and mechanism sets, an injective input
leg and an unrestricted output leg. Both representation round trips and the
exact correspondence of isomorphism classes are proved.

Composition is substitution along the actual output and input maps.
Its acyclicity is derived. Unique variable and attributed-apex mediators
establish the gluing/pushout universal property. The result has actual Mathlib
`Category`, `MonoidalCategory`, `SymmetricCategory` and `CopyDiscardCategory`
instances, including copied outputs, pass-through and empty feet.

Do not add a `MarkovCategory` instance to this raw syntax. Generator count is
invariant under its equality and increases under composition: postcomposing a
mechanism with discard does not identify it with mechanism-free discard.
`Examples.discard_not_natural` proves the obstruction.

The syntax construction needs no ecosystem proof dependency yet: it does not
use probabilities. A future interpretation functor can depend on the existing
finite-kernel semantics without moving the schema emitter or duplicating the
finite core. No Julia dependency is added.

## Consequences

- The new mathematical category is total on matching interfaces. The existing
  Julia `compose` wrapper has an additional name-freshness policy and can reject
  such a pair. That convenience wrapper is not certified as a total category
  operation by this result; it is unchanged.
- Signature types and labels can encode names, ordered state metadata,
  references and mechanism profiles, and structural isomorphisms preserve
  those values. A translation from validated Julia ACSets and their mutable
  part tables remains a separate refinement proof.
- The proof uses existence of rankings as its acyclicity presentation.
  Conversion of the Julia graph validator's output into a ranking certificate
  is not yet verified.
- The FinStoch interpretation functor for these general boundaries remains
  separate work. The new theorem is about syntax, not a numerical quotient.
- The package gets its own Lean CI workflow and a fail-closed axiom/source
  audit. Markdown, HTML and PDF are generated from the checked Lean source
  through `make docs`, following the ecosystem convention.
- The Catlab boundary of ADR 0009 is unchanged: this is a proof-project split,
  not an additional dependency of any Julia model or inference package.
