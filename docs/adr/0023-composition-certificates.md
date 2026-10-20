# ADR 0023: Composition certificates

Date: 2026-10-20. Status: accepted (2026-10-20).

## Context

ADR 0018 (decision 6) left the refinement from Julia to Lean open, and ADR 0019 (decision 5)
said what a reward certificate does not show. A certificate checks Julia's numbers for the
diagram it records against the Lean semantics of that diagram. It does not show that a
diagram Julia built with `compose` behaves as the composite of its parts: Catlab's colimit is
trusted. The Lean functoriality theorem (`interpret_comp`) is about `IDNet.comp`, the Lean
composite, and nothing linked a recorded composite to it.

A full syntactic refinement would decode the two parts and the composite and check an
isomorphism of nets between the composite and `IDNet.comp` of the parts, with Catlab's
cocone maps as the witness. That is a large development (net isomorphisms over decoded
records, utility generators, isomorphism invariance of `interpret` for `IDNet`, transport
between decoded interfaces). The semantic content, that the composite's reward kernel is the
composite of the parts' reward kernels, can be checked directly.

## Decision

1. **A composition certificate** (`OpenInfluence.CompositionCertificate/v1`) is a JSON object
   with exactly four keys:
   - `format`;
   - `left`, `right`: the reward certificates (ADR 0019) of two parts `A` and `B`;
   - `composite`: the reward certificate of `compose(A, B)`.

   All three are under one strategy. Julia writes it with `export_composition_certificate`.
   In rational mode it takes one table of exact companions per certificate.
2. **The checker** (`lake exe check_composition`, module `Certificate/Composition`) accepts
   when:
   - each certificate is accepted;
   - the interfaces line up by attributes: `A`'s outputs are `B`'s inputs, and the
     composite's inputs and outputs are `A`'s inputs and `B`'s outputs;
   - every recorded cell of the composite is within `τ` of the expectation-semiring
     composite of the parts' recorded cells, for `K` and for `W`.
3. **Soundness** (`composition_sound_K`, `composition_sound_W`). On acceptance, at every pair
   of coordinates, the decoded composite's exact tables are within an explicit bound of the
   composite of the decoded parts' exact tables. The bound is `2τ` plus `τ` times sums of
   recorded cells. With `interpret_K_eq` and `interpret_W_eq` these exact tables are the
   decoded diagrams' `interpret`. So Julia's composite is the Lean composite of its parts,
   for the semantics, cell by cell, up to the stated bound. One SA-Pass claim,
   `cid.composition-certificate`, ties the prose to it.
4. **Not claimed.** That the composite's syntax is `IDNet.comp` of the parts' (an isomorphism
   of nets), and anything for `glue`, `otimes` or `oapply`, which a later version could add
   the same way (the tensor's cells multiply).

## Consequences

- The trust in Catlab's colimit is replaced, for the semantics, by a checked relation between
  three certificates. The structure is still trusted.
- Fixtures: `proofs/composition-fixtures/` holds two accepted compositions (the umbrella and
  oil-wildcatter splits, rational) and three rejected at the stages a corruption should
  reach. `make check-compositions` checks them, in the Lean workflow too.
- Cost: the composition stage sums over the middle coordinates for every pair of outer
  coordinates, `|X| · |Y| · |Z|` cell products of recorded numbers, on top of the three
  certificates' own checks.
