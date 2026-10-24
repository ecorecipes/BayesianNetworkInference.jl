# ADR 0025: Colimit certificates and diagrams up to isomorphism

Date: 2026-10-21. Status: accepted (2026-10-21).

## Context

ADR 0023 checks a composite's numbers against its parts', up to a tolerance, for `compose`
and `otimes`. It left open three things:
- the structure: that Julia's composite *is* the colimit of its parts, which Catlab computes
  and nothing checks;
- `glue` and `oapply`, whose boundaries are neither a match nor a concatenation of the parts';
- the syntax: `IDNet` is not quotiented by isomorphism, so the semantics is a set of
  equations on representatives, not a functor out of a syntax category, and its invariance
  under isomorphism was not proved.

Catlab's colimits come with their cocone maps. Those maps are a witness that can be checked:
each part's variables, generators and utilities land in the composite. If the maps are
bijective on generators and utilities, keep labels, tables, parents and scopes, and join only
boundary variables, then the composite's semantics follows from the parts' exactly. Every
internal assignment of the composite then splits into the parts' internal assignments, and
each value comes from the part that generates it.

## Decision

1. **A colimit certificate** (`OpenInfluence.ColimitCertificate/v1`) is a JSON object with
   exactly five keys:
   - `format`;
   - `operation`, the operation's name;
   - `parts`, the reward certificates (ADR 0019) of the parts;
   - `composite`, the reward certificate of the composite;
   - `legs`, one per part, each `{variables, generators, utilities}`: the 0-based index in
     the composite's certificate of each of the part's variables, generators (chance
     mechanisms then decisions) and utilities.

   Julia writes it with `export_colimit_certificate`, for `compose`, `otimes` and `glue` (two
   parts) and for `oapply` (an undirected wiring diagram and its diagrams). The legs are
   Catlab's cocone maps, which the operations now expose internally. For `glue` the second
   part is recorded as glued, its glued inputs carrying their partners' names.
2. **The checker** (`lake exe check_composition`, which reads all three formats; module
   `Certificate/Colimit`) accepts when every reward certificate is accepted and the legs are a
   colimit witness (`Wired`):
   - the parts' generator maps together are a bijection onto the composite's generators, and
     likewise for utilities;
   - attributes, labels, targets, ordered parents, chance tables, recorded policies, ordered
     scopes and utility tables are preserved;
   - two distinct part variables share an image only when both are boundary rows of their
     parts;
   - every output row of the composite is one of its input rows or the image of a part's
     boundary row.
3. **Soundness is exact** (`colimit_exactK`, `colimit_exactW`, and on acceptance
   `colimit_sound_K`, `colimit_sound_W`). For all coordinates `x`, `y` of the composite:
   - `K_C(x, y) = Σ_t [t consistent] Π_i K_i(t_i)`;
   - `W_C(x, y) = Σ_t [t consistent] Σ_i W_i(t_i) Π_{j≠i} K_j(t_j)`;

   with no tolerance. Here `t` ranges over boundary coordinates of every part, and is
   consistent with `x`, `y` when rows with one image carry one state. This is the junction sum
   of the parts' semantics in the expectation semiring: for `compose`, the composite of reward
   kernels; for `otimes`, the tensor; for `glue` and `oapply`, the sum over the joined
   variables. With `interpret_K_eq` and `interpret_W_eq` (ADR 0019) the exact tables are the
   decoded diagrams' `interpret`. One SA-Pass claim, `cid.colimit-certificate`, ties the prose
   to it.
4. **Diagrams up to isomorphism** (Lean module `SyntaxCategory`):
   - `IDNet.Iso` is an isomorphism of the underlying nets with a bijection of utility
     generators keeping labels and scopes;
   - it is a congruence for serial composition, which is unital and associative up to it;
   - `interpret` and `interpretB` are invariant under it;
   - the semantics is a Mathlib `Functor` from the category `IDCat` of typed interfaces and
     diagrams up to isomorphism to `FinRew`, for every behavioural strategy.

   One SA-Pass claim, `cid.syntax-functor`, ties the prose to it.

## Consequences

- The trust in Catlab's colimit is replaced, for every operation, by a checked witness. The
  composite's semantics is then exact, where ADR 0023 bounds it by the tolerance. Composition
  and tensor certificates remain, since they check recorded cells and need no legs.
- The witness is checked on the decoded records. It is not turned into an `IDNet.Iso` between
  the decoded composite and `IDNet.comp` of the decoded parts: their interfaces are distinct
  Lean terms built from different records, and the semantic consequence is what the
  certificate proves.
- Only serial composition is quotiented; the monoidal structure of `IDCat` (tensor up to
  isomorphism with its coherence) is not developed.
- Fixtures: `proofs/composition-fixtures/colimit-*.json`, written by `colimit_fixtures` of
  `scripts/write_certificate_fixtures.jl`. Four are accepted: `compose`, `glue`, `oapply`
  with an outer port, and `otimes`. Three are rejected: a leg sending a generator to the
  wrong place, a composite whose recorded policy differs from its part's, and binary64.
- Cost: the structure check compares every pair of part variables, `O(V²)`, besides the
  reward certificates' own checks.

## Amendment (2026-10-24): the tensor up to isomorphism

The Lean module `SyntaxMonoidal` adds the tensor to the quotient, following
CategoricalBayesianNetworks' construction for nets and extending every bijection to utility
generators:
- isomorphism is a congruence for the tensor, and interchange holds up to isomorphism;
- the associator, unitors and symmetry are wiring along bijections of ports, and their
  naturality is a slide through wiring;
- `IDCat` is a symmetric monoidal category (`IDCat.monoidal`, `IDCat.symmetric`);
- for every behavioural strategy the semantic functor is braided strong monoidal
  (`braidedFunctor`), its comparison maps the canonical bijections of boundary assignments.

One SA-Pass claim, `cid.syntax-monoidal`, ties the prose to it. The colimit witness is still
checked on the decoded records, not turned into an `IDNet.Iso`. CategoricalBayesianNetworks'
certificate theorem takes such a net isomorphism as a hypothesis for the same reason: the
decoded diagrams' interfaces are distinct Lean terms, related by equalities that need casts.
