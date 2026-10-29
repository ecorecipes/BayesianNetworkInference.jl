# ADR 0027: Motifs, and what counts as an occurrence

Date: 2026-10-29. Status: accepted (2026-10-29).

## Context

ADR 0026 brought in `CausalLoopDiagrams.jl` and two of the three morphisms Baez and Chaudhuri
define between labelled graphs (arXiv:2506.23375, *Graphs With Polarities*): the simplifying
one is `coarsen`, the push-forward of labels along a map of factors, and the comparison of two
labellings of one graph is `compare_diagrams`. The third, **finding motifs**, was left out.

Leaving it out mattered more here than it would elsewhere, because the ecosystem had acquired
a corpus. The nineteen ARIA vignettes each open with a causal loop diagram drawn by hand for an
unrelated programme, and nobody had asked the collection a question that spans it: which shapes
recur, and does the same feedback appear in peatland hydrology and in bee disease? That is a
question about the subject matter rather than about any one model, and nothing in the stack
could pose it.

A motif search needs three decisions made explicitly, because each changes the answer and none
has an obviously right value.

## Decision

1. **A motif is an ordinary diagram, and an occurrence is a label-respecting embedding.** No new
   type: a motif is a small `CausalLoopDiagram` whose factor names are placeholders, which is
   exactly Baez's position that a motif is itself a labelled graph. `occurrences(motif, cld)`
   returns every injective map from the motif's factors to the diagram's under which each link
   of the motif is a link of the diagram with an admissible label.

   **Injective on factors**, which is the decision with teeth. Without it a three-factor chain
   occurs in a two-factor loop by visiting one factor twice, every long motif occurs in every
   short cycle, and the census becomes noise. The price is that a motif with a symmetry occurs
   once per symmetry -- a reinforcing pair contains two occurrences of `:mutual_amplification`,
   one each way round -- which is a property of counting maps and is documented rather than
   divided out.

2. **Labels match by containment in the subset order, not by equality.** The four polarities are
   the subsets `{0}`, `{+}`, `{-}` and `{0, +, -}` of the sign hyperfield (ADR 0026), so they
   carry a partial order for free, and `Base.issubset` is defined for it. A motif link matches a
   diagram's link when `issubset(diagram_label, motif_label)`.

   So a motif link labelled `INDETERMINATE` asks only that some link be there and matches
   whatever sign it carries, while one labelled `POSITIVE` matches a positive link alone. The
   wildcard of the search is therefore not a fifth value bolted on for the matcher; it is the
   top of the rig that was already there. The direction of the containment is the whole content
   of the decision, and it cuts the other way too: a diagram's link which is genuinely
   `INDETERMINATE` does **not** satisfy a motif asking for `POSITIVE`, because the diagram does
   not claim that sign. Getting this backwards is a silent failure -- every wildcard motif
   matches nothing -- and it is what the first implementation did.

3. **`occurrences` is not induced; `motif_census` is.** They answer different questions, and the
   defaults follow the question.

   `occurrences` allows the diagram further links among the matched factors, because the
   question is whether a shape is *present*. `induced = true` is available for the stricter
   reading.

   `motif_census(diagrams; size)` is induced, so that each set of `size` factors contributes
   exactly one shape -- the one it actually carries -- and the rows partition the sets counted.
   Were the census not induced, a set carrying a loop would be counted again as each chain
   inside it, and a shape's count would describe the vocabulary of shapes rather than the
   diagrams. The two defaults disagree on purpose, and each docstring says so.

   The census ranks by `support`, the number of diagrams containing a shape, rather than by
   total occurrences: a shape that appears forty times in one diagram is that diagram's
   business.

4. **Delays are not part of a shape unless asked.** `match_delay = false` throughout, because a
   motif is a question about shape, while the delay flag is a claim about representability
   (ADR 0026, decision 5) rather than about structure. `match_delay = true` makes a delayed and
   an instantaneous copy of one signed shape count separately, which is the right setting when
   the question is what a dynamic Bayesian network could hold.

5. **Canonical form by brute force, capped at seven factors.** Isomorphism classes come from
   minimising a sorted link list over every relabelling. For a motif that is at most `7! = 5040`
   comparisons; a canonical labelling algorithm would be a dependency and a correctness risk for
   no gain. Past seven factors `motif_key` raises `ArgumentError`, which also stops anyone
   mistaking it for a whole-diagram invariant.

6. **A library of ten named shapes**, `MOTIF_LIBRARY`, as an `OrderedDict`. It also gives
   `OrderedCollections` -- declared as a dependency since the package was created and until now
   unused anywhere in `src/` -- something to do.

   `:mutual_suppression` is in it deliberately: two negative links in series multiply to a
   positive loop, so mutual suppression is reinforcing, and that is the reading practitioners
   most often get wrong. `:opposed_pathways` and `:agreeing_pathways` are the same unlabelled
   shape with different signs, kept apart because `coarsen` merges the first to an
   `INDETERMINATE` link and the second to a signed one: a sketch that lumps them loses a
   trade-off it cannot then state.

## Consequences

- The census over the nineteen ARIA sketches yields three findings that no single vignette
  could, each cross-checked by a second route before being written down.

  **Not one of the nineteen sketches contains a self-link.** Zero, by the census at size one and
  again by reading the link tables directly. A self-link is the most compact way to say that a
  quantity persists, and `collapse_lags` *produces* them from dynamic templates, so the
  hand-drawn sketches are systematically externalising persistence into a second factor where
  the extracted diagrams would not.

  **Two-factor feedback is overwhelmingly reinforcing: eleven reinforcing pairs against three
  balancing.** Counted as motifs (`:mutual_amplification` in 7 sketches, `:mutual_suppression`
  in 4, `:predator_prey` in 3) and again as loops of length two. Over loops of every length the
  collection runs 21 reinforcing to 11 balancing, which agrees with the nineteen vignettes' own
  printed summaries.

  **`:opposed_pathways` occurs in exactly three sketches -- 12, 15 and 18.** Vignette 18 is the
  one where the lost trade-off was demonstrated by hand with `coarsen`; the search found it
  there and in two more, so 12 (`Deployment` to `Stability` through `Efficacy` and through
  `Harm`) and 15 (`Allele` to `Population` through `Protection` and through `Cost`) carry the
  same hazard without saying so.

- The limits are the formalism's rather than the search's, and they are worth stating because a
  census invites over-reading. A shape supported by every diagram describes the vocabulary and
  not the subject: the bare link, the chain and the collider are in all nineteen and say
  nothing. The counts are of factor sets, not of causal claims. And a polarity carries no
  magnitude, so a recurring shape is a question worth asking of each diagram rather than a
  finding on its own.

- `motif_census` is deliberately not a statistical test. The network-motif literature compares
  counts against a null model of rewired graphs, and nothing here does: nineteen diagrams of
  four to ten factors apiece, drawn by one hand, do not support that, and a *p*-value computed
  over them would be the most misleading number in the ecosystem. What the census reports is a
  count and the diagrams it came from, so that each can be read.

- The Lean project gains a module, `Order.lean`, holding six theorems about the new `subset`,
  all of them axiom-free and Mathlib-free like the rig laws. Four are the partial-order laws
  (reflexivity, transitivity, antisymmetry, `indeterminate` greatest) plus the incomparability
  of `positive` and `negative`. The other two are worth more than the bookkeeping: `mul_subset_mul`
  and `add_subset_add` say that **both rig operations respect the order**, so the matching
  relation is a rig congruence. That is what makes a motif of a *path* mean what it appears to:
  if each link of a path is admitted by the corresponding link of the motif, the path's polarity
  is admitted by the motif's path polarity. `add_subset_add` is the same statement for parallel
  links, which is what lets a motif survive `coarsen`. The Julia code relied on both and could
  not state either.

  The composition theorem of ADR 0026 -- that monotone kernels compose -- remains the one open
  gap, and is still the only thing in `Roadmap.lean`.
