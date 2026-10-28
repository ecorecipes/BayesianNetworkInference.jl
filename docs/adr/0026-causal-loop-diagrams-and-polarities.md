# ADR 0026: Causal loop diagrams, and four polarities rather than three

Date: 2026-10-28. Status: accepted (2026-10-28).

## Context

Every network in this ecosystem is acyclic, because that is what a Bayesian network is. Every
system the networks are built to describe has feedback in it. The gap is not a defect of the
models, but it is a gap that the models themselves cannot state, and until now each vignette
had to apologise for it in prose.

Baez formalises a causal loop diagram as a graph whose edges are labelled by elements of a
monoid of *polarities*, which makes a labelling the same thing as a functor from the free
category on the graph into the delooping of that monoid; with Chaudhuri he goes on to three
kinds of morphism between labelled graphs -- refining, simplifying and finding motifs -- and to
a homology that counts the feedback a composite acquires that neither part had
(arXiv:2506.23375, *Graphs With Polarities*). That is a formalism this ecosystem can carry,
and it meets the existing one at three points:

- a conditional probability table determines a direction, by stochastic monotonicity, and
  monotone kernels compose, so reading a sign off each mechanism really does give a labelling
  in Baez's sense rather than a convention;
- a dynamic Bayesian network's transition template is an acyclic graph whose lags carry the
  feedback, so quotienting the time index recovers the loops;
- the simplifying morphism is the push-forward of labels along a map of graphs, and it needs
  the labels to be *addable*, because merging two factors makes two links parallel.

The basic sign monoid is `{+, -}`. Baez adds an element for "indeterminate" to get `{+, 0, -}`.
Neither is enough here, for a reason specific to having tables rather than a whiteboard: a
table that does not depend on a parent at all and a table that depends on it in no consistent
direction are different facts, and both get written `0` in the three-element monoid. The first
is a structural zero (vignette 01: clearing the litter leaves dieback at exactly its previous
value, because the litter sits downstream of the woody damage). The second is an interaction
(vignette 15: wild relatives are helped by an allele that works and burdened by one that only
costs, so the sign depends on a latent variable). Conflating them loses the first finding and
misreports the second.

## Decision

1. **A new package, `CausalLoopDiagrams.jl`**, off the main dependency chain:
   `CausalLoopDiagrams` → `BayesianNetworks` → `FiniteKernels`. No Catlab and no
   MarkovCategories, as ADR 0009 requires of everything but those two packages. Nothing in the
   ecosystem depends on it; it is a reading layer, not a foundation.

2. **Four polarities.** `POSITIVE`, `NEGATIVE`, `NO_EFFECT` and `INDETERMINATE` are the subsets
   `{0}`, `{+}`, `{-}` and `{0, +, -}` of the sign hyperfield. Those four are closed under the
   hyperfield's multiplication and its multivalued addition, so they form a commutative rig: a
   sub-rig of the power-set rig Baez constructs from the sign hyperfield in part 4 of the
   *Polarities* series. `*` composes links along a path, with `POSITIVE` as its unit and
   `NO_EFFECT` absorbing; `+` combines links in parallel, with `NO_EFFECT` as its unit. A rig
   and not a monoid, because the push-forward needs the addition.

   The two 4x4 tables are written out in `src/polarity.jl` rather than derived, and the Lean
   project holds a second copy of the same tables. Neither is generated from the other, so
   neither is the source of truth for the other: change both together, as with `SchBayesNet`
   and its Lean copy.

3. **A polarity is read off a table, never asserted.** `polarity(kernel, parent)` tests
   stochastic monotonicity in the *declared* order of the states, holding the other parents
   fixed, with `atol` slack per comparison: increasing gives `POSITIVE`, decreasing
   `NEGATIVE`, a table independent of that parent `NO_EFFECT`, and anything else
   `INDETERMINATE`. Two keywords say where the declared order is not the quantity the diagram
   is about -- `reversed` for a variable declared best to worst, `unordered` for one whose states
   carry no order at all, every link touching which is `INDETERMINATE`.

4. **Two readings of one network, kept apart.** `mode = :structural` reads each mechanism's own
   table, so the sign is what intervening on the parent would move; `mode = :observational`
   reads `conditional(m, child, parent)`, so the sign is the direction the two move together.
   They are different operations and stay different, exactly as `do_intervention` and `observe`
   do. `compare_diagrams` reports where they disagree, which is where the network is
   confounded.

5. **Delays are marked, and they decide what can be modelled.** A link may be `delayed`. A loop
   with a delay in it is one a dynamic Bayesian network holds as it stands, the delayed link
   becoming a lagged input, and `causal_loop_diagram(::DynamicBayesModel)` recovers such a loop
   from a transition template by `collapse_lags`. A loop with no delay anywhere asserts
   simultaneity; nothing in this ecosystem can hold one, and unrolling does not help, because
   there is no slice boundary for the closing link to cross. `instantaneous_loops` names them.

6. **A diagram is an ACSet**, `SchCausalLoop`: objects `Factor` and `Link`, homs `link_source`
   and `link_target`, attributes `factor_name`, `link_polarity` and `link_delay`. That is Baez's
   definition of a graph rather than an encoding of it. It is a `BasicSchema` value, not a
   Catlab `@present`, following `SchBayesNet`. Parallel links and self-links are kept;
   combining them is `coarsen`'s job on writing or `link_polarity`'s on reading, never a silent
   effect of `add_link!`.

7. **Every ARIA vignette opens with one.** The sketch comes before the four views, drawn by
   hand with its loops intact, and its job is to say in advance which loops the networks will
   cut and which the dynamic view will close. It is not a sixth view: it is the thing the views
   are an improvement on, and it is where their simplifications are stated instead of implied.

## Consequences

- The limits of the formalism become statable rather than merely true. A polarity carries no
  magnitude, so a diagram cannot say which of two loops through one factor wins -- which is why
  Baez's series moves on to rings, hyperfields and rigs with more in them than signs, and why
  nothing here pretends to rank loops. It carries no interaction, so the superadditivity the
  community view found (vignette 04: levers worth more alongside the intervention than alone)
  has no expression in a diagram at all. It carries no unordered variable, so it has almost
  nothing to say about a choice between incomparable options, which several of these decisions
  are.
- Two vignettes acquire a result they did not have: 09 and 17 each contain a loop with no delay
  in it, so each contains a simultaneity that its networks resolve by splitting the idea into
  an ordered chain. That was an unstated assumption and is now a named one.
- The Lean project is Mathlib-free, which is a deliberate departure from the other proof
  projects. The rig is a four-element type, so its laws are decidable, and a 7 GB dependency to
  run `decide` would be absurd. The price is that the composition theorem -- monotone kernels
  compose, which is what makes the reading a functor -- is checked numerically in Julia and left
  in `Roadmap.lean` rather than proved. It is the one gap worth closing, and closing it means
  taking the Mathlib dependency for that module alone.
- The package is a consumer of `BayesianNetworks` and of nothing else in the ecosystem, so it
  cannot break anything below it. It is not part of the conformance surface and is deliberately
  absent from `conformance/run.py`'s `JULIA_PACKAGES`: there is no external implementation to
  be conformant with.
