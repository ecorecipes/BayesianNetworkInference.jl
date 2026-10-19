# ADR 0020: Lenses and open-game equilibria for open influence diagrams

Date: 2026-10-19. Status: accepted (2026-10-19).

## Context

ADR 0018 made optimisation of open influence diagrams contextual. A component `f : X → Y` is
valued against a prior `μ` on its inputs and a continuation `c` on its outputs, and the
contextual identity says what context each part of a serial composite `f ; g` receives:

`EU_{f;g}(σ, τ; μ, c) = EU_f(σ; μ, c_τ) = L_σ + EU_g(τ; ν_σ, c)`.

The proofs show that the optimum does not compose: components optimised one at a time, each
against the context it receives with nothing downstream, can compose to less than the optimum
(`isolated_optimisation_fails`). ADR 0018 recorded "multi-boundary lens or open-game
formulations of optimisation" as out of scope for the first release.

Open games (Ghani, Hedges, Winschel and Zahn, 2018) compose a weaker solution concept than the
optimum: equilibrium. Their contexts are pairs of a history and a continuation, which are the
`(μ, c)` above, and the transport of a context through a component is a lens: a forward map on
states and a backward map on values.

## Decision

1. **Each reward kernel is a lens.** For `f = (K, W) : X → Y` the forward map is the boundary law
   `μ ↦ ν = μ K` (`pushforward`, Julia's `boundary_law`) and the backward map is the
   continuation `c ↦ W 1 + K c` (`continuation`, Julia's `continuation_value`). Serial
   composition of reward kernels is sent to composition of lenses (the forward maps compose
   covariantly, the backward maps contravariantly), and identities to identities, provided the
   later kernel is normalised. The prefix reward `L_f(μ)` is the remaining term of the
   contextual identity and is not part of the lens.
2. **Open games over reward kernels, serial composition only.** A game `X → Y` has a type of
   strategies, a play (a normalised reward kernel for each strategy) and an equilibrium predicate
   over contexts `(μ, c)`. A *decision* game is one component whose decisions are controlled
   together: its equilibria in a context are the strategies that maximise `EU(σ; μ, c)` there.
   The serial composite `G ; H` plays the composite kernel, and `(σ, τ)` is an equilibrium in
   `(μ, c)` when `σ` is an equilibrium of `G` in `(μ, c_τ)` and `τ` an equilibrium of `H` in
   `(ν_σ, c)`. The tensor is left out: the context a tensor factor receives depends on the other
   factor's input state, which a context `(μ, c)` cannot express without a residual.
3. **What is proved, in the Lean project** (module `Lens`):
   - the lens laws of decision 1;
   - serial composition of games is associative and unital up to the reassociation of strategy
     pairs, because the induced contexts compose by the lens laws;
   - for two decision games, an equilibrium of the composite is exactly a pair from which neither
     component can gain by changing its own strategy, holding the other fixed, measured by the
     composite's expected utility (a person-by-person optimum);
   - a global optimum is an equilibrium, for every game built from decision games by serial
     composition;
   - the converse fails: a coordination witness has an equilibrium worth 1 where the optimum is 2;
   - in the obstruction witness of ADR 0018, the equilibria are exactly the optimal strategies,
     so the isolated choice is not an equilibrium.
4. **What Julia provides** (`CategoricalInfluenceDiagrams.jl`, `src/games.jl`), for a chain of
   components `f₁ ; … ; fₙ` composed with `⋅`:
   - `component_contexts`, the context `(μᵢ, cᵢ)` each component receives under a strategy of the
     whole chain, computed by the forward and backward passes of the lenses;
   - `regrets`, for each component the gain of its best response in its context over its current
     value, by `contextual_optimize`; `is_equilibrium` when every regret is at most a tolerance;
   - `best_response_dynamics`, which replaces one component's strategy at a time by a best
     response while that strictly improves it, and stops at an equilibrium. The composite's
     value never decreases, so it terminates.

   Each component owns its decisions: equally named decisions in two components are refused.
   Strategies are legal (rule 3) because they come from `contextual_optimize`. The chain is
   composed with `compose`, which checks rule 6 unless `no_forgetting = false`: limited-memory
   chains, where equilibria and optima differ most, need that option.
5. **Equilibrium is not optimisation.** The documentation says plainly that an equilibrium can
   be worse than the optimum, that best-response dynamics finds an equilibrium and not the
   optimum (the local optima of single policy updating in limited-memory influence diagrams,
   Lauritzen and Nilsson 2001), and that `optimize` on the closed composite remains the way to
   find the optimum. Equilibrium is what composes; the optimum is not.

## Consequences

- A component's context is computed compositionally for any number of boundaries, so a long
  chain can be analysed component by component without building the closed composite's joint
  law in one piece.
- The Lean results are about the reward-kernel semantics, like `Obstruction.lean`. That the
  play of a decision game is the `interpret` of a diagram is the existing interpretation
  theorem; that Julia's chain contexts are Lean's is checked numerically, not proved.
- Open: the tensor of games (decision 2), behavioural strategies, and subgame-perfect
  refinements that would rule out equilibria sustained by unreachable information states.
