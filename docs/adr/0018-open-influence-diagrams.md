# ADR 0018: Open influence diagrams and their compositional semantics

Date: 2026-10-16. Status: accepted (2026-10-16).

## Context

The categorical layer covers Bayesian networks only. `CategoricalBayesianNetworks.jl` composes
open networks as structured cospans and evaluates them in FinStoch (ADR 0009, ADR 0010). The
other parts of the ecosystem stop short of that:

- `InfluenceDiagrams.jl` has no open diagrams and no composition. Its "composing ecology and
  management" vignette builds one larger closed diagram. The original plan's
  `OpenInfluenceDiagram` was never built.
- The schema, `SchInfluenceDiagram`, extends `SchBayesNet` with `Decision`, `InformationInput`,
  `Utility`, `UtilityInput` and `DecisionPrecedence`. Every added hom points into `Variable`,
  `Decision` or `Utility`, and none leaves `Variable` or `State`. So Catlab's `OpenACSetTypes`
  over the variable-space feet that `CategoricalBayesianNetworks.jl` already uses applies
  unchanged.

The sealed research bundle `docs/categorical-extensions-2026-09-12/influence/` ("Serial
influence semantics", Lean, Rocq and HOL4, 14 September 2026) settles the mathematics for
serial composition within three declared source classes. Three of its results shape this
design.

- **Fixed strategies compose.** For a fixed legal strategy, an open diagram `f : X → Y`
  denotes a pair `(K_f, W_f)`. `K_f(x, y)` is the probability of reaching boundary state `y`,
  and `W_f(x, y)` the probability-weighted reward earned on the way. Serial composition is
  derived from the original source, not assumed:
  `K_{f;g} = K_f K_g` and `W_{f;g} = W_f K_g + K_f W_g`. This is kernel composition over the
  expectation semiring `(p, r)·(q, s) = (pq, ps + rq)`. It is the same division-free structure
  as the `(φ, ψ)` valuations of decision variable elimination, kept as products rather than
  ratios.
- **Optimisation does not compose component by component.** The bundle refutes "optimise each
  component on its own, then compose" with independent counterexamples: `1 < 4` (Lean),
  `1 < 3` (Rocq) and `1 < 10` (HOL4). It also proves that maximising pointwise at each hidden
  boundary state yields an illegal policy. What does hold is the contextual identity:
  `EU_{f;g}(σ, τ; μ, c) = EU_f(σ; μ, c_τ) = L_σ + EU_g(τ; ν_σ, c)`, where `c_τ` is the suffix's
  value function and `ν_σ` the prefix-induced joint law of hidden state and remembered
  information. Each component's optimum is attained relative to that context.
- **The boundary carries more than variables.** It carries the hidden state and the remembered
  (public) information, and the two must be told apart. A downstream decision may read only
  what is declared observable at the boundary.

The bundle states the limits plainly. None of the three tracks proves a category of
unrestricted open influence diagrams: arbitrary gluing, tensor, information-flow conditions in
general, evidence composition, and the refinement to the Julia implementation all remain open.

## Decision

1. **A new package, `CategoricalInfluenceDiagrams.jl`.**
   - It depends on `CategoricalBayesianNetworks.jl` (open-network machinery, Catlab) and
     `InfluenceDiagrams.jl` (schema, policies, expected utility, solvers).
   - This follows ADR 0009's rule: a module that needs Catlab lives in a categorical package.
     `InfluenceDiagrams.jl` stays Catlab-free.
   - `CategoricalBayesianNetworks.jl` does not take on the decision and inference stack.
   - The package is the decision-layer analogue of `CategoricalBayesianNetworks.jl`. It
     re-exports both parents' public names.
2. **Syntax: open influence diagrams as structured cospans.**
   - `OpenInfluenceDiagram = OpenACSetTypes(InfluenceDiagramUntyped, VariableSpaceUntyped)`,
     with the same multi-object feet as open Bayesian networks.
   - Each boundary variable additionally carries an **observability** flag: `:observed`
     (remembered information a later decision may read) or `:hidden`. Two feet with equal
     variables and states but different flags do not match (`InterfaceMismatchError`).
   - The typed-interface rule of `CategoricalBayesianNetworks.jl` extends as follows:
     1. Every variable has at most one generator, a chance mechanism or a decision.
     2. An input-foot variable has no generator in the apex.
     3. A decision's information inputs are internal variables or *observed* input-foot
        variables, never hidden ones. This is the bundle's information locality.
     4. Utility scopes may read any apex variable, inputs included.
     5. The derived causal-plus-information graph is acyclic.
     6. Composition preserves no-forgetting when both components and the boundary's observed
        set satisfy it. This is checked, not assumed.
   - Utilities and decisions stay inside components. Only variables cross the boundary; reward
     crosses only through the semantics.
3. **Semantics for a fixed strategy: the expectation-semiring kernel category.**
   - `interpret(o, strategy)` returns a `RewardKernel(K, W)`. `K` is a `FiniteKernel` from the
     input space to the output space, and `W` a nonnegative-or-signed table of the same shape.
   - Composition is `(K₁, W₁) ; (K₂, W₂) = (K₁K₂, W₁K₂ + K₁W₂)`.
   - Tensor is `(K₁⊗K₂, W₁⊗K₂ + K₁⊗W₂)`. Its proof is new; the bundle does not cover tensor.
   - Identities are `(id, 0)`. Copy and discard are lifted with zero reward.
   - Strategies compose by disjoint union of the components' policies.
   - The oracle is the closed composite: `instantiate` followed by `expected_utility`. Every
     composition law is tested against it.
4. **Optimisation: contextual, never isolated.**
   - The package provides no "optimise each part, then compose" operation.
   - It provides three things:
     - `optimize(compose(...))`, the existing solvers on the closed composite;
     - `contextual_value(f, σ; prior, continuation)` and `contextual_optimize(f; prior,
       continuation)`, which optimise a component against an explicit downstream value
       function `c` and an explicit boundary prior `μ`, following the contextual identity;
     - the bundle's isolated-optimisation counterexample, in a vignette and a test.
   - Randomised (behavioural) policies follow only where a theorem covers them. The first
     release uses deterministic policies, as `InfluenceDiagrams.jl` does.
5. **Proofs: a Lean project in the new package, built on the existing ones.**
   - `CategoricalInfluenceDiagrams.jl/proofs/` depends on the `CategoricalBayesianNetworks.jl`
     and `InfluenceDiagrams.jl` proof projects.
   - It extends ADR 0010's typed syntax with decision and utility generators. It proves:
     - the expectation-semiring kernel category, with monoidal structure;
     - functoriality of `interpret` for fixed strategies, under serial composition and tensor;
     - the contextual identity;
     - the isolated-optimisation obstruction as a theorem.
   - The bundle's Lean track is a guide and is cited by claim id (`INF-ISOLATED-OBSTRUCTION`,
     `INF-HIDDEN-OBSTRUCTION` and so on). It is not imported: the bundle is sealed, and its
     source class differs from this syntax.
   - SA-PASS claims cover every README sentence that states a theorem.
6. **Out of scope for the first release.** These are recorded as open, as the bundle records
   them:
   - evidence composition, which needs explicit positive evidence mass across a boundary;
   - stochastic-policy dominance;
   - history compression beyond declared observability;
   - multi-boundary "lens" or open-game formulations of optimisation;
   - the Julia-to-Lean refinement.

## Consequences

- **A ninth package** joins the workspace, with the usual layout, CI, docs on `gh-pages`, a
  Vignettes workflow and a Lean workflow. `scripts/scaffold.sh` creates the skeleton.
  `EcologicalBayesianNetworks.jl` may later use it for composed management examples, but does
  not have to.
- **Composition can go wrong in a new way.** A composed decision problem's value is not the sum
  of the parts' optimal values. The API makes the context explicit, so this cannot happen
  silently.
- **Interfaces must say what is observed.** Existing open Bayesian networks embed with every
  boundary variable `:hidden`, and nothing changes for them.
- **Reuse.** The `RewardKernel` category reuses `FiniteKernels.jl` for `K` and adds only the
  reward component. Its relation to the `(φ, ψ)` valuations is documented, not merged.
- **The bundle stays frozen.** This ADR supersedes none of it. It turns the parts of its
  results that fit a general typed syntax into package code and package proofs.
