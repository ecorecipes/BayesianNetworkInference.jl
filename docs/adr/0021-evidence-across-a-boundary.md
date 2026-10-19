# ADR 0021: Evidence across a boundary

Date: 2026-10-19. Status: accepted (2026-10-19).

## Context

ADR 0018 left evidence out of the first release of open influence diagrams. Its decision 6 says
that evidence composition "needs explicit positive evidence mass across a boundary", following
the research bundle, which proves nothing about it. So `interpret`, the contexts and the solvers
of `CategoricalInfluenceDiagrams.jl` refused a model with evidence.

Evidence breaks the two assumptions the contexts rest on:

- a reward kernel with evidence is not normalised: `K(y | x)` becomes the probability of
  reaching `y` *and* seeing the evidence;
- the contextual identity's first equation and the lens laws of ADR 0020 need the suffix's
  kernel normalised. The Lean theorem `eu_comp_general` gives the defect when it is not: the
  prefix's reward times the suffix's missing mass.

Conditioning each component on its own evidence and renormalising it does not repair this. The
likelihood of downstream evidence, as a function of the boundary state, is what updates
everything upstream of it, and renormalising discards it.

In `InfluenceDiagrams.jl` the expected utility of a model with evidence is the expected utility
given the evidence. Its exhaustive search maximises that ratio over strategies and skips those
under which the evidence is impossible. Its decision variable elimination needs evidence with no
action among its causal ancestors, since otherwise the evidence mass depends on the strategy.
ADR 0014 fixes what `ImpossibleEvidenceError` means and how a binary64 mass is trusted.

## Decision

1. **A component with evidence denotes a substochastic reward kernel.** Hard evidence on any
   apex variable, inputs and outputs included, multiplies the sum-product by the indicator of
   the observed state:
   - `K(y | x)` is the probability of reaching `y` with the evidence;
   - `W(y, x)` is the reward earned on those paths;
   - the column sums of `K` are the probability of the evidence given each input state.

   Nothing is normalised. The category of reward kernels does not need normalisation, so
   composition is unchanged, and the evidence of a composite is the union of its components'.
   Evidence on a shared boundary variable may be stated on either side or both, because an
   indicator is idempotent. Evidence on a decision's action is refused (`EvidenceOnActionError`),
   as `InfluenceDiagrams` refuses it.
2. **Contexts carry the mass.** A context on each side of a boundary becomes a pair in the
   expectation semiring:
   - a **state** `(ν, ρ)` on the inputs: the unnormalised law of the evidence so far and the
     reward-weighted law, pushed forward by `(K, W)` to `(ν K, ρ K + ν W)`;
   - an **effect** `(m, c)` on the outputs: the likelihood of the evidence still to come and
     the evidence-weighted value, pulled back to `(K m, W m + K c)`.

   Both are reward kernels, `I → X` and `Y → I`. A plain prior is the state `(μ, 0)` and a plain
   continuation the effect `(1, c)`, so the existing `prior` and `continuation` arguments take
   either. The pairing gives the evidence mass `Σ ν m` and the evidence-weighted value
   `Σ (ρ m + ν c)`. Both are the same on either side of every reward kernel, so they are the
   same at every cut of a chain. The **conditional value**, their ratio, is the expected
   utility given all of the evidence. With mass one everything reduces to ADR 0018 and
   ADR 0020.
3. **Julia** (`CategoricalInfluenceDiagrams.jl`):
   - `interpret(...; evidence)`, and the model forms, which use the model's evidence;
   - `boundary_state` and `continuation_effect`, the two passes;
   - `evidence_mass`;
   - `conditional_value`;
   - `contextual_value`, which with evidence is the evidence-weighted value;
   - `contextual_optimize(...; evidence)`, which maximises the conditional value over the
     legal strategies, skipping those of exactly zero mass, as the exhaustive search of
     `InfluenceDiagrams` does on a closed model;
   - `optimize(o, ...; evidence)` on a closed composite, which records the evidence on the
     model and calls `InfluenceDiagrams`' solvers with their rules;
   - the chain functions of ADR 0020, which take `evidence`; their contexts are then states and
     effects, and every regret is a gain in the chain's conditional value.
4. **ADR 0014 applies to `conditional_value`, the one entry point that divides by the mass.**
   - A binary64 run is trusted when its mass is finite and at least `floatmin` and none of its
     products of nonzero operands fell below `floatmin`, the rule the other packages share.
     The reward-kernel form checks the products conservatively, from the smallest nonzero
     entries of the state, the kernel and the effect. Otherwise the value is recomputed in
     `Rational{BigInt}` and rounded correctly (ADR 0016).
   - The diagram forms recompute the reward kernel from the tables, so a mass that only
     underflowed is resolved. The reward-kernel form has only its tables and decides on them.
   - Only an exactly zero mass raises `ImpossibleEvidenceError`.
   - Tolerated negative entries with evidence raise `IndeterminatePosteriorError` when the mass
     is within the budget `(1 + atol)^n - 1`, or when the exact fallback would be needed.
   - `evidence_mass` and `interpret` are unnormalised quantities, returned as computed, like
     ADR 0011's empty queries.
5. **Proofs** (Lean module `Evidence`):
   - the actions on states and effects are functorial with no normalisation hypothesis, and
     the mass and value pairings are the same on either side of a reward kernel;
   - the composite's evidence mass is the pairing of the prefix's boundary law with the
     suffix's likelihood, and its conditional value is computed at the boundary;
   - the first equation of the contextual identity holds without normalisation once the mass
     is carried (`eu_comp_pull`);
   - hard evidence on a boundary is idempotent and commutes;
   - the local-normalisation witness: a uniform binary state worth `1` when true, with
     evidence nine times likelier when it is true, has conditional value `9/10`, while the
     components normalised one by one give `1/2`;
   - for the syntax, an interpretation weighted by a likelihood on the generators' values,
     keyed by label as strategies are, is functorial under serial composition, and the
     likelihood one gives `interpret`.

   Five SA-Pass claims (`cid.evidence-*`, `cid.observe-idempotent`,
   `cid.local-normalisation`) tie the prose to these theorems.

## Consequences

- An observation anywhere in a chain can be conditioned on component by component, without
  building the closed composite's joint law. The two contexts at any boundary carry
  everything the rest of the chain needs.
- The API grows by four functions and one keyword, and the meaning of `prior` and
  `continuation` widens. Plain vectors keep their old meaning and arithmetic.
- The tests that expected evidence to be refused now check values against
  `InfluenceDiagrams`' `expected_utility` of the closed model with the evidence observed.
- **Cost.** With evidence, `contextual_optimize` scores every strategy through `interpret`.
  It does not delegate to `InfluenceDiagrams`' value-table search, whose `DecisionContext`
  semantics (`Σ_x μ(x) E[U + c | x]`) conditions on the prior's states, not on the evidence.
  It is exhaustive in the component's strategies, as before.
- Reward certificates (ADR 0019) still refuse evidence. A certificate for a substochastic
  reward kernel would need a checker that does not assume normalised chance rows.
- **Open:**
  - the tensor of evidence-weighted interpretations;
  - soft (likelihood) evidence in Julia, which the Lean weighting already covers;
  - a value-table search with evidence in a context;
  - the refinement of the Julia functions to the Lean ones, which remains a numerical check.
