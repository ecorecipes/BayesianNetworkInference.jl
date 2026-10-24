# ADR 0024: Shared policies along a chain

Date: 2026-10-21. Status: accepted (2026-10-21).

## Context

ADR 0022 left two things open: "mixed equilibria of chains whose components share policies,
and a randomised search for such chains". A policy is shared when several decisions must
use it, either because they cannot be told apart (the absent-minded driver of Piccione and
Rubinstein, 1997) or because one rule is to be applied at several places, such as a harvest
rule used in every year of a chain.

Two facts shape the decision.

- `compose` refuses two decisions of one name when both parts had unique names (ADR 0018), so
  a chain cannot share a policy by repeating a name. Sharing has to be declared.
- The equilibria of ADR 0020 are the wrong notion for a shared policy. There each component
  best responds on its own. At the optimum of the absent-minded driver, continuing with
  probability `2/3` at both junctions, the first junction alone would gain `2/3` by
  continuing for sure. The optimum is not an equilibrium in that sense.

What holds at an optimum is a first-order condition. Change the shared policy towards
another one. The gains from changing each use alone, each valued in the context that the
others induce, then sum to at most zero. This is the multi-self consistency of Piccione and
Rubinstein, as made precise by Aumann, Hart and Perry (1997). With one use, it is the
ordinary condition that no unilateral change gains.

## Decision

1. **Shared policies are declared.** The chain functions take `shared`, a list of groups of
   decision names, possibly in different components, that share one policy. The strategy
   gives a policy for the first decision of each group. The others draw from it position by
   position, as a `StochasticPolicy` on their own spaces. Their information states and
   actions must agree in size; anything else is an `ArgumentError`.
2. **Regrets are first-order.** `policy_regrets(chain, kernels, utilities, σ; shared)` returns,
   for each policy (keyed by its group's first decision; every other decision is a group of
   its own), the largest, over deterministic replacements, of the sum over its uses of the
   gain from switching that use alone. Each use's value is affine in the policy table, so the
   largest sum is found one information state at a time. `is_policy_equilibrium` thresholds
   the regrets. For a policy with one use the regret is exactly the gain of changing that
   decision alone.
3. **The search is over randomised strategies.** `shared_policy_search(chain, kernels,
   utilities; shared, grid)` searches the behavioural strategies whose probabilities are
   multiples of `1/grid`, valuing each on the composite. It refuses a search larger than
   `max_strategies` before starting (`PolicySearchTooLargeError`). The grid contains every
   deterministic strategy, and it contains the optimum when `grid` is a multiple of the
   optimum's denominators.
4. **Proofs** (Lean module `SharedPolicy`):
   - paths of reward kernels have first-order terms;
   - a segment `(1 - t) A + t B` has first-order term `B - A`;
   - serial composition obeys the product rule (`HasDeriv.comp`);
   - at an optimum along a path, the first-order term has nonpositive expected utility
     (`first_order_optimum`);
   - hence, for a policy with two uses separated by any fixed kernels, the per-use gains sum
     to at most zero at an optimum (`shared_optimum_first_order`);
   - at the driver's optimum the per-use gains are `2/3` and `-2/3` (`per_use_gains`).

   Two SA-Pass claims tie the prose to these: `cid.shared-first-order` and
   `cid.per-use-gains` (weak).

## Consequences

- A chain with a rule used in several components can be checked for the condition that its
  optimum must satisfy, and searched over randomised strategies.
- The first-order condition is necessary, not sufficient: a strategy can satisfy it and not
  be optimal. `shared_policy_search` finds the optimum on its grid; `optimize` on the closed
  composite does not apply, since it searches deterministic strategies of distinct decisions.
- **Not proved:** that Julia's `policy_regrets` computes the per-use gains. The test suite
  checks it against finite differences of the tied chain's value on random chains, and
  against the exact unilateral gain for decisions of their own. The Lean theorem is stated
  for two uses; more uses follow from the product rule, which holds for any serial composite.
- No evidence is taken by the shared-policy functions: with evidence the first-order term of
  a conditional value is a ratio's derivative, which is not implemented.

## Amendment (2026-10-24): given evidence

The shared-policy functions take `evidence`, and chain contexts that carry evidence mass, as
`regrets` does (ADR 0021). The value is then the chain's conditional value `V = N / Z`, a
ratio, and the gain of a use is the first-order change of `V` from changing that use alone,
`(N_u - V Z_u) / Z`. Here `N_u` and `Z_u` are the evidence-weighted value and the evidence mass
with the use changed, each in its component's context. Each use's `N_u` and `Z_u` are affine
in its policy table, so the regret is again found one information state at a time.

The Lean module `SharedPolicy` proves the condition. The evidence mass and value along a path
have first-order terms (`HasDeriv.mass`, `HasDeriv.value`). At an optimum of the ratio, the
first-order term's value less `V` times its mass is at most zero
(`first_order_conditional`). For two uses, the per-use first-order changes sum to at most zero
(`shared_optimum_first_order_conditional`). One SA-Pass claim,
`cid.shared-first-order-conditional`, ties the prose to it.

The gains divide by the chain's mass, so a mass below binary64's normal range is an
`ArgumentError`, and an impossible observation raises `ImpossibleEvidenceError`.
`shared_policy_search` skips strategies under which the evidence is impossible, and raises
`ImpossibleEvidenceError` when every strategy of its grid is one.
