# ADR 0022: Behavioural strategies for open influence diagrams

Date: 2026-10-19. Status: accepted (2026-10-19).

## Context

ADR 0018 (decision 4) kept randomised (behavioural) policies out of the first release of
`CategoricalInfluenceDiagrams.jl`: "randomised policies follow only where a theorem covers
them". `interpret` refused a `StochasticPolicy`, although `InfluenceDiagrams.jl` instantiates
one and its Lean project proves, for its finite model, that a nonnegative strategy is a
mixture of deterministic policy tables and never beats the best of them
(`exists_deterministic_optimal_all`, "including limited memory").

Two things were missing for open diagrams:
- a semantics for behavioural strategies that composes, so that a randomised component can be
  valued in its context;
- a statement of when deterministic strategies suffice, now that values may be conditional
  on evidence (ADR 0021), a ratio rather than an expectation.

There is also a case where they do not suffice. Decisions with equal names share one policy, in
the Julia package and in the Lean syntax alike. Sharing a policy is the absent-minded driver of
Piccione and Rubinstein (1997): two decisions that cannot be told apart. There the best policy
can be randomised.

## Decision

1. **Behavioural strategies are interpreted.** A behavioural strategy gives every decision a
   table of action probabilities given its information. It is drawn from independently at
   every use. Decisions become generators with those tables, and the semantics of an open
   influence diagram is again a reward kernel. Julia's `interpret` accepts a
   `StochasticPolicy`, and so does everything built on it: contexts, `contextual_value`,
   `conditional_value`, the chain functions and the oracle comparisons. Reward certificates
   (ADR 0019) still refuse one, because the format records deterministic policies.
2. **Proofs** (Lean module `Behaviour`):
   - the behavioural interpretation is stochastic, functorial under serial composition,
     identities and tensor, and satisfies the contextual identity;
   - a deterministic strategy is the behavioural strategy of its point masses, with the same
     interpretation, by definition (`rfl`);
   - a mixture of reward kernels composes affinely on either side;
   - the expected utility of a mixture never exceeds the best component's (`eu_mix_le`);
   - given evidence, the conditional value of a mixture never exceeds the best conditional
     value among components of positive mass (`conditionalValue_mix_le`, the mediant
     inequality);
   - the absent-minded driver: one policy shared by two junctions is worth `4/3` randomised
     against at most `1` deterministic, and with a policy each the deterministic pair is
     optimal at `4`.

   Three SA-Pass claims tie the prose to these theorems: `cid.behaviour-functor`,
   `cid.mixture-sufficiency`, and `cid.absent-minded` (weak).
3. **The optimum stays deterministic.** `optimize` and `contextual_optimize` search
   deterministic strategies. This is justified when every decision has its own policy:
   - for expected utility, by `InfluenceDiagrams`' `exists_deterministic_optimal_all` on the
     bound model that `contextual_optimize` builds, whose context, a prior factor and a
     terminal utility, is within that theorem's unnormalised chance kernels;
   - given evidence, by the mixture bound above, which is the same argument for a ratio.

   `contextual_optimize` already refuses repeated decision names, the case where the
   absent-minded witness shows the deterministic optimum can be beaten. Equilibria (ADR 0020)
   may start from a behavioural strategy. Best responses are deterministic, so best-response
   dynamics ends at a deterministic equilibrium.

## Consequences

- A randomised management rule, such as a survey commissioned with some probability, can be
  valued in context, composed and compared with deterministic ones.
- The documentation says plainly when randomising cannot help and when it can.
- **Not proved:** that a behavioural strategy of an `IDNet` with distinct decision labels *is*
  a mixture of deterministic ones (Kuhn's theorem for this syntax). The Lean proof of
  sufficiency for the syntax goes through `InfluenceDiagrams`' finite model, not through
  `IDNet`. The tests check sufficiency numerically on random composites: no random
  behavioural strategy beats the deterministic optimum, with or without evidence.
- Open: mixed equilibria of chains whose components share policies, and a randomised search
  for such chains.
