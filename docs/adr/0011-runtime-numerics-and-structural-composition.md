# ADR 0011: Separate structural composition and numerical solver contracts

Date: 2026-09-08. Status: accepted.

## Context

The exact finite-model DVE theorem and the general structural category do not
by themselves verify floating-point arrays or a name-restricting convenience
wrapper. The follow-up review found scale-dependent DVE evidence acceptance,
posterior feasibility shortcuts, misleading damped-BP convergence, repeated-slot
and reference-loss bugs, and a name-freshness restriction on composition.

Literal floating-point DVE/oracle equality is false: reassociation alone can
produce different answers. Natural discard is also false for syntax retaining
hidden mechanisms. Neither obstruction should be hidden by changing the meaning
of an existing theorem.

## Decision

Keep `compose` as the name-safe compatibility wrapper and expose
`compose_structural` as the structural sequential operation on matching valid
interfaces. Both perform the same pushout and validate the typed-interface rule;
only the compatibility wrapper demands unique names when both operands had them.
Unglued parts with equal names remain distinct. Code requiring unambiguous name
lookup must use explicit renaming or part ids; stable kernel references are not
rewritten implicitly.

Preserve every ordered input occurrence. Factor conversion identifies repeated
variable slots by diagonal extraction, while free expressions create one copy
per occurrence plus a retained wire. `VariablePort` includes `space_ref` in
equality, hashing, compatibility and reconstruction, with the old two-argument
constructor defaulting to `NoRef()`.

DVE checks no-forgetting and the absence of actions among external evidence's
causal ancestors before elimination. Unsupported policy-dependent conditioning
is rejected independently of likelihood scale, with exhaustive search named as
the alternative. Its numerical constancy diagnostic uses a relative tolerance
on each row; a large row cannot hide disagreement in a tiny row.

Every exact posterior entry point checks global nonzero mass, including
disconnected and all-observed cases. Empty inference queries retain their
unnormalized partition/evidence-mass meaning, so a returned zero scalar is legal.
Integer factor inputs promote to division-compatible posterior/message types.

BP convergence uses the undamped message-equation residual at the returned
iterate, not the damped step size. The residual is not a general marginal-error
bound and local support is not a complete feasibility test. An explicit
`check_evidence=true` option performs a VE feasibility pass, with its potentially
exponential cost visible; diagnostics record whether that check occurred.

Normalization tolerance is threaded through all model-facing inference and
oracle paths. The algebraic row-mass budget for a product of nonnegative kernels
is multiplicative, `(1 + atol)^n - 1`, evaluated with `log1p`/`expm1`, not `n*atol`
or an unchanged default. This is not a certified floating-point forward-error
bound. Tolerance acceptance does not silently renormalize a model or make
slightly negative/rounded entries into exact stochastic data. Exact theorems
retain their exact nonnegativity and normalization premises.

## Consequences

- Existing name-safe composition and name-lookup behavior remain available.
  The new structural operation may return duplicate names intentionally.
- Evidence rejected by DVE may still be a valid inference/decision problem for
  exhaustive search. Validation of a model remains distinct from a backend's
  supported domain.
- Approximate BP does not gain an unconditional exact-inference cost. Users
  choose whether global evidence feasibility must be certified numerically.
- Floating-point optimization results and oracle comparisons are numerical
  computations, not proofs of universal exact equality or policy stability.
  Error/optimality guarantees need explicit roundoff, denominator and gap
  hypotheses, or an exact/certified arithmetic implementation.
- Representation refinements, graph/message proofs and interpretation functors
  remain separately identified developments. No Catlab dependency moves into
  a model, inference or format package.
