# ADR 0002: Two axis conventions, one documented conversion

Date: 2026-09-06. Status: accepted.

## Context

Interchange formats (Netica, GeNIe, HUGIN, UAI) lay CPTs out row-major over `(parents..., child)` with the child fastest;
BIF `table` puts the child slowest; DSC enumerates the first parent fastest. Categorical composition is a matrix product
that wants outputs as rows. Mixing these silently is the most common bug class in BN software.

## Decision

- User-facing / IR convention (SPEC §8.2): a CPT for `P(Y | X1..Xk)` is an array with
  `size == (n(X1), ..., n(Xk), n(Y))`, normalised over the last axis. Parent order is the mechanism's `input_position`.
- `FiniteKernel` internal convention (`MarkovCategories.jl`): `table` has **output axes first, then input axes**;
  a state `I → X` is a vector; composition is a matrix product after reshaping.
- Exactly one pair of documented conversions, `cpt(parents, child, table)` and `cpt(kernel)`, moves between them.
  Format readers convert row-major data with `from_rowmajor(vals, dims) = permutedims(reshape(vals, reverse(dims)), N:-1:1)`.
- Kernel binding to a mechanism checks axis *labels* against the variable's `state_name`s in `state_position` order.

## Consequences

Every package's `CLAUDE.md` repeats the convention verbatim. Tests pin it with a brute-force joint (`asia`:
`P(dysp = yes) = 0.4360`). SPEC §10.2 is amended to state both conventions.
