# ADR 0007: MarkovCategories.jl deviations from the specification

Date: 2026-09-07. Status: accepted.

## Context

The 2026-09-07 review of the ecosystem found four places where `MarkovCategories.jl` departs from SPEC or from the
implementation plan without a record (SPEC §60 rule 23 requires deviations to be recorded in an ADR), and one place
where the implementation was weaker than both SPEC and the Lean model it claims to mirror:

- SPEC §54 lists `KernelNormalizationError` among the descriptive exception types; the package shipped
  `NotNormalizedError`. `BayesianNetworkFormats.jl` exports an unrelated `NotNormalizedError(id, index, total)` for a
  non-normalised row of a file table, so `using MarkovCategories, BayesianNetworkFormats` — the natural pair for a
  `BayesianNetworks.jl` user — left the name ambiguous and made `@test_throws NotNormalizedError` mean whichever
  binding won.
- `FiniteKernel(X, Y, [1.5 -0.5; -0.5 1.5])` was accepted and `is_normalized` returned `true`: only column sums were
  checked. Such an object is not a morphism of FinStoch, yet it composes silently, and the Lean model in `proofs/`
  defines `Stochastic := Nonneg ∧ Normalised` (`Finite/Kernel.lean`), so the laws proved there did not cover what
  Julia admitted. `NaN` and `Inf` were rejected only incidentally, through the column sums.
- SPEC §10.2 prescribes `FiniteKernel{T,N}` with `table::Array{T,N}`; the package had `FiniteKernel{T}` with an
  abstract `Array{T}` field, so every table access was a dynamic dispatch and `@inferred probability(k, …)` failed.
- SPEC §10.1 requires state spaces to be "serializable"; this package has no JSON layer.
- The plan's Lean layer for this package was `Markov/Syntax.lean` stated against Mathlib's abstract
  `CopyDiscardCategory`/`MarkovCategory` classes; what exists is a concrete finite model.

## Decision

1. **`KernelNormalizationError`.** The kernel normalisation exception carries SPEC §54's name. There is no
   `const NotNormalizedError = KernelNormalizationError` alias: an alias would re-export the clashing name and defeat
   the purpose of the rename. Downstream references (`BayesianNetworks.jl/src/semantics.jl`,
   `BayesianNetworkInference.jl`'s re-export, `factors.jl`, `junction_tree.jl` and their tests) were updated in the
   same change. `BayesianNetworkFormats.NotNormalizedError` keeps its name: it is a different failure (a row of a
   file table) with different fields.
2. **Nonnegativity is part of the contract.** A `FiniteKernel` built with the default `check=true` is a morphism of
   FinStoch: every entry is finite and at least `-atol`, and every column sums to one within `atol`. The first
   offending entry raises the new `KernelEntryError`, which carries the `CartesianIndex` in the internal
   `(codom axes..., dom axes...)` layout, the value, and the optional variable name; `NaN` and `±Inf` raise it as
   "is not finite" rather than being caught later by the column sums. Small negative entries are tolerated up to
   `-atol` because tables read from files and produced by floating-point arithmetic round below zero.
   `assert_normalized` performs both checks (it is the function the constructor calls), the exported predicate
   `is_stochastic` is their conjunction — the Lean `Stochastic` predicate — and `is_normalized` keeps its narrow
   meaning of "the columns sum to one". `check=false` skips *both* checks and nothing else: the shape is always
   checked. It remains supported, because the law tests need deliberately unnormalised kernels to show that discard
   is natural only for normalised ones, and because `compose`/`otimes` of checked kernels are normalised by
   construction and are not re-checked.
3. **`FiniteKernel{T,N}`.** The rank is a type parameter and the field is `table::Array{T,N}`, as in SPEC §10.2. `N`
   is deduced from the table passed to the constructor; the bare name `FiniteKernel` remains a `UnionAll` usable in
   type annotations such as `Dict{KernelRef,FiniteKernel}`, and no downstream package parameterises the type. The
   number of axes of a `FiniteSpace` is runtime data, so the *rank* of a composite cannot be inferred statically:
   `compose_kernel` and `tensor_kernel` infer `FiniteKernel{T}` (element type known, rank not), while
   `probability` and `kernel_matrix` are fully inferable. `probability` folds the strides by hand and asserts an
   `Int` linear index for that reason, and `tensor_kernel` asserts the element type of its broadcast result.
4. **Serialisation lives in `BayesianNetworks.jl`.** `FiniteSpace` and `FiniteKernel` have no JSON form here. The
   model layer serialises whole `BayesModel`s (schema, mechanisms and kernels together) and is the only place where
   a version tag for the serialised form makes sense; duplicating a kernel-level format would give two schemas to
   keep in step. This is the "serializable" requirement of SPEC §10.1 read as a property of the ecosystem, following
   the package split of ADR 0001, and it is why this package has no JSON dependency.
5. **The Lean layer here is a concrete finite model.** `proofs/MarkovCategoriesProofs` defines finite kernels over ℝ
   with `Nonneg`, `Normalised` and `Stochastic` predicates and proves the category, monoidal, comonoid, coherence and
   naturality laws for them, plus both directions of "discard is natural iff normalised" and "copy is natural iff
   deterministic". The abstract statements against Mathlib's classes were not written; the abstract-level work that
   the plan located here (open networks, structured cospans) lives in `BayesianNetworks.jl/proofs`. A concrete model
   is what the Julia code is: it lets each Julia testset name the theorem it mirrors, which the abstract classes
   could not.

## Consequences

- `NotNormalizedError` no longer exists in `MarkovCategories`; code that caught it catches
  `KernelNormalizationError`. Because no package has a release, no deprecation alias is provided.
- `assert_normalized` and the `FiniteKernel` constructor can now raise `KernelEntryError` where they previously
  succeeded. A caller that deliberately builds a signed or unnormalised table must pass `check=false`, as the law
  tests do. `BayesianNetworks._check_kernel` wraps only `KernelNormalizationError` into `UnnormalizedKernelError`,
  so a negative kernel bound to a mechanism surfaces as `KernelEntryError`.
- `normalize` on a kernel with a zero column now throws an `ArgumentError` naming the input index instead of a
  `KernelNormalizationError` carrying a fabricated `max_deviation` of `1.0` and `atol` of `0.0`.
- `Base.names(::FiniteSpace)` is superseded by `axis_names`; the `Base.names` method is kept, without a deprecation
  warning, because `BayesianNetworks.jl` and `InfluenceDiagrams.jl` call it.
- SPEC §54's name list and §10.2's type signature are now met exactly; SPEC §10.1's "serializable" is met at the
  model layer only, and the plan's `Markov/Syntax.lean` is not implemented.
