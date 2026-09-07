# ADR 0005: Lean 4 proofs live in `proofs/` inside each package

Date: 2026-09-06. Status: accepted; amended 2026-09-07 (see Amendment).

## Context

The ACSet schemas and the theorems about model manipulation (composition, intervention, policy substitution) are to be
checked in Lean 4. `CategoricalPopulationDynamics.jl` already keeps a Lake project in `proofs/`. Mathlib now provides
`CopyDiscardCategory`, `MarkovCategory` and `Stoch`; a Mathlib checkout is ~7 GB. The user chose the in-package layout
over a single separate Lean repository.

## Decision

`MarkovCategories.jl/proofs/`, `BayesianNetworks.jl/proofs/` and `InfluenceDiagrams.jl/proofs/` are Lake projects
pinned to `leanprover/lean4:v4.30.0` with Mathlib tag `v4.30.0` (the pin used by the open-systems Lean project). They share
one dependency checkout through Lake's `packagesDir` (`../../.lake-packages`, gitignored) so Mathlib is built once. The
ACSet schemas exist twice, once as Julia `@present` presentations and once as Lean `SchemaDesc` terms, and the two are
checked to agree: `lake exe emit_schema` writes `proofs/schemas/*.schema.json` in the ACSets.jl schema-JSON format,
`emit_schema --check` fails when the committed JSON has drifted from the Lean terms, and a Julia test compares that JSON
with `generate_json_acset_schema`. Each project has a `Makefile` (build, mdgen, html/pdf) and a `lean.yml` workflow that
runs `lake exe cache get` (through lean-action's `use-mathlib-cache`), `lake build --wfail`, the axiom audit and
`emit_schema --check`.

## Consequences

Proofs sit next to the code they describe and are CI-checked (unlike the CPD precedent). Shared definitions cross
projects through Lake path dependencies (`InfluenceDiagrams.jl/proofs` requires `BayesianNetworks.jl/proofs`).
If the shared `packagesDir` proves unreliable, each project falls back to its own checkout; olean downloads are
already shared via Mathlib's per-user cache directory.

## Amendment (2026-09-07)

Three claims in the original record were not true of what was built, and are corrected above and in the SPEC revision
note:

1. *"whose structured-cospan library is reused"* — no Lake project requires `open-systems` or vendors any
   structured-cospan file. The pin is shared; the library is not. The open-network closure theorem promised by SPEC §61
   was for a time absent as a result; it is now proved directly on the finite model, without any category-theoretic
   library, in `BayesianNetworks.jl/proofs/BayesianNetworksProofs/Finite/Open.lean`
   (`composeNet_target_injective`, `compose_input_exogenous`, `compose_exogenous_input`, `composeTopo`, `compose`,
   `closed_compose`), along with Proposition 3 in split form (`marg_joint_compose_split`). The decision recorded here
   stands; only the justification for the Mathlib pin changes.
2. *"the BayesianNetworks project is the source of truth for the ACSet schemas"* — the Lean terms and the Julia
   `@present` schemas are two hand-written definitions. Nothing generates one from the other, so the honest description
   is that they are *checked to agree*, and a schema change must be made on both sides.
3. *"a `lean.yml` workflow that runs `lake exe cache get`"* — no workflow did. The workflows now set lean-action's
   `use-mathlib-cache: 'true'` explicitly, which is that step.

Two further facts about the shared `packagesDir` were discovered when the workflows were fixed and are recorded here so
the fallback in *Consequences* is judged on the right evidence. Lake takes the packages directory from
`lake-manifest.json`, not from `lakefile.toml` (`Workspace.materializeDeps`, `Lake/Load/Resolve.lean` in the pinned
toolchain), and `leanprover/lean4:v4.30.0` offers no environment-variable override; CI therefore rewrites both files.
And `InfluenceDiagrams.jl/proofs` requires `BayesianNetworks.jl/proofs` by *path*, so a lone clone of that repository
cannot `lake build` — its workflow checks out the sibling at a pinned `main`.
