# ADR 0013: Exception roots, one type per condition per layer, and error boundaries

Date: 2026-09-26. Status: accepted.

## Context

Item 15 of the 2026-09-24 review asked for four things: a common abstract error root, one name for the
normalisation error, every package's exceptions in an `errors.jl`, and `checkdocs=:exports` in all eight
documentation builds. Checking each request against the code found that the first two cannot be met as written, and
that the problems behind the requests lie elsewhere. ADR 0012 settled the most serious of them, the two names for
evidence of zero mass. This record settles the rest.

- **76 exception types in four shapes.**
  - `BayesianNetworks` has an abstract root, `BayesNetError`, with 29 concrete types. `InfluenceDiagrams` extends
    it through `InfluenceDiagramError` (15 types), and `CategoricalBayesianNetworks` defines
    `ConflictingKernelError` under it.
  - `EcologicalBayesianNetworks` has an unrelated root, `ZooError <: Exception`, with 9 types.
  - `FiniteKernels` (5 types), `BayesianNetworkFormats` (7), `BayesianNetworkInference` (4) and `MarkovCategories`
    (1) subtype `Exception` directly, with no root.
  - The two errors of the stand-alone submodule `BayesianNetworks.Graphviz` subtype `Exception` directly.

  Catching every typed error of the ecosystem takes a `Union` of 21 members drawn from six packages.
- **No single root is possible.** `FiniteKernels` and `BayesianNetworkFormats` are leaves with no ecosystem
  dependencies (ADR 0003, ADR 0009), so no package sees all eight. Julia allows one supertype, and a `Union` cannot
  be one. A single root would need a Formats → FiniteKernels edge or a ninth package.
- **The three normalisation names belong to three layers,** and their fields do not overlap:
  - `BayesianNetworkFormats.NotNormalizedError(id, index, total)`: a row of a file table;
  - `FiniteKernels.KernelNormalizationError(msg, max_deviation, atol, name)`: a FinStoch kernel;
  - `BayesianNetworks.UnnormalizedKernelError(variable, max_deviation)`: a kernel bound to a mechanism.

  ADR 0007 kept the first two apart on purpose. One name would bring back the export clash that ADR removed, or need
  the same Formats → FiniteKernels edge.
- **Invalid entries have holes.**
  - `FactorGraph` (`BayesianNetworkInference/src/factor_graph.jl`) checks nothing, so variable elimination on a
    graph whose one factor has the entries `[1.5, -0.5]` returns `[1.5, -0.5]`.
  - `bind_cpt(...; renormalize=true)` (`BayesianNetworks/src/semantics.jl`) rescales before any check, so a row
    `[-1, -3]` silently becomes `[0.25, 0.75]`.
  - `_check_kernel` in the same file wraps `KernelNormalizationError` into `UnnormalizedKernelError` but lets
    `KernelEntryError` through, as ADR 0007 accepted. `semantic_errors` collects only `BayesNetError`s, so it
    throws that error instead of collecting it.
- **The roots promise more than they cover.**
  - `BayesNetError`'s docstring calls it the supertype of every exception `BayesianNetworks` throws. The Graphviz
    pair is not under it, and `FiniteKernels` and `BayesianNetworkFormats` errors pass through the package's API.
  - The header of `InfluenceDiagrams/src/errors.jl` says that all its errors are `BayesNetError`s. The package also
    throws Inference's `ScopeError`, which is not one, at nine sites in `decision_elimination.jl`,
    `execution_trace.jl` and `valuation.jl`.
  - `CategoricalBayesianNetworks` defines `ConflictingKernelError` in its `evaluation.jl`, against its own stated
    convention that the whole hierarchy lives in `BayesianNetworks`.
- **Exceptions are defined in many places:** in `errors.jl` in three packages, in `exceptions.jl` in Formats, and
  inline in seven other files: `spaces.jl`, `kernels.jl`, `finstoch_model.jl`, `factors.jl`, `compile.jl`,
  `log_variable_elimination.jl` and `evaluation.jl`.
- **`checkdocs=:exports` would loosen six packages.** Documenter's default is `:all`, and neither setting checks that
  an export has a docstring. The real gap was that every build set `warnonly` and nobody ran the builds. Strict
  builds and a docstring test in every package have since closed it.
- **Three names are frozen.** The pinned Rocq checkers compare exception names as bare strings: the domain
  certifier's `Rejections.v` compares `ImpossibleEvidenceError` and `IrregularDiagramError`, and the
  constructor-schedules checker's `NativeOrderRecords.v` compares `ScopeError`. The conformance adapters record
  `string(typeof(e))` after loading `BayesianNetworks`, `BayesianNetworkInference` and `InfluenceDiagrams` with
  `using`. So the names, their defining modules and their unqualified printing are fixed. The inspect adapter
  imports `BayesianNetworkFormats` qualified, and records its errors under qualified names such as
  `BayesianNetworkFormats.ParseError`.

## Decision

1. **Three roots, one per dependency tier.**
   - `FiniteKernelsError` for `FiniteKernels`, and for `MarkovCategories`' `UnboundGeneratorError`.
   - `BayesianNetworkFormatsError` for `BayesianNetworkFormats`.
   - `BayesNetError` for `BayesianNetworks` and every package that depends on it. That covers
     `InferenceError` (a new abstract type in `BayesianNetworkInference` over `ScopeError`, `ShapeError`,
     `CompileError`, `LogFactorDomainError` and the new `FactorEntryError`), `InfluenceDiagramError`, `ZooError`, and
     `CategoricalBayesianNetworks`' `ConflictingKernelError`.

   A root marks the dependency tier that introduced an error, not a kind of failure: `BayesNetError` means
   "introduced by `BayesianNetworks` or a package built on it", and its docstring says so.

   `BayesianNetworks` exports `AnyBayesNetError`, a `Union` of the three roots and the two stand-alone Graphviz
   errors, and every package that depends on it re-exports it. It is for catching and dispatch, never for
   subtyping. It covers every typed exception of the ecosystem. Base exceptions stay outside it, as rule (v) of
   decision 4 sets out.

   A new package subtypes the nearest root it depends on. If it defines two or more types of its own, it adds an
   abstract type of its own under that root.
2. **Stability.**
   - Only supertypes change. No type is renamed or moved to another module.
   - `ImpossibleEvidenceError` (defined in `BayesianNetworks`), `IrregularDiagramError` (`InfluenceDiagrams`) and
     `ScopeError` (`BayesianNetworkInference`) keep their names, their defining modules and their unqualified
     printing.
   - A re-export carries the same binding, and no package defines a different binding with any of these names.
   - Packages that the conformance adapters load with `using` (`BayesianNetworks`, `BayesianNetworkInference` and
     `InfluenceDiagrams`) re-export `BayesianNetworkFormatsError` but none of Formats' concrete types. Otherwise the
     inspect adapter's recorded names would silently lose their qualification.
3. **One type per condition per layer.** This reaffirms ADR 0007's first decision. The types are:

   | Condition | File row (Formats) | FinStoch kernel (`FiniteKernels`) | Bound mechanism (`BayesianNetworks`) | Factor graph (Inference) | JSON model document (`BayesianNetworks`) |
   |---|---|---|---|---|---|
   | Does not sum to one | `NotNormalizedError` (`validate`, `atol = 1e-6`) | `KernelNormalizationError` (`assert_normalized`, `atol = DEFAULT_ATOL = 1e-8`) | `UnnormalizedKernelError(variable, max_deviation, atol)` | Not checked: factors are unnormalised by design | `FormatError` |
   | Entry negative or not finite | `ValidationError` | `KernelEntryError` | `InvalidKernelEntryError` (new; wraps `KernelEntryError`) | `FactorEntryError` (new; at construction), and `LogFactorDomainError` for a tolerated entry in [−atol, 0) that the log domain cannot take | `FormatError` |
   | Zero evidence mass | — | `normalize`: `ArgumentError` | `ImpossibleEvidenceError` | `ImpossibleEvidenceError` from every backend (ADR 0012); `normalize(::Factor)`: `ArgumentError` | — |

   - `KernelNormalizationError` means only "a kernel's columns do not sum to one within `atol`". It is raised by
     kernel construction and `assert_normalized`, including `FiniteKernel(f::Factor, ...)`.
   - `UnnormalizedKernelError` gains an `atol` field, which explains the gap between a file's tolerance (1e-6) and
     a model's (1e-8).
   - Entries are checked with FinStoch's tolerance (ADR 0007's second decision): finite and at least −atol. Formats
     stays stricter, rejecting any negative entry of a file row.
   - `_check_kernel` wraps `KernelEntryError` into `InvalidKernelEntryError(variable, assignment, value, atol)`.
     `assignment` lists the parent-state pairs in input order, then the variable's own state, so it reads correctly
     whichever table layout the user wrote. `semantic_errors` then collects it, and `validate` throws the first
     collected error in mechanism order, which can now be an entry error.
   - `bind_cpt(...; renormalize=true)` checks entries before it rescales, and raises `InvalidKernelEntryError`. A
     row whose sum is not positive raises `UnnormalizedKernelError` instead of `FiniteKernels.normalize`'s
     `ArgumentError`.
   - `FactorGraph(factors; provenance=nothing, check=true, atol=DEFAULT_ATOL)` raises `FactorEntryError(vars,
     index, value, atol)` for an entry that is not finite or is below −atol. This applies FiniteKernel's
     `check`/`atol` contract to factors, at one pass per graph rather than per query. `compile` keeps the unchecked
     constructor, because the model was validated at its own `atol`, and so does any container that deliberately
     holds signed utilities rather than a measure.
   - `LogFactorDomainError` keeps its meaning, a tolerated entry that the log domain cannot take. That is a
     restriction of a backend's domain, which ADR 0011 keeps separate from validity.
4. **Boundaries.** At the boundary between two packages:
   - **(i)** Throw a lower package's type only for its documented condition. ADR 0012 applied this rule to
     `KernelNormalizationError`.
   - **(ii)** Wrap at a bridge when the upper layer adds information, such as the variable or the document.
     `_check_kernel` wraps the kernel errors into `UnnormalizedKernelError` and `InvalidKernelEntryError`, which name
     the variable; the JSON readers wrap into `FormatError`, which names the record.
   - **(iii)** Pass typed lower-layer errors through unchanged when nothing is added. Document the pass-through, and
     re-export the lower root. The pass-throughs are:
     - Formats errors through `read_bayesnet`, `read_influence_diagram`, `model_ir` and `load_model`, where wrapping
       would hide the row;
     - `SpaceMismatchError` from `JointTable`;
     - `UnboundGeneratorError` from `CategoricalBayesianNetworks`' `evaluate`;
     - Inference's factor-level `InvalidAxisError`;
     - `InfluenceDiagrams`' ordering and valuation `ScopeError`s.
   - **(iv)** Wrap a Base or third-party exception that signals a *data* condition, selectively and with no
     catch-all. JSON decoding in `BayesianNetworks` and `InfluenceDiagrams` raises `FormatError` for text that is not
     JSON, a missing key, an unknown `KernelRef` type, an unparsable time and an invalid kernel table, and lets every
     other exception, `BayesNetError`s included, propagate unchanged (`BayesianNetworks/src/serialization.jl`).
   - **(v)** Use Base types for the rest:
     - `ArgumentError` for invalid arguments, keywords, options and algebraic preconditions. Normalising a zero
       total is one, which keeps `FiniteKernels.normalize` and Inference's `normalize(::Factor)` as ADR 0007 and
       ADR 0012 left them.
     - `KeyError` for dictionary-style lookups.
     - `SystemError` for a missing or unreadable file, as the file-system call raises it, as in
       `BayesianNetworkFormats`' `read_network`. A package that already has a typed condition for a user-supplied
       path uses that type instead: `EcologicalBayesianNetworks` raises `NotAModelFileError` from `import_model`,
       as `check_model_file` does.
     - A wrongly typed keyword keeps Julia's own `TypeError`.

   **Known exceptions,** left to a follow-up: 28 domain-condition sites across the packages still raise a Base
   exception; a JSON value of the wrong type raises the error of the failed conversion (a `MethodError` from
   `String` for a number where a string is expected); and an error inside an ACSet body comes unchanged from
   ACSets' `parse_json_acset`.
5. **Re-exports.** A package re-exports:
   - the roots of the layers its API raises;
   - the concrete types that its own docstrings say its functions raise;
   - wholesale, every exception type of a package from which it re-exports a function.

   A type re-exported only for annotation, and a generic function extended only to avoid a name clash, do not bring
   in that package's exceptions. Adapter-loaded packages take only Formats' root (decision 2). Each package that
   re-exports wholesale has a test that it exports every exception type of its source packages, so that a new type
   cannot be missed.
6. **Files.** Every package module keeps its root and its exceptions, with their `showerror` methods, in
   `src/errors.jl`, in the module that defines them. `CategoricalBayesianNetworks` keeps `ConflictingKernelError` in
   its own `errors.jl` rather than moving it into `BayesianNetworks`. The stand-alone submodule
   `BayesianNetworks.Graphviz` is exempt, and its two errors stay in `graphviz.jl`.
7. **Equality.**
   - `BayesNetError` keeps its structural `==` and `hash`, now comparing fields with `isequal`, so that `==` agrees
     with `hash`. NaN fields then compare equal, and `-0.0` differs from `0.0`.
   - The types newly placed under it, Inference's four and the zoo's, gain structural equality and hashing. Nothing
     compares them today.
8. **Documentation.**
   - No docs build sets `warnonly`.
   - `checkdocs` stays at Documenter's default (`:all`), except in Formats (`:exports`), whose API page leaves its
     private docstrings out on purpose.
   - All eight documentation builds are run locally before the last commit of a series.
   - A test in every package checks that every export the package owns has a docstring.
   - A docstring names another package's type as a code span, never with `@ref`. Documenter includes only the
     docstrings of its own `modules`, so an `@ref` or an `@docs` entry for a foreign name fails the strict build.
     Re-exported foreign names are documented in prose on the API page.
9. **Conventions.** The eight packages' `CONTRIBUTING.md` and `CLAUDE.md` and the workspace's `scripts/scaffold.sh`
   state these rules: subtype the nearest root; keep exceptions in `src/errors.jl`; use `ArgumentError` for invalid
   arguments and keywords; pass typed lower-layer errors through, documented; name another package's type as a code
   span. A newly scaffolded package starts with a `src/errors.jl` whose header states them.
10. **Supersedes** ADR 0007's consequence that a negative kernel bound to a mechanism surfaces as `KernelEntryError`.

## Consequences

- Re-parenting and re-exports are not breaking. Re-parenting was checked on Julia 1.12.7: `isa`, `@test_throws` with
  the old type, `showerror` and dispatch all behave as before.
- **Breaking changes:**
  - `bind_*` and `validate` raise `InvalidKernelEntryError` where they raised `KernelEntryError`, and
    `semantic_errors` now collects it;
  - `bind_cpt(renormalize=true)` rejects a row with an invalid entry or a sum that is not positive;
  - `FactorGraph` rejects entries that are not finite or are below −atol, and `check=false` opts out;
  - `UnnormalizedKernelError` has a third field.
- Two changes of the same series landed before this record and already follow rules (iv) and (v): JSON decoding
  raises `FormatError` selectively, and `import_model` raises `NotAModelFileError` for a path with no model file.
- A user catches any typed ecosystem error with `e isa AnyBayesNetError`, and impossible evidence with
  `e isa ImpossibleEvidenceError` whichever backend answers. A bad probability table, at whichever layer it is
  detected, is caught with a `Union` of the file-row, kernel and mechanism types in the first two rows of decision
  3's table.
- Of the review's requests, (a) is met as three roots and one exported `Union` rather than one root, (c) is met,
  and (b) and (d) as written are declined, for the reasons in the Context.
- Frozen producers replay only against their pinned snapshots. The frozen Lean producer catches `KernelEntryError`
  and is not re-run against the live packages.
- SPEC §54 stays true: every type it names is still an `Exception`, now through a root.
- **Deferred to follow-ups:**
  - `ScopeError`'s uses outside scope errors: the trace budgets of Inference and `InfluenceDiagrams`, and the
    non-negativity checks of the trace and of stable decision elimination; also `InfluenceDiagrams`' rational size
    limit, which uses its own `UtilityScopeError`;
  - model-level label errors: `infer(::BayesModel)` and `tornado`'s target lookup should raise
    `UnknownStateError` and `UnknownVariableError`, as `BayesianNetworks.marginal` does;
  - the known exceptions of decision 4.
