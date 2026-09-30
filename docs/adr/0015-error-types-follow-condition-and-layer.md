# ADR 0015: Error types follow the condition and the layer

Date: 2026-09-29. Status: accepted.

## Context

ADR 0013 set the roots and the boundary rules. Its record of item 15 left four follow-ups, L2 to L5.

- **L2: `ScopeError` had uses that are not about scope.** It reported three other kinds of condition:
  - the cell budgets of both execution traces;
  - the Float64-only v1 trace profile;
  - the underflowed evidence mass that the trace cannot record (ADR 0014).

  It also reported the non-negativity checks of the trace inputs and of stable decision elimination. Those are the
  same kind of condition as `LogFactorDomainError`: a valid entry that one backend's arithmetic cannot take.
  `InfluenceDiagrams` used its own `UtilityScopeError` for the trace's rational digit limit. The conformance
  `schedule_adapter.jl` labelled every `ScopeError` as an invalid elimination order.
- **L3: a model-level query reported the factor layer's errors.** `infer`, `all_marginals` and the other methods on a
  `BayesModel` raised `ScopeError` for an unknown variable and `FiniteKernels`' `InvalidAxisError` for an unknown
  state. `BayesianNetworks.marginal` raises `UnknownVariableError` and `UnknownStateError` for the same query.
- **L4: data conditions still raised Base exceptions.** ADR 0013 counted 28 such sites and two known exceptions: a
  JSON value of the wrong type, and an error inside an ACSet body. The audit that listed the 28 sites was not kept,
  so the list below was rebuilt from the code on 2026-09-29. It covers the explicit throws of all eight packages and
  the implicit leaks at every reader of file, document or manifest content.
- **L5:** a pointer from SPEC §54 to ADR 0013, and a local pre-commit step. There are no remotes and no CI.

## Decision

1. **`ScopeError` means a scope problem.** It covers a variable or axis that is missing, repeated or misplaced in a
   factor operation or a query, and the argument checks built on them. Its name, module and unqualified printing
   stay frozen (ADR 0013). Two new `BayesianNetworkInference` types, under `InferenceError`, take its other uses:
   - **`TraceLimitError(trace, limit, detail, vars)`**: an execution trace cannot record this run. The `limit` is
     one of `:cells`, `:compilation_cells`, `:rational_digits`, `:scalar_type` or `:evidence_underflow`. Both
     traces raise it, and so does `InfluenceDiagrams`' trace, for the digit limit that used `UtilityScopeError`.
   - **`FactorDomainError(backend, vars, index, value)`**: a valid factor entry that `backend`'s arithmetic cannot
     take. It replaces `LogFactorDomainError`, which is removed. The backends are `:log_domain`,
     `:trace_variable_elimination` and `:stable_decision_elimination`. The packages are unregistered, so there is
     no deprecation binding.

   `conformance/adapters/julia/schedule_adapter.jl` is a frozen controller: the policy-identity audit checks its
   hash, so it is not edited. It still labels every `ScopeError` as `invalid_order`, but that label is now accurate.
   A trace limit or backend-domain failure is no longer a `ScopeError`, so the adapter rethrows it instead of
   mislabelling it. The only `ScopeError`s decision elimination can still raise are its ordering guards and the
   valuation algebra's internal scope checks. The frozen native-schedule records are unchanged: all five of their
   `invalid_order` rejections come from the ordering guards.
2. **A name is reported by the layer that named it.** Every method on a `BayesModel` checks its variable and state
   names after `compile` and raises `BayesianNetworks`' `UnknownVariableError` or `UnknownStateError`, as `marginal`
   does. The methods are `infer`, `posterior`, `all_marginals`, `log_evidence_probability`,
   `trace_variable_elimination`, `entropy`, `mutual_information`, `sensitivity`, `tornado`, `predict`, `baseline` and
   `evaluate`. A case in `predict` or `evaluate` is checked only for the target and the variables it enters. The
   methods on a `FactorGraph` keep `ScopeError` and `InvalidAxisError`, because a factor graph has no model to name.
3. **Data conditions are typed, by checking and not by catching.** A reader checks a value's JSON type, a token's
   integrality or an index's range before using it, and raises its package's typed error. It does not catch the
   `MethodError`, `InexactError` or `BoundsError` that an unchecked conversion would raise, because those also mean
   bugs.
   - **`BayesianNetworks` and `InfluenceDiagrams` JSON.** Typed reads (`_as_string`, `_as_int`, `_as_array` and the
     rest) raise an internal `_JSONShapeError`, and the record decoders convert it to `FormatError`, naming the
     record and key. `_field` and `_ref_field` check that a record is an object. The card envelope now checks its
     schema version, as its docstring said. An unknown provenance `source_type` and a non-string time are
     `FormatError`s too.
   - **The ACSet body.** It goes to ACSets' `parse_json_acset`, a third-party parser that reads only the document.
     `_parse_acset` converts anything it raises to `FormatError` except an `InterruptException`. This is the one
     scoped catch-all among the decoders, and `InfluenceDiagrams` uses it too.
   - **`BayesianNetworkFormats`.** Malformed file content raises `ParseError`, at the token where one is available.
     The cases are:
     - counts, sizes and indices that are fractional, negative or overflow, checked by `token_int` and
       `expect_int!`;
     - hexadecimal and `#k` literals that overflow;
     - BIF positions;
     - Netica `numstates`, visual centres and level intervals;
     - HUGIN states and positions;
     - GeNIe states without an id;
     - UAI counts, scopes and evidence indices;
     - the shape of an IR JSON document, field by field.

     `write_uai_evidence` raises `ValidationError` for an unknown variable, as it already did for an unknown state.
   - **Models and certificates.** Binding a kernel to a hard intervention is `KernelBindingError` with
     `what = :intervention`. A certificate reference the v1 profile has no tag for is `OpenCertificateError` with
     `what = :unsupported_reference`.
   - **`EcologicalBayesianNetworks`.** `read_manifest` checks the keys and value types of `[parser_options]`
     (`InvalidManifestError`). Before, a misspelt key surfaced only at load time, as an `ArgumentError`.
     `read_manifest` converts only `TOML.ParserError`.
4. **Some sites stay Base exceptions (ADR 0013, rule v).** They are invalid arguments, keywords, options and
   algebraic preconditions:
   - the keyword checks;
   - `normalize` of a zero total;
   - `KeyError` for a lookup by id (`variable`, `marginal` of a joint);
   - rank arguments and the arity of `oapply` in `CategoricalBayesianNetworks`;
   - AUC and ROC without both outcome classes, and a holdout from a single group;
   - IR `extras` holding a non-JSON value, which only a hand-built IR can contain;
   - the direct StructTypes path of `KernelRef`, which keeps `ArgumentError`, now also for a non-string field;
   - `EcologicalBayesianNetworks`' unknown loading keywords.
5. **L5.** `scripts/precommit.sh` runs the ADR, reference and vignette sync checks of all eight packages in seconds.
   With `--docs` it also builds every package's documentation strictly. `scripts/hooks/pre-commit` runs the fast
   checks, and a repository opts in with `git config core.hooksPath`. Nothing installs the hook, because the root
   repository is shared with another working session. The SPEC §54 pointer waits until that session has committed
   `docs/SPEC.md`.

## Consequences

- **Breaking, in unregistered packages:**
  - `LogFactorDomainError` becomes `FactorDomainError`;
  - trace budgets and profile limits are `TraceLimitError`, not `ScopeError` or `UtilityScopeError`;
  - model-level label errors are `BayesianNetworks`' types;
  - wrong-typed JSON, ACSet-body and malformed-file errors are typed.

  Tests that pinned the old types changed with them.
- `AnyBayesNetError` now covers every deliberate data condition at the readers and model-level entry points. What
  escapes it is an invalid argument (rule v), a missing file (`SystemError`) or a bug. ADR 0013's list of known
  exceptions is closed.
- A manifest `constructor` that names something that cannot be called without arguments is an
  `InvalidManifestError` when the model is loaded.
- **A residual risk:** a corrupt `.gz` in the zoo cache would escape as CodecZlib's error. The audit reasoned about
  it but did not reproduce it, and the checksum guards it.
- This record extends ADR 0013 decision 4 (boundaries) and its list of known exceptions, and amends ADR 0014 decision
  7: a negative entry in the log domain is `FactorDomainError`.
