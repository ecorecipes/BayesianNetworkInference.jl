# BayesianNetworkInference.jl

Exact and approximate inference for compositional Bayesian networks: factor
algebra, variable elimination with CliqueTrees.jl orderings, sampling,
junction trees and belief propagation.

This package sits above `BayesianNetworks.jl` in the ecorecipes ecosystem
(`FiniteKernels` → `BayesianNetworks` → `BayesianNetworkInference` →
`InfluenceDiagrams`). It works on [`FactorGraph`](@ref)s, bags of
[`Factor`](@ref)s over shared finite variables, and [`compile`](@ref)s a
`BayesModel` from `BayesianNetworks.jl` into one so that [`infer`](@ref),
[`posterior`](@ref) and [`ancestral_sample`](@ref) work directly on models,
including models read from Netica, GeNIe, HUGIN, BIF, DSC and UAI files.

## Factors versus kernels

A `FiniteKernel` is a morphism of FinStoch: it carries an input/output
partition and is normalised over its outputs. A [`Factor`](@ref) is an
unnormalised tensor over an ordered scope, closed under pointwise product
(`multiply`) and summation (`marginalize`). The two are different types with
explicit conversions:

- `Factor(k, inputs, output)` uses first-occurrence scope order and the
  parents-first table `cpt(k)` (ADR 0002), taking its diagonal when input
  slots repeat rather than treating repeated variables as independent axes;
- `FiniteKernel(f, inputs, outputs)` reorders to outputs-first and checks
  normalisation, throwing `KernelNormalizationError` otherwise.

## Quick start

```julia
using BayesianNetworkInference, FiniteKernels

rain = FiniteAxis(:Rain, [:yes, :no]); grass = FiniteAxis(:Grass, [:wet, :dry])
fg = FactorGraph([Factor(cpt(rain, [0.2, 0.8]), :Rain),
                  Factor(cpt(rain, grass, [0.9 0.1; 0.2 0.8]), [:Rain], :Grass)])

posterior, diagnostics = infer(fg, [:Rain]; evidence=Dict(:Grass => :wet))
posterior.table                      # P(Rain | Grass = wet)
brute_force_marginal(fg, [:Rain]; evidence=Dict(:Grass => :wet)) ≈ posterior   # true

# The same on a BayesianNetworks.jl model
using BayesianNetworks
m = read_bayesnet(fixture_path("dne/habitat_reference.dne"))   # or reference_habitat_model()
posterior(m, :Occupancy)                                       # Dict(:absent => 0.5238, :present => 0.4762)
posterior(do_intervention(m, :GrazingPressure => :low), :Occupancy)
infer(m, :Occupancy)[1] ≈ Factor(marginal(m, :Occupancy), :Occupancy)   # brute-force oracle: true
```

## Model bridge

[`compile`](@ref)`(m::BayesModel)` produces one factor per mechanism, reading
parents in `input_position` order, identifying repeated slots diagonally and
recording `(variable, mechanism, id)` in `fg.provenance`. `infer(m, query; evidence, backend)`
merges the explicit evidence with the evidence recorded by `observe`
(explicit entries win); hard and soft interventions are mechanism rewrites,
so an intervened model compiles and infers unchanged. Sampling
([`ancestral_sample`](@ref)`(m, n)`) draws from the prior or interventional
distribution, and `AncestralSamples(m, sample(m, n))` converts the output of
`BayesianNetworks.sample` for the same Monte Carlo checks.

## Elimination orderings

[`variable_elimination`](@ref) conditions every factor on the evidence,
builds the interaction graph of the remaining factors, asks CliqueTrees.jl for
a fill-reducing permutation with the query variables forced last
(`CompositeRotations`), eliminates, and normalises. Strategies:
[`MinFill`](@ref) (default), [`MinDegree`](@ref), [`AMDOrder`](@ref),
[`ExactTreewidth`](@ref) and [`UserOrder`](@ref); [`treewidth`](@ref) reports
the width each induces.

## Junction trees and belief propagation

[`JunctionTree`](@ref) builds a clique tree of the interaction graph with
`CliqueTrees.cliquetree(graph; alg, snd=Maximal())`
([`build_junction_tree`](@ref), cached per factor-graph identity and strategy) and
calibrates it by Shafer-Shenoy message passing ([`calibrate`](@ref): collect
towards the root, distribute back, no separator division). One calibration
gives every clique posterior ([`clique_beliefs`](@ref)) and every
single-variable posterior ([`all_marginals`](@ref), also on a `BayesModel`);
[`infer`](@ref) answers any query that lies in one clique and falls back to
variable elimination otherwise (reported by
`JunctionTreeDiagnostics.fallback`). [`BeliefPropagation`](@ref) runs sum-product
message passing on the factor graph ([`belief_propagation`](@ref)): its
sum-product fixed points on feasible conditioned trees ([`is_tree`](@ref))
are exact, while loopy beliefs are approximations. It has damping, a
tolerance, an iteration cap and
[`BPDiagnostics`](@ref). VE/JT posterior entry points reject globally zero mass,
including disconnected components and all-observed cases, with
`ImpossibleEvidenceError`. BP detects local zero-support failures, with the same
error, but otherwise leaves global feasibility unknown unless
`BeliefPropagation(check_evidence=true)` requests a VE feasibility pass.
`evidence_checked` records that choice. Its residual is measured before damping
in the message equations at the returned iterate, not from a tiny damped step,
and is not a general bound on marginal error.

## Exact finite-model proofs

The sibling BayesianNetworks Lean project proves evidence clamping equivalent
to indicator-factor elimination, normalized VE posterior correctness, actual
cached Shafer-Shenoy collect/distribute computation, and d-separation soundness
for the moralized ancestral graph. Tree validity requires structural running
intersection, complete factor assignment and variable coverage, not a
precomputed correct message trace. Grafting covers arbitrary branching and
empty-separator virtual roots cover disconnected forests.

These are not proofs of Julia arrays, CliqueTrees construction or iterative BP.
Formal forest beliefs include outside-component scalar masses; Julia stores
component-local beliefs and separately checks global mass before normalization.
Likewise, the empty `infer` query is an unnormalized mass API, not the formal
empty-query probability distribution. Numerical error bounds require an
explicit positive evidence-mass floor and error budget.

`BayesianNetworks.proof_certificate` captures the original ordered records,
references and exact bound numbers for the separate literal-Lean checker,
before this package's Float64 conversion. See the
[finite-model certificate guide](https://ecorecipes.github.io/BayesianNetworks.jl/certificates/).

## Validation and sensitivity

A model is not validated by inspecting it. [`Cases`](@ref) holds a dataset of
observations, [`predict`](@ref) withholds a target and predicts it from the
rest, and [`evaluate`](@ref) reports the proper scores
([`brier_score`](@ref), [`log_score`](@ref), [`spherical_score`](@ref)), the
[`calibration_curve`](@ref) and its [`calibration_error`](@ref),
[`roc_curve`](@ref) / [`auc`](@ref), and [`confusion_matrix`](@ref) /
[`accuracy`](@ref), each beside the marginal-prior [`baseline`](@ref) so that
"does the network beat always predicting the prior?" has an answer.
[`holdout`](@ref) and [`kfold`](@ref) produce index splits and can group by a
variable, so a spatial or temporal unit is never split across the two sides;
they split cases and do not refit CPTs. [`sensitivity`](@ref) ranks the
variables by [`mutual_information`](@ref) with a target (the
entropy-reduction metric of [Marcot2012](@cite)) and [`tornado`](@ref) by the
range a single finding could move the answer.

## Errors

All failures are typed exceptions carrying the offending names. The
package's own, [`ScopeError`](@ref), [`ShapeError`](@ref),
[`CompileError`](@ref) (a model that is open or lacks kernels),
[`FactorEntryError`](@ref) (a factor graph entry that is not finite or is below
`-atol`), [`FactorDomainError`](@ref) (a valid entry that a backend's arithmetic
cannot take, such as a tolerated negative entry in the log domain) and
[`TraceLimitError`](@ref) (a run that an execution trace cannot record: a cell
budget, a Float64-only profile, an underflowed evidence mass) subtype [`InferenceError`](@ref), which subtypes `BayesianNetworks`'
`BayesNetError` (ADR 0013). A name the model does not have is reported by the layer
that named it (ADR 0015): the methods on a `BayesModel` raise `BayesianNetworks`'
`UnknownVariableError` and `UnknownStateError`, as `BayesianNetworks.marginal` does,
while the methods on a [`FactorGraph`](@ref) raise `ScopeError` and `FiniteKernels`'
`InvalidAxisError`. `FiniteKernels`' `KernelNormalizationError`
(kernel columns that do not sum to one) and `InvalidAxisError`, and the
exceptions of `BayesianNetworks.validate`, pass through unchanged; the
re-exported `AnyBayesNetError` catches every one of them. Invalid arguments
and keywords raise `ArgumentError`, which is outside every root.

Evidence of probability exactly zero raises `BayesianNetworks`'
`ImpossibleEvidenceError`, which this package re-exports, carrying the
evidence. It is the one zero-mass error of every posterior entry point
(ADR 0012): variable elimination, brute force, the junction tree, belief
propagation (a zero message, belief or conditioned scalar, and
`check_evidence=true`), both log-domain backends, and everything built on them,
from [`posterior`](@ref) to [`predict`](@ref) and [`tornado`](@ref). It is the
same error `BayesianNetworks.marginal` raises, so one `catch` covers both:

```julia
try
    infer(m, :Rain; evidence=Dict(:Grass => :wet))
catch e
    e isa ImpossibleEvidenceError || rethrow()
    @info "no posterior" e.evidence
end
```

Evidence that is merely rare is answered, not rejected (ADR 0014). A mass that is
zero, subnormal or non-finite in binary64 does not decide impossibility, so
variable elimination, the junction tree, brute force and belief propagation
recompute such a query with [`LogVariableElimination`](@ref) or
[`LogJunctionTree`](@ref) and return that answer; the diagnostics' `log_fallback`
(or `exact_fallback` for belief propagation) records it. Only the log domain, where
[`log_evidence_probability`](@ref) is `-Inf` exactly for impossible evidence, may
raise `ImpossibleEvidenceError`. A model with tolerated entries in `[-atol, 0)` can
leave a posterior's sign to the rounding; that raises `BayesianNetworks`'
`IndeterminatePosteriorError` instead of returning a negative probability. An empty
query never raises: it returns the unnormalised mass, which may be zero.

## References

Variable elimination follows [Dechter1999](@cite), [ZhangPoole1994](@cite) and
[KollerFriedman2009](@cite); junction-tree calibration is Shafer-Shenoy
[ShaferShenoy1990](@cite), contrasted with the division-based architectures of
[LauritzenSpiegelhalter1988](@cite) and [JensenLauritzenOlesen1990](@cite);
sum-product message passing is [Pearl1988](@cite) in the factor-graph form of
[Kschischang2001](@cite), with the loopy case studied by
[MurphyWeissJordan1999](@cite). The orderings come from `CliqueTrees.jl`
[CliqueTrees](@cite). The scoring layer implements [Brier1950](@cite) and
[GneitingRaftery2007](@cite), and the sensitivity metric is
[Marcot2012](@cite), in the life cycle of [Marcot2006](@cite) and
[ChenPollino2012](@cite). Full entries are on the [References](references.md) page.

See the Tutorials section for rendered vignettes and the API Reference for
docstrings.
