"""
    BayesianNetworkInference

Exact and approximate inference for compositional Bayesian networks: factor
algebra, variable elimination with CliqueTrees.jl orderings, sampling, and
junction trees.

The package works on [`FactorGraph`](@ref)s, bags of [`Factor`](@ref)s over
shared finite variables. Factors are built from `FiniteKernels.FiniteKernel`s
with `Factor(k, inputs, output)` (scope `(inputs..., output)`, table in the
parents-first layout of ADR 0002) and converted back with
`FiniteKernel(f, inputs, outputs)`; the two types stay distinct because a
kernel carries an input/output partition and a normalisation invariant while a
factor is an unnormalised tensor under pointwise product and summation.

Inference: [`infer`](@ref) / [`variable_elimination`](@ref) with an
[`EliminationStrategy`](@ref) ([`MinFill`](@ref), [`MinDegree`](@ref),
[`ExactTreewidth`](@ref), [`AMDOrder`](@ref), [`UserOrder`](@ref)),
the brute-force oracle [`brute_force_marginal`](@ref), and
[`ancestral_sample`](@ref). The bridge from `BayesianNetworks.jl` is
[`compile`](@ref), which turns a `BayesModel` into a factor graph, and the
model-level methods of [`infer`](@ref), [`posterior`](@ref) and
[`ancestral_sample`](@ref). Two more backends work on the same factor graphs:
[`JunctionTree`](@ref) calibrates a clique tree from `CliqueTrees.cliquetree`
by Shafer-Shenoy message passing, so that [`all_marginals`](@ref) and
[`clique_beliefs`](@ref) come from one pass, and [`BeliefPropagation`](@ref)
runs sum-product on the factor graph (exact on trees, loopy otherwise, with
[`BPDiagnostics`](@ref)).

Validation: [`Cases`](@ref) holds a dataset of observations, [`predict`](@ref)
withholds a target and predicts it from the rest, and [`evaluate`](@ref)
scores those predictions against a marginal-prior [`baseline`](@ref) with the
proper scoring rules [`brier_score`](@ref), [`log_score`](@ref) and
[`spherical_score`](@ref), a [`calibration_curve`](@ref) and its
[`calibration_error`](@ref), [`roc_curve`](@ref) / [`auc`](@ref), and a
[`confusion_matrix`](@ref); [`holdout`](@ref) and [`kfold`](@ref) produce
index splits, optionally grouped so that a spatial or temporal unit is never
split. [`sensitivity`](@ref) ranks the variables by
[`mutual_information`](@ref) with a target and [`tornado`](@ref) by the range
a single finding could move it.

The algorithms are the standard ones: variable elimination and clique trees
follow [KollerFriedman2009](@cite) and [Dechter1999](@cite), junction-tree
calibration follows [ShaferShenoy1990](@cite), sum-product message passing
[Pearl1988](@cite) and [Kschischang2001](@cite), the proper scoring rules
[GneitingRaftery2007](@cite), and the entropy-reduction sensitivity metric
[Marcot2012](@cite).

Part of the ecorecipes compositional Bayesian-network ecosystem.
"""
module BayesianNetworkInference

using FiniteKernels
using FiniteKernels: labels, label_index
using BayesianNetworks: BayesianNetworks, BayesModel, syntax, variable_name, mechanism_of,
                        mechanism_name, inputs, topological_order, exogenous,
                        missing_kernels, kernel, states, BayesNetError, AnyBayesNetError,
                        BayesianNetworkFormatsError, ImpossibleEvidenceError,
                        IndeterminatePosteriorError
# Names shared with BayesianNetworks.jl are extended, not shadowed, so that
# `using BayesianNetworks, BayesianNetworkInference` never needs qualification.
import BayesianNetworks: variables, axis, empirical_marginal
using Graphs: SimpleGraph
using Graphs: Graphs
using SparseArrays: sparse
using Random: AbstractRNG, default_rng, shuffle
using CliqueTrees: CliqueTrees
using AMD: AMD                # activates the CliqueTrees AMD extension
using TreeWidthSolver: TreeWidthSolver    # activates the CliqueTrees BT extension
import LinearAlgebra: normalize
# `evaluate` (scores a model against a dataset) is this package's own function. It used
# to extend `MarkovCategories.evaluate`, which interprets a categorical expression in
# FinStoch, because `BayesianNetworks` re-exported that function; since the split
# (ADR 0009) the categorical layer lives in `CategoricalBayesianNetworks.jl`, this
# package depends on nothing categorical, and the two functions are unrelated. They
# clash under `using BayesianNetworkInference, CategoricalBayesianNetworks`; qualify
# the one you mean (`BayesianNetworkInference.evaluate`,
# `CategoricalBayesianNetworks.evaluate`).
import CliqueTrees: treewidth
import Graphs: is_tree            # extended with a FactorGraph method, never shadowed

# Factors
export Factor, scope, axis, unit_factor, multiply, marginalize, maximize, argmax_table,
       condition, normalize, reorder
# Factor graphs
export FactorGraph, variables, interaction_graph
# Orderings
export EliminationStrategy, MinFill, MinDegree, ExactTreewidth, AMDOrder, UserOrder,
       elimination_order, treewidth
# Inference
export InferenceBackend, VariableElimination, InferenceDiagnostics, variable_elimination,
       infer, joint_factor, brute_force_marginal, LogVariableElimination,
       LogInferenceDiagnostics, log_variable_elimination,
       log_evidence_probability
export trace_variable_elimination
# Junction trees and belief propagation
export JunctionTree, CompiledJunctionTree, CalibratedJunctionTree, JunctionTreeDiagnostics,
       LogJunctionTree, LogCalibratedJunctionTree, LogJunctionTreeDiagnostics,
       log_calibrate,
       build_junction_tree, calibrate, clique_beliefs, all_marginals, BeliefPropagation,
       BPDiagnostics, belief_propagation, is_tree
# Sampling
export AncestralSamples, ancestral_sample, empirical_marginal
# Model bridge (BayesianNetworks.jl)
export FactorGraphBackend, compile, posterior
# Validation and scoring
export Case, Cases, Predictions, predict, baseline, brier_score, log_score,
       spherical_score, CalibrationCurve, calibration_curve, calibration_error, ROCCurve,
       roc_curve, auc, ConfusionMatrix, confusion_matrix, accuracy, holdout, kfold,
       EvaluationResult, evaluate
# Sensitivity analysis
export entropy, mutual_information, sensitivity, tornado
# Exceptions (errors.jl): the root and the types this package defines
export InferenceError, ScopeError, ShapeError, CompileError, FactorDomainError,
       FactorEntryError, TraceLimitError
# Re-exported from FiniteKernels for convenience
export FiniteAxis, FiniteSpace, FiniteKernel, cpt
# Re-exported exception roots and types (ADR 0013). Factors and the kernel conversions
# raise FiniteKernels' errors, so every exception type FiniteKernels exports is
# re-exported (test/test_errors.jl checks for drift). From BayesianNetworks: the root
# `BayesNetError`, which `InferenceError` subtypes, the `AnyBayesNetError` union, and
# `ImpossibleEvidenceError`, the one error every posterior entry point raises for evidence
# of probability exactly zero (ADR 0012, 0014). Every binding is its owner's, never a second
# definition, so the names stay unambiguous and print unqualified. Of
# BayesianNetworkFormats only the root `BayesianNetworkFormatsError` is re-exported, never
# its concrete types (ADR 0013 decision 2): the conformance adapters load this package with
# `using` and serialise Formats' errors by their qualified names.
export FiniteKernelsError, InvalidAxisError, KernelShapeError, KernelEntryError,
       KernelNormalizationError, SpaceMismatchError
export BayesNetError, AnyBayesNetError, BayesianNetworkFormatsError,
       ImpossibleEvidenceError,
       IndeterminatePosteriorError

include("errors.jl")  # first: its field types all come from Base
include("factors.jl")
include("factor_graph.jl")
include("orderings.jl")
include("variable_elimination.jl")
include("junction_tree.jl")
include("log_variable_elimination.jl")
include("log_junction_tree.jl")
include("belief_propagation.jl")
include("sampling.jl")
include("compile.jl")
include("model_inference.jl")
include("execution_trace.jl")
include("scores.jl")
include("sensitivity.jl")

end # module
