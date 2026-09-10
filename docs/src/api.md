# API Reference

```@docs
BayesianNetworkInference
```

## Factors

```@docs
Factor
scope
axis
unit_factor
multiply
marginalize
maximize
argmax_table
condition
normalize
reorder
Base.:(==)(::Factor, ::Factor)
FiniteKernel(::Factor, ::AbstractVector{Symbol}, ::AbstractVector{Symbol})
```

## Factor graphs

```@docs
FactorGraph
variables
interaction_graph
```

## Elimination orderings

```@docs
EliminationStrategy
MinFill
MinDegree
ExactTreewidth
AMDOrder
UserOrder
elimination_order
treewidth
```

## Inference

```@docs
InferenceBackend
VariableElimination
InferenceDiagnostics
variable_elimination
infer
joint_factor
brute_force_marginal
LogVariableElimination
LogInferenceDiagnostics
LogFactorDomainError
log_variable_elimination
log_evidence_probability
trace_variable_elimination
```

## Junction trees

```@docs
JunctionTree
LogJunctionTree
LogCalibratedJunctionTree
LogJunctionTreeDiagnostics
log_calibrate
CompiledJunctionTree
build_junction_tree
CalibratedJunctionTree
calibrate
JunctionTreeDiagnostics
clique_beliefs
all_marginals
```

## Belief propagation

```@docs
BeliefPropagation
BPDiagnostics
belief_propagation
is_tree
```

## Sampling

```@docs
AncestralSamples
ancestral_sample
empirical_marginal
```

## Model bridge (BayesianNetworks.jl)

`compile` turns a `BayesianNetworks.BayesModel` into a [`FactorGraph`](@ref); the
model-level methods of [`infer`](@ref), [`ancestral_sample`](@ref),
[`AncestralSamples`](@ref) and [`empirical_marginal`](@ref) are documented with those
functions above.

```@docs
FactorGraphBackend
compile
posterior
```

## Validation and scoring

Out-of-sample validation of a model against a dataset: [`Cases`](@ref) holds the
observations, [`predict`](@ref) predicts a withheld target, and [`evaluate`](@ref)
scores those predictions against the marginal-prior [`baseline`](@ref).

```@docs
Case
Cases
Predictions
predict
baseline
brier_score
log_score
spherical_score
CalibrationCurve
calibration_curve
calibration_error
ROCCurve
roc_curve
auc
ConfusionMatrix
confusion_matrix
accuracy
holdout
kfold
EvaluationResult
evaluate
```

## Sensitivity analysis

```@docs
entropy
mutual_information
sensitivity
tornado
```

## Exceptions

```@docs
ScopeError
ShapeError
CompileError
```
