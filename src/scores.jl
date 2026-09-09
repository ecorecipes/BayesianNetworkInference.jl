# Out-of-sample, decision-focused validation of a Bayesian network: cases,
# predicted distributions for a withheld target, proper scoring rules,
# calibration curves, discrimination (ROC/AUC), a maximum-a-posteriori
# confusion matrix, a marginal-prior baseline, and index splits for holdout
# and k-fold validation.
#
# This layer answers the "verification and validation" row of the ecological
# Bayesian-network life cycle: a sensitivity analysis is not a validation, and
# apparent fit is not forecast quality. Nothing here refits a model; see
# [`holdout`](@ref).
#
# The literature is cited from the docstrings below with the keys of the
# shared bibliography (`docs/references.bib`): Brier1950, GneitingRaftery2007,
# Marcot2006, ChenPollino2012.

# Cases
# -----

"""
    Case

One observation: a `Dict{Symbol,Symbol}` from variable name to observed state
label. A case may be **complete** (every variable of the model) or **partial**
(any subset); the variables it omits are simply not entered as evidence.
"""
const Case = Dict{Symbol,Symbol}

"""
    Cases(rows::AbstractVector{<:AbstractDict{Symbol,Symbol}})
    Cases(samples::AncestralSamples)

A dataset for validation: a vector of [`Case`](@ref)s. This is the one
dataset convention of the package -- a column table is *not* accepted, because
partially observed cases are the normal situation in ecology and a missing
entry must be distinguishable from a state called `:missing`.

`Cases` behaves as an `AbstractVector{Case}`, so `length`, iteration and
`cases[i]` work as expected; indexing with a vector of indices (the output of
[`holdout`](@ref) or [`kfold`](@ref)) returns a `Cases` again.
[`variables`](@ref) lists every variable observed in at least one case, in a
deterministic order: within a case the names are visited sorted, and each is
appended the first time it is seen.

```jldoctest
julia> cs = Cases([Dict(:A => :a1, :B => :b1), Dict(:A => :a2)]);

julia> length(cs), variables(cs)
(2, [:A, :B])

julia> cs[[2]][1]
Dict{Symbol, Symbol} with 1 entry:
  :A => :a2
```
"""
struct Cases <: AbstractVector{Case}
    cases::Vector{Case}
    variables::Vector{Symbol}
end

function Cases(rows::AbstractVector{<:AbstractDict{Symbol,Symbol}})
    cases = Case[Case(r) for r in rows]
    vars = Symbol[]
    seen = Set{Symbol}()
    for c in cases, v in sort!(collect(keys(c)))
        v in seen && continue
        push!(seen, v)
        push!(vars, v)
    end
    return Cases(cases, vars)
end
Cases(c::Cases) = c

Base.size(c::Cases) = size(c.cases)
Base.getindex(c::Cases, i::Int) = c.cases[i]
Base.getindex(c::Cases, idx::AbstractVector) = Cases(c.cases[idx])
Base.IndexStyle(::Type{Cases}) = IndexLinear()

"""
    variables(c::Cases) -> Vector{Symbol}

Every variable observed in at least one case, in order of first appearance.
"""
variables(c::Cases) = c.variables

function Base.show(io::IO, c::Cases)
    return print(io, "Cases with ", length(c.cases), " observations over ",
                 length(c.variables), " variables")
end

# A dataset is usually long; print a summary and the first few rows rather than
# the whole vector, which the AbstractVector fallback would do.
function Base.show(io::IO, ::MIME"text/plain", c::Cases)
    show(io, c)
    for i in 1:min(3, length(c.cases))
        print(io, "\n  ", i, ": ", sort!(collect(c.cases[i]); by=first))
    end
    length(c.cases) > 3 && print(io, "\n  ... (", length(c.cases) - 3, " more)")
    return nothing
end

"""
    Cases(samples::AncestralSamples) -> Cases

The rows of a sample table as complete cases, so that
`Cases(ancestral_sample(m, n; rng))` simulates a validation dataset from a
model with a seeded generator.
"""
function Cases(samples::AncestralSamples)
    rows = Case[Case(v => samples.states[i, j] for (j, v) in enumerate(samples.vars))
                for i in 1:size(samples.states, 1)]
    return Cases(rows)
end

# Predictions
# -----------

"""
    Predictions

The predicted distributions of one target variable over a dataset, as produced
by [`predict`](@ref). Fields:

- `target`: the predicted variable;
- `axis`: its `FiniteKernels.FiniteAxis`, giving the column order of the table;
- `probabilities`: an `n x k` matrix whose row `i` is `P(target | evidence_i)`;
- `outcomes[i]`: the index in `axis` of the observed state of the target in
  case `i`, or `0` when the case does not record it;
- `evidence_variables`: the variables that were allowed to enter as evidence.

`size(p)` is `(n, k)`. Scoring functions require every outcome to be recorded.
"""
struct Predictions
    target::Symbol
    axis::FiniteAxis
    probabilities::Matrix{Float64}
    outcomes::Vector{Int}
    evidence_variables::Vector{Symbol}
end

Base.size(p::Predictions) = size(p.probabilities)
Base.length(p::Predictions) = size(p.probabilities, 1)

function Base.show(io::IO, p::Predictions)
    return print(io, "Predictions of ", repr(p.target), " for ", length(p), " cases over ",
                 size(p.probabilities, 2), " states")
end

# The column of `p` holding `state`.
function _state_index(p::Predictions, state::Symbol)
    return label_index(p.axis, state)
end

function _require_outcomes(p::Predictions, op::Symbol)
    missing_at = findall(iszero, p.outcomes)
    isempty(missing_at) ||
        throw(ScopeError(op,
                         "cases $(first(missing_at, 5)) do not record the target $(repr(p.target)); a score needs the observed outcome",
                         [p.target]))
    return nothing
end

"""
    predict(m::BayesModel, cases, target; evidence_vars=nothing, backend=VariableElimination())
        -> Predictions
    predict(fg::FactorGraph, cases, target; evidence_vars=nothing, backend=VariableElimination())

The posterior distribution of `target` in every case of `cases` (a [`Cases`](@ref)
or a vector of `Dict{Symbol,Symbol}`). The evidence entered for case `i` is the
case restricted to `evidence_vars` (every variable of the model except the
target when `evidence_vars` is `nothing`); the target is always withheld, so
the prediction is genuinely out of sample with respect to that variable. Any
variable a case does not record is simply not entered.

The model is [`compile`](@ref)d once, and the evidence recorded on it by
`observe` is merged into every case's evidence. On the [`FactorGraph`](@ref)
method there is no model to read that from, so the keyword `base_evidence`
carries it; pass it directly to condition every case on the same background
evidence.

Throws [`ScopeError`](@ref) if `target` or an evidence variable is not a
variable of the model, and `FiniteKernels.KernelNormalizationError`, naming
the case, if a case's evidence has probability zero under the model.
"""
function predict(fg::FactorGraph, cases, target::Symbol;
                 evidence_vars::Union{Nothing,AbstractVector{Symbol}}=nothing,
                 backend::InferenceBackend=VariableElimination(),
                 base_evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}())
    cs = Cases(cases)
    haskey(fg.axes, target) ||
        throw(ScopeError(:predict, "the target is not a variable of the model", [target]))
    ev_vars = if evidence_vars === nothing
        Symbol[v for v in variables(fg) if v != target]
    else
        collect(Symbol, evidence_vars)
    end
    unknown = [v for v in ev_vars if !haskey(fg.axes, v)]
    isempty(unknown) ||
        throw(ScopeError(:predict, "evidence variables are not variables of the model",
                         unknown))
    target in ev_vars &&
        throw(ScopeError(:predict, "the target cannot be one of the evidence variables",
                         [target]))
    ax = fg.axes[target]
    probs = Matrix{Float64}(undef, length(cs), length(ax))
    outcomes = zeros(Int, length(cs))
    for (i, case) in enumerate(cs)
        ev = merge(Dict{Symbol,Symbol}(base_evidence),
                   Dict{Symbol,Symbol}(v => case[v] for v in ev_vars if haskey(case, v)))
        delete!(ev, target)
        f = try
            first(infer(fg, [target]; evidence=ev, backend))
        catch e
            e isa KernelNormalizationError &&
                throw(KernelNormalizationError("the evidence of case $i has probability zero under the model, so $(repr(target)) has no posterior there",
                                               1.0, 0.0; name=target))
            rethrow()
        end
        probs[i, :] = f.table
        haskey(case, target) && (outcomes[i] = label_index(ax, case[target]))
    end
    return Predictions(target, ax, probs, outcomes, ev_vars)
end

function predict(m::BayesModel, cases, target::Symbol;
                 atol::Real=BayesianNetworks.DEFAULT_ATOL, kwargs...)
    return predict(compile(m; atol=atol), cases, target;
                   base_evidence=Dict{Symbol,Symbol}(BayesianNetworks.evidence(m)),
                   kwargs...)
end

"""
    baseline(m::BayesModel, cases, target; backend=VariableElimination()) -> Predictions
    baseline(fg::FactorGraph, cases, target; backend=VariableElimination())

The "always predict the prior" comparator: [`predict`](@ref) with no evidence
variables, so every case receives the marginal `P(target)` of the model. A
network that does not beat this baseline on the proper scores of
[`evaluate`](@ref) has not been shown to be useful for the target.
"""
function baseline(fg::FactorGraph, cases, target::Symbol; kwargs...)
    return predict(fg, cases, target; evidence_vars=Symbol[], kwargs...)
end
function baseline(m::BayesModel, cases, target::Symbol; kwargs...)
    return predict(m, cases, target; evidence_vars=Symbol[], kwargs...)
end

# Proper scoring rules
# --------------------

"""
    brier_score(p::Predictions) -> Float64
    brier_score(probabilities::AbstractMatrix, outcomes::AbstractVector{<:Integer})

The multi-category Brier score, negatively oriented (smaller is better):

```math
BS = \\frac{1}{n} \\sum_{i=1}^{n} \\sum_{k=1}^{K} (p_{ik} - y_{ik})^2,
```

with `y_{ik} = 1` when case `i` was observed in state `k` and `0` otherwise.
It lies in `[0, 2]` and is a strictly proper scoring rule
([Brier1950](@cite); [GneitingRaftery2007](@cite), section 4.1).

In the **binary special case** this is twice the familiar `mean((p - y)^2)`
over the probability of the positive state, because both columns contribute
the same squared error; halve it to compare with binary-only software.

Throws [`ScopeError`](@ref) if a case does not record the observed state.
"""
function brier_score(probabilities::AbstractMatrix{<:Real},
                     outcomes::AbstractVector{<:Integer})
    n, _ = size(probabilities)
    length(outcomes) == n ||
        throw(ShapeError(:brier_score, "one outcome per row of the prediction table", n,
                         length(outcomes)))
    total = 0.0
    for i in 1:n
        for k in axes(probabilities, 2)
            total += (probabilities[i, k] - (k == outcomes[i] ? 1.0 : 0.0))^2
        end
    end
    return total / n
end
function brier_score(p::Predictions)
    _require_outcomes(p, :brier_score)
    return brier_score(p.probabilities, p.outcomes)
end

"""
    log_score(p::Predictions; floor=1e-12) -> Float64
    log_score(probabilities::AbstractMatrix, outcomes::AbstractVector{<:Integer}; floor=1e-12)

The logarithmic score, **positively oriented** (larger is better):

```math
LS = \\frac{1}{n} \\sum_{i=1}^{n} \\log \\max(p_{i y_i}, \\texttt{floor}).
```

This is the mean predictive log-likelihood, the local strictly proper scoring
rule of [GneitingRaftery2007](@cite) (section 4.1); negate it for the mean
negative log-likelihood used by some software.

**Zero-probability outcomes.** A model that assigns probability zero to an
event that happened is infinitely wrong, and `floor = 0` reports exactly that
by returning `-Inf`. Because a single such case would then hide every other
difference, the default floors the probability at `1e-12` (about `-27.6`
nats per case); state the floor whenever a log score is reported.
"""
function log_score(probabilities::AbstractMatrix{<:Real},
                   outcomes::AbstractVector{<:Integer}; floor::Real=1e-12)
    n, _ = size(probabilities)
    length(outcomes) == n ||
        throw(ShapeError(:log_score, "one outcome per row of the prediction table", n,
                         length(outcomes)))
    floor >= 0 ||
        throw(ArgumentError("floor must be non-negative, got floor = $floor"))
    total = 0.0
    for i in 1:n
        total += log(max(probabilities[i, outcomes[i]], floor))
    end
    return total / n
end
function log_score(p::Predictions; kwargs...)
    _require_outcomes(p, :log_score)
    return log_score(p.probabilities, p.outcomes; kwargs...)
end

"""
    spherical_score(p::Predictions) -> Float64
    spherical_score(probabilities::AbstractMatrix, outcomes::AbstractVector{<:Integer})

The spherical score, positively oriented and bounded by `[0, 1]`:

```math
S = \\frac{1}{n} \\sum_{i=1}^{n} \\frac{p_{i y_i}}{\\lVert p_i \\rVert_2}.
```

A strictly proper rule ([GneitingRaftery2007](@cite), section 4.1) that, unlike
[`log_score`](@ref), stays finite when a realised outcome was given
probability zero. A row that is identically zero contributes zero.
"""
function spherical_score(probabilities::AbstractMatrix{<:Real},
                         outcomes::AbstractVector{<:Integer})
    n, _ = size(probabilities)
    length(outcomes) == n ||
        throw(ShapeError(:spherical_score, "one outcome per row of the prediction table",
                         n, length(outcomes)))
    total = 0.0
    for i in 1:n
        nrm = sqrt(sum(abs2, view(probabilities, i, :)))
        iszero(nrm) || (total += probabilities[i, outcomes[i]] / nrm)
    end
    return total / n
end
function spherical_score(p::Predictions)
    _require_outcomes(p, :spherical_score)
    return spherical_score(p.probabilities, p.outcomes)
end

# Calibration
# -----------

"""
    CalibrationCurve

A binned reliability diagram, as returned by [`calibration_curve`](@ref):
`edges` (`bins + 1` bin boundaries on `[0, 1]`), `centres` (the bin midpoints),
`predicted` (the mean predicted probability in each bin), `observed` (the
observed relative frequency in each bin) and `counts` (how many cases fell in
each bin). Empty bins carry `NaN` in `predicted` and `observed`.
"""
struct CalibrationCurve
    edges::Vector{Float64}
    centres::Vector{Float64}
    predicted::Vector{Float64}
    observed::Vector{Float64}
    counts::Vector{Int}
end

function Base.show(io::IO, c::CalibrationCurve)
    return print(io, "CalibrationCurve with ", length(c.counts), " bins over ",
                 sum(c.counts), " cases (ECE ", round(calibration_error(c); digits=4), ")")
end

function Base.show(io::IO, ::MIME"text/plain", c::CalibrationCurve)
    show(io, c)
    print(io, "\n  centre  predicted   observed  count")
    for i in eachindex(c.counts)
        print(io, "\n  ", lpad(round(c.centres[i]; digits=3), 6), "  ",
              lpad(round(c.predicted[i]; digits=4), 9), "  ",
              lpad(round(c.observed[i]; digits=4), 9), "  ", lpad(c.counts[i], 5))
    end
    return nothing
end

"""
    calibration_curve(predictions::AbstractVector{<:Real}, outcomes::AbstractVector{Bool}; bins=10)
        -> CalibrationCurve
    calibration_curve(p::Predictions, state::Symbol; bins=10)

A reliability diagram for a one-dimensional probabilistic forecast:
`predictions[i]` is the predicted probability of an event and `outcomes[i]`
says whether it happened. The unit interval is split into `bins` equal-width
bins, and each bin reports its mean predicted probability, the observed
relative frequency and the number of cases. A perfectly calibrated forecaster
has `observed ≈ predicted` in every populated bin.

The [`Predictions`](@ref) method scores the one-versus-rest event
`target == state`.
"""
function calibration_curve(predictions::AbstractVector{<:Real},
                           outcomes::AbstractVector{Bool}; bins::Integer=10)
    length(predictions) == length(outcomes) ||
        throw(ShapeError(:calibration_curve, "one outcome per prediction",
                         length(predictions), length(outcomes)))
    bins >= 1 || throw(ArgumentError("bins must be at least 1, got bins = $bins"))
    edges = collect(range(0.0, 1.0; length=bins + 1))
    centres = [(edges[i] + edges[i + 1]) / 2 for i in 1:bins]
    sum_p = zeros(Float64, bins)
    sum_y = zeros(Float64, bins)
    counts = zeros(Int, bins)
    for (p, y) in zip(predictions, outcomes)
        (0 <= p <= 1) ||
            throw(ArgumentError("predicted probabilities must lie in [0, 1], got $p"))
        b = clamp(ceil(Int, p * bins), 1, bins)
        counts[b] += 1
        sum_p[b] += p
        sum_y[b] += y
    end
    predicted = [counts[i] == 0 ? NaN : sum_p[i] / counts[i] for i in 1:bins]
    observed = [counts[i] == 0 ? NaN : sum_y[i] / counts[i] for i in 1:bins]
    return CalibrationCurve(edges, centres, predicted, observed, counts)
end
function calibration_curve(p::Predictions, state::Symbol; bins::Integer=10)
    _require_outcomes(p, :calibration_curve)
    k = _state_index(p, state)
    return calibration_curve(p.probabilities[:, k], p.outcomes .== k; bins)
end

"""
    calibration_error(c::CalibrationCurve) -> Float64
    calibration_error(predictions, outcomes; bins=10)
    calibration_error(p::Predictions, state::Symbol; bins=10)

The expected calibration error (ECE): the count-weighted mean absolute gap
between the observed frequency and the mean predicted probability of the bins
of a [`calibration_curve`](@ref),

```math
ECE = \\sum_b \\frac{n_b}{n} \\lvert \\bar{y}_b - \\bar{p}_b \\rvert .
```

Empty bins are skipped. Zero is perfect calibration; the value is sensitive to
the number of bins, so report `bins` alongside it.
"""
function calibration_error(c::CalibrationCurve)
    n = sum(c.counts)
    n == 0 && return NaN
    total = 0.0
    for i in eachindex(c.counts)
        c.counts[i] == 0 && continue
        total += c.counts[i] * abs(c.observed[i] - c.predicted[i])
    end
    return total / n
end
function calibration_error(predictions::AbstractVector{<:Real},
                           outcomes::AbstractVector{Bool}; kwargs...)
    return calibration_error(calibration_curve(predictions, outcomes; kwargs...))
end
function calibration_error(p::Predictions, state::Symbol; kwargs...)
    return calibration_error(calibration_curve(p, state; kwargs...))
end

# Discrimination
# --------------

"""
    ROCCurve

The receiver operating characteristic of a one-versus-rest forecast, as
returned by [`roc_curve`](@ref): `thresholds` (decreasing, starting at `Inf`),
`fpr` (false-positive rate) and `tpr` (true-positive rate), all of the same
length and starting at `(0, 0)`.
"""
struct ROCCurve
    thresholds::Vector{Float64}
    fpr::Vector{Float64}
    tpr::Vector{Float64}
end

function Base.show(io::IO, c::ROCCurve)
    return print(io, "ROCCurve with ", length(c.thresholds), " points (AUC ",
                 round(auc(c); digits=4), ")")
end

"""
    roc_curve(scores::AbstractVector{<:Real}, outcomes::AbstractVector{Bool}) -> ROCCurve
    roc_curve(p::Predictions, state::Symbol)

The ROC curve of the one-versus-rest problem "is the case in `state`?", ranked
by the predicted probability of that state. Every distinct score becomes one
point; tied scores share a point, so the curve is the correct step function
for a forecaster with ties.

ROC and [`auc`](@ref) describe **discrimination**, that is the quality of the
implied *ranking* of cases. They ignore calibration entirely and are only
meaningful when ranking cases is a sensible thing to do with the target (for
instance, prioritising sites for survey); a well-ranked but badly calibrated
network is still useless as a probability statement, which is what
[`calibration_curve`](@ref) and the proper scores measure.

Throws `ArgumentError` unless both outcomes occur.
"""
function roc_curve(scores::AbstractVector{<:Real}, outcomes::AbstractVector{Bool})
    length(scores) == length(outcomes) ||
        throw(ShapeError(:roc_curve, "one outcome per score", length(scores),
                         length(outcomes)))
    npos = count(outcomes)
    nneg = length(outcomes) - npos
    (npos == 0 || nneg == 0) &&
        throw(ArgumentError("roc_curve needs at least one positive and one negative outcome, got $npos positive of $(length(outcomes))"))
    ord = sortperm(collect(scores); rev=true)
    thresholds = Float64[Inf]
    fpr = Float64[0.0]
    tpr = Float64[0.0]
    tp = 0
    fp = 0
    i = 1
    while i <= length(ord)
        s = scores[ord[i]]
        while i <= length(ord) && scores[ord[i]] == s
            outcomes[ord[i]] ? (tp += 1) : (fp += 1)
            i += 1
        end
        push!(thresholds, s)
        push!(fpr, fp / nneg)
        push!(tpr, tp / npos)
    end
    return ROCCurve(thresholds, fpr, tpr)
end
function roc_curve(p::Predictions, state::Symbol)
    _require_outcomes(p, :roc_curve)
    k = _state_index(p, state)
    return roc_curve(p.probabilities[:, k], p.outcomes .== k)
end

# Mid-ranks, so that ties share their average rank.
function _tiedrank(x::AbstractVector{<:Real})
    ord = sortperm(collect(x))
    r = zeros(Float64, length(x))
    i = 1
    while i <= length(ord)
        j = i
        while j < length(ord) && x[ord[j + 1]] == x[ord[i]]
            j += 1
        end
        mid = (i + j) / 2
        for t in i:j
            r[ord[t]] = mid
        end
        i = j + 1
    end
    return r
end

"""
    auc(scores::AbstractVector{<:Real}, outcomes::AbstractVector{Bool}) -> Float64
    auc(p::Predictions, state::Symbol)
    auc(c::ROCCurve)

The area under the ROC curve. The vector method computes it exactly from the
Mann-Whitney statistic on mid-ranks, so tied scores contribute one half:
`auc` is the probability that a randomly chosen positive case is ranked above
a randomly chosen negative one, with ties counted as half a win. A perfect
ranking gives `1.0` and an uninformative forecast `0.5`. The
[`ROCCurve`](@ref) method integrates the stored curve by the trapezoidal rule
and agrees with it.

See [`roc_curve`](@ref) for when a ranking interpretation is appropriate.
"""
function auc(scores::AbstractVector{<:Real}, outcomes::AbstractVector{Bool})
    length(scores) == length(outcomes) ||
        throw(ShapeError(:auc, "one outcome per score", length(scores), length(outcomes)))
    npos = count(outcomes)
    nneg = length(outcomes) - npos
    (npos == 0 || nneg == 0) &&
        throw(ArgumentError("auc needs at least one positive and one negative outcome, got $npos positive of $(length(outcomes))"))
    r = _tiedrank(scores)
    return (sum(r[i] for i in eachindex(r) if outcomes[i]) - npos * (npos + 1) / 2) /
           (npos * nneg)
end
function auc(p::Predictions, state::Symbol)
    _require_outcomes(p, :auc)
    k = _state_index(p, state)
    return auc(p.probabilities[:, k], p.outcomes .== k)
end
function auc(c::ROCCurve)
    total = 0.0
    for i in 2:length(c.fpr)
        total += (c.fpr[i] - c.fpr[i - 1]) * (c.tpr[i] + c.tpr[i - 1]) / 2
    end
    return total
end

# Classification
# --------------

"""
    ConfusionMatrix

The maximum-a-posteriori classification table of a [`Predictions`](@ref):
`labels` are the states of the target and `counts[i, j]` is the number of
cases observed in state `i` and predicted (as the argmax of the posterior) to
be in state `j`. Rows are truth, columns are prediction.
"""
struct ConfusionMatrix
    labels::Vector{Symbol}
    counts::Matrix{Int}
end

function Base.show(io::IO, c::ConfusionMatrix)
    return print(io, "ConfusionMatrix over ", length(c.labels), " states (accuracy ",
                 round(accuracy(c); digits=4), ")")
end

function Base.show(io::IO, ::MIME"text/plain", c::ConfusionMatrix)
    show(io, c)
    w = maximum(length ∘ string, c.labels; init=8)
    print(io, "\n  ", lpad("observed \\ predicted", w + 2))
    for l in c.labels
        print(io, "  ", lpad(string(l), w))
    end
    for i in eachindex(c.labels)
        print(io, "\n  ", lpad(string(c.labels[i]), w + 2))
        for j in eachindex(c.labels)
            print(io, "  ", lpad(c.counts[i, j], w))
        end
    end
    return nothing
end

"""
    confusion_matrix(p::Predictions) -> ConfusionMatrix

The maximum-a-posteriori confusion matrix: each case is classified as the
state with the largest posterior probability (ties resolve to the first such
state, as in [`argmax_table`](@ref)) and counted against its observed state.
"""
function confusion_matrix(p::Predictions)
    _require_outcomes(p, :confusion_matrix)
    k = length(p.axis)
    counts = zeros(Int, k, k)
    for i in 1:length(p)
        counts[p.outcomes[i], argmax(view(p.probabilities, i, :))] += 1
    end
    return ConfusionMatrix(collect(Symbol, p.axis.labels), counts)
end

"""
    accuracy(c::ConfusionMatrix) -> Float64
    accuracy(p::Predictions)

The fraction of cases whose maximum-a-posteriori state is the observed one,
that is the trace of the [`confusion_matrix`](@ref) over its total. Accuracy
is not a proper scoring rule -- it ignores how confident the network was --
so report it beside [`brier_score`](@ref) and [`log_score`](@ref), never
instead of them.
"""
function accuracy(c::ConfusionMatrix)
    total = sum(c.counts)
    total == 0 && return NaN
    return sum(c.counts[i, i] for i in axes(c.counts, 1)) / total
end
accuracy(p::Predictions) = accuracy(confusion_matrix(p))

# Splits
# ------

"""
    holdout(cases; by=nothing, fraction=0.3, rng=Random.default_rng())
        -> (train::Vector{Int}, test::Vector{Int})

Split the indices of `cases` into a training and a test part. The two are
disjoint and their union is `1:length(cases)`.

With `by = nothing` the split is a uniformly random draw of
`round(Int, fraction * n)` cases into the test set. With `by = :X` the cases
are first grouped by their observed state of `X` and whole groups are drawn,
so **no group is ever split**; a shuffled group joins the test set only when
that brings its size closer to `fraction * n`, so the requested fraction is
approximate and a coarse grouping can miss it by a lot. That is how a spatial
or temporal holdout is expressed: record the region, catchment or year as a
variable of the case and hold out by it, as the ecological literature asks for
([ChenPollino2012](@cite)). Cases that do not record `by` form their own group,
and a grouping that puts every case in one group is rejected.

This function splits *cases*; it does not refit anything. The package has no
parameter estimation, so the honest use of a holdout here is to score a model
whose CPTs came from elsewhere (expert elicitation, a published table, a
learning package) on cases it did not see. Refitting per fold would mean
re-estimating every CPT from the training indices -- counts with a prior for
complete data, EM for incomplete data -- and rebuilding the `BayesModel` with
`BayesianNetworks.bind_cpt` before scoring each test fold.

Throws `ArgumentError` unless `0 < fraction < 1`.
"""
function holdout(cases; by::Union{Nothing,Symbol}=nothing, fraction::Real=0.3,
                 rng::AbstractRNG=default_rng())
    cs = Cases(cases)
    n = length(cs)
    0 < fraction < 1 ||
        throw(ArgumentError("fraction must lie strictly between 0 and 1, got fraction = $fraction"))
    n >= 2 ||
        throw(ArgumentError("a holdout needs at least two cases, got n = $n"))
    want = clamp(round(Int, fraction * n), 1, n - 1)
    test = Int[]
    if by === nothing
        test = sort!(shuffle(rng, 1:n)[1:want])
    else
        groups = _groups(cs, by)
        shuffled = shuffle(rng, collect(keys(groups)))
        for key in shuffled
            length(test) >= want && break
            g = groups[key]
            # take the group only when it moves the test set towards `want`,
            # so that one huge group does not swallow the whole dataset
            abs(length(test) + length(g) - want) <= abs(length(test) - want) || continue
            append!(test, g)
        end
        isempty(test) && append!(test, groups[argmin(k -> length(groups[k]), shuffled)])
        length(test) == n &&
            throw(ArgumentError("grouping by $(repr(by)) cannot produce a holdout: every case is in one group"))
        sort!(test)
    end
    train = setdiff(1:n, test)
    return train, test
end

"""
    kfold(cases; k=5, by=nothing, rng=Random.default_rng())
        -> Vector{Tuple{Vector{Int},Vector{Int}}}

`k` `(train, test)` index splits of `cases`. The test parts are disjoint and
cover `1:length(cases)`, and each `train` is its complement. With `by = :X`
the cases are grouped by their state of `X` and whole groups are assigned to
folds (largest group first, always to the currently smallest fold), so no
group is ever split across folds; the folds are then only approximately equal
in size.

As for [`holdout`](@ref), this splits cases and does not refit CPTs.

Throws `ArgumentError` unless `2 <= k <= length(cases)`.
"""
function kfold(cases; k::Integer=5, by::Union{Nothing,Symbol}=nothing,
               rng::AbstractRNG=default_rng())
    cs = Cases(cases)
    n = length(cs)
    2 <= k <= n ||
        throw(ArgumentError("k must satisfy 2 <= k <= $(n), got k = $k"))
    folds = [Int[] for _ in 1:k]
    if by === nothing
        for (i, c) in enumerate(shuffle(rng, 1:n))
            push!(folds[mod1(i, k)], c)
        end
    else
        groups = _groups(cs, by)
        keys_shuffled = shuffle(rng, collect(keys(groups)))
        sort!(keys_shuffled; by=key -> -length(groups[key]))
        for key in keys_shuffled
            f = argmin(map(length, folds))
            append!(folds[f], groups[key])
        end
    end
    foreach(sort!, folds)
    return [(setdiff(1:n, f), f) for f in folds]
end

# Case indices grouped by the observed state of `by`; cases that do not record
# it share the group `nothing`.
function _groups(cs::Cases, by::Symbol)
    groups = Dict{Union{Nothing,Symbol},Vector{Int}}()
    for (i, c) in enumerate(cs)
        push!(get!(groups, get(c, by, nothing), Int[]), i)
    end
    return groups
end

# The report
# ----------

"""
    EvaluationResult

Everything [`evaluate`](@ref) computed, for the network and for the
marginal-prior [`baseline`](@ref): `target`, `n`, `evidence_variables`, the
proper scores `brier` / `log` / `spherical`, `accuracy` and `confusion`, the
`calibration` curve and its `calibration_error` for the state `state`, the
`auc` for the same state, and the `baseline_*` counterparts. `show` prints
the two columns side by side.

An `auc` of `NaN` means the state did not occur (or always occurred) in the
data, so discrimination is undefined there.
"""
struct EvaluationResult
    target::Symbol
    state::Symbol
    n::Int
    evidence_variables::Vector{Symbol}
    predictions::Predictions
    brier::Float64
    log::Float64
    spherical::Float64
    accuracy::Float64
    confusion::ConfusionMatrix
    calibration::CalibrationCurve
    calibration_error::Float64
    auc::Float64
    baseline_predictions::Predictions
    baseline_brier::Float64
    baseline_log::Float64
    baseline_spherical::Float64
    baseline_accuracy::Float64
    baseline_calibration_error::Float64
    baseline_auc::Float64
end

function Base.show(io::IO, r::EvaluationResult)
    return print(io, "EvaluationResult for ", repr(r.target), " on ", r.n, " cases (Brier ",
                 round(r.brier; digits=4), " vs prior ", round(r.baseline_brier; digits=4),
                 ")")
end

function Base.show(io::IO, ::MIME"text/plain", r::EvaluationResult)
    print(io, "EvaluationResult for ", repr(r.target), " on ", r.n, " cases")
    print(io, "\n  evidence: ",
          isempty(r.evidence_variables) ? "(none)" : join(r.evidence_variables, ", "))
    rows = ["Brier score (lower better)" => (r.brier, r.baseline_brier),
            "log score (higher better)" => (r.log, r.baseline_log),
            "spherical score (higher)" => (r.spherical, r.baseline_spherical),
            "accuracy (MAP)" => (r.accuracy, r.baseline_accuracy),
            "calibration error (ECE)" => (r.calibration_error,
                                          r.baseline_calibration_error),
            "AUC($(repr(r.state)))" => (r.auc, r.baseline_auc)]
    w = maximum(length ∘ first, rows)
    print(io, "\n  ", rpad("metric", w), "     network       prior")
    for (name, (a, b)) in rows
        print(io, "\n  ", rpad(name, w), "  ", lpad(round(a; digits=4), 10), "  ",
              lpad(round(b; digits=4), 10))
    end
    return nothing
end

"""
    evaluate(m::BayesModel, cases, target; evidence_vars=nothing, state=nothing,
             backend=VariableElimination(), bins=10, floor=1e-12) -> EvaluationResult
    evaluate(fg::FactorGraph, cases, target; ...)

Score a network on a dataset: [`predict`](@ref) the withheld `target` from
every case, then compute the proper scores ([`brier_score`](@ref),
[`log_score`](@ref), [`spherical_score`](@ref)), the maximum-a-posteriori
[`accuracy`](@ref) and [`confusion_matrix`](@ref), the
[`calibration_curve`](@ref) and its [`calibration_error`](@ref), and the
[`auc`](@ref), each of them also for the marginal-prior [`baseline`](@ref) so
that "does the network beat always predicting the prior?" can be answered.

`state` selects the one-versus-rest state used for calibration and ROC; it
defaults to the last state of the target. `bins` and `floor` are passed to
[`calibration_curve`](@ref) and [`log_score`](@ref).

This function is unrelated to `CategoricalBayesianNetworks.evaluate`, which
interprets a categorical expression in FinStoch; `using` both packages needs
the name qualified.

```jldoctest
julia> using BayesianNetworks, Random

julia> m = reference_habitat_model();

julia> cases = Cases(ancestral_sample(m, 200; rng=MersenneTwister(1)));

julia> r = evaluate(m, cases, :Occupancy; evidence_vars=[:Vegetation]);

julia> r.n, r.brier < r.baseline_brier
(200, true)
```
"""
function evaluate(fg::FactorGraph, cases, target::Symbol;
                  evidence_vars::Union{Nothing,AbstractVector{Symbol}}=nothing,
                  state::Union{Nothing,Symbol}=nothing,
                  backend::InferenceBackend=VariableElimination(), bins::Integer=10,
                  floor::Real=1e-12,
                  base_evidence::AbstractDict{Symbol,Symbol}=Dict{Symbol,Symbol}())
    cs = Cases(cases)
    p = predict(fg, cs, target; evidence_vars, backend, base_evidence)
    b = predict(fg, cs, target; evidence_vars=Symbol[], backend, base_evidence)
    _require_outcomes(p, :evaluate)
    st = state === nothing ? last(p.axis.labels) : state
    k = _state_index(p, st)
    both = 0 < count(==(k), p.outcomes) < length(p)
    return EvaluationResult(target, st, length(p), p.evidence_variables, p,
                            brier_score(p), log_score(p; floor), spherical_score(p),
                            accuracy(p), confusion_matrix(p),
                            calibration_curve(p, st; bins), calibration_error(p, st; bins),
                            both ? auc(p, st) : NaN, b, brier_score(b),
                            log_score(b; floor), spherical_score(b), accuracy(b),
                            calibration_error(b, st; bins), both ? auc(b, st) : NaN)
end

function evaluate(m::BayesModel, cases, target::Symbol;
                  atol::Real=BayesianNetworks.DEFAULT_ATOL, kwargs...)
    return evaluate(compile(m; atol=atol), cases, target;
                    base_evidence=Dict{Symbol,Symbol}(BayesianNetworks.evidence(m)),
                    kwargs...)
end
