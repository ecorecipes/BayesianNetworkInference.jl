# Junction trees and belief propagation
Simon Frost

- [Overview](#overview)
- [Setup](#setup)
- [The junction tree of asia](#the-junction-tree-of-asia)
- [Calibration and all marginals](#calibration-and-all-marginals)
- [Belief propagation on a tree](#belief-propagation-on-a-tree)
- [Loopy belief propagation on asia](#loopy-belief-propagation-on-asia)
- [Timing: all marginals by junction tree versus variable
  elimination](#timing-all-marginals-by-junction-tree-versus-variable-elimination)
- [Summary](#summary)
- [References](#references)

## Overview

Variable elimination answers one query per run. When every marginal of a
network is wanted (an interactive “what does the evidence do to
everything” display, or a sensitivity sweep), two further backends work
on the same `FactorGraph`:

- `JunctionTree()` builds a clique tree of the interaction graph with
  [CliqueTrees.jl](https://github.com/AlgebraicJulia/CliqueTrees.jl)
  (`cliquetree(graph; alg=MF(), snd=Maximal())`), calibrates it by
  Shafer-Shenoy ([Shafer and Shenoy 1990](#ref-ShaferShenoy1990))
  message passing, and reads every marginal off the calibrated beliefs;
  the cost is two sweeps over the tree, whatever the number of variables
  queried.
- `BeliefPropagation()` runs sum-product message passing on the factor
  graph itself. On a tree-structured factor graph (a polytree network)
  it is exact; on a graph with loops it is the loopy approximation, with
  convergence diagnostics.

## Setup

``` julia
using BayesianNetworkInference
using FiniteKernels
using BayesianNetworks
using BayesianNetworkFormats: fixture_path
using Random
```

## The junction tree of asia

The asia network is read from the BIF fixture and compiled to a factor
graph, one factor per mechanism.

``` julia
asia = read_bayesnet(fixture_path("bif/asia.bif"))
fg = compile(asia)
```

    FactorGraph{Float64} with 8 factors over 8 variables

`build_junction_tree` triangulates the moral graph with a fill-reducing
order and returns the maximal cliques of the result as a rooted tree.
Each non-root clique lists the separator it shares with its parent.

``` julia
jt = build_junction_tree(fg)
```

    CompiledJunctionTree with 6 cliques (treewidth 2)
      1: [:asia, :tub]  --[:tub]--> 2
      2: [:tub, :lung, :either]  --[:lung, :either]--> 6
      3: [:either, :xray]  --[:either]--> 6
      4: [:bronc, :either, :dysp]  --[:bronc, :either]--> 5
      5: [:smoke, :bronc, :either]  --[:smoke, :either]--> 6
      6: [:smoke, :lung, :either]

``` julia
(n_cliques=length(jt), treewidth=jt.treewidth, roots=jt.roots)
```

    (n_cliques = 6, treewidth = 2, roots = [6])

Every factor is assigned to the smallest clique containing its scope,
and each variable records the smallest clique it lives in (where its
marginal is read from):

``` julia
[fg.provenance[i].variable => jt.cliques[c] for (i, c) in enumerate(jt.assignment)]
```

    8-element Vector{Pair{Symbol, Vector{Symbol}}}:
       :asia => [:asia, :tub]
        :tub => [:asia, :tub]
      :smoke => [:smoke, :bronc, :either]
       :lung => [:smoke, :lung, :either]
      :bronc => [:smoke, :bronc, :either]
     :either => [:tub, :lung, :either]
       :xray => [:either, :xray]
       :dysp => [:bronc, :either, :dysp]

``` julia
Dict(v => jt.cliques[c] for (v, c) in jt.home)
```

    Dict{Symbol, Vector{Symbol}} with 8 entries:
      :tub    => [:asia, :tub]
      :xray   => [:either, :xray]
      :bronc  => [:bronc, :either, :dysp]
      :either => [:either, :xray]
      :smoke  => [:smoke, :bronc, :either]
      :lung   => [:tub, :lung, :either]
      :dysp   => [:bronc, :either, :dysp]
      :asia   => [:asia, :tub]

## Calibration and all marginals

`calibrate` conditions the factors on the evidence, multiplies them into
their cliques, and passes messages from the leaves to the root and back.
The beliefs are then `P(clique, evidence)`, so their total mass is the
probability of the evidence.

``` julia
cal = calibrate(fg; evidence=Dict(:asia => :yes, :xray => :yes))
```

    CalibratedJunctionTree{Float64} with 6 cliques, P(evidence) = 0.001450925

``` julia
[scope(b) => round(sum(b.table); digits=6) for b in cal.beliefs]
```

    6-element Vector{Pair{Vector{Symbol}, Float64}}:
                        [:tub] => 0.001451
        [:tub, :either, :lung] => 0.001451
                     [:either] => 0.001451
      [:bronc, :either, :dysp] => 0.001451
     [:smoke, :bronc, :either] => 0.001451
      [:either, :smoke, :lung] => 0.001451

`all_marginals` does this once and returns every single-variable
posterior; observed variables appear as point masses. The junction-tree
answers agree with variable elimination to floating-point precision.

``` julia
ev = Dict(:asia => :yes, :xray => :yes)
jt_marginals = all_marginals(fg; evidence=ev)
ve_marginals = all_marginals(fg; evidence=ev, backend=VariableElimination())
sort([v => round.(m.table; digits=4) for (v, m) in jt_marginals]; by=first)
```

    8-element Vector{Pair{Symbol, Vector{Float64}}}:
       :asia => [1.0, 0.0]
      :bronc => [0.4911, 0.5089]
       :dysp => [0.6811, 0.3189]
     :either => [0.6906, 0.3094]
       :lung => [0.3715, 0.6285]
      :smoke => [0.637, 0.363]
        :tub => [0.3377, 0.6623]
       :xray => [1.0, 0.0]

``` julia
maximum(maximum(abs.(jt_marginals[v].table .- ve_marginals[v].table)) for v in variables(fg))
```

    1.1102230246251565e-16

Single queries that lie in one clique go through `infer` as with any
backend, with `JunctionTreeDiagnostics` telling which clique served the
query:

``` julia
post, diag = infer(fg, [:bronc, :either]; evidence=ev, backend=JunctionTree())
diag
```

    JunctionTreeDiagnostics(4, 6, 2, 8, 10, false)

``` julia
post ≈ infer(fg, [:bronc, :either]; evidence=ev)[1]
```

    true

A query whose variables are in no common clique falls back to variable
elimination (with a warning, and `diagnostics.fallback` set), so
`infer(fg, [:asia, :smoke]; backend=JunctionTree())` is still exact and
still returns a `JunctionTreeDiagnostics`. The model-level entry point
does the same on a `BayesModel`:

``` julia
m_ev = observe(asia, :asia => :yes)
round.(all_marginals(m_ev)[:tub].table; digits=4)
```

    2-element Vector{Float64}:
     0.05
     0.95

## Belief propagation on a tree

The SPEC section 45 habitat network is a polytree, so its factor graph
is a tree and sum-product belief propagation is exact: it converges in a
number of sweeps bounded by the diameter, and the residual drops to
zero.

``` julia
habitat = compile(reference_habitat_model())
is_tree(habitat)
```

    true

``` julia
bp_marginals, bp_diag = belief_propagation(habitat; evidence=Dict(:Vegetation => :dense))
bp_diag
```

    BPDiagnostics(4, true, 0.0, true)

``` julia
ve = all_marginals(habitat; evidence=Dict(:Vegetation => :dense), backend=VariableElimination())
maximum(maximum(abs.(bp_marginals[v].table .- ve[v].table)) for v in variables(habitat))
```

    1.1102230246251565e-16

## Loopy belief propagation on asia

Asia has loops (`smoke -> lung -> either -> dysp <- bronc <- smoke`), so
belief propagation is approximate there. It converges, and the marginals
are close to the exact ones but not equal:

``` julia
loopy, loopy_diag = belief_propagation(fg)
loopy_diag
```

    BPDiagnostics(5, true, 1.3877787807814457e-17, false)

``` julia
exact = all_marginals(fg)
sort([v => (bp=round(loopy[v].table[1]; digits=5), exact=round(exact[v].table[1]; digits=5))
      for v in variables(fg)]; by=first)
```

    8-element Vector{Pair{Symbol, @NamedTuple{bp::Float64, exact::Float64}}}:
       :asia => (bp = 0.01, exact = 0.01)
      :bronc => (bp = 0.45, exact = 0.45)
       :dysp => (bp = 0.43931, exact = 0.43597)
     :either => (bp = 0.06483, exact = 0.06483)
       :lung => (bp = 0.055, exact = 0.055)
      :smoke => (bp = 0.5, exact = 0.5)
        :tub => (bp = 0.0104, exact = 0.0104)
       :xray => (bp = 0.11029, exact = 0.11029)

Damping and the schedule change the path to the fixed point, not the
fixed point itself:

``` julia
for backend in (BeliefPropagation(), BeliefPropagation(; damping=0.5),
                BeliefPropagation(; schedule=:sequential))
    ms, d = belief_propagation(fg, backend)
    println((damping=backend.damping, schedule=backend.schedule, iterations=d.iterations,
             converged=d.converged, residual=round(d.max_residual; sigdigits=3),
             dysp=round(ms[:dysp].table[1]; digits=6)))
end
```

    (damping = 0.0, schedule = :flooding, iterations = 5, converged = true, residual = 1.39e-17, dysp = 0.43931)
    (damping = 0.5, schedule = :flooding, iterations = 36, converged = true, residual = 8.35e-9, dysp = 0.439311)
    (damping = 0.0, schedule = :sequential, iterations = 2, converged = true, residual = 0.0, dysp = 0.43931)

Evidence that cuts every loop makes the conditioned graph a tree again,
and the diagnostics report `tree = true`, so the fixed point of the
message equations is the exact marginal (the field is named `tree`
rather than `exact` because the returned iterate differs from that fixed
point by about `max_residual`, which under damping is the tolerance
rather than machine precision):

``` julia
_, cut_diag = belief_propagation(fg; evidence=Dict(:either => :no))
cut_diag
```

    BPDiagnostics(5, true, 0.0, true)

Evidence of probability zero is an error in all three backends: variable
elimination and the junction tree fail to normalise, and belief
propagation raises the same `KernelNormalizationError` rather than
returning a normalised belief with `converged = true`.

``` julia
try
    belief_propagation(fg; evidence=Dict(:lung => :yes, :either => :no))
catch e
    typeof(e)
end
```

    KernelNormalizationError

## Timing: all marginals by junction tree versus variable elimination

A random network with 60 binary variables (up to three parents among the
previous eight) shows where the junction tree pays off: variable
elimination repeats the elimination once per variable, the junction tree
calibrates once.

``` julia
ax(v) = FiniteAxis(v, [:a, :b])
function random_network(rng, n; window=8, maxparents=3)
    kernels = Pair{Symbol,FiniteKernel}[]
    parents = Dict{Symbol,Vector{Symbol}}()
    for i in 1:n
        v = Symbol(:X, i)
        cands = [Symbol(:X, j) for j in max(1, i - window):(i - 1)]
        ps = shuffle(rng, cands)[1:min(rand(rng, 0:maxparents), length(cands))]
        parents[v] = ps
        dom = FiniteSpace(FiniteAxis[ax(p) for p in ps])
        push!(kernels, v => random_kernel(rng, dom, FiniteSpace(ax(v))))
    end
    return kernels, parents
end

kernels, parents = random_network(MersenneTwister(2026), 60)
big = FactorGraph([Factor(k, parents[v], v) for (v, k) in kernels])
big_ev = Dict(:X60 => :a, :X31 => :b)
(treewidth=build_junction_tree(big).treewidth, n_cliques=length(build_junction_tree(big)))
```

    (treewidth = 4, n_cliques = 48)

``` julia
rows = map((JunctionTree(), VariableElimination(), BeliefPropagation())) do backend
    all_marginals(big; evidence=big_ev, backend)          # warm up compilation
    t = @elapsed ms = all_marginals(big; evidence=big_ev, backend)
    reference = all_marginals(big; evidence=big_ev, backend=VariableElimination())
    err = maximum(maximum(abs.(ms[v].table .- reference[v].table)) for v in variables(big))
    (backend=nameof(typeof(backend)), seconds=round(t; digits=4), max_error=round(err; sigdigits=3))
end
foreach(println, rows)
```

    (backend = :JunctionTree, seconds = 0.001, max_error = 2.22e-16)
    (backend = :VariableElimination, seconds = 0.0231, max_error = 0.0)
    (backend = :BeliefPropagation, seconds = 0.0055, max_error = 0.0521)

The junction tree gives the exact answers of variable elimination at a
fraction of its cost (one calibration instead of one elimination per
variable), and the gap widens with the number of variables. Loopy belief
propagation is cheap too, but on this loopy graph its marginals are off
by several percent; it is the backend for graphs too dense to
triangulate, not a substitute for the junction tree when the treewidth
is small.

## Summary

A junction tree turns the factor graph into a tree of cliques and
calibrates it with one collect-and-distribute sweep of Shafer-Shenoy
messages ([Shafer and Shenoy 1990](#ref-ShaferShenoy1990)), so every
single-variable marginal comes out of a single pass instead of one
elimination per variable; because the messages are never divided by a
separator potential, deterministic tables with zero entries need no
special treatment, which is the practical difference from the
division-based architectures of Lauritzen and Spiegelhalter
([1988](#ref-LauritzenSpiegelhalter1988)) and Jensen et al.
([1990](#ref-JensenLauritzenOlesen1990)). Sum-product belief propagation
([Pearl 1988](#ref-Pearl1988); [Kschischang et al.
2001](#ref-Kschischang2001)) on the factor graph itself is exact on a
tree and approximate with cycles, where the loopy fixed point is the
empirical object studied by Murphy et al.
([1999](#ref-MurphyWeissJordan1999)). The next vignette, *Validation and
scoring*, asks the separate question of whether the answers a calibrated
network gives are any good.

## References

<div id="refs" class="references csl-bib-body hanging-indent">

<div id="ref-JensenLauritzenOlesen1990" class="csl-entry">

Jensen, Finn V., Steffen L. Lauritzen, and Kristian G. Olesen. 1990.
“Bayesian Updating in Causal Probabilistic Networks by Local
Computations.” *Computational Statistics Quarterly* 4: 269–82.

</div>

<div id="ref-Kschischang2001" class="csl-entry">

Kschischang, Frank R., Brendan J. Frey, and Hans-Andrea Loeliger. 2001.
“Factor Graphs and the Sum-Product Algorithm.” *IEEE Transactions on
Information Theory* 47 (2): 498–519.
<https://doi.org/10.1109/18.910572>.

</div>

<div id="ref-LauritzenSpiegelhalter1988" class="csl-entry">

Lauritzen, Steffen L., and David J. Spiegelhalter. 1988. “Local
Computations with Probabilities on Graphical Structures and Their
Application to Expert Systems.” *Journal of the Royal Statistical
Society, Series B* 50 (2): 157–224.
<https://doi.org/10.1111/j.2517-6161.1988.tb01721.x>.

</div>

<div id="ref-MurphyWeissJordan1999" class="csl-entry">

Murphy, Kevin P., Yair Weiss, and Michael I. Jordan. 1999. “Loopy Belief
Propagation for Approximate Inference: An Empirical Study.” *Proceedings
of the Fifteenth Conference on Uncertainty in Artificial Intelligence
(UAI 1999)*, 467–75.

</div>

<div id="ref-Pearl1988" class="csl-entry">

Pearl, Judea. 1988. *Probabilistic Reasoning in Intelligent Systems:
Networks of Plausible Inference*. Morgan Kaufmann.

</div>

<div id="ref-ShaferShenoy1990" class="csl-entry">

Shafer, Glenn R., and Prakash P. Shenoy. 1990. “Probability
Propagation.” *Annals of Mathematics and Artificial Intelligence* 2:
327–51. <https://doi.org/10.1007/BF01531015>.

</div>

</div>
