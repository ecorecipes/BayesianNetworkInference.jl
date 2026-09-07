# ADR 0003: BayesianNetworkFormats.jl is a Catlab-free leaf

Date: 2026-09-06. Status: accepted.

## Context

Format parsers need only string/XML handling. Making them depend on `BayesianNetworks.jl` (and thus Catlab) would make
parser tests slow, force sibling checkouts for a leaf, and stop other Julia BN users from reusing the readers.

## Decision

`BayesianNetworkFormats.jl` defines a plain `NetworkIR` (variables with ordered states and parents, node kinds
chance/decision/utility, tables in the ADR 0002 convention, positions, extras) and readers/writers targeting it.
`BayesianNetworks.jl` and `InfluenceDiagrams.jl` depend on Formats and own the bridges
`BayesModel(::NetworkIR)` / `NetworkIR(::BayesModel)` and `InfluenceDiagramModel(::NetworkIR)`.
Formats exposes `fixture_path(name)` so downstream tests reuse its committed fixtures.

## Consequences

Formats has zero ecosystem dependencies and depends externally only on EzXML (and JSON3 for golden files).
The deepest CI chain shrinks by one level. `.neta` (binary Netica) is out of scope; GeNIe `<equation>`/`<noisymax>`
nodes are rejected in strict mode and skipped otherwise.
