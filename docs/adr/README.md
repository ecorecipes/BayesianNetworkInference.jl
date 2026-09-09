# Architecture decision records

One file per decision, `NNNN-short-title.md`, with sections Context, Decision, Consequences.
Add a new record rather than editing an old one when a decision changes.

The records describe the ecosystem as a whole, so all eight packages carry the same set. The
canonical copies live in `docs/adr/` at the root of the `bbn` workspace and are copied into each
package by `julia scripts/sync_adrs.jl` (`--check` exits non-zero when a copy is stale). Edit the
workspace copy, never a package copy.
