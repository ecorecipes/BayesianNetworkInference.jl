# ADR 0008: Model-zoo licence policy

Date: 2026-09-07. Status: accepted.

## Context

`EcologicalBayesianNetworks.jl` is a zoo of ecological Bayesian networks and influence diagrams collected from public
repositories: BNMA, bnlearn, Plexus Ecological, GitHub, Zenodo, Dryad, figshare and USGS ScienceBase. The models arrive
under a wide range of terms. BNMA labels each record `CC-BY`, `CC-BY-SA`, `CC-BY-ND`, `CC-BY-NC` or `CC-BY-NC-ND` with
no licence version; bnlearn states CC BY-SA 3.0; the Ballycanew tools are MIT; the Plexus files state nothing at all;
the USGS records are CC0 US Government works; and several catalogue entries are data or code archives rather than
network files.

A model zoo that ignores this ends up either useless (nothing committed, every example needing a download) or
infringing (files redistributed against their terms). The two clauses that matter are ND, which forbids distributing a
modified version, and NC, which forbids commercial use and therefore cannot be attached to an MIT-licensed package
whose users may be commercial.

Until now the policy existed only as prose in `models/README.md` and the project plan. `models/README.md` cited
ADR 0003 (Formats as a Catlab-free leaf) as its authority; that record contains no licence text. SPEC section 60
rule 23 requires a deviation or a policy decision of this weight to be recorded as an ADR. This record is that.

There is a second, subtler problem the policy has to solve. Whether a file may be redistributed cannot be decided by
searching the licence string for a permissive substring: `"CC BY-ND"` contains `"CC BY"`, and `"CC BY-NC-ND"` contains
it too. The test that was supposed to enforce the policy did exactly that (`occursin(r"CC BY|MIT|CC0"i, spec.licence)`)
and would have passed a no-derivatives model committed verbatim.

## Decision

**1. Verbatim redistribution only for CC BY, CC BY-SA, CC0 and MIT.** A model file is committed to the repository only
when its source states one of those licences. The file is stored byte-for-byte as downloaded (gzipped files stay
gzipped); only the file name may change, spaces becoming underscores, and the manifest `notes` say so. Every committed
file has a `LICENSE.txt` beside it carrying the attribution, the licence name and URL, and, for share-alike licences,
the statement that derivative works must carry the same licence.

**2. ND, NC and unstated licences are metadata plus on-demand fetch.** For those the repository holds a
`models/<name>/metadata.toml` and nothing else: no model file, no converted copy, no excerpt. The manifest records the
source URL, a direct `download_url` where the host offers one, `fetch_instructions` for the manual route, and the
SHA-256 of a test download. `fetch_model(name)` downloads the file into a cache outside the repository
(`cache_dir()`: a Scratch.jl space, or `ECOLOGICAL_BN_CACHE`, or a `with_cache_dir` block) and `import_model(name, path)`
installs one obtained by hand. The cache is per user and never committed. This is enforced structurally as well as by
policy: the test suite asserts that a fetch-only model's directory contains `metadata.toml` and nothing else.

**3. A converted derivative of an ND work is never committed.** More strongly, no converted derivative of *any* model
is committed, whatever its licence: the zoo stores what the authors published, and format conversion, renormalisation
and repair happen in memory when a model is loaded. This keeps rule 4 checkable and keeps the zoo honest about
provenance. Where a model needs a repair to be usable at all - the Song Sparrow finding nodes, for instance - the
repair lives in a vignette with its assumptions written out, never in the committed file.

**4. Attribution and checksums.** Every manifest carries `citation`, `doi` where one exists, `source_url`,
`retrieved` and `sha256`. `verify_checksums()` recomputes the digest of every committed file on every test run, so a
committed file that stops matching the download its checksum was taken from fails the suite. With
`include_cache = true` it checks the fetched files too, and `collect_mismatches = true` reports every bad file rather
than stopping at the first.

**5. A licence is recorded exactly as the source states it.** BNMA gives no licence version, so its manifests read
`licence = "CC-BY (version unstated)"`, not `"CC BY 4.0"`. The interpretation - which version of the licence text the
file is treated under - belongs in that model's `LICENSE.txt`, where it can be read as the judgement it is. The same
rule makes the USGS entries say `"CC0 1.0 (ScienceBase record); US Government work in the public domain"` rather than
the earlier `"record states no rights"`, which read as though no rights had been granted when in fact the record
grants all of them.

**6. The check is an allow-list plus a clause rejection, never a substring match.** `licence_permits_redistribution`
in `test/registry.jl` holds the exact set of permitted licence strings and, as a backstop against a bad allow-list
entry, rejects anything matching `\bN[CD]\b`. A negative test constructs a manifest that declares
`redistribution = "verbatim"` with `licence = "CC BY-ND"` and asserts that it fails the policy check while passing the
old substring test.

**7. Catalogue records that are not networks.** Dryad, Zenodo, figshare and source-repository packages, from which a
network can be reconstructed but which are not themselves network files, are recorded with `format = "package"` and
`redistribution = "reconstruction"`. They carry source URL, licence, citation and a note on what the archive holds;
nothing is committed and nothing is cached, and `model_path`, `model_ir` and `load_model` raise
`ReconstructionOnlyError` naming the record. A permissive licence on such an archive (the wild-pig materials are CC0)
does not make it committable, because there is no network file to commit; extracting one is the user's work, and the
result becomes a model of its own with its own licence check.

**8. A permissive licence permits committing a file; it does not oblige it.** The two USGS `.neta` records are CC0 and
could be committed, but they are binary Netica blobs that `BayesianNetworkFormats.jl` cannot read, so committing them
would add unreadable bytes with nothing to check them against beyond a checksum. They stay fetch-only, and their
manifest `notes` say that this is a choice about readability rather than a licence restriction. Any such choice must be
stated in `notes`; silence would leave a reader to infer a restriction that does not exist.

## Consequences

- Roughly a third of the catalogue cannot be exercised without a network call, so the download path needs its own
  tests. Those live in `test/fetch.jl` behind `ECOLOGICAL_BN_FETCH=true`, with `scripts/run_network_tests.jl` to run
  them and the slow builds together.
- A user who wants an ND or NC model must accept a download step and, for the records whose host has no direct
  endpoint, a manual export. The typed `ModelNotFetchedError` and `NoDownloadURLError` carry the instructions.
- The zoo's node and arc counts for fetch-only models come from a test download rather than from a committed file, so
  they are only as current as the `retrieved` date. A changed upstream file shows up as a `ChecksumMismatchError` on
  the next network run rather than silently.
- Recording a licence as the source states it makes the catalogue table less tidy - `"CC-BY (version unstated)"` is
  longer than `"CC BY 4.0"` - and that is the point: the untidiness is real and belongs in front of the reader.
- Committing only originals means the zoo cannot ship a fixed version of a model whose file has a defect. Two BNMA
  records (Polar Bear Stressor Phase I and II) did not parse when this record was written, because the Netica reader
  rejected empty entries in a parenthesised list; the manifests recorded that in `known_parse_issue` and the fix
  belonged in the reader. **Update (2026-09-25): the reader was fixed** -- `empty_list_entries.dne` is a dedicated
  fixture -- and `known_parse_issue` is empty for both records. The consequence stands as stated in general; this
  particular instance is resolved, and the resolution is the one the record predicted.
- The policy is enforced in three places that must stay in step: this record, `models/README.md`, and
  `licence_permits_redistribution` with its allow-list in `test/registry.jl`. Adding a permitted licence string means
  editing the allow-list, which makes the change visible in review.
