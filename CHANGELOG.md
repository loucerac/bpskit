# Changelog

All notable changes to bpskit. Versions follow
[Semantic Versioning](https://semver.org/): until 1.0, a minor release may
change the public API (the names in each module's `__all__`). From 0.4.0 on,
release-please writes each entry from the conventional commits it releases;
edit the entry in the release PR before merging when it needs more context.

## [0.5.0](https://github.com/loucerac/bpskit/compare/v0.4.0...v0.5.0) (2026-11-08)


### Features

* add Markov clustering, modularity and latent class analysis, and plant E11.2 in the synthetic delivery ([2ae456b](https://github.com/loucerac/bpskit/commit/2ae456bafd099071dd50d65d7cf3aec20c79943b))

## [0.4.0]

### Added
- `bpskit.universe.build_universe`: the people file into the universe table,
  with every delivered id (for the transfer check) and optional bootstrap buckets.
- `bpskit.ingest.ingest_source`: a diagnosis file into dated, ICD-10-harmonised
  events, episode flags, a harmonisation table and the ids not in the universe.
- `bpskit.quality`: the harmonisation table and implausible-record counts.

These are the ingest steps every BPS workflow starts with; the shared
Snakemake module bps-workflow runs them.

## [0.3.0]

### Added
- `bpskit.strata`: `Strata` (person-time and expected counts by sex, age band,
  calendar year and prior-year contact intensity) and `person_pairs`.
- `bpskit.stats.exact_contrast` and `idtn_from_ratio`: rate ratio, IDTN and rate
  difference of two groups with exact (Clopper-Pearson) intervals.
- `bpskit.icd.read_conditions`: read and validate a condition table.

These moved from the BPS sex-differences workflow, so a workflow needs no
helper modules of its own.

## [0.2.0]

### Added
- `bpskit.synthetic`: a synthetic BPS delivery with planted structure
  (`make_delivery`, `config`, `FILES`), and `bpskit synthetic` on the command line.
- `examples/quickstart.py`, a worked example run in CI.
- A declared public API (`__all__` in every module), type hints and `py.typed`.
- Packaging metadata, a release workflow publishing to PyPI by trusted
  publishing, `CITATION.cff`.

### Changed
- The version is read from `bpskit/__init__.py` (one place to bump).

## [0.1.0]

### Added
- `delivery`, `icd`, `persontime`, `stats` and `transfer`, extracted from the
  BPS sex-differences workflow, and the `bpskit pullable` / `bpskit check-logs`
  command line.

[0.4.0]: https://github.com/loucerac/bpskit/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/loucerac/bpskit/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/loucerac/bpskit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/loucerac/bpskit/releases/tag/v0.1.0
