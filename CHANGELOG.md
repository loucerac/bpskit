# Changelog

All notable changes to bpskit. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versions
[Semantic Versioning](https://semver.org/): until 1.0, a minor release may
change the public API (the names in each module's `__all__`).

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

[0.3.0]: https://github.com/loucerac/bpskit/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/loucerac/bpskit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/loucerac/bpskit/releases/tag/v0.1.0
