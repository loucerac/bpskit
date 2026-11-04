# Changelog

All notable changes to bpskit. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the versions
[Semantic Versioning](https://semver.org/): until 1.0, a minor release may
change the public API (the names in each module's `__all__`).

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

[0.2.0]: https://github.com/loucerac/bpskit/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/loucerac/bpskit/releases/tag/v0.1.0
