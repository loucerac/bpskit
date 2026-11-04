# bpskit

Shared tools for studies on a BPS (Base Poblacional de Salud, Servicio Andaluz
de Salud) delivery, used by the BPS projects' Snakemake workflows.

| module | what it does |
|---|---|
| `bpskit.delivery` | read the pipe-delimited delivery: text columns, sentinel codes, date formats, streaming in blocks |
| `bpskit.icd` | ICD-10 code specifications (`A15-A19, H35.3`), chapters, ICD-9 → ICD-10 mapping with the CMS GEM |
| `bpskit.persontime` | dates as days since 1970, age bands, Lexis splitting, streaming a person-sorted parquet |
| `bpskit.stats` | exact Poisson and binomial tests, BH and Bonferroni |
| `bpskit.transfer` | the check that no file leaving the data machine names a patient |

## Installing it in a project

Pin a release tag, in both the pixi environment and the rule environment:

```toml
# pixi.toml
[pypi-dependencies]
bpskit = { git = "https://github.com/loucerac/bpskit.git", tag = "v0.1.0" }
```

```yaml
# workflow/envs/<env>.yaml
dependencies:
  - pip
  - pip:
      - git+https://github.com/loucerac/bpskit.git@v0.1.0
```

**A new release reruns what depends on it.** Bumping the tag changes the rule
environment file, and Snakemake's `software-env` rerun trigger reruns the rules
that use that environment. That is why the package is pinned rather than
imported from a sibling folder: an imported module's edits are invisible to
Snakemake.

## The transfer rule

No row carrying a patient identifier leaves the machine that holds the
delivery. `bpskit.transfer` checks the bytes of each file against every
identifier the workflow read:

- `results/work/ids/universe.parquet`, written before anyone is dropped;
- `results/work/ids/<source>.parquet`, ids in a source file but not in the universe.

Where an id file is missing, it reads the delivery file instead. It fails
closed: an unreadable file, an unknown suffix or no identifiers all refuse.
From a workflow directory on the data machine:

```bash
bpskit pullable --strict results/out   # before copying results/out/ off the machine
bpskit check-logs .                    # writes results/work/logs_pullable.txt
```

`--strict` allows only csv/txt/png and refuses per-person columns: internal
keys, the delivery's id and date columns, identifier-like names. Both commands
read the workflow's `config/config.yaml` (`--config` to change it): its
`universe` and `sources` sections name the id columns.

## Developing

```bash
pixi run test     # pytest
pixi run lint     # ruff check + format check
pixi run fmt
```

Release: bump `version` in `pyproject.toml` and `__version__`, tag `vX.Y.Z`,
push the tag, then update the pin in each project.
