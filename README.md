# bpskit

Tools for studies on a BPS (Base Poblacional de Salud, Servicio Andaluz de
Salud) health-records delivery, and for working with pseudonymised
health-record extracts in a secure data environment generally.

- **Read the delivery.** Pipe-delimited files read as text, sentinel codes for
  "unknown", month- and day-precision dates, streaming in fixed-size blocks so a
  400-million-row file never sits in memory.
- **ICD codes.** Code specifications like `A15-A19, H35.3`, ICD-10 chapters,
  and ICD-9 → ICD-10 mapping with the CMS GEM. That includes the V/E
  supplementary codes, which are also valid ICD-10 and can only be placed by
  the record's date relative to a changeover.
- **Person-time.** Dates as days since 1970, age bands, Lexis splitting,
  person-sorted streaming.
- **Exact tests.** Poisson rate ratios (Garwood intervals), binomial tests,
  two-group rate contrasts with exact intervals, BH and Bonferroni.
- **Indirect standardisation.** Person-time and expected counts by sex, age,
  calendar year and prior-year contact intensity; within-person record pairs.
- **An output check for leaving the secure environment.** `bpskit.transfer`
  refuses any file that contains an identifier the workflow read, or a
  per-person column, and fails closed when it cannot read something.
- **A synthetic delivery** with BPS column names and planted structure, for
  tests and examples.

## Installing

```bash
pip install bpskit
```

In a pixi project (`pixi.toml`):

```toml
[pypi-dependencies]
bpskit = "==0.3.0"
```

In a Snakemake rule environment (`workflow/envs/*.yaml`):

```yaml
dependencies:
  - pip
  - pip:
      - bpskit==0.3.0
```

Pin a release. Bumping the tag changes the environment file, and Snakemake's
`software-env` rerun trigger reruns the rules that use it. A module imported
from a sibling folder gets no such tracking.

## A worked example

[`examples/quickstart.py`](examples/quickstart.py) runs in a few seconds with no
real data:

1. it writes a 5,000-person synthetic delivery;
2. it measures the incidence of type 2 diabetes (E11) in primary care in
   2018-2025;
3. it compares women with the whole population;
4. it runs the transfer check on the result.

The core of it:

```python
from bpskit import transfer
from bpskit.delivery import normalise_code, parse_date, read_delivery, stream_csv
from bpskit.icd import expand_spec, prefix_mask
from bpskit.persontime import day_of, follow_up, to_days
from bpskit.stats import poisson_rr
from bpskit.synthetic import make_delivery

cfg = make_delivery("work/delivery", n_persons=5_000)  # files + how to read them

# people: birth and death as days since 1970 (NaN = alive)
u = cfg["universe"]
people = read_delivery(f"work/delivery/{u['file']}", [u["id"], u["sex"], u["birth"], u["death"]])
birth = to_days(parse_date(people[u["birth"]], u["date_format"], ["", "-1"], mid_month_day=15))

# first E11 record per person, streamed block by block
ap = cfg["sources"]["ap"]
for block in stream_csv(
    f"work/delivery/{ap['file']}", [ap["id"], ap["date"], ap["code"]], 16, "utf-8"
):
    hits = block[prefix_mask(normalise_code(block[ap["code"]]), expand_spec("E11"))]
    ...

# follow-up, then observed vs expected with an exact interval
entry, exit_, in_period = follow_up(birth, death, day_of("2018-01-01"), day_of("2025-12-31"))
ratio, lo, hi, p = poisson_rr(observed, expected, alpha=0.05)

# nothing leaves the machine that names a patient
n, problems = transfer.check(
    ["work/results/out"],
    ids=set(people[u["id"]]),
    strict=True,
    forbidden=transfer.forbidden_columns(cfg),
)
```

```
E11 incidence 1,258 per 100,000 person-years; women vs all: 1.12 (0.98-1.27)
aggregate table: 1 file checked, 0 problem(s)
with a leaked file:
  .../first_cases.csv: contains 3 patient identifier(s)
  .../first_cases.csv: per-person or identifier column(s) ['patient']
```

## The transfer check

The rule it enforces: no row carrying a patient identifier leaves the machine
that holds the delivery. It checks the bytes of every file against the
identifiers the workflow actually read, not against a guessed id format. A
workflow writes those identifiers with `transfer.write_ids`:

- `results/work/ids/universe.parquet`: every id in the people file, written
  before anyone is dropped;
- `results/work/ids/<source>.parquet`: ids in a diagnosis file but not in the
  people file.

Where an id file is missing (a run that died early), the check reads the
delivery file instead. An unreadable file, an unknown suffix or an empty id set
all refuse. With `strict=True` (for a results folder copied off wholesale),
only csv/txt/png pass. CSVs whose header holds a per-person key, one of the
delivery's id or date columns, or an identifier-like name are refused too.

From a workflow directory on the data machine:

```bash
bpskit pullable --strict results/out   # exit 0 only if results/out/ may leave
bpskit check-logs .                    # list the logs that may leave
bpskit synthetic data/synthetic --persons 2000
```

`pullable` and `check-logs` read the workflow's `config/config.yaml`
(`--config` to change it). Its `universe` and `sources` sections, in the shape
`bpskit.synthetic.config()` returns, name the id columns.

## Developing

```bash
pixi run test       # pytest
pixi run lint       # ruff check + format check
pixi run example    # the worked example
pixi run build      # sdist + wheel, twine check
```

Releasing: bump `__version__` in `src/bpskit/__init__.py`, add the release to
`CHANGELOG.md`, commit, then tag `vX.Y.Z` and push the tag.
`.github/workflows/release.yml` checks that the tag matches the version,
builds, and publishes to PyPI by trusted publishing.

## Citing

See [`CITATION.cff`](CITATION.cff). MIT licensed.
