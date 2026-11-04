"""Reading a BPS delivery: pipe-delimited CSVs, every column read as text.

Columns stay text and nothing becomes null on its own, because the delivery
uses sentinel codes ("-1", "-2", "-9", "") for "unknown". Callers decide what
a sentinel means for each column: a missing date, or an unknown count.
"""

from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import yaml

__all__ = [
    "read_delivery",
    "read_header",
    "stream_csv",
    "parse_date",
    "normalise_code",
    "load_config",
    "delivery_root",
]

PROFILE = "profiles/default/config.yaml"


def read_delivery(path: str | Path, usecols: list[str]) -> pd.DataFrame:
    """Read a small delivery file whole, quotes and padding stripped.

    For the people file and similar. Diagnosis files can be hundreds of
    millions of rows: use `stream_csv` for those.

    Raises:
        KeyError: a needed column is not in the file. The message lists the
            header that is there, which is usually enough to fix the config.
    """
    path = Path(path)
    for enc in ("utf-8", "latin-1"):
        try:
            header = pd.read_csv(path, sep="|", nrows=0, encoding=enc).columns
            break
        except UnicodeDecodeError:
            continue
    missing = [c for c in usecols if c not in header]
    if missing:
        raise KeyError(f"{path.name}: column(s) {missing} not in file; have {list(header)}")
    df = pd.read_csv(path, sep="|", usecols=usecols, dtype=str, encoding=enc, keep_default_na=False)
    for c in usecols:
        df[c] = df[c].str.strip().str.strip('"').str.strip()
    print(f"[READ ] {path.name}: {len(df):,} rows")
    return df


def parse_date(s: pd.Series, fmt: str, sentinels, mid_month_day: int) -> pd.Series:
    """Parse delivery date strings, treating sentinel codes as missing.

    Month-precision formats (no `%d`) land on `mid_month_day`, so the average
    misplacement is zero rather than half a month early. Unparseable values
    become NaT, but a column in which EVERY non-missing value fails raises:
    that is a wrong `date_format` in config, not bad data.
    """
    s = s.where(~s.isin(sentinels))
    # An unquoted numeric column can arrive as "20150101.0".
    s = s.str.replace(r"\.0$", "", regex=True)
    out = pd.to_datetime(s, format=fmt, errors="coerce")
    if "%d" not in fmt:
        # pd.to_timedelta, not pd.Timedelta: NumPy deprecates the latter's
        # unitless conversion when the column holds NaT.
        out = out + pd.to_timedelta(mid_month_day - 1, unit="D")
    if len(out) and out.isna().all() and s.notna().any():
        raise ValueError(f"every value failed to parse as {fmt!r}, e.g. {s.dropna().iloc[0]!r}")
    return out


def read_header(path: str | Path, encoding: str) -> list[str]:
    """Column names of a delivery file, without reading any data."""
    with open(path, encoding=encoding) as fh:
        return [h.strip().strip('"') for h in fh.readline().rstrip("\r\n").split("|")]


def stream_csv(
    path: str | Path, usecols: list[str], block_mb: float, encoding: str
) -> Iterator[pd.DataFrame]:
    """Yield a delivery file as pandas blocks of text columns.

    pyarrow parses `block_mb` of the file at a time, so memory stays flat
    whatever the file size. Block boundaries carry no meaning: a person's
    records can be split across blocks, and callers must not assume otherwise.

    Raises:
        KeyError: a needed column is not in the header.
    """
    import pyarrow as pa
    import pyarrow.csv as pacsv

    path = Path(path)
    header = read_header(path, encoding)
    missing = [c for c in usecols if c not in header]
    if missing:
        raise KeyError(f"{path.name}: column(s) {missing} not in file; have {header}")
    reader = pacsv.open_csv(
        path,
        read_options=pacsv.ReadOptions(block_size=int(float(block_mb) * 2**20), encoding=encoding),
        parse_options=pacsv.ParseOptions(delimiter="|"),
        convert_options=pacsv.ConvertOptions(
            include_columns=usecols,
            column_types={c: pa.string() for c in usecols},
            strings_can_be_null=False,
        ),
    )
    n = 0
    for batch in reader:
        df = batch.to_pandas()
        for c in usecols:
            df[c] = df[c].str.strip().str.strip('"').str.strip()
        n += len(df)
        yield df
    print(f"[READ ] {path.name}: {n:,} rows streamed")


def normalise_code(s: pd.Series) -> pd.Series:
    """ICD codes as upper case with dots, spaces and quotes removed ('e11.9' -> 'E119')."""
    return s.astype(str).str.upper().str.replace(r"[.\s\"]", "", regex=True)


# --------------------------------------------------------------------------- #
# Config, for the command-line tools (rules get theirs from Snakemake)
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict:
    """A workflow's config file (YAML)."""
    return yaml.safe_load(Path(path).read_text())


def delivery_root(root: Path, cfg: dict) -> Path:
    """Where the delivery is, resolved the way Snakemake resolves it.

    A workflow profile at <root>/profiles/default/config.yaml overrides the
    config's own `delivery_root`. A `--config delivery_root=...` given to one
    run is invisible here, so a tool run after such a run must be given the
    same profile.
    """
    root = Path(root)
    prof = root / "profiles/default/config.yaml"
    entries = (yaml.safe_load(prof.read_text()) or {}).get("config") if prof.exists() else None
    entries = entries or []
    # A profile may write `config:` as a list of "key=value" or as a mapping.
    if isinstance(entries, dict):
        entries = [f"{k}={v}" for k, v in entries.items()]
    for e in entries:
        k, _, v = str(e).partition("=")
        if k.strip() == "delivery_root":
            return (root / v.strip()).resolve()
    return (root / cfg["delivery_root"]).resolve()
