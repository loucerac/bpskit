"""Person-time: dates as numbers, age bands, Lexis splitting, streaming persons.

Every per-person array holds dates as float days since 1970-01-01, so
person-time is a subtraction and NaN means "no date" (alive, never recorded).
Ages are in years of DAYS_PER_YEAR days.
"""

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25
EPOCH = np.datetime64("1970-01-01", "D")


def to_days(s: pd.Series) -> np.ndarray:
    """Datetimes to float days since 1970-01-01; NaT becomes NaN."""
    v = s.to_numpy("datetime64[D]")
    out = ((v - EPOCH) / np.timedelta64(1, "D")).astype(float)
    out[np.isnat(v)] = np.nan
    return out


def day_of(date) -> float:
    """One date (string or Timestamp) as days since 1970-01-01."""
    return to_days(pd.Series([pd.Timestamp(date)]))[0]


def follow_up(birth_d, death_d, start_d, end_d):
    """Entry day, exit day and whether any follow-up falls inside [start, end].

    A person enters at the later of the period start and birth, and leaves at
    the earlier of death and the period end; NaN death means alive.
    """
    entry = np.maximum(start_d, birth_d)
    exit_ = np.minimum(end_d, np.where(np.isnan(death_d), np.inf, death_d))
    return entry, exit_, exit_ > entry


def age_bands(width: int, open_from: int) -> pd.DataFrame:
    """Age bands of `width` years from 0, with a last open band from `open_from`.

    Columns: age_band (label), age_lo, age_hi (200 for the open band), age_mid.
    """
    lo = np.arange(0, open_from + 1, width)
    hi = np.append(lo[1:], 200.0).astype(float)
    label = [f"{a}-{a + width - 1}" for a in lo[:-1]] + [f"{lo[-1]}+"]
    mid = np.append(lo[:-1] + width / 2, lo[-1] + width / 2)
    return pd.DataFrame({"age_band": label, "age_lo": lo, "age_hi": hi, "age_mid": mid})


def lexis(age_start: np.ndarray, age_end: np.ndarray, bands: pd.DataFrame) -> np.ndarray:
    """Person-years each interval [age_start, age_end) spends in each band (N x B).

    An empty or reversed interval contributes zero everywhere.
    """
    lo = bands["age_lo"].to_numpy(float)[None, :]
    hi = bands["age_hi"].to_numpy(float)[None, :]
    a = np.asarray(age_start, float)[:, None]
    b = np.asarray(age_end, float)[:, None]
    return np.clip(np.minimum(b, hi) - np.maximum(a, lo), 0.0, None)


def band_index(age: np.ndarray, bands: pd.DataFrame) -> np.ndarray:
    """Band of each age; ages outside the bands are clipped to the first or last."""
    lo = bands["age_lo"].to_numpy(float)
    return np.clip(np.searchsorted(lo, np.asarray(age, float), side="right") - 1, 0, len(lo) - 1)


def accumulate(acc: np.ndarray, key: np.ndarray, values: np.ndarray) -> None:
    """acc[key[i], b] += values[i, b] for every row i and column b, in place."""
    for b in range(values.shape[1]):
        acc[:, b] += np.bincount(key, weights=values[:, b], minlength=acc.shape[0])


def person_chunks(path, batch_rows: int, columns=None):
    """Stream a parquet sorted by `idx` as DataFrames that never split a person.

    Each yielded frame holds whole persons; the last person of a batch is held
    back and joined to the next one. The file MUST be sorted by idx, or a
    person can come out in two chunks.
    """
    import pyarrow.parquet as pq

    left = None
    for batch in pq.ParquetFile(path).iter_batches(batch_size=int(batch_rows), columns=columns):
        df = batch.to_pandas()
        if left is not None:
            df = pd.concat([left, df], ignore_index=True)
        cut = int(np.searchsorted(df.idx.to_numpy(), df.idx.iat[-1], side="left"))
        if cut:
            yield df.iloc[:cut].reset_index(drop=True)
        left = df.iloc[cut:]
    if left is not None and len(left):
        yield left.reset_index(drop=True)


def read_universe(path, columns: list[str]) -> pd.DataFrame:
    """Read universe columns, refusing a file written by an older workflow version.

    Snakemake does not rebuild an intermediate whose rule did not change, so a
    universe.parquet from an older layout can outlive the code that wrote it.
    """
    import pyarrow.parquet as pq

    have = set(pq.read_schema(path).names)
    missing = [c for c in columns if c not in have]
    if missing:
        raise RuntimeError(
            f"{path} has no column(s) {missing}: it was written by an older version of "
            "this workflow. Rebuild everything downstream of it: snakemake -c 20 --forceall"
        )
    return pd.read_parquet(path, columns=columns)
