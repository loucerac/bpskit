"""Data-quality tables: ICD harmonisation by year, and records that cannot be
true for the person they are attached to."""

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from bpskit.persontime import DAYS_PER_YEAR, read_universe

__all__ = ["PLAUSIBILITY", "harmonisation", "plausibility"]

# Records that cannot be true for the person they are attached to. A handful
# is ordinary miscoding; many point at a sex or birth-date problem in the
# people file. Each rule gets (codes, is_female, age in years).
PLAUSIBILITY = {
    "obstetric code (O) in a man": lambda c, f, a: c.str.startswith("O") & ~f,
    "obstetric code (O) in a woman aged 55+": lambda c, f, a: c.str.startswith("O") & f & (a >= 55),
    "perinatal code (P) at age >= 1": lambda c, f, a: c.str.startswith("P") & (a >= 1),
    "breast cancer (C50) in a man": lambda c, f, a: c.str.startswith("C50") & ~f,
    "prostate cancer (C61) in a woman": lambda c, f, a: c.str.startswith("C61") & f,
    "record before birth": lambda c, f, a: a < 0,
}


def harmonisation(qc_paths: Iterable[str | Path]) -> pd.DataFrame:
    """The per-source ICD harmonisation tables (rows by year and coding system), stacked."""
    return pd.concat([pd.read_csv(f) for f in qc_paths])


def plausibility(universe_path: str | Path, events: dict, batch_rows: int) -> pd.DataFrame:
    """Rows and distinct persons failing each PLAUSIBILITY rule, per source.

    Args:
        events: {source name: events parquet}, streamed in `batch_rows` batches.
    """
    u = read_universe(universe_path, ["is_f", "birth_d"])
    is_f, birth = u.is_f.to_numpy(), u.birth_d.to_numpy()
    del u
    out = []
    for src, path in events.items():
        n_rows = {k: 0 for k in PLAUSIBILITY}
        persons = {k: set() for k in PLAUSIBILITY}
        for batch in pq.ParquetFile(path).iter_batches(
            batch_size=int(batch_rows), columns=["idx", "day", "code"]
        ):
            e = batch.to_pandas()
            i = e.idx.to_numpy()
            f = is_f[i]
            a = (e.day.to_numpy(float) - birth[i]) / DAYS_PER_YEAR
            for k, rule in PLAUSIBILITY.items():
                m = np.asarray(rule(e.code, f, a))
                n_rows[k] += int(m.sum())
                persons[k].update(i[m].tolist())
        out += [
            {"source": src, "check": k, "rows": n_rows[k], "persons": len(persons[k])}
            for k in PLAUSIBILITY
        ]
    table = pd.DataFrame(out)
    print(table.to_string(index=False))
    return table
