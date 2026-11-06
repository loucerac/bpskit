"""The universe: one row per person, the denominator of every rate.

After this step no per-person file carries the patient identifier. Everything
joins on `idx`, the person's row in the universe, which also makes joins plain
array indexing.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from bpskit.delivery import parse_date, read_delivery
from bpskit.persontime import to_days

__all__ = ["build_universe"]


def build_universe(
    path: str | Path,
    spec: dict,
    sentinels: list[str],
    mid_month_day: int,
    n_buckets: int | None = None,
    seed: int = 1,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Read the people file into the universe table.

    Args:
        spec: the `universe` section of a workflow config: file columns, the
            sex labels (`sex_values`) and the date format.
        n_buckets, seed: if given, persons are assigned at random to
            `n_buckets` groups (column `bucket`) for a grouped bootstrap.

    Returns:
        (universe, every delivered id, a sex x birth-decade summary).
        The ids come back BEFORE persons with unknown sex or birth date are
        dropped: a dropped person is still a person, and the transfer check
        must look for their id too.

    Raises:
        ValueError: the people file repeats an id.
    """
    df = read_delivery(path, [spec["id"], spec["sex"], spec["birth"], spec["death"]])
    out = pd.DataFrame(
        {
            "pid": df[spec["id"]],
            "sex": df[spec["sex"]].map(spec["sex_values"]),
            "birth": parse_date(df[spec["birth"]], spec["date_format"], sentinels, mid_month_day),
            "death": parse_date(df[spec["death"]], spec["date_format"], sentinels, mid_month_day),
        }
    )
    del df
    all_ids = out.pid.copy()
    if out.pid.duplicated().any():
        raise ValueError(f"{out.pid.duplicated().sum()} duplicated patient ids in the universe")
    unknown = out.sex.isna() | out.birth.isna()
    if unknown.any():
        print(f"[DROP ] {unknown.sum():,} persons with unknown sex or birth date")
    out = out[~unknown].reset_index(drop=True)

    out["idx"] = np.arange(len(out), dtype=np.int64)
    out["is_f"] = out.sex.eq("F")
    out["birth_d"] = to_days(out.birth)
    out["death_d"] = to_days(out.death)
    if n_buckets:
        G = int(n_buckets)
        out["bucket"] = np.random.default_rng(int(seed)).integers(0, G, len(out)).astype(np.int32)
        per_bucket = len(out) / G
        print(
            f"[BOOT ] {G:,} buckets, {per_bucket:,.1f} persons per bucket"
            + ("  <-- under 20: raise the universe or lower n_buckets" if per_bucket < 20 else "")
        )

    summary = (
        out.assign(birth_decade=(out.birth.dt.year // 10) * 10, dead=out.death.notna())
        .groupby(["sex", "birth_decade"])
        .agg(n=("idx", "size"), n_dead=("dead", "sum"))
        .reset_index()
    )
    print(
        f"[DONE ] universe: {len(out):,} persons "
        f"({out.is_f.sum():,} F / {(~out.is_f).sum():,} M), "
        f"{out.death.notna().mean():.1%} with a death date"
    )
    return out, all_ids, summary
