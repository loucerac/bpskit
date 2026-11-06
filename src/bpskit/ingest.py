"""Ingest one diagnosis file: dated, ICD-10-harmonised records for the universe."""

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from bpskit.delivery import normalise_code, parse_date, stream_csv
from bpskit.icd import Icd9Mapper, harmonise, prefix_mask
from bpskit.persontime import day_of, read_universe, to_days
from bpskit.transfer import write_ids

__all__ = ["QC_COLUMNS", "ingest_source"]

QC_COLUMNS = ["year", "rows", "icd10", "icd9_mapped", "icd9_unmapped", "not_icd10_shaped"]


def ingest_source(
    path,
    universe_path,
    spec: dict,
    *,
    name: str,
    sentinels,
    mid_month_day: int,
    map_icd9: bool,
    washout_from,
    last_date,
    scale: dict,
    events_out,
    episodes_out,
    qc_out,
    ids_out,
) -> None:
    """Stream one source file into an events parquet; write its side tables.

    Kept rows are those of persons in the universe, dated inside
    [washout_from, last_date], with an ICD-10-shaped code after mapping.

    Args:
        spec: the source's entry under `sources:` in a workflow config.
            `position` and `primary_position` (optional) mark the principal
            diagnosis. `episode_flags` (optional, {flag: [code prefixes]})
            names admission types to record, e.g. obstetric.
        events_out: idx, day, code (+ is_primary), in row groups of one
            streamed block each.
        episodes_out: (idx, day, flag) for every record matching a flag. Kept
            separate because one admission's rows can fall in different blocks,
            so whole episodes are dropped later by key.
        qc_out: rows per calendar year by coding system, for the quality table.
        ids_out: ids in this file that are not in the universe. Their rows are
            dropped, but the ids were read, so the transfer check needs them.
    """
    has_pos = bool(spec.get("position"))
    flags = {
        f: [p.replace(".", "") for p in prefixes]
        for f, prefixes in (spec.get("episode_flags") or {}).items()
    }

    u = read_universe(universe_path, ["pid", "idx"])
    pid_index = pd.Index(u.pid)
    del u

    washout, last = day_of(washout_from), day_of(last_date)
    changeover = day_of(spec["icd9_changeover"]) if spec.get("icd9_changeover") else None
    mapper = Icd9Mapper(map_icd9)

    schema = pa.schema(
        [("idx", pa.int64()), ("day", pa.int32()), ("code", pa.string())]
        + ([("is_primary", pa.bool_())] if has_pos else [])
    )
    writer = pq.ParquetWriter(events_out, schema)
    qc_parts, episode_parts = [], []
    n_in = n_univ = n_kept = 0
    orphans: set[str] = set()

    cols = [spec["id"], spec["date"], spec["code"]] + ([spec["position"]] if has_pos else [])
    for raw in stream_csv(path, cols, scale["csv_block_mb"], scale["encoding"]):
        n_in += len(raw)
        idx = pid_index.get_indexer(raw[spec["id"]])
        orphans.update(raw[spec["id"]][idx < 0].unique())
        raw = raw[idx >= 0]
        idx = idx[idx >= 0]
        n_univ += len(raw)
        if raw.empty:
            continue
        date = parse_date(raw[spec["date"]], spec["date_format"], sentinels, mid_month_day)
        df = pd.DataFrame(
            {"idx": idx, "day": to_days(date), "code": normalise_code(raw[spec["code"]]).to_numpy()}
        )
        if has_pos:
            primary = spec.get("primary_position", 1)
            df["is_primary"] = (
                pd.to_numeric(raw[spec["position"]], errors="coerce").eq(primary).to_numpy()
            )

        system = harmonise(df, changeover, mapper)
        shape_ok = df.code.str.match(r"^[A-Z]\d")

        year = pd.Series(pd.to_datetime(df.day, unit="D", errors="coerce").dt.year, index=df.index)
        qc_parts.append(
            pd.DataFrame(
                {
                    "year": year,
                    "icd10": system.eq("icd10"),
                    "icd9_mapped": system.eq("icd9_mapped"),
                    "icd9_unmapped": system.eq("icd9_unmapped"),
                    "not_icd10_shaped": ~shape_ok & system.ne("icd9_unmapped"),
                }
            )
            .groupby("year", dropna=False)
            .agg(
                rows=("icd10", "size"),
                icd10=("icd10", "sum"),
                icd9_mapped=("icd9_mapped", "sum"),
                icd9_unmapped=("icd9_unmapped", "sum"),
                not_icd10_shaped=("not_icd10_shaped", "sum"),
            )
        )

        keep = df.day.notna() & system.ne("icd9_unmapped") & shape_ok
        keep &= (df.day >= washout) & (df.day <= last)
        df = df[keep]
        for flag, prefixes in flags.items():
            hit = prefix_mask(df.code, prefixes)
            episode_parts.append(df.loc[hit, ["idx", "day"]].drop_duplicates().assign(flag=flag))
        df = df.astype({"day": "int32"})
        writer.write_table(pa.Table.from_pandas(df, schema=schema, preserve_index=False))
        n_kept += len(df)
    writer.close()
    write_ids(sorted(orphans), ids_out)

    qc = (
        pd.concat(qc_parts).groupby(level=0, dropna=False).sum().reset_index()
        if qc_parts
        else pd.DataFrame(columns=QC_COLUMNS)
    )
    qc.insert(0, "source", name)
    qc.to_csv(qc_out, index=False)

    if episode_parts:
        episodes = pd.concat(episode_parts).drop_duplicates()
    else:
        episodes = pd.DataFrame(
            {
                "idx": pd.Series(dtype="int64"),
                "day": pd.Series(dtype="float64"),
                "flag": pd.Series(dtype="string"),
            }
        )
    episodes.astype({"idx": "int64", "day": "int32"}).to_parquet(episodes_out, index=False)
    for flag in flags:
        print(f"[EPIS ] {int((episodes.flag == flag).sum()):,} admissions flagged {flag}")
    print(
        f"[DONE ] {name}: {n_in:,} rows read, {n_univ:,} in the universe "
        f"({len(orphans):,} ids not in it), {n_kept:,} kept; "
        f"ICD-9 mapped {int(qc.icd9_mapped.sum()):,}, unmapped {int(qc.icd9_unmapped.sum()):,}"
    )
