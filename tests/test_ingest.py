import pandas as pd

from bpskit.ingest import ingest_source
from bpskit.quality import harmonisation, plausibility
from bpskit.synthetic import make_delivery
from bpskit.universe import build_universe

SENTINELS = ["-1", "-2", "-9", ""]
SCALE = {"csv_block_mb": 1, "encoding": "utf-8"}


def _universe(tmp_path, **kw):
    cfg = make_delivery(tmp_path / "d", n_persons=400, seed=2)
    u, ids, summary = build_universe(
        tmp_path / "d" / cfg["universe"]["file"], cfg["universe"], SENTINELS, 15, **kw
    )
    u.to_parquet(tmp_path / "universe.parquet", index=False)
    return cfg, u, ids, summary


def test_universe_keeps_every_delivered_id_and_buckets_are_optional(tmp_path):
    cfg, u, ids, summary = _universe(tmp_path)
    assert len(ids) == 400 and "bucket" not in u
    assert list(u.idx) == list(range(len(u)))
    assert summary.n.sum() == len(u)
    _, u2, _, _ = _universe(tmp_path, n_buckets=10, seed=3)
    assert u2.bucket.between(0, 9).all()


def test_ingest_writes_harmonised_events_and_orphan_ids(tmp_path):
    cfg, u, _, _ = _universe(tmp_path)
    spec = cfg["sources"]["cmbd"] | {"primary_position": 1, "episode_flags": {"obs": ["O"]}}
    out = {k: tmp_path / f"{k}.out" for k in ("events", "episodes", "qc", "ids")}
    ingest_source(
        tmp_path / "d" / spec["file"],
        tmp_path / "universe.parquet",
        spec,
        name="cmbd",
        sentinels=SENTINELS,
        mid_month_day=15,
        map_icd9=False,
        washout_from="2015-01-01",
        last_date="2025-12-31",
        scale=SCALE,
        events_out=out["events"],
        episodes_out=out["episodes"],
        qc_out=out["qc"],
        ids_out=out["ids"],
    )
    ev = pd.read_parquet(out["events"])
    assert {"idx", "day", "code", "is_primary"} <= set(ev.columns)
    assert ev.idx.max() < len(u) and ev.code.str.match(r"^[A-Z]\d").all()
    assert pd.read_parquet(out["ids"]).pid.empty  # every synthetic id is in the universe
    qc = harmonisation([out["qc"]])
    assert qc.rows.sum() >= len(ev)
    table = plausibility(tmp_path / "universe.parquet", {"cmbd": out["events"]}, 10_000)
    assert set(table.columns) == {"source", "check", "rows", "persons"}
