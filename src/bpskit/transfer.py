"""Does a file name a patient? The check behind every transfer off the data machine.

The rule: no row carrying a patient identifier leaves the machine that holds
the delivery. It is tested on the bytes, against the identifiers the workflow
actually READ, not against a guessed id format:

    results/work/ids/universe.parquet   every id in the people file, written
                                        before the universe drops anyone
    results/work/ids/<source>.parquet   ids in a diagnosis file that are not in
                                        the universe (ingest drops their rows)

An id the workflow never read cannot be in anything it wrote, so this set is
complete. The check fails closed: an unreadable file, an unknown suffix or an
empty id set refuses, because a guard that waves through what it could not
parse is trusted and wrong.

A small aggregate cell is not a row and names nobody, so cell counts are not
checked here; a publication floor is a separate, later decision.
"""

import re
from pathlib import Path

import pandas as pd

from bpskit.delivery import delivery_root, stream_csv

IDS_DIR = "results/work/ids"
# Shorter ids would match ordinary counts in every table. How many are skipped
# is printed, so a delivery with short ids cannot weaken the check unnoticed.
MIN_ID_LEN = 4
# results/out/ is copied wholesale, so only formats this check can read (and
# figures, which are drawn from tables that must pass on their own).
OUT_SUFFIXES = {".csv", ".txt", ".png"}
TEXT_SUFFIXES = {".csv", ".tsv", ".txt", ".log", ".md", ".json"}
OPAQUE_SUFFIXES = {".png"}
# A CSV with one of these columns is a per-person table whatever else it holds.
INTERNAL_KEYS = {"pid", "idx", "bucket", "birth", "death", "birth_d", "death_d", "day"}
ID_LIKE = re.compile(r"(^|_)(pid|patient|cod_id|id_paciente|nuhsa)($|_)", re.I)
# Deliberately wide: an id inside a delimited row, a quoted cell, `key=value`,
# a traceback repr or a sentence must come out as a token of its own.
TOKENISERS = (
    re.compile(r"[^\s,;|\"'\[\](){}:=<>/\\]+"),
    re.compile(r"[A-Za-z0-9_-]+"),
    re.compile(r"[A-Za-z0-9]+"),
    re.compile(r"\d+"),
)


# --------------------------------------------------------------------------- #
# Identifiers
# --------------------------------------------------------------------------- #
def _clean(values) -> set[str]:
    s = pd.Series(list(values), dtype="string").dropna().str.strip().str.strip('"').str.strip()
    return set(s[s != ""])


def _finish(ids: set[str], where: str) -> set[str]:
    short = sum(len(i) < MIN_ID_LEN for i in ids)
    ids = {i for i in ids if len(i) >= MIN_ID_LEN}
    if not ids:
        raise SystemExit(f"[REFUSE] no patient identifiers loaded from {where}; nothing is cleared")
    warn = f"  [WARN ] {short:,} shorter than {MIN_ID_LEN} chars are NOT checked" if short else ""
    print(f"[IDS  ] {len(ids):,} identifiers from {where}{warn}")
    return ids


def write_ids(ids, path) -> None:
    """Write an id list for the transfer check (one `pid` column)."""
    pd.DataFrame({"pid": pd.Series(list(ids), dtype="string")}).to_parquet(path, index=False)


def load_ids(paths) -> set[str]:
    """The identifier set from id files, as declared inputs of a rule."""
    paths = list(paths)
    ids: set[str] = set()
    for p in paths:
        ids |= _clean(pd.read_parquet(p, columns=["pid"]).pid)
    return _finish(ids, f"{len(paths)} id file(s)")


def ids_for_root(root: Path, cfg: dict) -> set[str]:
    """Every identifier the workflow at `root` has read, for the hand-run checks.

    Uses the id files where they exist. Where one is missing (a run that died
    before `universe` or inside an `ingest_source`), it reads that file's id
    column from the delivery itself, because the job may have printed a row
    before dying.

    Raises:
        SystemExit: an id file is missing and the delivery file cannot be read.
    """
    root = Path(root)
    ids_dir = root / IDS_DIR
    wanted = {"universe": cfg["universe"]} | dict(cfg["sources"])
    ids: set[str] = set()
    fallback = []
    for name, spec in wanted.items():
        f = ids_dir / f"{name}.parquet"
        if f.exists():
            ids |= _clean(pd.read_parquet(f, columns=["pid"]).pid)
            continue
        path = delivery_root(root, cfg) / spec["file"]
        sc = cfg["scale"]
        try:
            for block in stream_csv(path, [spec["id"]], sc["csv_block_mb"], sc["encoding"]):
                ids |= _clean(block[spec["id"]].unique())
        except Exception as exc:  # noqa: BLE001 -- fail closed: no ids, nothing cleared
            raise SystemExit(
                f"[REFUSE] no id file for {name!r} and its delivery file could not be read "
                f"({exc}); nothing is cleared"
            ) from exc
        fallback.append(name)
    where = IDS_DIR + (
        f" + delivery files for {', '.join(fallback)} (no id file yet)" if fallback else ""
    )
    return _finish(ids, where)


def forbidden_columns(cfg: dict) -> set[str]:
    """Lower-cased column names that make a CSV per-person.

    The internal keys plus the delivery's own id and date columns, taken from
    config/config.yaml so a new source is covered without editing this.
    """
    cols = set(INTERNAL_KEYS)
    u = cfg["universe"]
    cols |= {u["id"], u["birth"], u["death"]}
    for s in cfg["sources"].values():
        cols |= {s["id"], s["date"]}
    return {c.lower() for c in cols}


# --------------------------------------------------------------------------- #
# Files
# --------------------------------------------------------------------------- #
def _tokens(text: str) -> set[str]:
    out: set[str] = set()
    for t in TOKENISERS:
        out.update(t.findall(text))
    return out


def check_file(path: Path, ids: set[str], strict: bool, forbidden: set[str]) -> list[str]:
    """Reasons this file may not leave; empty if it may.

    `strict` adds the rules for results/out/: an allowed suffix, and no CSV
    header column in `forbidden` or shaped like an identifier. Without it
    (logs), only the identifier scan applies.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if strict and suffix not in OUT_SUFFIXES:
        return [f"suffix {suffix or '(none)'} is not one of {sorted(OUT_SUFFIXES)}"]
    if suffix in OPAQUE_SUFFIXES:
        return []
    if suffix not in TEXT_SUFFIXES:
        return [f"suffix {suffix or '(none)'} cannot be checked"]
    bad = []
    text = path.read_text(encoding="utf-8", errors="replace")
    if hits := ids & _tokens(text):
        bad.append(f"contains {len(hits)} patient identifier(s)")
    if strict and suffix == ".csv" and text:
        header = [c.strip().strip('"').strip() for c in text.split("\n", 1)[0].split(",")]
        if cols := [c for c in header if c.lower() in forbidden or ID_LIKE.search(c)]:
            bad.append(f"per-person or identifier column(s) {cols}")
    return bad


def check(
    paths, ids: set[str], strict: bool, forbidden: set[str], skip=()
) -> tuple[int, list[str]]:
    """Check files and directories (recursively).

    Returns:
        (number of files checked, one "path: reason" line per problem). A
        listed path that does not exist is skipped: not built yet is not a leak.
    """
    skip = {Path(p).resolve() for p in skip}
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.rglob("*") if f.is_file())
        elif p.exists():
            files.append(p)
    problems, n = [], 0
    for f in files:
        if f.resolve() in skip:
            continue
        # Snakemake's empty marker inside a directory output; sync.sh never copies it.
        if f.name == ".snakemake_timestamp" and f.stat().st_size == 0:
            continue
        n += 1
        try:
            why = check_file(f, ids, strict, forbidden)
        except Exception as exc:  # noqa: BLE001 -- fail closed: ANY error refuses
            why = [f"could not be checked: {exc}"]
        problems += [f"{f}: {w}" for w in why]
    return n, problems
