"""The transfer checks, run on the data machine before anything is copied off it.

    bpskit pullable [--strict] <path> ...   exit 0 only if no file names a patient
    bpskit check-logs [workflow_dir]        list the logs that may leave
    bpskit synthetic <out_dir>              write a synthetic delivery (bpskit.synthetic)

pullable and check-logs run from the workflow directory and read its config
(default config/config.yaml) for the delivery's id columns and the id files
under results/work/ids/.
"""

import argparse
import sys
from pathlib import Path

from bpskit import transfer
from bpskit.delivery import load_config


def pullable(args: argparse.Namespace) -> int:
    root = Path.cwd()
    cfg = load_config(root / args.config)
    ids = transfer.ids_for_root(root, cfg)
    n, problems = transfer.check(args.paths, ids, args.strict, transfer.forbidden_columns(cfg))
    print(f"[CHECK] {n} file(s), {len(problems)} problem(s)")
    for p in problems:
        print(f"[REFUSE] {p}")
    if problems:
        print("No transfer: these files stay on this machine.")
        return 1
    print("[PASS ] no file names a patient")
    return 0


def check_logs(args: argparse.Namespace) -> int:
    """Write the clean logs to results/work/logs_pullable.txt (which sync.sh copies).

    A traceback can print a data row, so a log containing an id is WITHHELD,
    not redacted. The list is emptied first, so any failure leaves nothing listed.
    """
    root = Path(args.workflow_dir).resolve()
    out = root / "results/work/logs_pullable.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("")
    ids = transfer.ids_for_root(root, load_config(root / args.config))
    candidates = sorted(
        list((root / "results/work/logs").rglob("*.log"))
        + list((root / ".snakemake/log").glob("*.snakemake.log"))
        + [p for p in [root / "results_run.txt"] if p.exists()]
    )
    clean, withheld = [], []
    for f in candidates:
        try:
            why = transfer.check_file(f, ids, strict=False, forbidden=set())
        except Exception as exc:  # noqa: BLE001 -- fail closed: any error withholds
            why = [f"could not be checked: {exc}"]
        (withheld if why else clean).append((f, why))
    out.write_text("".join(f"{f.relative_to(root).as_posix()}\n" for f, _ in clean))
    for f, why in withheld:
        print(f"WITHHELD ({'; '.join(why)}): {f.relative_to(root)}")
    print(f"{len(clean)} of {len(candidates)} logs cleared")
    return 0


def synthetic(args: argparse.Namespace) -> int:
    from bpskit.synthetic import make_delivery

    make_delivery(args.out_dir, args.persons, args.seed)
    print(f"{args.persons:,} synthetic persons -> {args.out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bpskit", description=__doc__.split("\n")[0])
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="workflow config, relative to the workflow directory",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("pullable", help="refuse files that name a patient")
    p.add_argument(
        "--strict",
        action="store_true",
        help="results/out/ rules: csv/txt/png only, no per-person columns",
    )
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=pullable)
    c = sub.add_parser("check-logs", help="list the logs free of patient ids")
    c.add_argument("workflow_dir", nargs="?", default=".")
    c.set_defaults(func=check_logs)
    s = sub.add_parser("synthetic", help="write a synthetic BPS delivery")
    s.add_argument("out_dir")
    s.add_argument("--persons", type=int, default=50_000)
    s.add_argument("--seed", type=int, default=1)
    s.set_defaults(func=synthetic)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
