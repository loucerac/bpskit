import pandas as pd
import pytest

from bpskit import transfer

IDS = {"P00000007", "ORPHAN99", "P00000123"}
FORBIDDEN = {"idx", "cod_id", "birth"}


def _check(tmp_path, name, text, strict=True):
    f = tmp_path / name
    f.write_text(text)
    return transfer.check_file(f, IDS, strict, FORBIDDEN)


@pytest.mark.parametrize(
    "text",
    [
        "a,b\nx,P00000007\n",  # an id in a cell
        "traceback: key=ORPHAN99)\n",  # inside key=value and a bracket
        "row ['P00000123', 'I10']\n",  # a printed list
    ],
)
def test_an_id_anywhere_refuses(tmp_path, text):
    assert _check(tmp_path, "t.txt", text)


def test_per_person_columns_refuse_in_strict_mode(tmp_path):
    assert _check(tmp_path, "a.csv", "idx,n\n1,3\n")
    assert _check(tmp_path, "b.csv", "COD_ID_count,n\n1,3\n")
    assert not _check(tmp_path, "c.csv", "birth_decade,n\n1950,3\n")
    assert not _check(tmp_path, "d.csv", "idx,n\n1,3\n", strict=False)


def test_unknown_suffix_and_unreadable_file_refuse(tmp_path):
    assert _check(tmp_path, "rows.parquet", "")
    n, problems = transfer.check([tmp_path / "missing_dir"], IDS, True, FORBIDDEN)
    assert (n, problems) == (0, [])
    (tmp_path / "bad.csv").mkdir()
    (tmp_path / "bad.csv" / "x.txt").write_text("fine\n")
    bad = tmp_path / "unreadable.csv"
    bad.write_bytes(b"a,b\n")
    bad.chmod(0)
    n, problems = transfer.check([bad], IDS, True, FORBIDDEN)
    bad.chmod(0o644)
    assert problems and "could not be checked" in problems[0]


def test_clean_tree_passes_and_skips_snakemake_marker(tmp_path):
    (tmp_path / "t.csv").write_text("condition,n\nE11,40\n")
    (tmp_path / ".snakemake_timestamp").write_text("")
    n, problems = transfer.check([tmp_path], IDS, True, FORBIDDEN)
    assert (n, problems) == (1, [])


def test_ids_round_trip_and_empty_set_refuses(tmp_path):
    transfer.write_ids(["P00000007", "ORPHAN99"], tmp_path / "ids.parquet")
    assert transfer.load_ids([tmp_path / "ids.parquet"]) == {"P00000007", "ORPHAN99"}
    transfer.write_ids([], tmp_path / "none.parquet")
    with pytest.raises(SystemExit):
        transfer.load_ids([tmp_path / "none.parquet"])
    assert pd.read_parquet(tmp_path / "ids.parquet").columns.tolist() == ["pid"]
