import hashlib

from bpskit.delivery import read_delivery, stream_csv
from bpskit.synthetic import FILES, make_delivery


def _digests(path):
    return {f.name: hashlib.md5(f.read_bytes()).hexdigest() for f in sorted(path.iterdir())}


def test_writes_readable_files_matching_its_config(tmp_path):
    cfg = make_delivery(tmp_path, n_persons=300, seed=3)
    assert set(_digests(tmp_path)) == set(FILES.values())
    u = cfg["universe"]
    people = read_delivery(tmp_path / u["file"], [u["id"], u["sex"], u["birth"], u["death"]])
    assert len(people) == 300 and set(people[u["sex"]]) <= set(u["sex_values"])
    for spec in cfg["sources"].values():
        cols = [spec["id"], spec["date"], spec["code"]]
        rows = sum(len(b) for b in stream_csv(tmp_path / spec["file"], cols, 1, "utf-8"))
        assert rows > 0


def test_same_seed_same_bytes(tmp_path):
    make_delivery(tmp_path / "a", n_persons=200, seed=5)
    make_delivery(tmp_path / "b", n_persons=200, seed=5)
    assert _digests(tmp_path / "a") == _digests(tmp_path / "b")
