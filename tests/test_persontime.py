import itertools

import numpy as np
import pandas as pd

from bpskit.persontime import age_bands, band_index, lexis, person_chunks


def test_lexis_splits_an_interval_without_losing_time():
    bands = age_bands(5, 95)
    pt = lexis(np.array([3.0, 92.0, 10.0]), np.array([17.5, 101.0, 9.0]), bands)
    np.testing.assert_allclose(pt.sum(axis=1), [14.5, 9.0, 0.0])
    assert pt[0, 0] == 2.0 and pt[0, 3] == 2.5
    np.testing.assert_array_equal(band_index([-1, 0, 4.99, 120], bands), [0, 0, 0, 19])


def test_person_chunks_never_split_a_person(tmp_path):
    idx = np.repeat(np.arange(50), np.arange(50) % 7 + 1)
    pd.DataFrame({"idx": idx, "v": np.arange(len(idx))}).to_parquet(tmp_path / "x.parquet")
    chunks = list(person_chunks(tmp_path / "x.parquet", batch_rows=9))
    assert sum(len(c) for c in chunks) == len(idx)
    seen = [set(c.idx) for c in chunks]
    assert all(not (a & b) for a, b in itertools.combinations(seen, 2))
