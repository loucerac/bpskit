import itertools

import numpy as np

from bpskit.persontime import age_bands
from bpskit.strata import Strata, person_pairs


def test_person_pairs_is_every_ordered_pair_within_a_person():
    person = np.array([0, 0, 0, 1, 2, 2, 5, 5, 5, 5])
    got = set()
    for a, b in person_pairs(person, max_pairs=5):
        got |= set(zip(a.tolist(), b.tolist()))
    want = {
        (i, j) for i, j in itertools.permutations(range(len(person)), 2) if person[i] == person[j]
    }
    assert got == want


def test_strata_integrate_matches_accumulate():
    bands = age_bands(5, 95)
    ys = np.array([0.0, 365.0, 730.0, 1095.0])
    st = Strata(bands, ys)
    rng = np.random.default_rng(0)
    n = 40
    birth = rng.uniform(-30000, -10000, n)
    entry = rng.uniform(0, 600, n)
    exit_ = entry + rng.uniform(0, 600, n)
    sex = rng.integers(0, 2, n)
    person = np.arange(n)
    acc = np.zeros((1, st.S))
    st.accumulate(acc, np.zeros(n, np.int64), sex, birth, entry, exit_, person)
    rate = rng.uniform(0, 1, (st.S, 1))
    got = st.integrate(rate, np.zeros(n, np.int64), sex, birth, entry, exit_, person).sum()
    np.testing.assert_allclose(got, acc[0] @ rate[:, 0])
