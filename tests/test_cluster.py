import numpy as np

from bpskit.cluster import adjusted_rand, jaccard, mcl, modularity
from bpskit.mixture import bernoulli_mixture, match_classes


def test_jaccard_by_hand():
    m = np.array([[1, 1, 0], [1, 0, 0], [0, 0, 0]], bool)
    j = jaccard(m)
    assert j[0, 1] == 0.5 and j[0, 0] == 1.0 and j[2, 2] == 0.0 and j[0, 2] == 0.0


def _planted(rng, sizes, p_in=0.8, p_out=0.05):
    lab = np.repeat(np.arange(len(sizes)), sizes)
    n = len(lab)
    same = lab[:, None] == lab[None, :]
    w = (rng.random((n, n)) < np.where(same, p_in, p_out)).astype(float)
    w = np.triu(w, 1)
    return w + w.T, lab


def test_mcl_recovers_planted_blocks_and_orders_labels_by_size():
    rng = np.random.default_rng(1)
    w, lab = _planted(rng, [12, 8, 5])
    perm = rng.permutation(len(lab))
    got = mcl(w[np.ix_(perm, perm)], inflation=2.0)
    assert adjusted_rand(got, lab[perm]) == 1.0
    assert np.bincount(got).tolist() == [12, 8, 5]


def test_mcl_higher_inflation_never_gives_fewer_clusters_here():
    rng = np.random.default_rng(2)
    w, _ = _planted(rng, [10, 10], p_in=0.5, p_out=0.1)
    n = [len(np.unique(mcl(w, inflation=i))) for i in (1.4, 2.0, 4.0)]
    assert n == sorted(n)


def test_modularity_by_hand():
    # two triangles joined by one edge
    w = np.zeros((6, 6))
    for a, b in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]:
        w[a, b] = w[b, a] = 1
    q = modularity(w, [0, 0, 0, 1, 1, 1])
    assert abs(q - 5 / 14) < 1e-12
    assert modularity(w, [0] * 6) == 0.0


def test_adjusted_rand_is_label_free_and_near_zero_for_noise():
    assert adjusted_rand([0, 0, 1, 1], [5, 5, 2, 2]) == 1.0
    rng = np.random.default_rng(3)
    assert abs(adjusted_rand(rng.integers(0, 3, 3000), rng.integers(0, 3, 3000))) < 0.01


def _planted_mixture(rng, n, prof, share, missing):
    z = rng.choice(len(share), size=n, p=share)
    x = (rng.random((n, prof.shape[1])) < prof[z]).astype(float)
    obs = rng.random(x.shape) >= missing
    return x, obs, z


PROF = np.array(
    [
        [0.9, 0.9, 0.1, 0.1, 0.1, 0.5],
        [0.1, 0.1, 0.9, 0.9, 0.1, 0.5],
        [0.1, 0.1, 0.1, 0.1, 0.9, 0.5],
    ]
)


def test_mixture_recovers_planted_classes_with_missing_cells():
    rng = np.random.default_rng(4)
    x, obs, z = _planted_mixture(rng, 3000, PROF, [0.5, 0.3, 0.2], 0.3)
    x_seen = np.where(obs, x, 0.0)  # a missing cell must not be read as 0
    fit = bernoulli_mixture(x_seen, obs, 3, starts=5, seed=0)
    assert np.abs(fit.weights - [0.5, 0.3, 0.2]).max() < 0.03
    assert np.abs(fit.profiles - PROF).max() < 0.06
    # Assignments as good as those from the true parameters (the best possible).
    share = np.array([0.5, 0.3, 0.2])
    lr = np.log(share) + (x_seen * obs) @ np.log(PROF).T + ((1 - x_seen) * obs) @ np.log1p(-PROF).T
    best = adjusted_rand(lr.argmax(axis=1), z)
    assert adjusted_rand(fit.posterior.argmax(axis=1), z) > best - 0.02


def test_bic_picks_the_true_number_of_classes():
    rng = np.random.default_rng(5)
    x, obs, _ = _planted_mixture(rng, 3000, PROF, [0.5, 0.3, 0.2], 0.2)
    bic = [bernoulli_mixture(x, obs, k, starts=5, seed=0).bic for k in range(1, 6)]
    assert int(np.argmin(bic)) + 1 == 3


def test_row_weights_equal_repeated_rows_and_classes_match():
    rng = np.random.default_rng(6)
    x, _, _ = _planted_mixture(rng, 400, PROF, [0.5, 0.3, 0.2], 0.0)
    rows, counts = np.unique(x, axis=0, return_counts=True)
    a = bernoulli_mixture(x, None, 2, starts=5, seed=0)
    b = bernoulli_mixture(rows, None, 2, row_weights=counts, starts=5, seed=0)
    assert abs(a.loglik - b.loglik) < 1e-4 * abs(a.loglik)
    perm = match_classes(a.profiles, b.profiles[::-1])
    np.testing.assert_allclose(b.profiles[::-1][perm], a.profiles, atol=1e-3)
