import numpy as np

from bpskit.cluster import adjusted_rand, jaccard, mcl, modularity
from bpskit.mixture import (
    bernoulli_mixture,
    categorical_mixture,
    categorical_posterior,
    match_classes,
)


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


# Three classes, four items, four ordered categories each ("first recorded in
# window 0, 1, 2, or not by the end"), shaped like the subtype features.
THETA = np.array(
    [
        [[0.7, 0.1, 0.1, 0.1], [0.7, 0.1, 0.1, 0.1], [0.05, 0.05, 0.1, 0.8], [0.1, 0.1, 0.1, 0.7]],
        [[0.05, 0.05, 0.1, 0.8], [0.1, 0.6, 0.2, 0.1], [0.1, 0.6, 0.2, 0.1], [0.1, 0.1, 0.1, 0.7]],
        [
            [0.05, 0.05, 0.1, 0.8],
            [0.05, 0.05, 0.1, 0.8],
            [0.1, 0.1, 0.1, 0.7],
            [0.2, 0.2, 0.5, 0.1],
        ],  # noqa: E501
    ]
)


def _planted_categorical(rng, n, share, censored):
    """Rows of `allowed`, right-censored like follow-up: a censored row was seen
    only up to window `cut`, so a category at or after `cut` stays possible."""
    k, p, c = THETA.shape
    z = rng.choice(k, size=n, p=share)
    cat = np.array([[rng.choice(c, p=THETA[zi, j]) for j in range(p)] for zi in z])
    allowed = np.zeros((n, p, c), bool)
    allowed[np.arange(n)[:, None], np.arange(p)[None, :], cat] = True
    cut = np.where(rng.random(n) < censored, rng.integers(0, c - 1, n), c)
    late = np.arange(c)[None, None, :] >= cut[:, None, None]
    unseen = cat >= cut[:, None]  # its category falls after the end of follow-up
    allowed = np.where(unseen[:, :, None], late, allowed)
    return allowed, z


def test_categorical_mixture_recovers_planted_classes_under_censoring():
    rng = np.random.default_rng(7)
    allowed, z = _planted_categorical(rng, 4000, [0.5, 0.3, 0.2], censored=0.4)
    assert (allowed.sum(axis=2) > 1).mean() > 0.1  # censoring is really there
    fit = categorical_mixture(allowed, 3, starts=5, seed=0)
    perm = match_classes(THETA, fit.profiles)
    assert np.abs(fit.weights[perm] - [0.5, 0.3, 0.2]).max() < 0.04
    assert np.abs(fit.profiles[perm] - THETA).max() < 0.07
    # Reading a censored cell as "not by the end" is what the sets avoid: it
    # pushes the last category up and the profiles far off.
    naive = allowed.copy()
    multi = naive.sum(axis=2) > 1
    naive[multi] = False
    naive[multi, -1] = True
    bad = categorical_mixture(naive, 3, starts=5, seed=0)
    assert np.abs(bad.profiles[match_classes(THETA, bad.profiles)] - THETA).max() > 0.3
    post = categorical_posterior(fit, allowed)
    np.testing.assert_allclose(post, fit.posterior, atol=1e-6)
    assert adjusted_rand(post.argmax(axis=1), z) > 0.5


def test_categorical_bic_picks_the_true_number_of_classes():
    rng = np.random.default_rng(8)
    allowed, _ = _planted_categorical(rng, 4000, [0.5, 0.3, 0.2], censored=0.3)
    bic = [categorical_mixture(allowed, k, starts=4, seed=0).bic for k in range(1, 6)]
    assert int(np.argmin(bic)) + 1 == 3


def test_two_categories_with_unknown_cells_is_the_bernoulli_mixture():
    rng = np.random.default_rng(9)
    x, obs, _ = _planted_mixture(rng, 1500, PROF, [0.5, 0.3, 0.2], 0.2)
    allowed = np.stack([(x == 0) | ~obs, (x == 1) | ~obs], axis=2)
    b = bernoulli_mixture(x, obs, 3, starts=5, seed=0)
    c = categorical_mixture(allowed, 3, starts=5, seed=0)
    assert abs(b.loglik - c.loglik) < 1e-3 * abs(b.loglik)
    assert b.n_params == c.n_params
    perm = match_classes(b.profiles, c.profiles[:, :, 1])
    np.testing.assert_allclose(c.profiles[perm, :, 1], b.profiles, atol=0.01)


def test_a_category_no_row_allows_gets_no_probability_and_no_parameter():
    rng = np.random.default_rng(10)
    allowed, _ = _planted_categorical(rng, 500, [0.5, 0.3, 0.2], censored=0.0)
    padded = np.concatenate([allowed, np.zeros(allowed.shape[:2] + (1,), bool)], axis=2)
    a = categorical_mixture(allowed, 2, starts=3, seed=0)
    b = categorical_mixture(padded, 2, starts=3, seed=0)
    assert a.n_params == b.n_params
    assert (b.profiles[:, :, -1] == 0).all()
    assert abs(a.loglik - b.loglik) < 1e-6 * abs(a.loglik)
