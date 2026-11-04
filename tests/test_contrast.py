import numpy as np
from scipy import stats

from bpskit.stats import exact_contrast, idtn_from_ratio


def test_irr_interval_is_exact_clopper_pearson():
    cf, cm, pyf, pym = 7.0, 3.0, 1000.0, 2000.0
    out = exact_contrast([cf], [pyf], [cm], [pym], alpha=0.05).iloc[0]
    ci = stats.binomtest(int(cf), int(cf + cm)).proportion_ci(0.95, method="exact")
    to_irr = lambda p: p / (1 - p) * pym / pyf  # noqa: E731
    np.testing.assert_allclose([out.irr_lo, out.irr_hi], [to_irr(ci.low), to_irr(ci.high)])
    np.testing.assert_allclose(out.irr, (cf / pyf) / (cm / pym))
    np.testing.assert_allclose(out.idtn, idtn_from_ratio(out.irr))


def test_empty_cells_are_nan_not_infinite():
    out = exact_contrast([0, 5], [100, 0], [0, 1], [100, 100], alpha=0.05)
    assert out[["irr", "idtn"]].isna().all().all()
    assert np.isnan(out.rate_f.iloc[1])


def test_idtn_of_infinite_ratio_is_one():
    assert idtn_from_ratio(np.inf) == 1.0
    assert idtn_from_ratio(1.0) == 0.0
