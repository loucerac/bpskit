"""Exact tests and multiplicity adjustment, vectorised over many tests at once.

Missing inputs give NaN p-values, and NaN p-values stay out of the adjustment
family rather than counting as tests.
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

__all__ = ["bh", "adjust", "poisson_rr", "binom_half_p", "binom_p"]


def bh(p: pd.Series) -> pd.Series:
    """Benjamini-Hochberg q-values over the non-missing p-values."""
    p = pd.Series(p, dtype=float)
    ok = p.notna()
    q = pd.Series(np.nan, index=p.index)
    if ok.sum():
        v = p[ok].to_numpy()
        order = np.argsort(v)
        m = len(v)
        ranked = v[order] * m / np.arange(1, m + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        out = np.empty(m)
        out[order] = np.minimum(ranked, 1.0)
        q[ok] = out
    return q


def adjust(p: pd.Series, method: str) -> pd.Series:
    """BH q-values, or Bonferroni-adjusted p-values, over the non-missing entries."""
    if method == "bh":
        return bh(p)
    if method == "bonferroni":
        p = pd.Series(p, dtype=float)
        return (p * p.notna().sum()).clip(upper=1.0)
    raise ValueError(f"multiplicity must be 'bh' or 'bonferroni', not {method!r}")


def poisson_rr(
    obs: ArrayLike, exp: ArrayLike, alpha: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Observed/expected ratio, exact (Garwood) interval and two-sided exact p.

    `exp` is treated as known. exp == 0 gives an infinite or NaN ratio.
    """
    obs, exp = np.asarray(obs, float), np.asarray(exp, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        rr = obs / exp
        lo = np.where(obs > 0, stats.chi2.ppf(alpha / 2, 2 * obs) / 2, 0.0) / exp
        hi = stats.chi2.ppf(1 - alpha / 2, 2 * obs + 2) / 2 / exp
        p = np.minimum(
            1.0, 2 * np.minimum(stats.poisson.sf(obs - 1, exp), stats.poisson.cdf(obs, exp))
        )
    return rr, lo, hi, p


def binom_half_p(k: ArrayLike, n: ArrayLike) -> np.ndarray:
    """Two-sided exact binomial test of k successes in n against p = 0.5; NaN when n = 0."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    with np.errstate(invalid="ignore"):
        p = np.minimum(1.0, 2 * stats.binom.cdf(np.minimum(k, n - k), n, 0.5))
    return np.where(n > 0, p, np.nan)


def binom_p(k: ArrayLike, n: ArrayLike, p0: ArrayLike) -> np.ndarray:
    """Two-sided exact binomial test of k in n against p0 (doubled smaller tail).

    NaN when n = 0 or p0 is 0 or 1, where the test is undefined.
    """
    k, n, p0 = np.asarray(k, float), np.asarray(n, float), np.asarray(p0, float)
    with np.errstate(invalid="ignore"):
        p = np.minimum(1.0, 2 * np.minimum(stats.binom.cdf(k, n, p0), stats.binom.sf(k - 1, n, p0)))
    return np.where((n > 0) & (p0 > 0) & (p0 < 1), p, np.nan)
