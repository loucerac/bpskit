"""Exact tests and multiplicity adjustment, vectorised over many tests at once.

Missing inputs give NaN p-values, and NaN p-values stay out of the adjustment
family rather than counting as tests.
"""

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats

__all__ = [
    "bh",
    "adjust",
    "poisson_rr",
    "binom_half_p",
    "binom_p",
    "idtn_from_ratio",
    "exact_contrast",
]


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


def idtn_from_ratio(r: ArrayLike) -> np.ndarray:
    """IDTN from a rate ratio: (r - 1) / (r + 1), with an infinite ratio giving 1."""
    r = np.asarray(r, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (r - 1.0) / (r + 1.0)
    return np.where(np.isinf(r), 1.0, out)


def exact_contrast(
    cf: ArrayLike, pyf: ArrayLike, cm: ArrayLike, pym: ArrayLike, alpha: float
) -> pd.DataFrame:
    """Rate ratio, IDTN and rate difference of two groups, with exact intervals.

    Written for female (f) against male (m), the names the output columns carry,
    but any two groups work.

    Conditional on the total count, female cases are binomial with
    p = IRR*pyf / (IRR*pyf + pym). A Clopper-Pearson interval on p therefore
    maps to an exact interval on the IRR, and through the monotone
    (r-1)/(r+1) onto IDTN. Exact matters here: age bands at the extremes hold
    a handful of cases, where a Wald interval on log IRR is badly wrong.

    Rates are per 100,000 person-years. Cells with no cases or no person-time
    in either sex come back NaN rather than as 0 or +-inf.
    """
    cf, cm = np.asarray(cf, float), np.asarray(cm, float)
    pyf, pym = np.asarray(pyf, float), np.asarray(pym, float)
    n = cf + cm
    with np.errstate(invalid="ignore", divide="ignore"):
        rf, rm = cf / pyf, cm / pym
        irr = rf / rm
        idtn = (rf - rm) / (rf + rm)
        p_lo = np.where(cf > 0, stats.beta.ppf(alpha / 2, cf, n - cf + 1), 0.0)
        p_hi = np.where(cf < n, stats.beta.ppf(1 - alpha / 2, cf + 1, n - cf), 1.0)
        k = pym / pyf
        irr_lo = p_lo / (1 - p_lo) * k
        irr_hi = np.where(p_hi >= 1, np.inf, p_hi / (1 - p_hi) * k)
    bad = (n == 0) | (pyf <= 0) | (pym <= 0)
    out = pd.DataFrame(
        {
            "rate_f": rf * 1e5,
            "rate_m": rm * 1e5,
            "rd_per_100k": (rf - rm) * 1e5,
            "irr": irr,
            "irr_lo": irr_lo,
            "irr_hi": irr_hi,
            "idtn": idtn,
            "idtn_lo": idtn_from_ratio(irr_lo),
            "idtn_hi": idtn_from_ratio(irr_hi),
        }
    )
    out.loc[bad, ["irr", "irr_lo", "irr_hi", "idtn", "idtn_lo", "idtn_hi"]] = np.nan
    out.loc[(pyf <= 0) | (pym <= 0), ["rate_f", "rate_m", "rd_per_100k"]] = np.nan
    return out
