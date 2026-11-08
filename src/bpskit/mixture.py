"""Latent class analysis: a mixture of independent Bernoullis fitted by EM, with
cells that were not observed left out of the likelihood rather than read as 0.

Written for subtyping the patients of one disease by which diagnoses they
receive in windows after it, where a window not fully followed (death, end of
data) is unknown, not "no diagnosis". Rows can carry weights, so identical rows
are fitted once and a bootstrap is a reweighting.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

__all__ = ["MixtureFit", "bernoulli_mixture", "match_classes"]


@dataclass
class MixtureFit:
    """One fitted mixture. Classes are ordered by decreasing weight.

    weights    (k,)     class shares
    profiles   (k, p)   probability of each item being 1 in each class
    posterior  (n, k)   probability of each row belonging to each class
    loglik, bic, icl    fit; lower BIC / ICL is better. ICL adds twice the
                        classification entropy, penalising classes that overlap
    entropy             relative entropy, 1 = rows assigned without doubt, 0 = no separation
    n_params, n_obs     free parameters; total row weight
    converged
    """

    k: int
    weights: np.ndarray
    profiles: np.ndarray
    posterior: np.ndarray
    loglik: float
    bic: float
    icl: float
    entropy: float
    n_params: int
    n_obs: float
    converged: bool


def _loglik_rows(x, obs, logp, log1p, logw):
    # (n, k): log weight + sum over observed items of the Bernoulli log density
    return logw[None, :] + (x * obs) @ logp.T + ((1 - x) * obs) @ log1p.T


def _fit_once(x, obs, rw, k, rng, max_iter, tol, eps):
    n, p = x.shape
    tau = rng.dirichlet(np.ones(k), size=n)
    ll_prev = -np.inf
    converged = False
    for _ in range(max_iter):
        wt = tau * rw[:, None]
        weights = wt.sum(axis=0) / rw.sum()
        num = wt.T @ (x * obs) + eps
        den = wt.T @ obs + 2 * eps
        prof = num / den
        logp, log1p = np.log(prof), np.log1p(-prof)
        logw = np.log(np.maximum(weights, 1e-300))
        lr = _loglik_rows(x, obs, logp, log1p, logw)
        norm = logsumexp(lr, axis=1)
        ll = float((rw * norm).sum())
        tau = np.exp(lr - norm[:, None])
        if abs(ll - ll_prev) < tol * max(1.0, abs(ll)):
            converged = True
            break
        ll_prev = ll
    return ll, weights, prof, tau, converged


def bernoulli_mixture(
    x: ArrayLike,
    observed: ArrayLike | None,
    k: int,
    row_weights: ArrayLike | None = None,
    starts: int = 10,
    seed: int = 0,
    max_iter: int = 1000,
    tol: float = 1e-8,
    eps: float = 1e-3,
) -> MixtureFit:
    """Fit a k-class Bernoulli mixture by EM from `starts` random starts and keep
    the best likelihood.

    x            (n, p) 0/1 values; ignored where `observed` is False
    observed     (n, p) bool, None = everything observed
    row_weights  (n,) non-negative weights, e.g. the count of identical rows or
                 bootstrap multiplicities; None = 1 each
    eps          pseudo-count keeping every probability inside (0, 1), so a
                 class that never shows an item does not give log(0)

    BIC counts (k - 1) + k p parameters and log(sum of row weights) per parameter.
    """
    x = np.asarray(x, dtype=float)
    n, p = x.shape
    obs = np.ones_like(x) if observed is None else np.asarray(observed, dtype=float)
    if obs.shape != x.shape:
        raise ValueError("observed must have the shape of x")
    rw = np.ones(n) if row_weights is None else np.asarray(row_weights, dtype=float)
    if k < 1:
        raise ValueError("k must be at least 1")
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(max(1, int(starts) if k > 1 else 1)):
        fit = _fit_once(x, obs, rw, k, rng, max_iter, tol, eps)
        if best is None or fit[0] > best[0]:
            best = fit
    ll, weights, prof, tau, converged = best
    order = np.argsort(-weights, kind="stable")
    weights, prof, tau = weights[order], prof[order], tau[:, order]
    n_obs = float(rw.sum())
    n_params = (k - 1) + k * p
    bic = -2 * ll + n_params * np.log(n_obs)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = float(-(rw[:, None] * np.where(tau > 0, tau * np.log(tau), 0.0)).sum())
    rel = 1.0 - ent / (n_obs * np.log(k)) if k > 1 else 1.0
    return MixtureFit(
        k=k,
        weights=weights,
        profiles=prof,
        posterior=tau,
        loglik=ll,
        bic=float(bic),
        icl=float(bic + 2 * ent),
        entropy=float(rel),
        n_params=n_params,
        n_obs=n_obs,
        converged=converged,
    )


def match_classes(reference: ArrayLike, profiles: ArrayLike) -> np.ndarray:
    """The order of `profiles`' classes that best matches `reference` (Hungarian
    assignment on the summed absolute difference of profiles):
    profiles[match_classes(ref, profiles)] lines up with ref row by row."""
    ref, prof = np.asarray(reference, float), np.asarray(profiles, float)
    cost = np.abs(ref[:, None, :] - prof[None, :, :]).sum(axis=2)
    rows, cols = linear_sum_assignment(cost)
    return cols[np.argsort(rows)]
