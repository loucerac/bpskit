"""Latent class analysis by EM, with cells that were not fully observed kept in
the likelihood only for what is known about them.

    bernoulli_mixture     items are 0/1; a cell is observed or not
    categorical_mixture   items have several categories; a cell carries the set
                          of categories it may be in (one when known, several
                          when censored, all when unknown)

Written for subtyping the patients of one disease by when they first receive
each associated diagnosis: before the index, in one of the windows after it, or
not by the last. Each diagnosis is one categorical item, so its windows are not
fitted as separate 0/1 items that nest inside each other ("by 1 year" implies
"by 3 years"), which breaks the independence the model assumes within a class.
A patient followed for two years without the diagnosis may still be in a later
window or none: those categories stay allowed. Rows can carry weights, so
identical rows are fitted once and a bootstrap is a reweighting.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp

__all__ = [
    "MixtureFit",
    "bernoulli_mixture",
    "categorical_mixture",
    "categorical_posterior",
    "match_classes",
]


@dataclass
class MixtureFit:
    """One fitted mixture. Classes are ordered by decreasing weight.

    weights    (k,)     class shares
    profiles   (k, p)   probability of each item being 1 in each class
               (k, p, c) for a categorical mixture: of each category of each item
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
    return _finish(k, *best, rw, (k - 1) + k * p)


def _finish(k, ll, weights, prof, tau, converged, rw, n_params):
    order = np.argsort(-weights, kind="stable")
    weights, prof, tau = weights[order], prof[order], tau[:, order]
    n_obs = float(rw.sum())
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


def _cat_loglik_rows(a, theta, logw):
    # (n, k): log weight + sum over items of log P(the cell's allowed categories)
    out = np.empty((a.shape[0], theta.shape[0]))
    for c in range(theta.shape[0]):
        out[:, c] = logw[c] + np.log(np.einsum("ijc,jc->ij", a, theta[c])).sum(axis=1)
    return out


def _cat_fit_once(a, ever, rw, k, rng, max_iter, tol, eps):
    n = a.shape[0]
    tau = rng.dirichlet(np.ones(k), size=n)
    # First M-step: each cell spread evenly over its allowed categories.
    share = a / a.sum(axis=2, keepdims=True)
    counts = np.einsum("ik,ijc->kjc", tau * rw[:, None], share)
    ll_prev = -np.inf
    converged = False
    for _ in range(max_iter):
        weights = (tau * rw[:, None]).sum(axis=0) / rw.sum()
        num = (counts + eps) * ever[None]
        theta = num / num.sum(axis=2, keepdims=True)
        logw = np.log(np.maximum(weights, 1e-300))
        lr = _cat_loglik_rows(a, theta, logw)
        norm = logsumexp(lr, axis=1)
        ll = float((rw * norm).sum())
        tau = np.exp(lr - norm[:, None])
        if abs(ll - ll_prev) < tol * max(1.0, abs(ll)):
            converged = True
            break
        ll_prev = ll
        # M-step counts: within class c, a cell's weight goes to its allowed
        # categories in proportion to theta (expected category of a censored cell).
        for c in range(k):
            lik = np.einsum("ijc,jc->ij", a, theta[c])
            counts[c] = theta[c] * np.einsum("ij,ijc->jc", (tau[:, c] * rw)[:, None] / lik, a)
    return ll, weights, theta, tau, converged


def categorical_mixture(
    allowed: ArrayLike,
    k: int,
    row_weights: ArrayLike | None = None,
    starts: int = 10,
    seed: int = 0,
    max_iter: int = 1000,
    tol: float = 1e-8,
    eps: float = 1e-3,
) -> MixtureFit:
    """Fit a k-class mixture of independent categorical items by EM from `starts`
    random starts and keep the best likelihood.

    allowed      (n, p, c) bool: the categories each cell may be in. One True is a
                 known cell, several a censored one (the likelihood is the class's
                 probability of that set), all True an unknown one (no
                 information). Every cell needs at least one True. Items with
                 fewer categories than c leave the rest False in every row: a
                 category no row allows gets probability 0 and no parameter.
    row_weights  (n,) non-negative weights; None = 1 each
    eps          pseudo-count per category, so a class never showing a category
                 does not give log(0)

    The censoring is taken as uninformative: being followed to a given day says
    nothing about the category beyond excluding the ones already ruled out.
    BIC counts (k - 1) + k * sum over items of (categories allowed anywhere - 1).
    """
    a = np.asarray(allowed, dtype=bool)
    if a.ndim != 3:
        raise ValueError("allowed must be (rows, items, categories)")
    if not a.any(axis=2).all():
        raise ValueError("every cell must allow at least one category")
    if k < 1:
        raise ValueError("k must be at least 1")
    n = a.shape[0]
    rw = np.ones(n) if row_weights is None else np.asarray(row_weights, dtype=float)
    ever = a.any(axis=0)
    af = a.astype(float)
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(max(1, int(starts) if k > 1 else 1)):
        fit = _cat_fit_once(af, ever, rw, k, rng, max_iter, tol, eps)
        if best is None or fit[0] > best[0]:
            best = fit
    n_params = (k - 1) + k * int((ever.sum(axis=1) - 1).sum())
    return _finish(k, *best, rw, n_params)


def categorical_posterior(fit: MixtureFit, allowed: ArrayLike) -> np.ndarray:
    """(n, k) class probabilities of new rows under a fitted categorical mixture."""
    a = np.asarray(allowed, dtype=float)
    lr = _cat_loglik_rows(a, fit.profiles, np.log(np.maximum(fit.weights, 1e-300)))
    return np.exp(lr - logsumexp(lr, axis=1)[:, None])


def match_classes(reference: ArrayLike, profiles: ArrayLike) -> np.ndarray:
    """The order of `profiles`' classes that best matches `reference` (Hungarian
    assignment on the summed absolute difference of profiles, of any shape with
    classes first): profiles[match_classes(ref, profiles)] lines up with ref."""
    ref, prof = np.asarray(reference, float), np.asarray(profiles, float)
    ref, prof = ref.reshape(len(ref), -1), prof.reshape(len(prof), -1)
    cost = np.abs(ref[:, None, :] - prof[None, :, :]).sum(axis=2)
    rows, cols = linear_sum_assignment(cost)
    return cols[np.argsort(rows)]
