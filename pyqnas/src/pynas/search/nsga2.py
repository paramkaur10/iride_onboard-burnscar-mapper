"""Selection on several objectives at once: NSGA-II (Deb, Pratap, Agarwal and Meyarivan 2002).

NSGA-II sorts a population into Pareto fronts (no candidate of a front is beaten by another on every objective at
once), keeps the earlier fronts first and, inside the last front it keeps, the candidates in the emptier regions
(largest crowding distance), so that the front stays spread along the trade-off.

Every objective here is minimised (a maximised one is passed negated). A candidate without a value (NaN or inf)
ends up behind every candidate that has one.
"""
from __future__ import annotations

import numpy as np


def _clean(F):
    """Objective values with every row that lacks a value (NaN) set to inf on all objectives: a candidate without a
    score must not survive on its other objectives (e.g. a failed training with a low latency)."""
    F = np.asarray(F, dtype=float)
    if F.ndim == 1:
        F = F[:, None]
    bad = np.isnan(F).any(1) | np.isinf(F).any(1)
    F = F.copy()
    F[bad] = np.inf
    return F


def dominates(a, b):
    """a is at least as good as b on every objective and better on one."""
    return bool(np.all(a <= b) and np.any(a < b))


def fronts(F):
    """Non-dominated sorting: the candidates' indices front by front, the best front first."""
    F = _clean(F)
    n = len(F)
    le = (F[:, None, :] <= F[None, :, :]).all(-1)          # le[p, q]: p at least as good as q everywhere
    lt = (F[:, None, :] < F[None, :, :]).any(-1)
    dom = le & lt                                          # dom[p, q]: p dominates q
    count = dom.sum(0)                                     # how many dominate q
    out, current = [], np.flatnonzero(count == 0)
    remaining = count.copy()
    while len(current):
        out.append(current)
        nxt = []
        for p in current:
            for q in np.flatnonzero(dom[p]):
                remaining[q] -= 1
                if remaining[q] == 0:
                    nxt.append(q)
        current = np.array(sorted(nxt), dtype=int)
    return out


def crowding(F):
    """Crowding distance of every candidate of one front: the sum over objectives of the gap between its two
    neighbours, relative to the range of the front. The two ends of each objective get infinity."""
    F = _clean(F)
    n, m = F.shape
    if n <= 2:
        return np.full(n, np.inf)
    d = np.zeros(n)
    for j in range(m):
        o = np.argsort(F[:, j], kind="stable")
        lo, hi = F[o[0], j], F[o[-1], j]
        d[o[0]] = d[o[-1]] = np.inf
        if np.isfinite(hi) and np.isfinite(lo) and hi > lo:
            d[o[1:-1]] += (F[o[2:], j] - F[o[:-2], j]) / (hi - lo)
    return d


def rank(F):
    """Front number (0 is the best) and crowding distance of every candidate."""
    F = _clean(F)
    front = np.zeros(len(F), dtype=int)
    crowd = np.zeros(len(F))
    for k, idx in enumerate(fronts(F)):
        front[idx] = k
        crowd[idx] = crowding(F[idx])
    return front, crowd


def select(F, k):
    """The k candidates NSGA-II keeps: whole fronts in order, the last one cut by crowding distance."""
    F = _clean(F)
    keep = []
    for idx in fronts(F):
        if len(keep) + len(idx) <= k:
            keep += idx.tolist()
        else:
            d = crowding(F[idx])
            keep += idx[np.argsort(-d, kind="stable")[:k - len(keep)]].tolist()
            break
    return np.array(keep, dtype=int)


def tournament(F, rng, n):
    """n parents, each the better of two candidates drawn at random: lower front, then larger crowding."""
    front, crowd = rank(F)
    a = rng.integers(0, len(front), n)
    b = rng.integers(0, len(front), n)
    better_a = (front[a] < front[b]) | ((front[a] == front[b]) & (crowd[a] >= crowd[b]))
    return np.where(better_a, a, b)


def hypervolume_2d(F, ref):
    """Area dominated by a set of points of two minimised objectives, up to the reference point."""
    F = _clean(F)[:, :2]
    F = F[np.all(F < np.asarray(ref, float), axis=1)]
    if not len(F):
        return 0.0
    F = F[np.argsort(F[:, 0], kind="stable")]
    hv, best_y = 0.0, ref[1]
    for x, y in F:
        if y < best_y:
            hv += (ref[0] - x) * (best_y - y)
            best_y = y
    return float(hv)
