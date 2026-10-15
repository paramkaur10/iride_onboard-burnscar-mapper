"""NSGA-II ranking and selection on known points; the BatchNorm folding formula on a direct convolution (NumPy,
grouped and depthwise included)."""
import numpy as np

from pynas.search import nsga2


def test_fronts_and_selection():
    # minimise both: (1,5) (2,2) (5,1) form front 0; (3,3) and (6,2) are dominated by (2,2) / (5,1)
    F = np.array([[1, 5], [2, 2], [5, 1], [3, 3], [6, 2], [7, 7]], float)
    fr = nsga2.fronts(F)
    assert sorted(fr[0].tolist()) == [0, 1, 2]
    assert sorted(fr[1].tolist()) == [3, 4] and fr[2].tolist() == [5]
    keep = nsga2.select(F, 4)
    assert set(keep[:3].tolist()) == {0, 1, 2} and len(keep) == 4


def test_nan_ranks_last():
    F = np.array([[1, 1], [np.nan, 0], [2, 2]], float)
    front, _ = nsga2.rank(F)
    assert front[1] == front.max()


def test_crowding_keeps_ends():
    F = np.array([[0, 4], [1, 3], [2, 2], [3, 1], [4, 0]], float)
    d = nsga2.crowding(F)
    assert np.isinf(d[0]) and np.isinf(d[4]) and np.all(np.isfinite(d[1:4]))


def _conv(x, w, b, groups):
    """Direct 2-D convolution, stride 1, 'same' padding. x (C, H, W), w (O, C/groups, k, k)."""
    C, H, W = x.shape
    O, cg, k, _ = w.shape
    p = k // 2
    xp = np.pad(x, ((0, 0), (p, p), (p, p)))
    og = O // groups
    y = np.zeros((O, H, W))
    for o in range(O):
        g = o // og
        for i in range(H):
            for j in range(W):
                y[o, i, j] = (xp[g * cg:(g + 1) * cg, i:i + k, j:j + k] * w[o]).sum()
    return y + b[:, None, None]


def _fold(w, b, gamma, beta, mean, var, eps):
    scale = gamma / np.sqrt(var + eps)
    return w * scale[:, None, None, None], (b - mean) * scale + beta


def test_folding_formula():
    rng = np.random.default_rng(0)
    for groups, cin, cout in ((1, 3, 4), (2, 4, 6), (4, 4, 4)):          # plain, grouped, depthwise
        x = rng.normal(size=(cin, 6, 6))
        w = rng.normal(size=(cout, cin // groups, 3, 3))
        b = np.zeros(cout)
        gamma, beta = rng.normal(size=cout), rng.normal(size=cout)
        mean, var, eps = rng.normal(size=cout), rng.uniform(0.01, 2, cout), 1e-5
        y = _conv(x, w, b, groups)
        bn = (y - mean[:, None, None]) / np.sqrt(var + eps)[:, None, None] * gamma[:, None, None] + beta[:, None, None]
        wf, bf = _fold(w, b, gamma, beta, mean, var, eps)
        assert np.allclose(_conv(x, wf, bf, groups), bn, atol=1e-9)


if __name__ == "__main__":
    from _run import run
    run(dict(globals()))
