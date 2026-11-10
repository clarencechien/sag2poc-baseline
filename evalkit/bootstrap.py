"""Paired bootstrap over questions (handoff §5.2): 10,000 resamples, 95% percentile interval."""
import numpy as np


def paired_bootstrap(a, b, qids=None, metric="R@5", n_resamples=10_000, seed=20261110, percent=True):
    """Δ = mean(a) − mean(b) on the same qids, resampling questions with replacement.

    a, b: {qid: {metric: value}}. Returns {"n", "a", "b", "delta", "ci_low", "ci_high", "p_a_gt_b"}.
    """
    qids = [q for q in a if q in b] if qids is None else list(qids)
    missing = [q for q in qids if q not in a or q not in b]
    if missing:
        raise KeyError(f"{len(missing)} question(s) not present in both systems, e.g. {missing[:3]}")
    if not qids:
        raise ValueError("no paired questions")
    x = np.array([a[q][metric] for q in qids], dtype=np.float64)
    y = np.array([b[q][metric] for q in qids], dtype=np.float64)
    diff = x - y
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.empty(n_resamples)
    step = 1000
    for start in range(0, n_resamples, step):
        size = min(step, n_resamples - start)
        idx = rng.integers(0, n, size=(size, n))
        means[start:start + size] = diff[idx].mean(axis=1)
    scale = 100.0 if percent else 1.0
    low, high = np.percentile(means, [2.5, 97.5])
    return {"n": n, "metric": metric, "a": scale * x.mean(), "b": scale * y.mean(), "delta": scale * diff.mean(),
            "ci_low": scale * low, "ci_high": scale * high, "p_a_gt_b": float((means > 0).mean())}
