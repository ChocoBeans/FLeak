"""Z-score standardization of the node splits, using training data only (no leakage).

Both functions take and return the layout of `utils.preprocess.make_splits`:
test = (X, y, node_per_sample), nodes = {node: {"train": (X, y), "val": (X, y)}}.
"""
import numpy as np


def _zscore(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)


def standardize_pooled(test, nodes):
    """Centralized setting: z-score every split with mean/std of the pooled training data."""
    Xtr = np.vstack([d["train"][0] for d in nodes.values()])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
    nodes = {s: {k: (_zscore(Xs, mu, sd), ys) for k, (Xs, ys) in d.items()} for s, d in nodes.items()}
    return (_zscore(test[0], mu, sd), *test[1:]), nodes


def standardize_local(test, nodes):
    """Federated setting: each node z-scores with the mean/std of its own training data.

    Each test patient is scaled with the statistics of their own cohort's node, as if the
    global model were evaluated at that site.
    """
    out, Xte = {}, test[0].copy()
    for s, d in nodes.items():
        Xtr = d["train"][0]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-8
        out[s] = {k: (_zscore(Xs, mu, sd), ys) for k, (Xs, ys) in d.items()}
        m = test[2] == s
        Xte[m] = _zscore(test[0][m], mu, sd)
    return (Xte.astype(np.float32), *test[1:]), out
