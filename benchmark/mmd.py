"""Maximum Mean Discrepancy (MMD / MMD²) between joint-configuration distributions — paper Table IV.

MMD measures how close a solver's returned batch of IK solutions is to a ground-truth distribution of
solutions (paper: TRAC-IK seeded samples). Lower = the solver covers the solution manifold better.

Gaussian kernel  k(a,b) = exp(-||a-b||² / (2σ²)).  Biased V-statistic estimate:
    MMD²(X,Y) = mean(k(X,X)) + mean(k(Y,Y)) − 2·mean(k(X,Y))
σ via the median heuristic over pooled pairwise distances (shared across targets for stability).

NOTE (confirm vs paper for an exact-number match): kernel choice + bandwidth fix the absolute MMD. The
median-heuristic Gaussian here is the standard default; the *ranking* (which solver is lowest) is robust
to this, which is the claim in Table IV. Stdlib + numpy only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


# ---- config-dump I/O (shared by the harnesses + run_mmd.py) ----
def save_config_dump(path, solver, configs_per_target, num_dof=None):
    """Write a config dump: per-target lists of K joint vectors. See run_mmd.py for the schema."""
    data = {
        "solver": str(solver),
        "num_dof": int(num_dof) if num_dof is not None else None,
        "configs": [[[float(v) for v in q] for q in tgt] for tgt in configs_per_target],
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def load_config_dump(path):
    return json.loads(Path(path).read_text())


def _sq_dists(A, B):
    """Pairwise squared Euclidean distances, A:(n,d) B:(m,d) -> (n,m)."""
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    return (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T


def median_sigma(samples):
    """Median-heuristic bandwidth σ from a pool of points (n,d)."""
    Z = np.asarray(samples, float)
    if len(Z) < 2:
        return 1.0
    d2 = _sq_dists(Z, Z)
    iu = np.triu_indices(len(Z), k=1)
    med = float(np.median(d2[iu]))
    return float(np.sqrt(med / 2.0)) if med > 0 else 1.0


def mmd2_gaussian(X, Y, sigma):
    """Biased MMD² (V-statistic) with a Gaussian kernel of bandwidth σ."""
    X = np.asarray(X, float)
    Y = np.asarray(Y, float)
    if len(X) == 0 or len(Y) == 0:
        return float("nan")
    g = 1.0 / (2.0 * sigma * sigma)
    kxx = np.exp(-g * _sq_dists(X, X)).mean()
    kyy = np.exp(-g * _sq_dists(Y, Y)).mean()
    kxy = np.exp(-g * _sq_dists(X, Y)).mean()
    return float(kxx + kyy - 2.0 * kxy)


def mmd_over_targets(solver_configs, gt_configs, sigma=None):
    """Mean MMD² / MMD over matched per-target config sets.

    solver_configs, gt_configs: lists (one entry per target) of (k, dof) joint-config arrays.
    Returns dict {mmd2, mmd, n_targets, sigma}. σ is shared across targets (median heuristic over the
    pooled solver+gt points) unless given.
    """
    pairs = [(np.asarray(s, float), np.asarray(g, float))
             for s, g in zip(solver_configs, gt_configs)
             if len(s) and len(g)]
    if not pairs:
        return dict(mmd2=float("nan"), mmd=float("nan"), n_targets=0, sigma=float("nan"))

    if sigma is None:
        pool = np.vstack([np.vstack([s, g]) for s, g in pairs])
        sigma = median_sigma(pool)

    vals = [mmd2_gaussian(s, g, sigma) for s, g in pairs]
    vals = [v for v in vals if np.isfinite(v)]
    mmd2 = float(np.mean(vals)) if vals else float("nan")
    return dict(mmd2=mmd2, mmd=float(np.sqrt(max(mmd2, 0.0))), n_targets=len(vals), sigma=float(sigma))
