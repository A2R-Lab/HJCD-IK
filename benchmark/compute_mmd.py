#!/usr/bin/env python3
"""
compute_mmd_ikflow.py

IKFlow-style MMD between two sets of joint-space solutions.

- Compares distributions in JOINT space (not EE pose).
- Kernel: inverse multi-quadric (IMQ): k(x,y) = (||x-y||^2 + c^2)^(-beta), default beta=0.5
- Bandwidths: c = median_pairwise_distance * scale for scale in {0.2, 0.5, 1, 2, 5} (averaged across scales)
- Estimator: unbiased MMD^2 (Gretton et al.)
- Optional grouping by a column (e.g. pose_id) and mean over groups.
"""

import argparse
import re
import numpy as np
import pandas as pd
from typing import Iterable, List, Tuple, Optional

# ------------------------ column handling ------------------------

_QNUM_RE = re.compile(r"^q([0-9]+)$", re.I)
_NONJOINT_DEFAULTS = {
    "solver", "sample_id", "time_ms", "pos_err_mm", "ori_err_rad",
    "x", "y", "z", "qw", "qx", "qy", "qz",
}

def _is_numeric_like(s: str) -> bool:
    try:
        float(str(s).strip())
        return True
    except Exception:
        return False

def _q_index(c: str) -> Optional[int]:
    m = _QNUM_RE.match(c)
    return int(m.group(1)) if m else None

def _longest_contiguous(start: int, idxs: List[int]) -> List[int]:
    run = []
    expect = start
    for v in idxs:
        if v == expect:
            run.append(v); expect += 1
        elif v > expect:
            break
    return run

def _find_joint_cols(df: pd.DataFrame, n_joints: Optional[int]) -> List[str]:
    qnum = sorted([c for c in df.columns if _QNUM_RE.match(c)], key=lambda c: _q_index(c))
    if not qnum:
        # fall back to numeric columns minus known non-joints
        num = df.select_dtypes(include=[np.number]).drop(columns=list(_NONJOINT_DEFAULTS), errors="ignore").columns.tolist()
        return num

    idxs = [_q_index(c) for c in qnum]
    run1 = _longest_contiguous(1, idxs)
    run0 = _longest_contiguous(0, idxs)
    chosen = run1 if len(run1) >= len(run0) else run0
    if not chosen:
        chosen = idxs

    if n_joints is not None:
        chosen = chosen[:n_joints]

    cols = [f"q{i}" for i in chosen]
    # ensure they exist (they should, since they came from qnum)
    return [c for c in cols if c in df.columns]

def _select_columns(df: pd.DataFrame, cols: Optional[List[str]], n_joints: Optional[int]) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Return (feature_df, used_cols, available_qcols)."""
    qnum_all = sorted([c for c in df.columns if _QNUM_RE.match(c)], key=lambda c: _q_index(c))

    # 1) Explicit user columns
    if cols:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            # try remap q1..qN -> q0..q{N-1}
            zero_try, ok = [], True
            for c in cols:
                m = _QNUM_RE.match(c)
                if not m:
                    ok = False; break
                idx = int(m.group(1)) - 1
                alt = f"q{idx}"
                if alt not in df.columns:
                    ok = False; break
                zero_try.append(alt)
            if ok:
                used = zero_try
            else:
                # soft-trim if at least 2 available
                existing = [c for c in cols if c in df.columns]
                if len(existing) >= 2:
                    print(f"[note] Missing {missing}; proceeding with {existing}")
                    used = existing
                else:
                    raise ValueError(f"Columns not found: {missing}")
        else:
            used = cols
        return df[used].astype(float), used, qnum_all

    # 2) n_joints hint
    if n_joints is not None:
        used = _find_joint_cols(df, n_joints)
        if len(used) < 2:
            raise ValueError(f"Found too few joint columns for --njoints {n_joints}. "
                             f"Available numbered joint columns: {qnum_all}")
        if len(used) < n_joints:
            print(f"[note] --njoints {n_joints} requested but only {len(used)} contiguous joint columns found: {used}")
        return df[used].astype(float), used, qnum_all

    # 3) Auto: numbered q-columns if present, else numeric columns minus known non-joints
    if qnum_all:
        return df[qnum_all].astype(float), qnum_all, qnum_all

    num = df.select_dtypes(include=[np.number]).drop(columns=list(_NONJOINT_DEFAULTS), errors="ignore")
    if num.shape[1] == 0:
        raise ValueError("No numeric joint columns found.")
    used = list(num.columns)
    return num.astype(float), used, qnum_all

def _read_csv_flexible(path: str) -> pd.DataFrame:
    """
    Read CSV; if columns look like a misinterpreted headerless first row,
    re-read with header=None and assign q1..qN.
    """
    df = pd.read_csv(path)

    # If we already have q-columns, we're done
    if any(_QNUM_RE.match(c) for c in df.columns):
        return df

    # If *all* column names look numeric-like (or Unnamed), it's likely headerless
    colnames = list(df.columns)
    looks_headerless = all(_is_numeric_like(c) or str(c).startswith("Unnamed") for c in colnames)

    if looks_headerless:
        df2 = pd.read_csv(path, header=None)
        n = df2.shape[1]
        df2.columns = [f"q{i+1}" for i in range(n)]
        print(f"[note] Detected headerless CSV at '{path}'. Assigned columns: {list(df2.columns)}")
        return df2

    return df

def _load_and_select(path: str, cols: Optional[List[str]], n_joints: Optional[int]):
    df = _read_csv_flexible(path)
    feats, used_cols, avail_q = _select_columns(df, cols, n_joints)
    return df, feats, used_cols, avail_q

# ------------------------ MMD core ------------------------

def _pairwise_sq_dists(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    X2 = np.sum(X * X, axis=1, keepdims=True)
    Y2 = np.sum(Y * Y, axis=1, keepdims=True).T
    D2 = X2 + Y2 - 2.0 * (X @ Y.T)
    np.maximum(D2, 0.0, out=D2)
    return D2

def _median_pairwise_dist(X: np.ndarray, Y: np.ndarray) -> float:
    Z = np.vstack([X, Y])
    n = Z.shape[0]
    D2 = _pairwise_sq_dists(Z, Z)
    iu = np.triu_indices(n, k=1)
    vals = D2[iu]
    if vals.size == 0:
        return 1.0
    med = np.sqrt(np.median(vals))
    if not np.isfinite(med) or med <= 0:
        med = 1.0
    return float(med)

def _imq_kernel_multi(X: np.ndarray, Y: np.ndarray, c_list: Iterable[float], beta: float) -> np.ndarray:
    D2 = _pairwise_sq_dists(X, Y)
    K = np.zeros_like(D2)
    count = 0
    for c in c_list:
        c2 = float(c) ** 2
        K += np.power(D2 + c2, -beta)
        count += 1
    if count > 1:
        K /= count
    return K

def _unbiased_mmd2(Kxx: np.ndarray, Kyy: np.ndarray, Kxy: np.ndarray) -> float:
    m = Kxx.shape[0]
    n = Kyy.shape[0]
    if m < 2 or n < 2:
        return float("nan")
    sum_xx = (np.sum(Kxx) - np.sum(np.diag(Kxx))) / (m * (m - 1))
    sum_yy = (np.sum(Kyy) - np.sum(np.diag(Kyy))) / (n * (n - 1))
    sum_xy = np.sum(Kxy) / (m * n)
    mmd2 = sum_xx + sum_yy - 2.0 * sum_xy
    return float(max(mmd2, 0.0))

def compute_mmd_ikflow(
    X: np.ndarray,
    Y: np.ndarray,
    beta: float = 0.5,
    scales: Iterable[float] = (0.2, 0.5, 1.0, 2.0, 5.0),
) -> Tuple[float, float]:
    med = _median_pairwise_dist(X, Y)
    c_list = [max(med * s, 1e-12) for s in scales]
    Kxx = _imq_kernel_multi(X, X, c_list, beta)
    Kyy = _imq_kernel_multi(Y, Y, c_list, beta)
    Kxy = _imq_kernel_multi(X, Y, c_list, beta)
    mmd2 = _unbiased_mmd2(Kxx, Kyy, Kxy)
    return mmd2, float(np.sqrt(mmd2) if np.isfinite(mmd2) and mmd2 >= 0 else np.nan)

# ------------------------ Runner ------------------------

def run(
    ref_path: str,
    cmp_path: str,
    cols: Optional[List[str]],
    n_joints: Optional[int],
    group_col: Optional[str],
    beta: float,
    scales: List[float],
    save_csv: Optional[str],
) -> None:
    df_ref, Xdf_ref, used_ref, qall_ref = _load_and_select(ref_path, cols, n_joints)
    df_cmp, Xdf_cmp, used_cmp, qall_cmp = _load_and_select(cmp_path, cols, n_joints)

    # Align on common joint columns if needed
    if list(used_ref) != list(used_cmp):
        ref_set = set(used_ref)
        cmp_set = set(used_cmp)
        common = sorted(ref_set & cmp_set, key=lambda c: _q_index(c) if _QNUM_RE.match(c) else c)
        if n_joints is not None:
            common = common[:n_joints]
        if len(common) >= 2:
            if list(common) != list(used_ref) or list(common) != list(used_cmp):
                print(f"[note] Aligning to common joint columns: {common}")
            Xdf_ref = df_ref[common].astype(float)
            Xdf_cmp = df_cmp[common].astype(float)
        else:
            raise ValueError(
                "Could not find ≥2 common joint columns between files.\n"
                f"ref joints: {used_ref}\n"
                f"cmp joints: {used_cmp}"
            )

    if group_col and group_col in df_ref.columns and group_col in df_cmp.columns:
        ids_ref = set(df_ref[group_col].unique())
        ids_cmp = set(df_cmp[group_col].unique())
        common_ids = sorted(list(ids_ref & ids_cmp))
        if not common_ids:
            raise ValueError(f"No overlapping '{group_col}' between files.")

        rows = []
        for gid in common_ids:
            Xg = Xdf_ref[df_ref[group_col] == gid].to_numpy(dtype=float)
            Yg = Xdf_cmp[df_cmp[group_col] == gid].to_numpy(dtype=float)
            if len(Xg) < 2 or len(Yg) < 2:
                m2, m = float("nan"), float("nan")
            else:
                m2, m = compute_mmd_ikflow(Xg, Yg, beta=beta, scales=scales)
            rows.append((gid, m2, m))

        out = pd.DataFrame(rows, columns=[group_col, "mmd2", "mmd"])
        mean_mmd2 = out["mmd2"].mean(skipna=True)
        mean_mmd = np.sqrt(mean_mmd2) if np.isfinite(mean_mmd2) and mean_mmd2 >= 0 else np.nan

        print(f"# Groups compared: {len(out)}")
        print(out.to_string(index=False))
        print("\n# IKFlow-style average (mean of MMD^2 over groups):")
        print(f"MMD^2_mean = {mean_mmd2:.6g}")
        print(f"MMD_mean   = {mean_mmd:.6g}")

        if save_csv:
            out.to_csv(save_csv, index=False)
            print(f"\nPer-group results saved to: {save_csv}")
    else:
        X = Xdf_ref.to_numpy(dtype=float)
        Y = Xdf_cmp.to_numpy(dtype=float)
        if X.shape[1] != Y.shape[1]:
            raise ValueError(
                f"Dim mismatch after alignment: {X.shape[1]} vs {Y.shape[1]}.\n"
                f"ref cols: {list(Xdf_ref.columns)}\n"
                f"cmp cols: {list(Xdf_cmp.columns)}"
            )
        if X.shape[0] < 2 or Y.shape[0] < 2:
            raise ValueError("Need at least 2 samples in each set for unbiased MMD.")

        mmd2, mmd = compute_mmd_ikflow(X, Y, beta=beta, scales=scales)
        print(f"MMD^2 = {mmd2:.6g}")
        print(f"MMD   = {mmd:.6g}")

def parse_args():
    ap = argparse.ArgumentParser(description="IKFlow-style MMD (IMQ kernel) between two joint-space solution sets.")
    ap.add_argument("--ref", required=True, help="Reference CSV (e.g., TRAC-IK / GT samples)")
    ap.add_argument("--cmp", required=True, help="Comparison CSV (e.g., IKFlow / your solver samples)")
    ap.add_argument("--cols", nargs="+", default=None, help="Explicit joint columns (e.g. q1 q2 ...)")
    ap.add_argument("--njoints", type=int, default=None, help="Use up to N from a contiguous run (q1.. or q0..)")
    ap.add_argument("--group_col", default=None, help="Optional group column to compute per-group MMD then average")
    ap.add_argument("--beta", type=float, default=0.5, help="IMQ exponent")
    ap.add_argument("--scales", type=str, default="0.2,0.5,1,2,5", help="Comma-separated scale list for c = median_dist*scale")
    ap.add_argument("--save_csv", default=None, help="If set, save per-group results here")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    scales = [float(s.strip()) for s in args.scales.split(",") if s.strip()]
    run(
        ref_path=args.ref,
        cmp_path=args.cmp,
        cols=args.cols,
        n_joints=args.njoints,
        group_col=args.group_col,
        beta=args.beta,
        scales=scales,
        save_csv=args.save_csv,
    )
