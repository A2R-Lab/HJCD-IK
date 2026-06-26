#!/usr/bin/env python3
"""Generate the TRAC-IK ground-truth config dump for MMD / Table IV.

For each shared target pose, solve IK from many random seeds with TRAC-IK and keep the distinct
solutions — the "randomly seeded ground-truth samples" the paper compares solver batches against
(lower MMD = a solver's returned batch better matches this manifold). Writes the config-dump schema
in benchmark/mmd.py, so benchmark/run_mmd.py consumes it directly.

Needs TRAC-IK python bindings (`pip install tracikpy`) + the URDF. Tip link should be the shared EE
frame used everywhere else (default `panda_hand`).

  python benchmark/gen_groundtruth_tracik.py --targets benchmark/targets/panda_open.json \
      --urdf csrc/urdf/panda.urdf --base panda_link0 --tip panda_hand \
      --num-samples 50 --out benchmark/results/dumps/groundtruth.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmd import save_config_dump  # noqa: E402


def _wxyz_to_R(w, x, y, z):
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = (v / n for v in (w, x, y, z))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def _load_targets(path):
    """Read the shared targets (.json filtered-targets or .yml goal_file) -> list of [x,y,z,qw,qx,qy,qz]."""
    path = Path(path)
    if path.suffix == ".json":
        d = json.loads(path.read_text())
        items = d["targets"] if isinstance(d, dict) and "targets" in d else d
        return [list(it["target"]) for it in items]
    # .yml goal_file: {goals: [{position:[3], quaternion:[wxyz]}]}
    import yaml
    d = yaml.safe_load(path.read_text())
    return [list(g["position"]) + list(g["quaternion"]) for g in d["goals"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--targets", required=True, help="shared targets file (.json or .yml)")
    ap.add_argument("--urdf", default=str(root / "csrc" / "urdf" / "panda.urdf"))
    ap.add_argument("--base", default="panda_link0")
    ap.add_argument("--tip", default="panda_hand", help="EE frame; keep == the shared-target frame")
    ap.add_argument("--num-samples", type=int, default=50, help="distinct solutions to keep per target")
    ap.add_argument("--num-seeds", type=int, default=400, help="max random seeds attempted per target")
    ap.add_argument("--timeout", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(root / "benchmark" / "results" / "dumps" / "groundtruth.json"))
    args = ap.parse_args()

    try:
        from tracikpy import TracIKSolver
    except ImportError:
        sys.exit("TRAC-IK not installed. `pip install tracikpy` (or via scripts/setup/install_baselines.sh).")

    targets = _load_targets(args.targets)
    solver = TracIKSolver(args.urdf, args.base, args.tip, timeout=args.timeout, solve_type="Distance")
    lo, hi = np.asarray(solver.joint_limits[0]), np.asarray(solver.joint_limits[1])
    dof = int(solver.number_of_joints)
    rng = np.random.default_rng(args.seed)

    per_target = []
    for tgt in targets:
        pose = np.eye(4)
        pose[:3, 3] = tgt[:3]
        pose[:3, :3] = _wxyz_to_R(*tgt[3:7])
        sols = []
        for _ in range(args.num_seeds):
            q = solver.ik(pose, qinit=lo + rng.random(dof) * (hi - lo))
            if q is not None:
                sols.append([float(v) for v in q])
                if len(sols) >= args.num_samples:
                    break
        per_target.append(sols)

    save_config_dump(args.out, "groundtruth", per_target, num_dof=dof)
    got = [len(s) for s in per_target]
    print(f"[OK] wrote {args.out}: {len(per_target)} targets, "
          f"{min(got) if got else 0}-{max(got) if got else 0} sols/target (target {args.num_samples})")


if __name__ == "__main__":
    main()
