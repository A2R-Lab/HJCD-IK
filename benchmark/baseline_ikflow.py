#!/usr/bin/env python3
"""IKFlow baseline (paper Tables I + IV) — normalizing-flow generative IK.

Standalone (only torch + ikflow; no cuRobo/PyRoki) so it can be installed/run independently. Emits the
same CSV schema as benchmark/baseline_bench.py (solver=ikflow) and the same MMD config dump as
benchmark/run_mmd.py, so its results merge straight into make_tables.py / plot_pareto.py / run_mmd.py.

Open-world only (IKFlow is unconstrained; the paper does not run it collision-free). Targets come from the
shared goal file so it's head-to-head with the other solvers.

  python benchmark/baseline_ikflow.py --goal_file benchmark/targets/panda_open.yml \
      --model panda_full_tpm --seed_list 1,10,100,1000,2000 --csv-out benchmark/results/open_ikflow.csv
  python benchmark/baseline_ikflow.py --goal_file benchmark/targets/panda_open.yml --mmd-dump dumps/ikflow.json

Requires IKFlow (`pip install ikflow`) + a pretrained model. NOTE: pin the IKFlow API/model against the
co-author's setup — `get_ik_solver` / `generate_ik_solutions` names + the model's EE frame (should be the
shared `panda_hand`) are the things to confirm. The error eval uses IKFlow's own robot FK, so it is
self-consistent regardless; the EE-frame check (benchmark/check_ee_frames.py) catches a frame mismatch.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mmd import save_config_dump  # noqa: E402


def _parse_int_list(s):
    return [int(x) for x in s.replace(",", " ").split()]


def _load_targets(path):
    """Shared targets (.yml goal_file or .json filtered-targets) -> (N,7) [x,y,z,qw,qx,qy,qz]."""
    path = Path(path)
    if path.suffix == ".json":
        d = json.loads(path.read_text())
        items = d["targets"] if isinstance(d, dict) and "targets" in d else d
        rows = [list(it["target"]) for it in items]
    else:
        import yaml
        d = yaml.safe_load(path.read_text())
        rows = [list(g["position"]) + list(g["quaternion"]) for g in d["goals"]]
    return np.asarray(rows, dtype=np.float64)


def _pose_errors(fk_pose7, target7):
    """fk_pose7, target7: [x,y,z, qw,qx,qy,qz]. Returns (pos_mm, ori_rad)."""
    pos_mm = float(np.linalg.norm(fk_pose7[:3] - target7[:3]) * 1000.0)
    qd = abs(float(np.dot(fk_pose7[3:7] / np.linalg.norm(fk_pose7[3:7]),
                          target7[3:7] / np.linalg.norm(target7[3:7]))))
    ori_rad = float(2.0 * np.arccos(min(1.0, qd)))
    return pos_mm, ori_rad


def _load_solver(model_name):
    from ikflow.model_loading import get_ik_solver  # noqa: confirm name vs co-author env
    ik_solver, _ = get_ik_solver(model_name)
    return ik_solver


def _solve(ik_solver, target7, n):
    """Return (n, ndof) joint configs for one target pose [x,y,z,qw,qx,qy,qz]."""
    sols = ik_solver.generate_ik_solutions(target7, n, refine_solutions=False)
    return np.asarray(sols.detach().cpu().numpy() if hasattr(sols, "detach") else sols, dtype=np.float64)


def _fk(ik_solver, qs):
    """Batched FK via IKFlow's own robot model -> (n,7) [x,y,z,qw,qx,qy,qz]."""
    return np.asarray(ik_solver.robot.forward_kinematics(qs), dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goal_file", required=True, help="shared targets (.yml or .json)")
    ap.add_argument("--model", default="panda_full_tpm", help="pretrained IKFlow model name")
    ap.add_argument("--seed_list", default="1,10,100,1000,2000", help="batch sizes (n solutions per target)")
    ap.add_argument("--csv-out", default="", help="write per-(solver,batch,target) rows here")
    ap.add_argument("--mmd-dump", default="", help="write K best-of-2000 configs/target (MMD) then exit")
    ap.add_argument("--mmd-batch", type=int, default=2000)
    ap.add_argument("--solutions-count", type=int, default=50)
    args = ap.parse_args()

    targets = _load_targets(args.goal_file)
    ik_solver = _load_solver(args.model)

    # ---- MMD dump: K best (by pose error) of mmd-batch per target ----
    if args.mmd_dump:
        K, Bd = int(args.solutions_count), int(args.mmd_batch)
        _ = _solve(ik_solver, targets[0], Bd)  # warm
        per_target = []
        for t in targets:
            qs = _solve(ik_solver, t, Bd)
            fk = _fk(ik_solver, qs)
            err = np.array([_pose_errors(fk[i], t)[0] for i in range(len(qs))])
            keep = qs[np.argsort(err)[:K]]
            per_target.append([[float(v) for v in q] for q in keep])
        dof = len(per_target[0][0]) if (per_target and per_target[0]) else None
        save_config_dump(args.mmd_dump, "ikflow", per_target, num_dof=dof)
        print(f"[OK] wrote MMD dump {args.mmd_dump} ({len(per_target)} targets x up to {K})")
        return

    # ---- open-world timing/accuracy sweep ----
    data = defaultdict(list)
    for n in _parse_int_list(args.seed_list):
        _ = _solve(ik_solver, targets[0], n)  # warm/compile
        print(f"  ikflow n={n}")
        for ti, t in enumerate(targets):
            t0 = time.perf_counter()
            qs = _solve(ik_solver, t, n)
            dt_ms = (time.perf_counter() - t0) * 1e3
            fk = _fk(ik_solver, qs)
            errs = [_pose_errors(fk[i], t) for i in range(len(qs))]
            best = min(errs, key=lambda e: e[0] + e[1])  # best returned solution
            data["solver"].append("ikflow")
            data["Batch-Size"].append(n)
            data["IK-time(ms)"].append(dt_ms)
            data["Pos-Error(mm)"].append(best[0])
            data["Ori-Error"].append(best[1])

    if args.csv_out:
        cols = ["solver", "Batch-Size", "IK-time(ms)", "Pos-Error(mm)", "Ori-Error"]
        p = Path(args.csv_out); p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(cols)
            for row in zip(*[data[c] for c in cols]):
                w.writerow(row)
        print(f"[OK] wrote {args.csv_out} ({len(data['solver'])} rows)")


if __name__ == "__main__":
    main()
