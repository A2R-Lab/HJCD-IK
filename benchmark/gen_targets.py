#!/usr/bin/env python3
"""Generate a shared, solver-independent set of open-world IK targets.

A *neutral* Halton sampler in joint space + a pure-numpy URDF forward kinematics to the named
end-effector frame (default `panda_hand`, matching the baseline scripts' open-world EE) produce a
fixed set of target poses. The SAME targets are emitted in both harness formats so HJCD-IK and the baselines
solve identical poses (fair, head-to-head open-world comparison):

  * HJCD-IK  :  python benchmark/hjcd_ik_bench.py --skip-grid-codegen --filtered-targets <out>.json
  * baselines:  python benchmark/baseline_bench.py --mode <pyroki|curobo> --goal_file <out>.yml

Core deps only (numpy + stdlib xml/json) — no GPU and no PyRoki/cuRobo/JAX needed to regenerate
targets. The numpy FK here is the same independent reference validated in tests/test_fk_equivalence.py
(<0.1 mm / <1e-3 quat vs the GRiD kernel), so the emitted grasptarget poses match HJCD-IK's EE frame.

Halton spec (co-author): one prime per joint dimension — Panda {2,3,5,7,11,13,17} — with random
scrambling, across the FULL joint limits. Implemented here as a seeded digit-scramble (see `_halton`);
`--no-scramble` reverts to raw Halton. No self-collision/reachability filter is applied (a fair shared
set, not necessarily the paper's exact 100 poses — that would also need their seed).
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# Enough primes for high-DoF variants (24-DoF Panda etc.).
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
           59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113]


def _rpy_to_R(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    return Rz @ Ry @ Rx


def _axis_R(axis, q):
    a = axis / np.linalg.norm(axis)
    c, s = np.cos(q), np.sin(q)
    x, y, z = a
    return np.array([
        [c + x * x * (1 - c), x * y * (1 - c) - z * s, x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s, c + y * y * (1 - c), y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s, z * y * (1 - c) + x * s, c + z * z * (1 - c)],
    ])


def _quat_from_R(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)  # wxyz


def _parse_joints(urdf_path):
    root = ET.parse(urdf_path).getroot()
    joints = {}
    for j in root.findall("joint"):
        o, a, lim = j.find("origin"), j.find("axis"), j.find("limit")
        xyz = np.array([float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()])
        rpy = np.array([float(v) for v in (o.get("rpy") if o is not None else "0 0 0").split()])
        ax = np.array([float(v) for v in a.get("xyz").split()]) if a is not None else None
        lower = float(lim.get("lower")) if (lim is not None and lim.get("lower") is not None) else None
        upper = float(lim.get("upper")) if (lim is not None and lim.get("upper") is not None) else None
        joints[j.get("name")] = dict(type=j.get("type"), parent=j.find("parent").get("link"),
                                     child=j.find("child").get("link"), xyz=xyz, rpy=rpy, axis=ax,
                                     lower=lower, upper=upper)
    return joints


def _chain_to_target(joints, target_name):
    """Ordered base->target list of joints reaching the named fixed target frame."""
    by_child = {v["child"]: (k, v) for k, v in joints.items()}
    if target_name not in joints:
        cands = [k for k in joints if target_name in k]
        if len(cands) != 1:
            raise SystemExit(f"target '{target_name}' not found uniquely in URDF; candidates={cands}")
        target_name = cands[0]
    chain = []
    cur = joints[target_name]["child"]
    while cur in by_child:
        name, v = by_child[cur]
        chain.append(v)
        cur = v["parent"]
    chain.reverse()
    return chain


def _actuated_limits(chain):
    """(lower, upper) arrays for the actuated joints along the chain, in q order."""
    lo, hi = [], []
    for v in chain:
        if v["type"] in ("revolute", "prismatic"):
            l, u = v["lower"], v["upper"]
            if l is None or u is None:           # continuous / unlimited
                l, u = -np.pi, np.pi
            lo.append(l); hi.append(u)
        elif v["type"] == "continuous":
            lo.append(-np.pi); hi.append(np.pi)
    return np.array(lo), np.array(hi)


def _fk(chain, q):
    """World 4x4 transform of the target frame for actuated config q (along the chain order)."""
    T = np.eye(4)
    qi = 0
    for v in chain:
        Tj = np.eye(4)
        Tj[:3, :3] = _rpy_to_R(*v["rpy"])
        Tj[:3, 3] = v["xyz"]
        if v["type"] in ("revolute", "continuous"):
            Rj = np.eye(4); Rj[:3, :3] = _axis_R(v["axis"], q[qi]); qi += 1
            Tj = Tj @ Rj
        elif v["type"] == "prismatic":
            d = np.eye(4); d[:3, 3] = v["axis"] * q[qi]; qi += 1
            Tj = Tj @ d
        T = T @ Tj
    return T


def _halton(n, dim, skip, scramble=True, seed=0):
    """Scrambled Halton: one prime per joint dim (co-author spec: Panda {2,3,5,7,11,13,17}).

    Scrambling = a seeded random permutation of the digit alphabet {0..base-1} per dimension applied to
    every digit (random linear / Faure-Tezuka-style digit scramble) — breaks the well-known correlation
    artifacts of raw van-der-Corput while staying low-discrepancy and fully reproducible from `seed`.
    """
    if dim > len(_PRIMES):
        raise SystemExit(f"need {dim} Halton dims but only {len(_PRIMES)} primes available")
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(_PRIMES[d]) if scramble else np.arange(_PRIMES[d]) for d in range(dim)]
    pts = np.empty((n, dim))
    for k in range(n):
        i = k + skip + 1                          # skip index 0 (always the origin)
        for d in range(dim):
            base, perm, f, r, ii = _PRIMES[d], perms[d], 1.0, 0.0, i
            while ii > 0:
                f /= base
                r += f * perm[ii % base]          # scramble the digit
                ii //= base
            pts[k, d] = r
    return pts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--urdf", type=str, default=str(root / "include" / "test_urdf" / "panda.urdf"))
    ap.add_argument("--target", type=str, default="panda_hand_joint",
                    help="EE fixed-frame/joint name to FK to (the common EE frame for all solvers). "
                         "Default panda_hand_joint = the `panda_hand` link, matching the baselines' "
                         "open-world EE (PyRoki ik_beam_hand). Use panda_grasptarget_hand for the TCP frame.")
    ap.add_argument("--num-targets", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0,
                    help="seed for the Halton digit-scramble (deterministic).")
    ap.add_argument("--skip", type=int, default=0, help="Halton start-index offset.")
    ap.add_argument("--no-scramble", action="store_true",
                    help="disable random digit-scrambling (raw Halton; not recommended).")
    ap.add_argument("--out", type=str, default=str(root / "benchmark" / "targets" / "panda_open"),
                    help="Output path prefix; writes <out>.json (HJCD) and <out>.yml (baselines).")
    args = ap.parse_args()

    joints = _parse_joints(args.urdf)
    chain = _chain_to_target(joints, args.target)
    lo, hi = _actuated_limits(chain)
    dof = len(lo)
    print(f"[gen_targets] {args.urdf} -> '{args.target}': {dof} actuated joints on chain")

    u = _halton(args.num_targets, dof, args.skip, scramble=not args.no_scramble, seed=args.seed)
    configs = lo + u * (hi - lo)

    targets = []   # [x,y,z, qw,qx,qy,qz]
    for q in configs:
        T = _fk(chain, q)
        pos = T[:3, 3]
        quat = _quat_from_R(T[:3, :3])   # wxyz
        targets.append([float(pos[0]), float(pos[1]), float(pos[2]),
                        float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # HJCD-IK format: --filtered-targets JSON
    json_path = out.with_suffix(".json")
    json_path.write_text(json.dumps(
        {"targets": [{"problem_idx": i, "target": t} for i, t in enumerate(targets)]}, indent=2))

    # baseline format: --goal_file YAML  ({goals: [{position:[xyz], quaternion:[wxyz]}]})
    yml_path = out.with_suffix(".yml")
    lines = ["goals:"]
    for t in targets:
        lines.append(f"  - position: [{t[0]!r}, {t[1]!r}, {t[2]!r}]")
        lines.append(f"    quaternion: [{t[3]!r}, {t[4]!r}, {t[5]!r}, {t[6]!r}]")
    yml_path.write_text("\n".join(lines) + "\n")

    print(f"[gen_targets] wrote {len(targets)} targets:\n  {json_path}\n  {yml_path}")


if __name__ == "__main__":
    main()
