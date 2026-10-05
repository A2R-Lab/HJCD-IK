"""One record per attempted query, including empty results; shared by scoring and timing."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

POS_OK_MM, ORI_OK_RAD = 5.0, 0.05


def best_candidate(result):
    """Prefer a jointly accurate candidate, then minimum position/orientation error."""
    n = int(result["count"])
    if n == 0:
        return None
    pe, oe = np.asarray(result["pos_errors"]), np.asarray(result["ori_errors"])
    if pe.shape != (n,) or oe.shape != (n,) or not np.isfinite(pe).all() or not np.isfinite(oe).all():
        raise ValueError("invalid candidate error arrays")
    return min(range(n), key=lambda i: (not (pe[i] < POS_OK_MM and oe[i] < ORI_OK_RAD), pe[i], oe[i]))


def query_record(result, *, solver, problem_set, problem_idx, batch, target, ee_target, elapsed_ms):
    i = best_candidate(result)
    return dict(schema_version=1, solver=solver, problem_set=problem_set,
                problem_idx=int(problem_idx), batch=int(batch), count=int(result["count"]),
                status="returned" if i is not None else "empty", target=list(map(float, target)),
                ee_target=ee_target, time_ms=None if elapsed_ms is None else float(elapsed_ms),
                q=None if i is None else np.asarray(result["joint_config"][i], float).tolist(),
                pos_err_mm=None if i is None else float(result["pos_errors"][i]),
                ori_err_rad=None if i is None else float(result["ori_errors"][i]))


def append_record(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, allow_nan=False) + "\n")


def validate_groups(records, problems):
    """Reject truncated, duplicated, or mixed-protocol groups instead of inflating rates."""
    from collections import defaultdict
    groups = defaultdict(list)
    seen = set()
    for r in records:
        key = (r["problem_set"], r["solver"], int(r["batch"]))
        idx = int(r["problem_idx"])
        if (*key, idx) in seen:
            raise ValueError(f"duplicate query: {(*key, idx)}")
        seen.add((*key, idx))
        if key[0] not in problems or not 0 <= idx < len(problems[key[0]]):
            raise ValueError(f"unknown problem: {key[0]}[{idx}]")
        groups[key].append(r)
    if not groups:
        raise ValueError("no query records")
    solvers = {key[1] for key in groups}
    batches = {key[2] for key in groups}
    for pset in {key[0] for key in groups}:
        for solver in solvers:
            for batch in batches:
                if (pset, solver, batch) not in groups:
                    raise ValueError(f"missing query group: {pset}, {solver}, {batch}")
    for key, rows in groups.items():
        if len(rows) != len(problems[key[0]]):
            raise ValueError(f"incomplete query group {key}: {len(rows)} / {len(problems[key[0]])}")
        frames = {r.get("ee_target") for r in rows}
        if len(frames) != 1:
            raise ValueError(f"mixed target frames in {key}")
    return groups


def pose_errors(q, target, ee_target="panda_hand_joint"):
    """Independent URDF FK, with position in mm and SO(3) geodesic angle in radians."""
    from gen_targets import _fk
    from collision_check import quat_wxyz_to_rot
    chain = _pose_chain(ee_target)
    q = np.asarray(q, dtype=float)
    target = np.asarray(target, dtype=float)
    if q.shape != (7,) or target.shape != (7,) or not np.isfinite(q).all() or not np.isfinite(target).all():
        raise ValueError("expected finite Panda q[7] and target[7]")
    T = _fk(chain, q)
    pe = float(np.linalg.norm(T[:3, 3] - target[:3]) * 1000)
    R = quat_wxyz_to_rot(target[3:])
    oe = float(np.arccos(np.clip((np.trace(R.T @ T[:3, :3]) - 1) / 2, -1, 1)))
    return pe, oe


@lru_cache(maxsize=4)
def _pose_chain(ee_target):
    from gen_targets import _parse_joints, _chain_to_target
    return _chain_to_target(_parse_joints(Path(__file__).resolve().parents[1] / "csrc/urdf/panda.urdf"), ee_target)


def panda_hand_target(pose, ee_target):
    """Express the same requested hand pose as a named compiled Panda tool pose."""
    from gen_targets import _fk, _quat_from_R
    from collision_check import quat_wxyz_to_rot
    if ee_target not in ("panda_hand_joint", "panda_grasptarget_hand"):
        raise ValueError("expected a Panda hand or grasptarget build")
    hand = _fk(_pose_chain("panda_hand_joint"), np.zeros(7))
    tool = _fk(_pose_chain(ee_target), np.zeros(7))
    world = np.eye(4)
    world[:3, :3] = quat_wxyz_to_rot(pose[3:])
    world[:3, 3] = pose[:3]
    target = world @ np.linalg.inv(hand) @ tool
    return [*target[:3, 3], *_quat_from_R(target[:3, :3])]
