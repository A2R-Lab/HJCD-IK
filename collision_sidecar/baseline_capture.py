"""Checkpoint 3, Section 1 -- baseline capture of the collision-DISABLED G1 solver.

Records branch/HEAD/status/extension path/hashes, confirms NUM_JOINTS/NUM_TARGETS, runs the
current solver on fixed targets+seeds, and saves outputs/counts/iters/timings/q arrays so later
integration stages can prove off-mode is numerically identical. Writes generated/baseline_g1_solver.json.
"""
from __future__ import annotations
import hashlib, json, os, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
GEN = os.path.join(HJCD, "generated")
OUT = os.path.join(GEN, "baseline_g1_solver.json")


def sh(*a):
    return subprocess.run(a, cwd=HJCD, capture_output=True, text=True).stdout.strip()


def file_hash(p):
    return hashlib.sha1(open(p, "rb").read()).hexdigest()[:16] if os.path.exists(p) else None


def main():
    import hjcdik
    state = {
        "branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "head": sh("git", "rev-parse", "--short", "HEAD"),
        "extension_path": os.path.abspath(hjcdik.__file__),
        "num_joints": hjcdik.num_joints(),
        "num_targets": hjcdik.num_targets(),
        "num_frames": hjcdik.num_frames(),
        "hashes": {f: file_hash(os.path.join(HJCD, f)) for f in (
            "csrc/generated/grid.cuh", "csrc/kernel/hjcd_kernel.cu",
            "csrc/kernel/hjcd_settings.h", "csrc/bindings/pybind_module.cpp")},
        "grid_collision_baked": "namespace grid_collision" in open(
            os.path.join(HJCD, "csrc/generated/grid.cuh")).read(),
    }
    assert state["num_joints"] == 29, state["num_joints"]
    assert state["num_targets"] == 4, state["num_targets"]

    def mat2quat(R):  # -> [qw,qx,qy,qz]
        t = np.trace(R)
        if t > 0:
            s = np.sqrt(t + 1.0) * 2; w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s; y = (R[0, 2] - R[2, 0]) / s; z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w = (R[2, 1] - R[1, 2]) / s; x = 0.25 * s; y = (R[0, 1] + R[1, 0]) / s; z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w = (R[0, 2] - R[2, 0]) / s; x = (R[0, 1] + R[1, 0]) / s; y = 0.25 * s; z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w = (R[1, 0] - R[0, 1]) / s; x = (R[0, 2] + R[2, 0]) / s; y = (R[1, 2] + R[2, 1]) / s; z = 0.25 * s
        q = np.array([w, x, y, z]); return q / np.linalg.norm(q)

    # reachable multi-target problems: derive 4-frame targets by FK from reference configs
    rng = np.random.default_rng(3)
    qrefs = np.clip(rng.normal(0, 0.35, (4, 29)), -1.2, 1.2)
    T = np.asarray(hjcdik.target_transforms(np.ascontiguousarray(qrefs)))   # [4,4,4,4]
    runs = {}
    for label, B in [("B256", 256), ("B1024", 1024)]:
        cap = []
        for rep in range(2):
            per_prob = []
            t0 = time.perf_counter()
            for pi in range(len(qrefs)):
                tpos1 = T[pi, :, :3, 3]
                tquat1 = np.stack([mat2quat(T[pi, k, :3, :3]) for k in range(4)])
                tpos = np.ascontiguousarray(np.broadcast_to(tpos1, (B, 4, 3)))
                tquat = np.ascontiguousarray(np.broadcast_to(tquat1, (B, 4, 4)))
                # warm-ish reproducible batch: reference config + spread (realistic diverse seeds)
                seed_q = np.ascontiguousarray(qrefs[pi][None] + np.random.default_rng(100 + pi).normal(0, 0.2, (B, 29)))
                out = hjcdik.solve(seed_q, tpos, tquat, active_target_mask=None, seed=42,
                                   precision="float32", position_tol=0.02, orientation_tol=0.1)
                q = np.asarray(out["joint_config"], float)
                succ = np.asarray(out.get("success", [])).astype(bool)
                per_prob.append(dict(
                    prob=pi, n_success=int(succ.sum()) if succ.size else None,
                    q_hash=hashlib.sha1(np.ascontiguousarray(q).tobytes()).hexdigest()[:16],
                    q_shape=list(q.shape)))
            cap.append(dict(rep=rep, wall_s=time.perf_counter() - t0, per_prob=per_prob))
        repro = all(cap[0]["per_prob"][i]["q_hash"] == cap[1]["per_prob"][i]["q_hash"]
                    for i in range(len(qrefs)))
        runs[label] = dict(batch_size=B, reproducible=repro,
                           n_success=[p["n_success"] for p in cap[0]["per_prob"]],
                           q_hashes=[p["q_hash"] for p in cap[0]["per_prob"]],
                           wall_s=[cap[0]["wall_s"], cap[1]["wall_s"]])

    result = dict(state=state, runs=runs, n_problems=len(qrefs))
    json.dump(result, open(OUT, "w"), indent=2)
    print(f"branch {state['branch']} @ {state['head']}  ext {state['extension_path']}")
    print(f"NUM_JOINTS={state['num_joints']} NUM_TARGETS={state['num_targets']} "
          f"grid_collision_baked={state['grid_collision_baked']}")
    for label, r in runs.items():
        print(f"  {label}: n_success={r['n_success']} reproducible={r['reproducible']} "
              f"wall={r['wall_s'][0]*1000:.0f}/{r['wall_s'][1]*1000:.0f} ms")
    print(f"wrote {os.path.relpath(OUT, HJCD)}")
    ok = all(r["reproducible"] for r in runs.values())
    print("BASELINE reproducible:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
