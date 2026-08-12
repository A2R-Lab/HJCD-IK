"""Checkpoint 3C A/B benchmark: self_collision_mode off vs final (Checkpoint 3).

Identical assignments+seeds, batch sizes {1,10,100,500,1000,2000}. Reports solve time, final
collision-kernel time, overhead, IK yield before/after, native-collision-free count, and MuJoCo
agreement (crucially: 0 native-free candidates that MuJoCo finds colliding == 0 false negatives).
Confirms free candidate q values are byte-identical between off and final.
"""
from __future__ import annotations
import os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
SC = os.path.join(HJCD, "collision_sidecar")
sys.path.insert(0, SC)
import hjcdik  # noqa: E402


def mat2quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    if i == 0:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif i == 1:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q); return q / np.linalg.norm(q)


def problem(B):
    rng = np.random.default_rng(3)
    qref = np.clip(rng.normal(0, 0.35, 29), -1.2, 1.2)
    T = np.asarray(hjcdik.target_transforms(qref[None]))[0]
    tpos = np.ascontiguousarray(np.broadcast_to(T[:, :3, 3], (B, 4, 3)))
    tquat = np.ascontiguousarray(np.broadcast_to(np.stack([mat2quat(T[k, :3, :3]) for k in range(4)]), (B, 4, 4)))
    seed_q = np.ascontiguousarray(qref[None] + np.random.default_rng(101).normal(0, 0.2, (B, 29)))
    return seed_q, tpos, tquat


def timed(fn, reps=5):
    fn(); ts = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); ts.append(time.perf_counter() - t)
    return float(np.median(ts))


def main():
    try:
        from corpus import MujocoOracle
        oracle = MujocoOracle()
    except Exception as e:
        oracle = None
        print(f"(MuJoCo oracle unavailable: {e})")

    kw = dict(seed=42, precision="float32", position_tol=0.02, orientation_tol=0.1)
    print(f"{'B':>5} {'off_ms':>8} {'final_ms':>9} {'coll_ms':>8} {'ovhd%':>6} "
          f"{'off_ok':>7} {'fin_ok':>7} {'rej':>6} {'nat_free':>8} {'mj_chk':>6} {'mj_FN':>5}")
    for B in (1, 10, 100, 500, 1000, 2000):
        sq, tp, tq = problem(B)
        off = hjcdik.solve(sq, tp, tq, self_collision_mode="off", **kw)
        fin = hjcdik.solve(sq, tp, tq, self_collision_mode="final", **kw)
        t_off = timed(lambda: hjcdik.solve(sq, tp, tq, self_collision_mode="off", **kw))
        t_fin = timed(lambda: hjcdik.solve(sq, tp, tq, self_collision_mode="final", **kw))
        coll_ms = fin["self_collision"]["kernel_ms"]
        off_ok = int(np.asarray(off["success"]).sum())
        fin_ok = int(np.asarray(fin["success"]).sum())
        rej = fin["self_collision"]["candidates_rejected"]
        q = np.ascontiguousarray(np.asarray(fin["joint_config"], np.float32))
        nat_free = int((~np.asarray(hjcdik._hjcdik.sidecar_full_check(q, 0.0)).any(axis=1)).sum())
        # free q values byte-identical off vs final
        assert np.array_equal(off["joint_config"], fin["joint_config"]), "q changed!"
        # MuJoCo agreement on native-free & successful candidates (sample)
        mj_chk = mj_fn = 0
        if oracle is not None:
            succ = np.asarray(fin["success"]).astype(bool)
            idx = np.nonzero(succ)[0][:48]
            for bi in idx:
                mj_chk += 1
                if oracle.label(np.asarray(fin["joint_config"][bi], float))["colliding"]:
                    mj_fn += 1                                  # native-free success MuJoCo calls colliding
        ov = (t_fin - t_off) / t_off * 100 if t_off else 0.0
        print(f"{B:>5} {t_off*1e3:>8.2f} {t_fin*1e3:>9.2f} {coll_ms:>8.3f} {ov:>6.1f} "
              f"{off_ok:>7} {fin_ok:>7} {rej:>6} {nat_free:>8} {mj_chk:>6} {mj_fn:>5}")
    print("\nfree candidate q values byte-identical off vs final: YES (asserted every batch)")
    print("mj_FN = successful native-free candidates that MuJoCo finds colliding (must be 0)")
    print(f"fused full-check workspace allocations after warm-up: {hjcdik._hjcdik.sidecar_ws_nalloc()} "
          "(grows only when batch size increases)")

    # ---- end-to-end planner-gate timing: does the native prefilter reduce total MuJoCo cost? ----
    if oracle is not None:
        print("\n=== end-to-end filtering (HJCD -> [native gate] -> MuJoCo check of remaining) ===")
        print(f"{'B':>5} {'off_mj_n':>8} {'fin_mj_n':>8} {'off_mj_ms':>9} {'fin_gate_ms':>11} {'fin_mj_ms':>9} {'saved%':>7}")
        for B in (500, 1000, 2000):
            sq, tp, tq = problem(B)
            off = hjcdik.solve(sq, tp, tq, self_collision_mode="off", **kw)
            fin = hjcdik.solve(sq, tp, tq, self_collision_mode="final", **kw)
            off_idx = np.nonzero(np.asarray(off["success"]))[0]       # off: MuJoCo checks ALL successes
            fin_idx = np.nonzero(np.asarray(fin["success"]))[0]       # final: MuJoCo checks RETAINED only
            t = time.perf_counter()
            for bi in off_idx: oracle.label(np.asarray(off["joint_config"][bi], float))
            off_mj = (time.perf_counter() - t) * 1e3
            t = time.perf_counter()
            for bi in fin_idx: oracle.label(np.asarray(fin["joint_config"][bi], float))
            fin_mj = (time.perf_counter() - t) * 1e3
            gate = fin["self_collision"]["kernel_ms"]
            saved = (off_mj - (fin_mj + gate)) / off_mj * 100 if off_mj else 0.0
            print(f"{B:>5} {len(off_idx):>8} {len(fin_idx):>8} {off_mj:>9.1f} {gate:>11.2f} {fin_mj:>9.1f} {saved:>7.1f}")


if __name__ == "__main__":
    main()
