#!/usr/bin/env python3
"""Multi-warp LM correctness sweep: run identical targets at HJCD_LM_WARPS in {1,2,4,8}
and verify the refined solutions match W=1 (each warp's LM is independent of W, so with
num_solutions>1 / early-stop off the per-candidate outputs should be ~identical).

Run via the env-driven knob: this script re-execs itself once per W with HJCD_LM_WARPS set,
writes results to /tmp, then the W=1 parent compares. Usage: python wsweep_correctness.py
"""
import os, sys, json, subprocess
import numpy as np

NUM_TARGETS = 8
BATCH = 2000
NSOL = 4          # >1 => stop_on_first=0 => deterministic, W-independent
SEED = 0

def run_one(W):
    import hjcdik
    ts = hjcdik.sample_targets(num_targets=NUM_TARGETS, seed=SEED)
    res = []
    for t in ts:
        out = hjcdik.generate_solutions(t, batch_size=BATCH, num_solutions=NSOL)
        res.append({
            "count": int(out["count"]),
            "pos_errors": np.array(out["pos_errors"], dtype=float).tolist(),
            "ori_errors": np.array(out["ori_errors"], dtype=float).tolist(),
            "pose": np.array(out["pose"], dtype=float).ravel().tolist(),
            "joint_config": np.array(out["joint_config"], dtype=float).ravel().tolist(),
        })
    return res

if "WSWEEP_CHILD_W" in os.environ:
    W = int(os.environ["WSWEEP_CHILD_W"])
    data = run_one(W)
    with open(f"/tmp/wsweep_W{W}.json", "w") as f:
        json.dump(data, f)
    print(f"[child W={W}] done")
    sys.exit(0)

# parent: spawn a child per W (fresh process so the env knob takes at launch)
Ws = [1, 2, 4, 8]
for W in Ws:
    env = dict(os.environ, HJCD_LM_WARPS=str(W), WSWEEP_CHILD_W=str(W))
    subprocess.run([sys.executable, __file__], env=env, check=True)

ref = json.load(open("/tmp/wsweep_W1.json"))
print(f"\n=== compare vs W=1 ({NUM_TARGETS} targets, batch={BATCH}, num_solutions={NSOL}) ===")
ok_all = True
for W in Ws:
    cur = json.load(open(f"/tmp/wsweep_W{W}.json"))
    max_pos_d = max_ori_d = max_pose_d = 0.0
    cnt_mismatch = 0
    for a, b in zip(ref, cur):
        if a["count"] != b["count"]:
            cnt_mismatch += 1
        n = min(len(a["pos_errors"]), len(b["pos_errors"]))
        if n:
            max_pos_d = max(max_pos_d, float(np.max(np.abs(np.array(a["pos_errors"][:n]) - np.array(b["pos_errors"][:n])))))
            max_ori_d = max(max_ori_d, float(np.max(np.abs(np.array(a["ori_errors"][:n]) - np.array(b["ori_errors"][:n])))))
        m = min(len(a["pose"]), len(b["pose"]))
        if m:
            max_pose_d = max(max_pose_d, float(np.max(np.abs(np.array(a["pose"][:m]) - np.array(b["pose"][:m])))))
    # also report absolute best pos err at this W
    best = min((min(c["pos_errors"]) if c["pos_errors"] else 9e9) for c in cur)
    status = "OK" if (max_pos_d < 1e-3 and max_ori_d < 1e-4 and cnt_mismatch == 0) else "DIFF"
    if status != "OK":
        ok_all = False
    print(f"W={W}: dpos(mm)={max_pos_d:.3e} dori={max_ori_d:.3e} dpose={max_pose_d:.3e} "
          f"count_mismatch={cnt_mismatch} best_pos(mm)={best:.3e}  [{status}]")

print("\nRESULT:", "ALL MATCH W=1" if ok_all else "MISMATCH DETECTED")
sys.exit(0 if ok_all else 1)
