#!/usr/bin/env python3
"""Solve multi-target IK on the Unitree G1 (29-DoF, four end-effector frames).

Build first (this swaps the compiled robot; it stays until you regenerate for another URDF):

    python scripts/codegen/generate_grid.py csrc/urdf/g1_29dof_rev_1_0.urdf \
        -t left_hand_palm_joint \
        --target "name=left_hand;fixed=left_hand_palm_joint" \
        --target "name=right_hand;fixed=right_hand_palm_joint" \
        --target "name=left_foot;anchor=left_ankle_roll_joint;xyz=0.035,0,-0.035" \
        --target "name=right_foot;anchor=right_ankle_roll_joint;xyz=0.035,0,-0.035"
    scripts/setup/rebuild.sh

Target order is FIXED:  0 = left hand, 1 = right hand, 2 = left foot, 3 = right foot.

Note: generate_solutions() (the single-target API) does NOT work on a G1 build -- a 7-vector target
is ambiguous when there are four target frames. Use solve() / refine() / coarse_search().
"""
import numpy as np

import hjcdik

N = hjcdik.num_joints()      # 29
K = hjcdik.num_targets()     # 4
LIM = hjcdik.joint_limits()  # [N, 2] lower/upper, exactly what the solver clamps to
LO, HI = LIM[:, 0], LIM[:, 1]


def quat_from_R(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25*s, (R[2,1]-R[1,2])/s, (R[0,2]-R[2,0])/s, (R[1,0]-R[0,1])/s]
    else:
        i = int(np.argmax([R[0,0], R[1,1], R[2,2]]))
        if i == 0:
            s = np.sqrt(1+R[0,0]-R[1,1]-R[2,2])*2
            q = [(R[2,1]-R[1,2])/s, 0.25*s, (R[0,1]+R[1,0])/s, (R[0,2]+R[2,0])/s]
        elif i == 1:
            s = np.sqrt(1+R[1,1]-R[0,0]-R[2,2])*2
            q = [(R[0,2]-R[2,0])/s, (R[0,1]+R[1,0])/s, 0.25*s, (R[1,2]+R[2,1])/s]
        else:
            s = np.sqrt(1+R[2,2]-R[0,0]-R[1,1])*2
            q = [(R[1,0]-R[0,1])/s, (R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s, 0.25*s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def main():
    meta = hjcdik.target_metadata()
    print(f"robot: {N} joints, {K} target frames")
    print(f"  anchors (joint ids): {meta['anchor_jid']}")
    print(f"  order: 0=left_hand 1=right_hand 2=left_foot 3=right_foot\n")

    rng = np.random.default_rng(0)
    B = 128                                     # candidate seeds (one IK problem per seed here)

    # --- make a REACHABLE target set: pick a pose, read off where its four frames land -----------
    span = HI - LO
    q_true = rng.uniform(LO + 0.15*span, HI - 0.15*span)
    T = hjcdik.target_transforms(q_true[None, :])[0]        # [K, 4, 4] world poses
    tgt_pos = np.repeat(T[:, :3, 3][None], B, axis=0)                       # [B, K, 3]
    tgt_quat = np.repeat(np.stack([quat_from_R(T[k][:3, :3]) for k in range(K)])[None], B, axis=0)

    # --- seeds: random restarts inside the joint limits ------------------------------------------
    seeds = rng.uniform(LO, HI, size=(B, N))

    # --- solve. active_target_mask picks WHICH frames to hit (one bit per target) -----------------
    #     0b0011 = both hands    0b1100 = both feet    0b1111 = all four
    for label, mask in (("both hands", 0b0011), ("both feet", 0b1100), ("all four", 0b1111)):
        out = hjcdik.solve(
            seeds, tgt_pos, tgt_quat,
            active_target_mask=np.full(B, mask, dtype=np.uint32),
            position_tol=1e-4, orientation_tol=1e-3,      # 0.1 mm, 1 mrad
            coarse_mode="auto",                            # popcount>=2 -> coarse -> LM
            coarse_iters=120, lm_iters=60,
            diagnostics=True,
        )
        act = [k for k in range(K) if (mask >> k) & 1]
        pos = out["position_errors"][:, act].max(axis=1) * 1000     # mm, worst active target
        ori = out["orientation_errors"][:, act].max(axis=1)         # rad
        best = int(np.argmin(pos))
        print(f"{label:>11}: best candidate  pos={pos[best]:.5f} mm  ori={ori[best]:.2e} rad")
        print(f"             coarse iters={out['coarse_iterations'][best]}, "
              f"accepted={out['accepted_coarse_steps'][best]}  |  "
              f"LM iters={out['lm_iterations'][best]}")

    # --- per-target breakdown for the all-four case ----------------------------------------------
    out = hjcdik.solve(seeds, tgt_pos, tgt_quat, position_tol=1e-4, orientation_tol=1e-3,
                       coarse_iters=120, lm_iters=60)
    b = int(np.argmin(out["position_errors"].max(axis=1)))
    names = ["left_hand", "right_hand", "left_foot", "right_foot"]
    print("\nper-target error of the best all-four solution:")
    for k in range(K):
        print(f"  {names[k]:>11}: pos={out['position_errors'][b,k]*1000:8.5f} mm   "
              f"ori={out['orientation_errors'][b,k]:.2e} rad")
    print(f"\njoint config (29 values):\n{np.round(out['joint_config'][b], 4)}")


if __name__ == "__main__":
    main()
