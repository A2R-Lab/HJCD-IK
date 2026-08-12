"""Milestone 5: backward compatibility + production validation.

The batched path and the legacy solve() share ONE coarse and ONE LM implementation -- the batched
kernels coarse_search_mt_kernel / lm_multi_target_kernel, which solve() drives with
seeds_per_problem=1 (P=B) and solve_problems drives with the real (P,S). There is no separate
single-problem math. These tests pin the exact equivalence

    solve(seeds, targets, mask)  ==  solve_problems(targets[None], mask[None], seeds[None])   (P=1,S=B)

and the mixed-mask correctness that Variant A (branch-in-kernel on the candidate's own mask) must
provide, plus the zero-allocation steady state shared by both APIs.
"""
import json
from pathlib import Path

import numpy as np
import pytest

import hjcdik

N = hjcdik.num_joints()
K = hjcdik.num_targets()
LIM = hjcdik.joint_limits()
LO, HI = LIM[:, 0], LIM[:, 1]
PTOL, OTOL = 1e-4, 1e-3


def _quat(R):
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


def _pose7(qc):
    T = hjcdik.target_transforms(qc[None, :])[0]
    o = np.zeros((K, 7))
    for k in range(K):
        o[k, :3] = T[k][:3, 3]
        o[k, 3:] = _quat(T[k][:3, :3])
    return o


# --- exact regression: legacy solve() == batched P=1,S=B ------------------------------------------
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("mask", ([(1 << K) - 1] if K == 1 else [0b0001, 0b0011, 0b1100, 0b1111]))
def test_solve_equals_batched_P1(precision, mask):
    """The candidate arrays from solve_problems(P=1,S=B) reproduce legacy solve() BITWISE, for every
    mask and both precisions. Proves the two APIs run the identical shared kernels."""
    B = 256
    rng = np.random.default_rng(int(mask) * 7 + (precision == "float32"))
    q_true = rng.uniform(LO, HI)
    pose = _pose7(q_true)
    seeds = np.clip(q_true + rng.normal(scale=0.3, size=(B, N)), LO, HI)

    old = hjcdik.solve(seeds, np.repeat(pose[None, :, :3], B, axis=0),
                       np.repeat(pose[None, :, 3:], B, axis=0),
                       active_target_mask=np.full(B, mask, dtype=np.uint32),
                       position_tol=PTOL, orientation_tol=OTOL,
                       coarse_iters=120, lm_iters=60, precision=precision)
    new = hjcdik.solve_problems(pose[None], np.array([mask], dtype=np.uint32), seeds[None],
                                position_tol=PTOL, orientation_tol=OTOL,
                                coarse_iters=120, lm_iters=60, precision=precision,
                                return_all_candidates=True)
    np.testing.assert_array_equal(new["all_joint_config"][0], old["joint_config"],
                                  err_msg=f"{precision} mask={mask:04b}: config differs")
    np.testing.assert_array_equal(new["all_position_errors"][0], old["position_errors"])
    np.testing.assert_array_equal(new["all_success"][0], old["success"])


# --- mixed masks: each candidate honours ITS problem's mask ---------------------------------------
@pytest.mark.skipif(K < 4, reason="mixed masks need G1")
def test_mixed_masks_match_per_mask_solves():
    """A batch mixing K=1..4 masks must give each problem the SAME result as solving it alone with its
    own mask -- i.e. Variant A (branch on the candidate's own active[pid]) is correct."""
    rng = np.random.default_rng(7)
    masks = np.array([0b0001, 0b0011, 0b0111, 0b1111], dtype=np.uint32)
    P, S = 4, 48
    poses = np.zeros((P, K, 7)); seeds = np.zeros((P, S, N))
    for p in range(P):
        qt = rng.uniform(LO, HI)
        poses[p] = _pose7(qt)
        seeds[p] = np.clip(qt + rng.normal(scale=0.05, size=(S, N)), LO, HI)

    mixed = hjcdik.solve_problems(poses, masks, seeds, coarse_iters=120, lm_iters=60,
                                  return_all_candidates=True)
    for p in range(P):
        alone = hjcdik.solve_problems(poses[p:p+1], masks[p:p+1], seeds[p:p+1],
                                      coarse_iters=120, lm_iters=60, return_all_candidates=True)
        np.testing.assert_array_equal(
            mixed["all_joint_config"][p], alone["all_joint_config"][0],
            err_msg=f"problem {p} (mask {masks[p]:04b}) differs in a mixed vs homogeneous batch")


# --- zero-allocation steady state, both APIs share the workspace ----------------------------------
def test_both_apis_share_workspace_zero_alloc():
    sv = hjcdik.HJCDSolver()
    rng = np.random.default_rng(3)
    qt = rng.uniform(LO, HI); pose = _pose7(qt)
    B = 512
    seeds = np.clip(qt + rng.normal(scale=0.1, size=(B, N)), LO, HI)
    P, S = 16, 32
    poses = np.repeat(pose[None], P, axis=0)
    pseeds = np.clip(qt + rng.normal(scale=0.1, size=(P, S, N)), LO, HI)
    masks_b = np.full(B, (1 << K) - 1, dtype=np.uint32)
    masks_p = np.full(P, (1 << K) - 1, dtype=np.uint32)

    def one_round():
        sv.solve(seeds, np.repeat(pose[None, :, :3], B, axis=0),
                 np.repeat(pose[None, :, 3:], B, axis=0),
                 active_target_mask=masks_b, coarse_iters=60, lm_iters=30)
        sv.solve_problems(poses, masks_p, pseeds, coarse_iters=60, lm_iters=30)

    # Warm up: solve() and solve_problems() have slightly different arena needs, so the shared arena
    # grows to the max over the first couple of interleaved rounds (a grow does one cudaFree +
    # cudaMalloc). Once it settles, steady state must allocate AND free nothing.
    for _ in range(3):
        one_round()
    n0 = sv.workspace_stats()["cuda_mallocs"]
    f0 = sv.workspace_stats()["cuda_frees"]
    for _ in range(5):
        one_round()
    assert sv.workspace_stats()["cuda_mallocs"] == n0, "a steady-state call allocated"
    assert sv.workspace_stats()["cuda_frees"] == f0, "a steady-state call freed/reallocated"


# --- collision regression: solve() == batched P=1 on the Panda shelf ------------------------------
@pytest.mark.skipif(K != 1, reason="collision build is Panda")
def test_solve_equals_batched_P1_collision():
    SET = "bookshelf_small_panda"
    probs = (Path(__file__).resolve().parents[1] / "tests" / "mb_problems.json").read_text()
    inst = json.loads(probs)["problems"][SET]
    inst = inst[0] if isinstance(inst, list) else inst
    gp = inst["goal_pose"]
    pos = np.asarray(gp.get("position_xyz", gp.get("position")), float)
    quat = np.asarray(gp.get("quaternion_wxyz", gp.get("quat_wxyz")), float); quat /= np.linalg.norm(quat)
    rng = np.random.default_rng(5)
    cand = np.stack([rng.uniform(LO, HI) for _ in range(4000)])
    seeds = cand[hjcdik.collision_free(cand, probs, SET, 0)][:128]
    B = len(seeds)

    old = hjcdik.solve(seeds, np.repeat(pos[None, None], B, axis=0), np.repeat(quat[None, None], B, axis=0),
                       active_target_mask=np.ones(B, dtype=np.uint32),
                       position_tol=PTOL, orientation_tol=OTOL, coarse_iters=120, lm_iters=60,
                       problems_json_text=probs, problem_set_name=SET, problem_idx=0)
    pose = np.concatenate([pos, quat])
    new = hjcdik.solve_problems(pose[None, None], np.array([1], dtype=np.uint32), seeds[None],
                                position_tol=PTOL, orientation_tol=OTOL, coarse_iters=120, lm_iters=60,
                                problems_json_text=probs, problem_set_name=SET, problem_idx=0,
                                return_all_candidates=True)
    np.testing.assert_array_equal(new["all_joint_config"][0], old["joint_config"])
    np.testing.assert_array_equal(new["all_collision_free"][0], old["collision_free"])
    np.testing.assert_array_equal(new["all_used_coarse_fallback"][0], old["used_coarse_fallback"])
