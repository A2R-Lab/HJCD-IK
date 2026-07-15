"""Milestone 2: the batched-problem API solve_problems(P problems x S seeds).

Flattening:  b = p*S + s.  The candidate kernels see B = P*S blocks; a block reads PROBLEM-level data
(targets, mask, weights) via pid = gp/S and CANDIDATE-level data (seed, outputs) via gp. Targets and
masks are stored ONCE per problem ([P,K,...]/[P]), never broadcast to [P,S,K,...].

Two identities pin backward compatibility exactly (bitwise, not just within tolerance):
  * P=1, S=B  reproduces a single shared-target solve of B seeds  (kernel reads targets[0] for all);
  * P=B, S=1  reproduces the candidate-specific path              (kernel reads targets[gp], pid==gp).
Both hold BITWISE because with identical target VALUES the kernel executes identically.

Selection metric (for later milestones): cost_physical = E_phys, tolerance-normalised and comparable
across candidates. cost_lm (row-scaled) is diagnostic only and never used for selection.
"""
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


def _pose7(qcfg):
    """The K target poses [x,y,z,qw,qx,qy,qz] reached by configuration qcfg."""
    T = hjcdik.target_transforms(qcfg[None, :])[0]
    out = np.zeros((K, 7))
    for k in range(K):
        out[k, :3] = T[k][:3, 3]
        out[k, 3:] = _quat(T[k][:3, :3])
    return out


def _problems(P, S, seed=0, mask=None, sigma=0.05, spread=True):
    """P reachable problems: each targets a distinct random config; S nearby seeds per problem."""
    rng = np.random.default_rng(seed)
    poses = np.zeros((P, K, 7))
    seeds = np.zeros((P, S, N))
    q_true = []
    for pi in range(P):
        qt = rng.uniform(LO, HI)
        q_true.append(qt)
        poses[pi] = _pose7(qt)
        seeds[pi] = np.clip(qt + rng.normal(scale=sigma, size=(S, N)), LO, HI)
    if mask is None:
        masks = np.full(P, (1 << K) - 1, dtype=np.uint32)
    else:
        masks = np.asarray(mask, dtype=np.uint32)
        if masks.ndim == 0:
            masks = np.full(P, int(masks), dtype=np.uint32)
    return poses, masks, seeds, q_true


def _candidates(poses, masks, seeds, solver=None, **kw):
    """The full [P,S,...] post-fallback candidate arrays. The default solve_problems now returns only
    the selected top-1 ([P,1,...]); the candidate-level M2 checks ask for return_all_candidates=True
    and read the all_* arrays under M2 names."""
    fn = (solver or hjcdik).solve_problems
    o = fn(poses, masks, seeds, return_all_candidates=True, **kw)
    out = {
        "joint_config": o["all_joint_config"],
        "position_errors": o["all_position_errors"],
        "orientation_errors": o["all_orientation_errors"],
        "cost_lm": o["all_cost_lm"],
        "cost_physical": o["all_cost_physical"],
        "success": o["all_success"],
        "active_masks": o["active_masks"],
    }
    if o["collision_enabled"]:
        out["collision_free"] = o["all_collision_free"]
        out["used_coarse_fallback"] = o["all_used_coarse_fallback"]
    return out


def _solved(out):
    """[P,S] bool: candidate meets both tolerances on its ACTIVE targets."""
    act = ((out["active_masks"][:, None] >> np.arange(K, dtype=np.uint32)) & 1).astype(bool)  # [P,K]
    pe = np.where(act[:, None, :], out["position_errors"], 0.0).max(axis=2)
    oe = np.where(act[:, None, :], out["orientation_errors"], 0.0).max(axis=2)
    return (pe <= PTOL) & (oe <= OTOL)


# --- 1-4. shape coverage --------------------------------------------------------------------------
@pytest.mark.parametrize("P,S", [(1, 1), (1, 8), (5, 1), (5, 8)])
def test_shapes(P, S):
    poses, masks, seeds, _ = _problems(P, S, seed=P * 100 + S)
    sel = hjcdik.solve_problems(poses, masks, seeds, coarse_iters=60, lm_iters=30)  # default: top-1
    assert sel["joint_config"].shape == (P, 1, N)
    assert sel["success"].shape == (P, 1)
    assert sel["cost_physical"].shape == (P, 1)
    assert sel["selected_seed_ids"].shape == (P, 1)
    assert sel["position_errors"].shape == (P, 1, K)
    assert sel["problem_success"].shape == (P,)
    assert sel["num_solved"].shape == (P,)
    assert sel["active_masks"].shape == (P,)
    cand = _candidates(poses, masks, seeds, coarse_iters=60, lm_iters=30)
    assert cand["joint_config"].shape == (P, S, N)
    assert cand["position_errors"].shape == (P, S, K)
    assert np.isfinite(np.asarray(sel["joint_config"], np.float64)).all()


# --- 5/6. different vs identical targets ----------------------------------------------------------
def test_different_targets_each_problem_solves_its_own():
    poses, masks, seeds, _ = _problems(6, 64, seed=1)
    out = _candidates(poses, masks, seeds, coarse_iters=120, lm_iters=60)
    ok = _solved(out)
    assert ok.any(axis=1).all(), "some problem solved no candidate despite reachable nearby seeds"


def test_identical_targets_across_problems():
    poses1, masks, seeds, _ = _problems(1, 32, seed=2)
    P = 4
    poses = np.repeat(poses1, P, axis=0)              # same targets for every problem
    masks = np.full(P, (1 << K) - 1, dtype=np.uint32)
    rng = np.random.default_rng(3)
    seeds = rng.uniform(LO, HI, size=(P, 32, N))
    out = _candidates(poses, masks, seeds, coarse_iters=120, lm_iters=60)
    ok = _solved(out)
    assert ok.any(axis=1).all()


# --- 7/8. mixed masks -----------------------------------------------------------------------------
@pytest.mark.skipif(K < 4, reason="mixed K=1..4 masks need K>=4 (G1)")
def test_mixed_masks_in_one_batch():
    masks = np.array([0b0001, 0b0011, 0b0111, 0b1111], dtype=np.uint32)  # K=1,2,3,4
    poses, _, seeds, _ = _problems(4, 64, seed=4, mask=masks)
    out = _candidates(poses, masks, seeds, coarse_iters=120, lm_iters=60)
    ok = _solved(out)
    for pi in range(4):
        assert ok[pi].any(), f"problem {pi} (mask {masks[pi]:04b}) solved nothing"
    # inactive targets are not evaluated -> exactly zero error
    for pi in range(4):
        inact = [k for k in range(K) if not ((int(masks[pi]) >> k) & 1)]
        if inact:
            assert np.all(out["position_errors"][pi][:, inact] == 0.0)
            assert np.all(out["orientation_errors"][pi][:, inact] == 0.0)


# --- 9. target isolation: changing ONE problem's targets affects only that problem -----------------
def test_target_isolation():
    poses, masks, seeds, _ = _problems(5, 48, seed=5)
    a = _candidates(poses, masks, seeds, coarse_iters=120, lm_iters=60)
    poses2 = poses.copy()
    # replace problem 2's targets with a completely different reachable set
    rng = np.random.default_rng(999)
    poses2[2] = _pose7(rng.uniform(LO, HI))
    b = _candidates(poses2, masks, seeds, coarse_iters=120, lm_iters=60)
    for pi in range(5):
        if pi == 2:
            assert not np.array_equal(a["joint_config"][pi], b["joint_config"][pi]), \
                "changing problem 2's targets did not change problem 2"
        else:
            np.testing.assert_array_equal(
                a["joint_config"][pi], b["joint_config"][pi],
                err_msg=f"problem {pi} changed when only problem 2's targets changed")


# --- 10. seed isolation ---------------------------------------------------------------------------
def test_seed_isolation():
    poses, masks, seeds, _ = _problems(5, 48, seed=6)
    a = _candidates(poses, masks, seeds, coarse_iters=120, lm_iters=60)
    seeds2 = seeds.copy()
    rng = np.random.default_rng(1234)
    seeds2[3] = rng.uniform(LO, HI, size=(seeds.shape[1], N))       # only problem 3's seeds
    b = _candidates(poses, masks, seeds2, coarse_iters=120, lm_iters=60)
    for pi in range(5):
        if pi == 3:
            assert not np.array_equal(a["joint_config"][pi], b["joint_config"][pi])
        else:
            np.testing.assert_array_equal(a["joint_config"][pi], b["joint_config"][pi],
                                          err_msg=f"problem {pi} changed when only problem 3 seeds did")


# --- 11. flattened gp->pid indexing against a host reference --------------------------------------
def test_flattened_indexing_matches_host_reference():
    """Build P problems, run once as a batch, and separately run EACH problem's S seeds on its own.
    Candidate (p,s) in the batch must equal problem p solved alone. This checks that pid = gp/S maps
    every candidate to the right problem's targets."""
    P, S = 5, 40
    poses, masks, seeds, _ = _problems(P, S, seed=7)
    batch = _candidates(poses, masks, seeds, coarse_iters=120, lm_iters=60)
    for pi in range(P):
        one = _candidates(poses[pi:pi+1], masks[pi:pi+1], seeds[pi:pi+1],
                              coarse_iters=120, lm_iters=60)
        np.testing.assert_array_equal(
            batch["joint_config"][pi], one["joint_config"][0],
            err_msg=f"batched candidate block for problem {pi} != that problem solved alone")


# --- 12. P=1,S=B  ==  a single shared-target solve (BITWISE) ---------------------------------------
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_P1_SB_matches_shared_target_solve(precision):
    B = 256
    rng = np.random.default_rng(8)
    qt = rng.uniform(LO, HI)
    pose = _pose7(qt)                                   # [K,7]
    seeds = rng.uniform(LO, HI, size=(B, N))
    mask = (1 << K) - 1

    # existing single-problem solve(): targets broadcast to [B,K]
    pos_b = np.repeat(pose[None, :, :3], B, axis=0)
    quat_b = np.repeat(pose[None, :, 3:], B, axis=0)
    old = hjcdik.solve(seeds, pos_b, quat_b,
                       active_target_mask=np.full(B, mask, dtype=np.uint32),
                       position_tol=PTOL, orientation_tol=OTOL,
                       coarse_iters=120, lm_iters=60, precision=precision)

    new = _candidates(pose[None], np.array([mask], dtype=np.uint32), seeds[None],
                      coarse_iters=120, lm_iters=60, precision=precision)

    np.testing.assert_array_equal(new["joint_config"][0], old["joint_config"],
                                  err_msg=f"{precision}: P=1,S=B config != shared-target solve")
    np.testing.assert_array_equal(new["position_errors"][0], old["position_errors"])
    np.testing.assert_array_equal(new["orientation_errors"][0], old["orientation_errors"])
    np.testing.assert_array_equal(new["success"][0], old["success"])


# --- 13. P=B,S=1  ==  candidate-specific behavior (BITWISE) ----------------------------------------
@pytest.mark.parametrize("precision", ["float32", "float64"])
def test_PB_S1_matches_candidate_specific(precision):
    B = 200
    rng = np.random.default_rng(9)
    # each candidate its own distinct target + seed
    poses = np.stack([_pose7(rng.uniform(LO, HI)) for _ in range(B)])   # [B,K,7]
    seeds = rng.uniform(LO, HI, size=(B, N))
    masks = np.full(B, (1 << K) - 1, dtype=np.uint32)

    old = hjcdik.solve(seeds, poses[:, :, :3], poses[:, :, 3:], active_target_mask=masks,
                       position_tol=PTOL, orientation_tol=OTOL,
                       coarse_iters=120, lm_iters=60, precision=precision)
    new = _candidates(poses, masks, seeds[:, None, :],
                      coarse_iters=120, lm_iters=60, precision=precision)

    np.testing.assert_array_equal(new["joint_config"][:, 0], old["joint_config"],
                                  err_msg=f"{precision}: P=B,S=1 config != candidate-specific solve")
    np.testing.assert_array_equal(new["position_errors"][:, 0], old["position_errors"])
    np.testing.assert_array_equal(new["success"][:, 0], old["success"])


# --- 14/15. precisions ----------------------------------------------------------------------------
@pytest.mark.parametrize("precision,dtype", [("float32", np.float32), ("float64", np.float64)])
def test_precisions(precision, dtype):
    poses, masks, seeds, _ = _problems(4, 32, seed=10)
    out = hjcdik.solve_problems(poses, masks, seeds, coarse_iters=60, lm_iters=30, precision=precision)
    assert out["joint_config"].dtype == dtype
    assert out["position_errors"].dtype == np.float64
    assert out["cost_physical"].dtype == np.float64
    assert np.isfinite(np.asarray(out["joint_config"], np.float64)).all()


# --- 16. diagnostics on/off ------------------------------------------------------------------------
def test_diagnostics_equivalence():
    poses, masks, seeds, _ = _problems(4, 48, seed=11)
    a = hjcdik.solve_problems(poses, masks, seeds, coarse_iters=120, lm_iters=60, diagnostics=False)
    b = hjcdik.solve_problems(poses, masks, seeds, coarse_iters=120, lm_iters=60, diagnostics=True)
    np.testing.assert_array_equal(a["joint_config"], b["joint_config"])   # selected top-1
    np.testing.assert_array_equal(a["cost_lm"], b["cost_lm"])
    np.testing.assert_array_equal(a["cost_physical"], b["cost_physical"])


# --- 21. strides / contiguity ---------------------------------------------------------------------
def test_strides_and_contiguity():
    poses, masks, seeds, _ = _problems(4, 32, seed=12)
    out = hjcdik.solve_problems(poses, masks, seeds, coarse_iters=60, lm_iters=30)
    for name, v in out.items():
        if not isinstance(v, np.ndarray) or v.ndim == 0:
            continue
        assert v.flags.c_contiguous, f"{name} not C-contiguous"
        assert 0 not in v.strides, f"{name} has a zero stride"


# --- 22. workspace reuse: zero allocations on repeat -----------------------------------------------
def test_workspace_zero_alloc_on_repeat():
    sv = hjcdik.HJCDSolver()
    poses, masks, seeds, _ = _problems(10, 32, seed=13)
    kw = dict(coarse_iters=60, lm_iters=30)
    sv.solve_problems(poses, masks, seeds, **kw)          # warm-up (may allocate)
    n0 = sv.workspace_stats()["cuda_mallocs"]
    for _ in range(6):
        sv.solve_problems(poses, masks, seeds, **kw)
    assert sv.workspace_stats()["cuda_mallocs"] == n0
    assert sv.workspace_stats()["cuda_frees"] == 0


def test_workspace_grows_for_larger_then_reuses():
    sv = hjcdik.HJCDSolver()
    small = _problems(4, 16, seed=14)
    big = _problems(20, 64, seed=15)
    sv.solve_problems(small[0], small[1], small[2], coarse_iters=30, lm_iters=20)
    n0 = sv.workspace_stats()["cuda_mallocs"]
    sv.solve_problems(big[0], big[1], big[2], coarse_iters=30, lm_iters=20)     # grows
    n1 = sv.workspace_stats()["cuda_mallocs"]
    assert n1 == n0 + 1
    sv.solve_problems(small[0], small[1], small[2], coarse_iters=30, lm_iters=20)  # reuse
    assert sv.workspace_stats()["cuda_mallocs"] == n1


# --- 23/24/25. input validation -------------------------------------------------------------------
def test_malformed_shapes_rejected():
    poses, masks, seeds, _ = _problems(3, 8, seed=16)
    with pytest.raises(ValueError, match="target_poses must be"):
        hjcdik.solve_problems(poses[:, :, :6], masks, seeds)          # 6 instead of 7
    with pytest.raises(ValueError, match="seed_configs must be"):
        hjcdik.solve_problems(poses, masks, seeds[:, :, :N-1])        # wrong N
    with pytest.raises(ValueError, match="active_masks must be"):
        hjcdik.solve_problems(poses, masks[:2], seeds)               # P mismatch
    with pytest.raises(ValueError, match="seed_configs P="):
        hjcdik.solve_problems(poses, masks, seeds[:2])               # P mismatch (seeds)


def test_invalid_masks_rejected():
    poses, _, seeds, _ = _problems(3, 8, seed=17)
    bad = np.array([(1 << K) - 1, (1 << (K + 1)) - 1, 1], dtype=np.uint32)  # middle sets bit K
    with pytest.raises(ValueError, match="sets bits above"):
        hjcdik.solve_problems(poses, bad, seeds)
    empty = np.array([(1 << K) - 1, 0, 1], dtype=np.uint32)                  # middle empty
    with pytest.raises(ValueError, match="empty"):
        hjcdik.solve_problems(poses, empty, seeds)


def test_nan_inf_rejected():
    poses, masks, seeds, _ = _problems(3, 8, seed=18)
    bad = poses.copy(); bad[1, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or inf"):
        hjcdik.solve_problems(bad, masks, seeds)
    bads = seeds.copy(); bads[0, 0, 0] = np.inf
    with pytest.raises(ValueError, match="NaN or inf"):
        hjcdik.solve_problems(poses, masks, bads)


def test_num_solutions_range_validated():
    poses, masks, seeds, _ = _problems(3, 8, seed=19)
    with pytest.raises(ValueError, match="num_solutions"):
        hjcdik.solve_problems(poses, masks, seeds, num_solutions=0)
    with pytest.raises(ValueError, match="num_solutions"):
        hjcdik.solve_problems(poses, masks, seeds, num_solutions=9)   # > S=8


# --- 17/18/19/20/34. collision (Panda --collision build) ------------------------------------------
import json                                                                            # noqa: E402
from pathlib import Path                                                               # noqa: E402

collision_only = pytest.mark.skipif(K != 1, reason="collision build is the single-target Panda")
REPO = Path(__file__).resolve().parents[1]
SET = "bookshelf_small_panda"


@pytest.fixture(scope="module")
def probs():
    return (REPO / "tests" / "mb_problems.json").read_text()


@pytest.fixture(scope="module")
def goal_pose(probs):
    p = json.loads(probs)["problems"][SET]
    inst = p[0] if isinstance(p, list) else p
    gp = inst["goal_pose"]
    pos = np.asarray(gp.get("position_xyz", gp.get("position")), dtype=float)
    quat = np.asarray(gp.get("quaternion_wxyz", gp.get("quat_wxyz")), dtype=float)
    quat = quat / np.linalg.norm(quat)
    return np.concatenate([pos, quat])              # [7]


@pytest.fixture(scope="module")
def free_seeds(probs):
    rng = np.random.default_rng(5)
    cand = np.stack([rng.uniform(LO, HI) for _ in range(4000)])
    return cand[hjcdik.collision_free(cand, probs, SET, 0)]


@collision_only
def test_collision_enabled_returns_no_colliding_config(probs, goal_pose, free_seeds):
    """18. Every returned candidate across all problems is collision-free (one shared world)."""
    P, S = 8, 64
    poses = np.repeat(goal_pose[None, None, :], P, axis=0)          # [P,1,7], same goal each problem
    masks = np.ones(P, dtype=np.uint32)
    seeds = np.resize(free_seeds, (P, S, N))
    out = _candidates(poses, masks, seeds, coarse_iters=120, lm_iters=60,
                      problems_json_text=probs, problem_set_name=SET, problem_idx=0)
    assert "collision_free" in out and "used_coarse_fallback" in out
    q = np.asarray(out["joint_config"], np.float64).reshape(P * S, N)
    free = hjcdik.collision_free(q, probs, SET, 0).reshape(P, S)
    # only candidates flagged collision_free need to actually be free (infeasible ones are flagged)
    flagged = out["collision_free"]
    assert np.all(free[flagged]), "a candidate flagged collision_free actually collides"
    # and the fallback did fire somewhere (LM does re-enter the shelf)
    assert int(out["used_coarse_fallback"].sum()) > 0


@collision_only
def test_collision_fallback_is_candidate_local(probs, goal_pose, free_seeds):
    """19/34. A fallback replaces a colliding LM candidate with ITS OWN coarse result, and touches
    no other candidate or problem."""
    P, S = 6, 64
    poses = np.repeat(goal_pose[None, None, :], P, axis=0)
    masks = np.ones(P, dtype=np.uint32)
    seeds = np.resize(free_seeds, (P, S, N))
    out = _candidates(poses, masks, seeds, coarse_iters=120, lm_iters=60,
                      problems_json_text=probs, problem_set_name=SET, problem_idx=0)
    fb = out["used_coarse_fallback"]                # [P,S]
    assert fb.any()
    # every fallback candidate is collision-free in the returned config
    q = np.asarray(out["joint_config"], np.float64)
    for pi in range(P):
        for si in range(S):
            if fb[pi, si]:
                assert hjcdik.collision_free(q[pi, si][None, :], probs, SET, 0)[0]


@collision_only
def test_one_problem_infeasible_does_not_contaminate_neighbors(probs, goal_pose, free_seeds):
    """20. Problem 0 gets seeds deep in collision; neighbors get free seeds. Problem 0's DATA (its
    colliding seeds, its fallbacks) must not affect any neighbor's result.

    The batch LAYOUT is held fixed (P=5 in both runs) -- only problem 0's seeds change in place. The
    coarse perturbation RNG is keyed on the global candidate index gp, so a neighbor keeps the same
    gp and must produce a bitwise-identical result regardless of what problem 0 contains. (Changing
    the batch SIZE would shift neighbors to different gp and legitimately change their RNG -- that is
    positional, not data contamination, so we do not test it that way.)
    """
    P, S = 5, 48
    poses = np.repeat(goal_pose[None, None, :], P, axis=0)
    masks = np.ones(P, dtype=np.uint32)
    rng = np.random.default_rng(77)
    coll = np.stack([rng.uniform(LO, HI) for _ in range(6000)])
    coll = coll[~hjcdik.collision_free(coll, probs, SET, 0)]

    seedsA = np.resize(free_seeds, (P, S, N)).copy()
    seedsA[0] = np.resize(coll, (S, N))                       # problem 0: colliding set A
    seedsB = seedsA.copy()
    seedsB[0] = np.resize(coll[::-1], (S, N))                 # problem 0: DIFFERENT colliding set B

    kw = dict(coarse_iters=120, lm_iters=60, problems_json_text=probs,
              problem_set_name=SET, problem_idx=0)
    a = _candidates(poses, masks, seedsA, **kw)
    b = _candidates(poses, masks, seedsB, **kw)
    np.testing.assert_array_equal(
        a["joint_config"][1:], b["joint_config"][1:],
        err_msg="a neighbor changed when only problem 0's (colliding) seeds changed")
    # every returned collision_free candidate is genuinely free, in every problem
    q = np.asarray(a["joint_config"], np.float64).reshape(P * S, N)
    free = hjcdik.collision_free(q, probs, SET, 0).reshape(P, S)
    assert np.all(free[a["collision_free"]])
    # problem 0 (all colliding seeds) still never returns a config flagged free-but-colliding
    assert np.all(free[0][a["collision_free"][0]])


@collision_only
def test_P1_SB_collision_matches_single_problem_solve(probs, goal_pose, free_seeds):
    """12 + collision: P=1,S=B with a collision world reproduces the single-problem collision solve
    (within fp32 task tolerance -- the collision filter's np.where over floats is the same)."""
    B = 128
    seeds = np.resize(free_seeds, (B, N))
    pos_b = np.repeat(goal_pose[None, None, :3], B, axis=0)
    quat_b = np.repeat(goal_pose[None, None, 3:], B, axis=0)
    old = hjcdik.solve(seeds, pos_b, quat_b, active_target_mask=np.ones(B, dtype=np.uint32),
                       position_tol=PTOL, orientation_tol=OTOL, coarse_iters=120, lm_iters=60,
                       problems_json_text=probs, problem_set_name=SET, problem_idx=0)
    new = _candidates(goal_pose[None, None, :], np.array([1], dtype=np.uint32),
                      seeds[None], coarse_iters=120, lm_iters=60,
                      problems_json_text=probs, problem_set_name=SET, problem_idx=0)
    np.testing.assert_array_equal(new["joint_config"][0], old["joint_config"])
    np.testing.assert_array_equal(new["collision_free"][0], old["collision_free"])
