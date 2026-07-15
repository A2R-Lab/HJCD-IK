"""Milestone 4: deterministic per-problem top-M GPU selection.

top-M runs M rounds of masked segmented argmin. Round m picks the m-th best candidate under the same
three-class key (class, E_phys, seed) used for top-1, skipping the m already-selected seeds, so the M
winners are DISTINCT candidate IDs. When a problem has fewer than M valid candidates, the leftover
slots are INVALID PADS (selected_seed_id=-1, cost=+inf, valid=False) -- never duplicates of a real
candidate. M=1 reproduces top-1 exactly.
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


def _pose7(qc):
    T = hjcdik.target_transforms(qc[None, :])[0]
    o = np.zeros((K, 7))
    for k in range(K):
        o[k, :3] = T[k][:3, 3]
        o[k, 3:] = _quat(T[k][:3, :3])
    return o


def _problems(P, S, seed=0, mask=None, sigma=0.06):
    rng = np.random.default_rng(seed)
    poses = np.zeros((P, K, 7))
    seeds = np.zeros((P, S, N))
    for pi in range(P):
        qt = rng.uniform(LO, HI)
        poses[pi] = _pose7(qt)
        seeds[pi] = np.clip(qt + rng.normal(scale=sigma, size=(S, N)), LO, HI)
    masks = np.full(P, (1 << K) - 1, dtype=np.uint32) if mask is None else \
        np.asarray(mask, dtype=np.uint32)
    return poses, masks, seeds


def _cpu_topm(allo, M, precision="float32", cc=False):
    """CPU reference: for each problem, sort candidates by the three-class key and take the top M
    DISTINCT valid candidates, padding with -1 (invalid) if fewer than M valid exist."""
    masks = allo["active_masks"]
    pe = allo["all_position_errors"]; oe = allo["all_orientation_errors"]
    cfg = np.asarray(allo["all_joint_config"], np.float64)
    P, S = pe.shape[0], pe.shape[1]
    ep = float(np.float32(PTOL)) if precision == "float32" else PTOL
    eo = float(np.float32(OTOL)) if precision == "float32" else OTOL
    act = ((masks[:, None] >> np.arange(K, dtype=np.uint32)) & 1).astype(bool)
    out = np.full((P, M), -1)
    for p in range(P):
        keys = []
        for s in range(S):
            a = act[p]
            pev = pe[p, s][a]; oev = oe[p, s][a]
            finite = np.isfinite(cfg[p, s]).all() and np.isfinite(pev).all() and np.isfinite(oev).all()
            feas = bool(allo["all_collision_free"][p, s]) if cc else True
            ephys = float(((pev / ep) ** 2 + (oev / eo) ** 2).sum())
            within = finite and np.all(pev <= ep) and np.all(oev <= eo)
            cls = 2 if (not finite or not feas) else (0 if within else 1)
            e = np.inf if cls == 2 else ephys
            keys.append((cls, e, s))
        order = sorted(range(S), key=lambda s: keys[s])
        rank = 0
        for s in order:
            if keys[s][0] == 2:            # no more valid candidates -> pad the rest
                break
            if rank < M:
                out[p, rank] = s
                rank += 1
    return out


def _run(poses, masks, seeds, M, precision="float32", **kw):
    sel = hjcdik.solve_problems(poses, masks, seeds, num_solutions=M, precision=precision,
                                return_all_candidates=False, **kw)
    allo = hjcdik.solve_problems(poses, masks, seeds, num_solutions=M, precision=precision,
                                 return_all_candidates=True, **kw)
    return sel, allo


# --- shapes + M=1 reproduces top-1 -----------------------------------------------------------------
@pytest.mark.parametrize("M", [1, 2, 5, 10, 32])
def test_shapes(M):
    poses, masks, seeds = _problems(5, 64, seed=M)
    sel = hjcdik.solve_problems(poses, masks, seeds, num_solutions=M, coarse_iters=60, lm_iters=30)
    assert sel["joint_config"].shape == (5, M, N)
    assert sel["success"].shape == (5, M)
    assert sel["cost_physical"].shape == (5, M)
    assert sel["selected_seed_ids"].shape == (5, M)
    assert sel["position_errors"].shape == (5, M, K)


def test_M1_equals_top1():
    poses, masks, seeds = _problems(6, 64, seed=100)
    m1 = hjcdik.solve_problems(poses, masks, seeds, num_solutions=1, coarse_iters=120, lm_iters=60)
    mm = hjcdik.solve_problems(poses, masks, seeds, num_solutions=5, coarse_iters=120, lm_iters=60)
    # the first column of top-5 is the top-1
    np.testing.assert_array_equal(m1["selected_seed_ids"][:, 0], mm["selected_seed_ids"][:, 0])
    np.testing.assert_array_equal(m1["joint_config"][:, 0], mm["joint_config"][:, 0])


# --- CPU segmented-sort agreement ------------------------------------------------------------------
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("M", [2, 5, 10, 32])
def test_topm_matches_cpu_segmented_sort(precision, M):
    poses, masks, seeds = _problems(8, 96, seed=M * 3 + 7, sigma=0.08)
    sel, allo = _run(poses, masks, seeds, M, precision=precision, coarse_iters=120, lm_iters=60)
    cpu = _cpu_topm(allo, M, precision=precision, cc=False)
    np.testing.assert_array_equal(sel["selected_seed_ids"], cpu,
                                  err_msg=f"{precision} M={M}: top-M seed ids != CPU segmented sort")


# --- distinctness + ranked order (monotone E_phys, class non-decreasing) ---------------------------
@pytest.mark.parametrize("M", [5, 10, 32])
def test_no_duplicate_ids_and_ranked(M):
    poses, masks, seeds = _problems(6, 100, seed=M + 50)
    sel = hjcdik.solve_problems(poses, masks, seeds, num_solutions=M, coarse_iters=120, lm_iters=60)
    ids = sel["selected_seed_ids"]; valid = sel["valid"]; eph = sel["cost_physical"]
    for p in range(6):
        vids = ids[p][valid[p]]
        assert len(set(vids.tolist())) == len(vids), f"duplicate seed id in problem {p}"
        assert np.all(vids >= 0)
        e = eph[p][valid[p]]
        assert np.all(np.diff(e) >= -1e-9), f"E_phys not non-decreasing in problem {p}"
    # selected config/metrics of each slot come from that slot's candidate
    allo = hjcdik.solve_problems(poses, masks, seeds, num_solutions=M, coarse_iters=120, lm_iters=60,
                                 return_all_candidates=True)
    for p in range(6):
        for m in range(M):
            s = int(sel["selected_seed_ids"][p, m])
            if s < 0:
                continue
            np.testing.assert_array_equal(sel["joint_config"][p, m], allo["all_joint_config"][p, s])
            np.testing.assert_array_equal(sel["position_errors"][p, m], allo["all_position_errors"][p, s])


# --- distinct valid ids only, then invalid pads (no real-candidate duplication) -------------------
def test_valid_ids_are_distinct_padding_is_minus_one():
    """Every VALID slot is a distinct candidate; any leftover slots are invalid pads (seed=-1). On
    open-world G1 all candidates are valid, so with M <= S every slot is a distinct valid candidate
    and there is no padding -- which still exercises 'no duplicate real ids' at the largest M."""
    poses, masks, seeds = _problems(5, 40, seed=200)
    M = 32                                            # < S, all candidates valid open-world
    sel = hjcdik.solve_problems(poses, masks, seeds, num_solutions=M, coarse_iters=120, lm_iters=60)
    ids = sel["selected_seed_ids"]; valid = sel["valid"]
    for p in range(5):
        assert valid[p].all(), "an open-world candidate came back invalid"
        assert len(set(ids[p].tolist())) == M, f"duplicate seed id in problem {p}"
        assert np.all(ids[p] >= 0)


# --- num_solutions validation ---------------------------------------------------------------------
def test_num_solutions_range():
    poses, masks, seeds = _problems(3, 8, seed=300)
    with pytest.raises(ValueError, match="num_solutions"):
        hjcdik.solve_problems(poses, masks, seeds, num_solutions=0)
    with pytest.raises(ValueError, match="num_solutions"):
        hjcdik.solve_problems(poses, masks, seeds, num_solutions=9)   # > S=8


# --- return_all does not change the selected top-M ------------------------------------------------
@pytest.mark.parametrize("M", [1, 5, 32])
def test_return_all_invariant(M):
    poses, masks, seeds = _problems(5, 64, seed=400 + M)
    off = hjcdik.solve_problems(poses, masks, seeds, num_solutions=M, coarse_iters=120, lm_iters=60,
                                return_all_candidates=False)
    on = hjcdik.solve_problems(poses, masks, seeds, num_solutions=M, coarse_iters=120, lm_iters=60,
                               return_all_candidates=True)
    for key in ("joint_config", "selected_seed_ids", "cost_physical", "success", "valid"):
        np.testing.assert_array_equal(off[key], on[key])


# --- strides / contiguity + zero-alloc reuse for varying M ----------------------------------------
def test_strides_and_workspace():
    sv = hjcdik.HJCDSolver()
    poses, masks, seeds = _problems(8, 64, seed=500)
    sv.solve_problems(poses, masks, seeds, num_solutions=32, coarse_iters=60, lm_iters=30)  # warm big M
    n0 = sv.workspace_stats()["cuda_mallocs"]
    for M in (1, 5, 10, 32):
        sel = sv.solve_problems(poses, masks, seeds, num_solutions=M, coarse_iters=60, lm_iters=30)
        for name, v in sel.items():
            if isinstance(v, np.ndarray) and v.ndim > 0:
                assert v.flags.c_contiguous and 0 not in v.strides, name
    assert sv.workspace_stats()["cuda_mallocs"] == n0    # smaller M reuses the M=32 arena


# --- collision top-M (Panda) ----------------------------------------------------------------------
collision_only = pytest.mark.skipif(K != 1, reason="collision build is the single-target Panda")


@collision_only
@pytest.mark.parametrize("M", [1, 5, 10])
def test_topm_collision(M):
    import json
    from pathlib import Path
    REPO = Path(__file__).resolve().parents[1]
    SET = "bookshelf_small_panda"
    probs = (REPO / "tests" / "mb_problems.json").read_text()
    p = json.loads(probs)["problems"][SET]
    inst = p[0] if isinstance(p, list) else p
    gp = inst["goal_pose"]
    pos = np.asarray(gp.get("position_xyz", gp.get("position")), float)
    quat = np.asarray(gp.get("quaternion_wxyz", gp.get("quat_wxyz")), float); quat /= np.linalg.norm(quat)
    goal = np.concatenate([pos, quat])
    rng = np.random.default_rng(5)
    cand = np.stack([rng.uniform(LO, HI) for _ in range(4000)])
    free = cand[hjcdik.collision_free(cand, probs, SET, 0)]
    P, S = 6, 64
    poses = np.repeat(goal[None, None, :], P, axis=0)
    masks = np.ones(P, dtype=np.uint32)
    seeds = np.resize(free, (P, S, N))
    sel = hjcdik.solve_problems(poses, masks, seeds, num_solutions=M, coarse_iters=120, lm_iters=60,
                                problems_json_text=probs, problem_set_name=SET, problem_idx=0)
    # every valid selected slot across all M is genuinely collision-free
    q = np.asarray(sel["joint_config"], np.float64).reshape(P * M, N)
    v = sel["valid"].reshape(P * M)
    free_out = hjcdik.collision_free(q, probs, SET, 0)
    assert np.all(free_out[v]), "a valid top-M slot collides"
    # CPU agreement on the collision path
    allo = hjcdik.solve_problems(poses, masks, seeds, num_solutions=M, coarse_iters=120, lm_iters=60,
                                 return_all_candidates=True,
                                 problems_json_text=probs, problem_set_name=SET, problem_idx=0)
    cpu = _cpu_topm(allo, M, precision="float32", cc=True)
    np.testing.assert_array_equal(sel["selected_seed_ids"], cpu)


@collision_only
def test_topm_fewer_valid_pads_with_invalid():
    """A problem seeded entirely with colliding configs yields fewer than M valid candidates: the
    valid slots are distinct real ids, the rest are invalid pads (seed=-1), never duplicates."""
    import json
    from pathlib import Path
    REPO = Path(__file__).resolve().parents[1]
    SET = "bookshelf_small_panda"
    probs = (REPO / "tests" / "mb_problems.json").read_text()
    p = json.loads(probs)["problems"][SET]
    inst = p[0] if isinstance(p, list) else p
    gp = inst["goal_pose"]
    pos = np.asarray(gp.get("position_xyz", gp.get("position")), float)
    quat = np.asarray(gp.get("quaternion_wxyz", gp.get("quat_wxyz")), float); quat /= np.linalg.norm(quat)
    goal = np.concatenate([pos, quat])
    rng = np.random.default_rng(9)
    cand = np.stack([rng.uniform(LO, HI) for _ in range(8000)])
    colliding = cand[~hjcdik.collision_free(cand, probs, SET, 0)][:64]
    P, S, M = 3, 64, 16
    poses = np.repeat(goal[None, None, :], P, axis=0)
    masks = np.ones(P, dtype=np.uint32)
    seeds = np.stack([np.resize(colliding, (S, N)) for _ in range(P)])
    sel = hjcdik.solve_problems(poses, masks, seeds, num_solutions=M, coarse_iters=120, lm_iters=60,
                                problems_json_text=probs, problem_set_name=SET, problem_idx=0)
    ids = sel["selected_seed_ids"]; valid = sel["valid"]
    for pi in range(P):
        vids = ids[pi][valid[pi]]
        assert len(set(vids.tolist())) == len(vids), "duplicate real id among valid slots"
        for m in range(M):
            if not valid[pi, m]:
                assert ids[pi, m] == -1
                assert np.isinf(sel["cost_physical"][pi, m])
                assert not sel["collision_free"][pi, m]
                assert np.isfinite(np.asarray(sel["joint_config"][pi, m], np.float64)).all()
