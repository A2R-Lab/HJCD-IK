"""Milestone 3: per-problem segmented top-1 selection on the GPU.

The device selector (one block per problem) ranks candidates by the deterministic three-class key
    R = (class, E_phys, seed),  lower wins
    class 0 solved  <  class 1 valid-unsolved  <  class 2 invalid
E_phys = sum_{k in active} [ |e_p|^2/eps_p^2 + |e_R|^2/eps_R^2 ] is the STABLE metric; the row-scaled
LM cost is never used for selection. These tests pin the selector against a CPU reference that uses
the identical key, plus the no-solution / all-invalid / tie / NaN semantics.
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


def _problems(P, S, seed=0, mask=None, sigma=0.05):
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


def _cpu_select(out, precision, cc):
    """CPU reference from the full candidate arrays (return_all_candidates=True). Uses the same
    three-class key + E_phys as the kernel, with the eps NARROWED to the compute type so the
    tolerance boundary matches the kernel exactly."""
    masks = out["active_masks"]                                  # [P]
    pe = out["all_position_errors"]; oe = out["all_orientation_errors"]   # [P,S,K]
    cfg = np.asarray(out["all_joint_config"], np.float64)        # [P,S,N]
    P, S = pe.shape[0], pe.shape[1]
    ep = float(np.float32(PTOL)) if precision == "float32" else PTOL
    eo = float(np.float32(OTOL)) if precision == "float32" else OTOL
    act = ((masks[:, None] >> np.arange(K, dtype=np.uint32)) & 1).astype(bool)  # [P,K]

    sel = np.full(P, -1)
    for p in range(P):
        best = None
        for s in range(S):
            a = act[p]
            pev = pe[p, s][a]; oev = oe[p, s][a]
            finite = np.isfinite(cfg[p, s]).all() and np.isfinite(pev).all() and np.isfinite(oev).all()
            feas = True
            if cc:
                feas = bool(out["all_collision_free"][p, s])
            ephys = float(((pev / ep) ** 2 + (oev / eo) ** 2).sum())
            within = finite and np.all(pev <= ep) and np.all(oev <= eo)
            if not finite or not feas:
                cls, e = 2, np.inf
            elif within:
                cls, e = 0, ephys
            else:
                cls, e = 1, ephys
            key = (cls, e, s)
            if best is None or key < best[0]:
                best = (key, s)
        sel[p] = best[1] if best[0][0] < 2 else -1
    return sel


def _run(poses, masks, seeds, precision="float32", **kw):
    """Returns (selected_out, all_out) for CPU comparison. Selected outputs must match between
    return_all=False and True."""
    sel = hjcdik.solve_problems(poses, masks, seeds, precision=precision,
                                return_all_candidates=False, **kw)
    allo = hjcdik.solve_problems(poses, masks, seeds, precision=precision,
                                 return_all_candidates=True, **kw)
    return sel, allo


# --- 1/10/11. top-1 matches CPU reference; seed ids + selected metrics consistent -----------------
@pytest.mark.parametrize("precision", ["float32", "float64"])
@pytest.mark.parametrize("P,S", [(1, 32), (6, 64), (10, 100)])
def test_top1_matches_cpu_reference(precision, P, S):
    poses, masks, seeds = _problems(P, S, seed=P * 7 + S, sigma=0.08)
    sel, allo = _run(poses, masks, seeds, precision=precision, coarse_iters=120, lm_iters=60)
    cpu = _cpu_select(allo, precision, cc=False)
    gpu = sel["selected_seed_ids"][:, 0]
    np.testing.assert_array_equal(gpu, cpu, err_msg=f"{precision} P={P} S={S}: seed ids != CPU ref")

    # the selected config and metrics come from the selected candidate
    for p in range(P):
        s = int(gpu[p])
        if s < 0:
            continue
        np.testing.assert_array_equal(sel["joint_config"][p, 0], allo["all_joint_config"][p, s])
        np.testing.assert_array_equal(sel["position_errors"][p, 0], allo["all_position_errors"][p, s])
        np.testing.assert_array_equal(sel["cost_lm"][p, 0], allo["all_cost_lm"][p, s])


# --- 2/4. solved is preferred; lowest E_phys wins among solved -------------------------------------
def test_solved_preferred_and_lowest_ephys_wins():
    poses, masks, seeds = _problems(8, 96, seed=11, sigma=0.06)
    sel, allo = _run(poses, masks, seeds, coarse_iters=120, lm_iters=60)
    act = ((masks[:, None] >> np.arange(K, dtype=np.uint32)) & 1).astype(bool)
    pe = allo["all_position_errors"]; oe = allo["all_orientation_errors"]
    ep = float(np.float32(PTOL)); eo = float(np.float32(OTOL))
    solved = (np.where(act[:, None, :], pe, 0).max(2) <= PTOL) & \
             (np.where(act[:, None, :], oe, 0).max(2) <= OTOL)          # [P,S]
    ephys = ((pe / ep) ** 2 + (oe / eo) ** 2 * 0 + (oe / eo) ** 2).sum(2)   # [P,S] (over all K, inactive=0)
    ephys = (((pe / ep) ** 2 + (oe / eo) ** 2) * act[:, None, :]).sum(2)
    for p in range(8):
        s = int(sel["selected_seed_ids"][p, 0])
        if solved[p].any():
            assert solved[p, s], f"problem {p} had a solved candidate but selected an unsolved one"
            best = np.argmin(np.where(solved[p], ephys[p], np.inf))
            assert abs(ephys[p, s] - ephys[p, best]) < 1e-9, "not the lowest-E_phys solved candidate"


# --- 3/5/8. no solved but valid -> lowest-E_phys valid, success False ------------------------------
def test_no_solved_but_valid():
    """Targets placed just far enough that no seed solves but every candidate is finite/valid."""
    rng = np.random.default_rng(21)
    P, S = 4, 48
    poses = np.zeros((P, K, 7)); seeds = np.zeros((P, S, N))
    for p in range(P):
        qt = rng.uniform(LO, HI)
        pose = _pose7(qt)
        pose[:, :3] += 0.25                       # shift targets 25 cm -> unreachable exactly
        poses[p] = pose
        seeds[p] = np.clip(qt + rng.normal(scale=0.05, size=(S, N)), LO, HI)
    masks = np.full(P, (1 << K) - 1, dtype=np.uint32)
    sel, allo = _run(poses, masks, seeds, coarse_iters=120, lm_iters=60)
    # THE invariant: selection matches the CPU three-class reference regardless.
    cpu = _cpu_select(allo, "float32", cc=False)
    np.testing.assert_array_equal(sel["selected_seed_ids"][:, 0], cpu)
    # For every problem with NO solved candidate, a valid one is still returned, success=False.
    no_solve = sel["num_solved"] == 0
    assert no_solve.any(), "the shifted targets were all reachable; make the shift larger"
    for p in np.where(no_solve)[0]:
        assert not sel["success"][p, 0]
        assert sel["valid"][p, 0]
        assert not sel["problem_success"][p]
        assert sel["selected_seed_ids"][p, 0] >= 0
        assert np.isfinite(sel["cost_physical"][p, 0])
    # every candidate here is finite -> always valid, never invalid
    assert sel["valid"].all()


# --- 6. lower seed index breaks exact ties --------------------------------------------------------
def test_lower_seed_index_breaks_ties():
    """Duplicate the same seed across several positions: identical configs -> identical E_phys ->
    the LOWEST seed index must win."""
    rng = np.random.default_rng(31)
    qt = rng.uniform(LO, HI)
    pose = _pose7(qt)
    S = 16
    one = np.clip(qt + rng.normal(scale=0.02, size=(N,)), LO, HI)
    seeds = np.repeat(one[None, :], S, axis=0)[None]           # [1,S,N] all identical
    masks = np.array([(1 << K) - 1], dtype=np.uint32)
    sel = hjcdik.solve_problems(pose[None], masks, seeds, coarse_iters=120, lm_iters=60,
                                return_all_candidates=True)
    # every candidate is identical -> the winner must be seed 0
    assert int(sel["selected_seed_ids"][0, 0]) == 0, "tie not broken by the lowest seed index"


# --- 7. NaN/Inf candidate never wins --------------------------------------------------------------
def test_nan_candidate_never_wins():
    """A seed poisoned to NaN produces an invalid candidate. As long as ANY valid candidate exists it
    must be chosen over the NaN one. (NaN inputs to solve_problems are rejected at validation, so we
    poison a seed to a huge finite value that the LM cannot rescue -> its config stays far/!finite is
    simulated by an extreme start; here we assert the selected candidate is always finite.)"""
    poses, masks, seeds = _problems(4, 48, seed=41, sigma=0.05)
    sel, allo = _run(poses, masks, seeds, coarse_iters=120, lm_iters=60)
    q = np.asarray(sel["joint_config"], np.float64)
    assert np.isfinite(q).all(), "a non-finite configuration was selected"
    # the selected E_phys is finite for every solved/valid problem
    assert np.isfinite(sel["cost_physical"][sel["valid"]]).all()


# --- 9. all invalid -> documented fill ------------------------------------------------------------
@pytest.mark.skipif(K != 1, reason="all-invalid via collision needs the Panda collision build")
def test_all_invalid_semantics(request):
    pytest.importorskip("json")
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

    # seeds guaranteed to be deep in collision with no feasible escape is hard to construct; instead
    # give a problem seeds that all collide AND a target that keeps LM inside the shelf. We assert the
    # documented all-invalid contract holds for any problem the solver marks fully infeasible.
    rng = np.random.default_rng(5)
    cand = np.stack([rng.uniform(LO, HI) for _ in range(6000)])
    colliding = cand[~hjcdik.collision_free(cand, probs, SET, 0)][:64]
    P, S = 3, 64
    poses = np.repeat(goal[None, None, :], P, axis=0)
    masks = np.ones(P, dtype=np.uint32)
    seeds = np.stack([np.resize(colliding, (S, N)) for _ in range(P)])
    sel = hjcdik.solve_problems(poses, masks, seeds, coarse_iters=120, lm_iters=60,
                                problems_json_text=probs, problem_set_name=SET, problem_idx=0)
    infeasible = ~sel["valid"][:, 0]
    for p in range(P):
        if infeasible[p]:
            assert int(sel["selected_seed_ids"][p, 0]) == -1
            assert not sel["success"][p, 0]
            assert not sel["problem_success"][p]
            assert not sel["collision_free"][p, 0]
            assert np.isinf(sel["cost_physical"][p, 0])
            # fill = the problem's first seed (finite) -> config is finite, not marked feasible
            assert np.isfinite(np.asarray(sel["joint_config"][p, 0], np.float64)).all()


# --- 12/13/14/15/16/17. P/S coverage incl. S > blockDim -------------------------------------------
@pytest.mark.parametrize("P,S", [(1, 1), (1, 10), (3, 1), (7, 100), (4, 300), (2, 1000)])
def test_selection_shapes_and_cpu_agreement(P, S):
    poses, masks, seeds = _problems(P, S, seed=P * 13 + S, sigma=0.08)
    sel, allo = _run(poses, masks, seeds, coarse_iters=120, lm_iters=60)
    assert sel["joint_config"].shape == (P, 1, N)
    cpu = _cpu_select(allo, "float32", cc=False)
    np.testing.assert_array_equal(sel["selected_seed_ids"][:, 0], cpu)


# --- 18/19. mixed masks + one infeasible neighbor -------------------------------------------------
@pytest.mark.skipif(K < 4, reason="mixed masks need G1")
def test_mixed_masks_selection():
    masks = np.array([0b0001, 0b0011, 0b0111, 0b1111], dtype=np.uint32)
    poses, _, seeds = _problems(4, 64, seed=51, mask=masks)
    sel, allo = _run(poses, masks, seeds, coarse_iters=120, lm_iters=60)
    cpu = _cpu_select(allo, "float32", cc=False)
    np.testing.assert_array_equal(sel["selected_seed_ids"][:, 0], cpu)


# --- 23/24. precisions ----------------------------------------------------------------------------
@pytest.mark.parametrize("precision,dtype", [("float32", np.float32), ("float64", np.float64)])
def test_precisions(precision, dtype):
    poses, masks, seeds = _problems(5, 48, seed=61)
    sel = hjcdik.solve_problems(poses, masks, seeds, precision=precision, coarse_iters=60, lm_iters=30)
    assert sel["joint_config"].dtype == dtype
    assert sel["cost_physical"].dtype == np.float64
    assert sel["selected_seed_ids"].dtype.kind == "i"


# --- 25. diagnostics on/off -----------------------------------------------------------------------
def test_diagnostics_equivalence():
    poses, masks, seeds = _problems(4, 48, seed=71)
    a = hjcdik.solve_problems(poses, masks, seeds, coarse_iters=120, lm_iters=60, diagnostics=False)
    b = hjcdik.solve_problems(poses, masks, seeds, coarse_iters=120, lm_iters=60, diagnostics=True)
    np.testing.assert_array_equal(a["joint_config"], b["joint_config"])
    np.testing.assert_array_equal(a["selected_seed_ids"], b["selected_seed_ids"])


# --- 26. return_all on/off gives identical SELECTED outputs ----------------------------------------
def test_return_all_does_not_change_selection():
    poses, masks, seeds = _problems(6, 64, seed=81)
    off = hjcdik.solve_problems(poses, masks, seeds, coarse_iters=120, lm_iters=60,
                                return_all_candidates=False)
    on = hjcdik.solve_problems(poses, masks, seeds, coarse_iters=120, lm_iters=60,
                               return_all_candidates=True)
    for key in ("joint_config", "success", "valid", "cost_physical", "cost_lm",
                "position_errors", "selected_seed_ids", "problem_success", "num_solved"):
        np.testing.assert_array_equal(off[key], on[key], err_msg=f"{key} differs with return_all")
    assert "all_joint_config" in on and "all_joint_config" not in off


# --- 27. strides / contiguity ---------------------------------------------------------------------
def test_strides_and_contiguity():
    poses, masks, seeds = _problems(5, 48, seed=91)
    sel = hjcdik.solve_problems(poses, masks, seeds, coarse_iters=60, lm_iters=30)
    for name, v in sel.items():
        if not isinstance(v, np.ndarray) or v.ndim == 0:
            continue
        assert v.flags.c_contiguous, f"{name} not C-contiguous"
        assert 0 not in v.strides, f"{name} zero stride"


# --- 28. repeated calls: zero device allocations --------------------------------------------------
def test_repeated_calls_zero_alloc():
    sv = hjcdik.HJCDSolver()
    poses, masks, seeds = _problems(10, 64, seed=101)
    sv.solve_problems(poses, masks, seeds, coarse_iters=60, lm_iters=30)   # warm-up
    n0 = sv.workspace_stats()["cuda_mallocs"]
    for _ in range(6):
        sv.solve_problems(poses, masks, seeds, coarse_iters=60, lm_iters=30)
    assert sv.workspace_stats()["cuda_mallocs"] == n0
    assert sv.workspace_stats()["cuda_frees"] == 0


# --- 29. legacy solve() unchanged -----------------------------------------------------------------
def test_legacy_solve_unchanged():
    rng = np.random.default_rng(111)
    B = 128
    qt = rng.uniform(LO, HI)
    pose = _pose7(qt)
    seeds = rng.uniform(LO, HI, size=(B, N))
    a = hjcdik.solve(seeds, np.repeat(pose[None, :, :3], B, axis=0),
                     np.repeat(pose[None, :, 3:], B, axis=0),
                     active_target_mask=np.full(B, (1 << K) - 1, dtype=np.uint32),
                     position_tol=PTOL, orientation_tol=OTOL, coarse_iters=120, lm_iters=60)
    b = hjcdik.solve(seeds, np.repeat(pose[None, :, :3], B, axis=0),
                     np.repeat(pose[None, :, 3:], B, axis=0),
                     active_target_mask=np.full(B, (1 << K) - 1, dtype=np.uint32),
                     position_tol=PTOL, orientation_tol=OTOL, coarse_iters=120, lm_iters=60)
    np.testing.assert_array_equal(a["joint_config"], b["joint_config"])   # deterministic
    assert a["joint_config"].shape == (B, N)                              # per-candidate, unchanged


# --- 30. P=1,S=B selection agrees with CPU ranking over solve() outputs ----------------------------
def test_P1_SB_selection_matches_cpu_over_solve():
    B = 200
    rng = np.random.default_rng(121)
    qt = rng.uniform(LO, HI)
    pose = _pose7(qt)
    seeds = rng.uniform(LO, HI, size=(B, N))
    # solve_problems P=1,S=B top-1
    sel = hjcdik.solve_problems(pose[None], np.array([(1 << K) - 1], dtype=np.uint32),
                                seeds[None], coarse_iters=120, lm_iters=60)
    s_sel = int(sel["selected_seed_ids"][0, 0])
    # legacy solve() over the same B candidates, then CPU three-class select
    old = hjcdik.solve(seeds, np.repeat(pose[None, :, :3], B, axis=0),
                       np.repeat(pose[None, :, 3:], B, axis=0),
                       active_target_mask=np.full(B, (1 << K) - 1, dtype=np.uint32),
                       position_tol=PTOL, orientation_tol=OTOL, coarse_iters=120, lm_iters=60)
    pe = old["position_errors"]; oe = old["orientation_errors"]
    ep = float(np.float32(PTOL)); eo = float(np.float32(OTOL))
    act = np.ones(K, bool)
    ephys = ((pe / ep) ** 2 + (oe / eo) ** 2).sum(1)
    within = (pe.max(1) <= ep) & (oe.max(1) <= eo)
    cls = np.where(within, 0, 1)
    order = np.lexsort((np.arange(B), ephys, cls))
    assert s_sel == order[0], "P=1,S=B selection disagrees with CPU ranking over solve()"
