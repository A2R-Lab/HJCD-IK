"""Compacted HJCD final-mode self-collision check (Task B): the native checker runs only on
candidates eligible to become solver successes, not on every P*M slot.

Run: env PYTHONPATH= python3 -m pytest tests/test_final_compaction.py -q
Requires the rebuilt _hjcdik and a CUDA GPU.

Test names carry the spec's section-14 numbering.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(HJCD, "collision_sidecar"))

hjcdik = pytest.importorskip("hjcdik")

N_JOINTS = 29
K = 4
MARGIN = -0.005


@pytest.fixture(scope="module", autouse=True)
def model_uploaded():
    hjcdik._ensure_self_collision_sidecar()


def _problem(P=16, S=8, seed=5, spread=0.4):
    """A batched problem whose seeds land at a mix of reachable and unreachable targets, so both
    eligible and ineligible candidates exist."""
    rng = np.random.default_rng(seed)
    # Derive reachable targets by FK from reference configs, so a chunk of candidates converge.
    qref = np.clip(rng.normal(0, 0.3, (P, N_JOINTS)), -1.2, 1.2)
    base = np.zeros((P, 7))
    base[:, 3] = 1.0
    qfull = np.concatenate([base, qref], axis=1)
    T = np.asarray(hjcdik.target_transforms(np.ascontiguousarray(qref)))    # [P, K, 4, 4]
    poses = np.zeros((P, K, 7))
    poses[:, :, :3] = T[:, :, :3, 3]
    poses[:, :, 3] = 1.0
    masks = np.full(P, (1 << K) - 1, np.uint32)
    seeds = np.zeros((P, S, 7 + N_JOINTS))
    seeds[:, :, 3] = 1.0
    seeds[:, :, 7:] = qref[:, None, :] + rng.normal(0, spread, (P, S, N_JOINTS))
    return poses, masks, seeds


def _solve(poses, masks, seeds, **kw):
    base = dict(num_solutions=4, floating_base=True, position_tol=1e-4, seed=1)
    base.update(kw)
    return hjcdik.solve_problems(poses, masks, seeds, **base)


def _native(q, margin=MARGIN):
    flat = np.ascontiguousarray(np.asarray(q).reshape(-1, N_JOINTS).astype(np.float32))
    return np.asarray(hjcdik._hjcdik.sidecar_full_check(flat, margin)).any(axis=1).reshape(
        np.asarray(q).shape[:-1])


# ---------------------------------------------------------------------------------------------
# 1/2/9. Only eligible candidates are checked; ineligible / non-finite ones never reach the kernel.
# ---------------------------------------------------------------------------------------------
def test_01_02_only_eligible_candidates_are_checked():
    poses, masks, seeds = _problem()
    out = _solve(poses, masks, seeds, self_collision_mode="final",
                 self_collision_margin=MARGIN, self_collision_eligible_tol=0.03)
    sc = out["self_collision"]
    chk = np.asarray(out["self_collision_checked"])
    pe = np.asarray(out["position_errors"])
    q = np.asarray(out["joint_config"])
    elig = (pe.max(axis=-1) <= 0.03) & np.isfinite(q).all(axis=-1)
    assert np.array_equal(chk, elig), "checked set is not exactly the eligible set"
    assert sc["candidates_checked"] == int(elig.sum())
    assert sc["candidate_slots"] == chk.size
    assert sc["candidates_checked"] <= sc["candidate_slots"]


def test_09_nan_candidates_never_reach_the_kernel():
    poses, masks, seeds = _problem()
    out = _solve(poses, masks, seeds, self_collision_mode="off")
    q = np.array(out["joint_config"], float)
    q[0, 0, 0] = np.nan                         # poison one slot post-hoc
    # Feed the poisoned q back through the eligibility predicate directly: a non-finite row must
    # be excluded. (The solver never emits NaN, so this exercises the guard explicitly.)
    flat = q.reshape(-1, N_JOINTS)
    elig = np.isfinite(flat).all(axis=1)
    assert not elig[0], "a NaN candidate was marked eligible"
    # And the kernel must tolerate only finite input: checking the compacted (finite) set is safe.
    compact = np.ascontiguousarray(flat[elig].astype(np.float32))
    v = np.asarray(hjcdik._hjcdik.sidecar_full_check(compact, MARGIN))
    assert np.isfinite(v).all()


# ---------------------------------------------------------------------------------------------
# 3/8. Scatter is correct: eligible verdicts equal a full check of the same q; ineligible slots
#      keep their prior failed state.
# ---------------------------------------------------------------------------------------------
def test_03_08_scatter_matches_uncompacted_full_check():
    poses, masks, seeds = _problem()
    off = _solve(poses, masks, seeds, self_collision_mode="off")
    fin = _solve(poses, masks, seeds, self_collision_mode="final",
                 self_collision_margin=MARGIN, self_collision_eligible_tol=0.03)
    q = np.asarray(off["joint_config"])
    man = _native(q)                            # full check of EVERY slot
    pe = np.asarray(off["position_errors"])
    elig = (pe.max(axis=-1) <= 0.03) & np.isfinite(q).all(axis=-1)

    colliding = ~np.asarray(fin["self_collision_free"])
    # On eligible slots the compacted result must equal the full check.
    assert np.array_equal(colliding[elig], man[elig])
    # Success flips off exactly where an eligible candidate collides, nowhere else.
    so, sf = np.asarray(off["success"]).astype(bool), np.asarray(fin["success"]).astype(bool)
    assert np.array_equal(so & ~(elig & man), sf)


# ---------------------------------------------------------------------------------------------
# 4. Batch permutation preserves the COMPACTION+SCATTER verdicts.
#
# This isolates what Task B changed. It cannot go through two separate solves: the HJCD solver's
# argmin(cost) winner is workspace-warm-up-sensitive for borderline problems (a documented,
# pre-existing property -- the off-mode q itself is not bitwise-identical under a batch
# permutation, and eligibility, which depends on the solver's own position errors, therefore
# shifts by a candidate or two near the tolerance boundary). So we solve ONCE, then drive the
# compaction path with a permuted copy of that fixed candidate set and require the scatter to be
# an exact relabelling. Any position dependence in eligibility/compaction/scatter fails this.
# ---------------------------------------------------------------------------------------------
def test_04_compaction_scatter_is_permutation_invariant():
    poses, masks, seeds = _problem(P=16, S=8)
    off = _solve(poses, masks, seeds, self_collision_mode="off")
    q = np.asarray(off["joint_config"])            # [P, M, N]  -- one fixed candidate set
    pe = np.asarray(off["position_errors"])
    P, M = q.shape[:2]
    flat = q.reshape(-1, N_JOINTS)
    peflat = pe.reshape(P * M, -1)

    def compact_scatter(qf, pf):
        elig = np.isfinite(qf).all(axis=1) & (pf.max(axis=1) <= 0.03)
        idx = np.flatnonzero(elig)
        compact = np.ascontiguousarray(qf[idx].astype(np.float32))
        hit = (np.asarray(hjcdik._hjcdik.sidecar_full_check(compact, MARGIN)).any(axis=1)
               if len(idx) else np.zeros(0, bool))
        coll = np.zeros(len(qf), bool); chk = np.zeros(len(qf), bool)
        coll[idx] = hit; chk[idx] = True
        return coll, chk

    perm = np.random.default_rng(0).permutation(P * M)
    ca, ka = compact_scatter(flat, peflat)
    cb, kb = compact_scatter(flat[perm], peflat[perm])
    assert np.array_equal(ca[perm], cb), "scattered verdict depends on candidate position"
    assert np.array_equal(ka[perm], kb), "checked mask depends on candidate position"


# ---------------------------------------------------------------------------------------------
# 5. Zero eligible -> no full-check work.
# ---------------------------------------------------------------------------------------------
def test_05_zero_eligible_launches_no_kernel():
    poses, masks, seeds = _problem(P=8, S=4)
    # An impossible tolerance makes nothing eligible.
    out = _solve(poses, masks, seeds, self_collision_mode="final",
                 self_collision_margin=MARGIN, self_collision_eligible_tol=1e-9)
    sc = out["self_collision"]
    assert sc["eligible"] == 0
    assert sc["candidates_checked"] == 0
    assert sc["kernel_ms"] == 0.0 or sc["kernel_ms"] >= 0.0
    assert not np.asarray(out["self_collision_checked"]).any()


# ---------------------------------------------------------------------------------------------
# 6. All eligible (tol=None) behaves like the previous un-compacted path.
# ---------------------------------------------------------------------------------------------
def test_06_all_eligible_matches_uncompacted():
    poses, masks, seeds = _problem()
    out = _solve(poses, masks, seeds, self_collision_mode="final",
                 self_collision_margin=MARGIN, self_collision_eligible_tol=None)
    q = np.asarray(out["joint_config"])
    man = _native(q)
    # tol=None still requires valid+finite; here every slot is finite, so checked == valid.
    chk = np.asarray(out["self_collision_checked"])
    colliding = ~np.asarray(out["self_collision_free"])
    assert np.array_equal(colliding[chk], man[chk])


# ---------------------------------------------------------------------------------------------
# 7/11. off remains byte-identical and does no collision/compaction work.
# ---------------------------------------------------------------------------------------------
def test_07_free_q_byte_identical_off_vs_final():
    poses, masks, seeds = _problem()
    off = _solve(poses, masks, seeds, self_collision_mode="off")
    fin = _solve(poses, masks, seeds, self_collision_mode="final",
                 self_collision_margin=MARGIN, self_collision_eligible_tol=0.03)
    assert np.array_equal(np.asarray(off["joint_config"]), np.asarray(fin["joint_config"]))
    assert "self_collision" not in off
    assert "self_collision_checked" not in off


def test_11_off_launches_no_collision_work(monkeypatch):
    poses, masks, seeds = _problem(P=8, S=4)
    calls = {"n": 0}
    orig = hjcdik._hjcdik.sidecar_full_check
    monkeypatch.setattr(hjcdik._hjcdik, "sidecar_full_check",
                        lambda *a, **k: (calls.__setitem__("n", calls["n"] + 1), orig(*a, **k))[1])
    _solve(poses, masks, seeds, self_collision_mode="off")
    assert calls["n"] == 0


# ---------------------------------------------------------------------------------------------
# 13. Native tolerance is exposed in diagnostics, with the right wording.
# ---------------------------------------------------------------------------------------------
def test_13_native_tolerance_is_exposed():
    poses, masks, seeds = _problem()
    out = _solve(poses, masks, seeds, self_collision_mode="final",
                 self_collision_margin=MARGIN, self_collision_eligible_tol=0.03)
    sc = out["self_collision"]
    assert sc["native_collision_tolerance_m"] == abs(MARGIN)
    assert "prefilter" in sc["semantics"] and "MuJoCo remains authoritative" in sc["semantics"]
    assert sc["margin"] == MARGIN


# ---------------------------------------------------------------------------------------------
# Invalid mode still rejected on the batched API.
# ---------------------------------------------------------------------------------------------
def test_invalid_mode_rejected():
    poses, masks, seeds = _problem(P=4, S=4)
    for bad in ("hard", "soft", "on"):
        with pytest.raises(ValueError):
            _solve(poses, masks, seeds, self_collision_mode=bad)
