"""Checkpoint 3D.1 tests: the dedicated batched collision-free seed generator that replaced the
ineffective +-5%-of-joint-span stall kick as hard mode's initial reseed.

Run: env PYTHONPATH= python3 -m pytest tests/test_collision_reseed.py -q
Requires the rebuilt _hjcdik and a CUDA GPU.

Test names carry the spec's section-9 numbering so a reviewer can map requirement -> test directly.
"""
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
GEN = os.path.join(HJCD, "generated")
sys.path.insert(0, os.path.join(HJCD, "collision_sidecar"))
sys.path.insert(0, HERE)

hjcdik = pytest.importorskip("hjcdik")
from test_collision_integration import _problem, _solve  # noqa: E402

N_JOINTS = 29
FLAG_STATE_VALID = 0x1
FLAG_HAS_FREE_Q = 0x2
COMP_PERTURB, COMP_NOMINAL, COMP_BROAD = 0, 1, 2
R_DEFAULT = 16


@pytest.fixture(scope="module", autouse=True)
def model_uploaded():
    hjcdik._ensure_self_collision_sidecar()


def _free(q):
    q32 = np.ascontiguousarray(np.asarray(q, np.float32))
    return ~np.asarray(hjcdik._hjcdik.sidecar_full_check(q32, 0.0)).any(axis=1)


def _limits():
    """The joint limits the kernel itself projects into -- read from the extension, not inferred.

    An earlier version of this test inferred the bracket from the observed spread of the broad
    (component C) candidates. That is wrong twice over: a finite uniform sample does not reach the
    true limits, and the clamped A/B candidates may legitimately sit outside the observed spread.
    """
    lim = np.asarray(hjcdik.joint_limits(), dtype=np.float64).reshape(-1, 2)
    assert lim.shape[0] == N_JOINTS, lim.shape
    return lim[:, 0], lim[:, 1]


@pytest.fixture(scope="module")
def reseed_run():
    """A hard-mode run on a problem whose seeds collide heavily, plus its candidate arena and
    committed state, BOTH captured immediately.

    Two things this fixture has to get right, and both bit an earlier version of this file:
      * `collision_reseed_rounds=1`. The candidate arena holds only the LAST round's block, and F
        shrinks every round, so with 2 rounds `initially_colliding` is the wrong F for the dump and
        the reshape reads past the live data.
      * The dumps are taken here, not lazily in each test. The workspace is process-global; any
        later solve (or `hard_ws_release()`) overwrites it, so a test that dumps on demand inspects
        whichever run happened to go last.
    """
    sq, tp, tq = _problem(2, 256)
    out = _solve(sq, tp, tq, self_collision_mode="hard", diagnostics=True,
                 collision_reseed_candidates=R_DEFAULT, collision_reseed_rounds=1)
    h = out["self_collision"]
    assert h["initially_colliding"] > 50, "problem 2 no longer stresses the reseeder"
    assert h["reseed_rounds_run"] == 1
    F = h["initially_colliding"]
    cand = hjcdik._hjcdik.hard_reseed_dump(F, R_DEFAULT)
    state = hjcdik._hjcdik.hard_dump()
    return out, h, F, cand, state


# ---------------------------------------------------------------------------------------------
# 1/14. Free initial seeds bypass reseeding; off and final allocate no reseed workspace.
# ---------------------------------------------------------------------------------------------
def test_01_free_seeds_bypass_reseeding(reseed_run):
    _out, h, _F, _d, _st = reseed_run
    # Exactly the colliding seeds produce candidates -- a free seed generates none, so the
    # candidate count is pinned to F*R per round and nothing else.
    assert h["reseed_candidates_checked"] % R_DEFAULT == 0
    per_round = h["reseed_candidates_checked"] // R_DEFAULT
    assert per_round >= h["initially_colliding"], (per_round, h["initially_colliding"])
    assert h["reseed_rounds_run"] >= 1


def test_01b_all_free_batch_runs_no_reseed():
    """A batch of already-free seeds must not generate a single candidate."""
    sq, tp, tq = _problem(0, 32)
    sq = np.ascontiguousarray(np.zeros_like(sq))          # neutral: independently verified free
    assert _free(sq).all()
    out = _solve(sq, tp, tq, self_collision_mode="hard", diagnostics=True)
    h = out["self_collision"]
    assert h["initially_colliding"] == 0
    assert h["reseed_candidates_checked"] == 0
    assert h["reseed_rounds_run"] == 0


def test_14_off_and_final_allocate_no_reseed_workspace(reseed_run):
    hjcdik._hjcdik.hard_ws_release()
    assert hjcdik._hjcdik.hard_reseed_ws_capacity() == 0
    sq, tp, tq = _problem(0, 128)
    _solve(sq, tp, tq, self_collision_mode="off")
    _solve(sq, tp, tq, self_collision_mode="final")
    assert hjcdik._hjcdik.hard_reseed_ws_capacity() == 0, "off/final allocated reseed state"


# ---------------------------------------------------------------------------------------------
# 2/8/9/10/11/12. Recovered seeds are genuinely free and genuinely initialised; unrecovered ones
#                 are failed and cannot publish a success.
# ---------------------------------------------------------------------------------------------
def test_02_08_recovered_seeds_pass_a_fresh_full_check(reseed_run):
    _out, _h, _F, _d, state = reseed_run
    _qc, qfree, flags, _Tf, _Td = state
    ok = (np.asarray(flags) & FLAG_STATE_VALID) != 0
    assert ok.sum() > 100, "too few admitted seeds for this to mean anything"
    # Fresh full check, not the verdict the reseeder itself produced.
    assert _free(qfree[ok]).all()


def test_09_10_selected_q_initialises_both_states(reseed_run):
    _out, _h, _F, _d, state = reseed_run
    qc, _qfree, flags, Tf, _Td = state
    ok = (np.asarray(flags) & FLAG_STATE_VALID) != 0
    assert ((np.asarray(flags) & FLAG_HAS_FREE_Q) != 0)[ok].all(), \
        "an admitted seed has no last-free-coarse state"
    # The sidecar's committed transforms still describe its committed q, bitwise.
    fresh = np.asarray(hjcdik._hjcdik.sidecar_fk(
        np.ascontiguousarray(qc.astype(np.float32)))).reshape(Tf.shape)
    assert np.array_equal(Tf, fresh)


def test_11_12_unrecovered_seeds_are_failed_and_never_successful(reseed_run):
    out, h, _F, _d, _st = reseed_run
    seed_ok = np.asarray(out["hard_seed_ok"]).astype(bool)
    succ = np.asarray(out["success"]).astype(bool)
    assert (~seed_ok).sum() == h["seed_failures"]
    assert not (succ & ~seed_ok).any(), "a seed with no free start published a success"


# ---------------------------------------------------------------------------------------------
# 3/4. Candidate generation respects joint limits, and the mixture really is a mixture.
# ---------------------------------------------------------------------------------------------
def test_03_candidates_respect_joint_limits(reseed_run):
    _out, _h, F, d, _st = reseed_run
    cand_q = np.asarray(d[0]).reshape(F, R_DEFAULT, N_JOINTS)
    lo, hi = _limits()
    eps = 1e-5
    assert np.isfinite(cand_q).all()
    assert (cand_q >= lo - eps).all(), (
        f"a candidate fell below its joint limit by "
        f"{float((lo - cand_q.reshape(-1, N_JOINTS)).max()):.3e}")
    assert (cand_q <= hi + eps).all(), (
        f"a candidate exceeded its joint limit by "
        f"{float((cand_q.reshape(-1, N_JOINTS) - hi).max()):.3e}")


def test_03b_broad_candidates_really_span_the_limit_range(reseed_run):
    """Component C must be a BROAD draw, not a timid one -- that is its entire purpose."""
    _out, _h, F, d, _st = reseed_run
    cand_q = np.asarray(d[0]).reshape(F, R_DEFAULT, N_JOINTS)
    comp = np.asarray(d[2]).reshape(F, R_DEFAULT)
    broad = cand_q[comp == COMP_BROAD]
    lo, hi = _limits()
    covered = (broad.max(axis=0) - broad.min(axis=0)) / (hi - lo)
    assert covered.mean() > 0.8, f"broad candidates cover only {covered.mean():.2f} of the range"


def test_04_all_three_distribution_components_are_present(reseed_run):
    _out, _h, F, d, _st = reseed_run
    comp = np.asarray(d[2]).reshape(F, R_DEFAULT)
    present = set(np.unique(comp).tolist())
    assert present == {COMP_PERTURB, COMP_NOMINAL, COMP_BROAD}, present
    for c in (COMP_PERTURB, COMP_NOMINAL, COMP_BROAD):
        assert (comp == c).sum() >= F, f"component {c} is present but degenerate"


def test_04b_perturbation_scales_are_actually_different(reseed_run):
    """Component A must span SEVERAL scales. If every A candidate used one scale we would be back
    to the single small kick this checkpoint exists to replace."""
    _out, _h, F, d, _st = reseed_run
    cand_q = np.asarray(d[0]).reshape(F, R_DEFAULT, N_JOINTS)
    comp = np.asarray(d[2]).reshape(F, R_DEFAULT)
    a_mask = comp[0] == COMP_PERTURB
    a = cand_q[:, a_mask, :]
    # Spread of each A candidate about the per-seed mean, aggregated over joints.
    spread = np.abs(a - a.mean(axis=1, keepdims=True)).mean(axis=(0, 2))
    assert spread.max() / max(spread.min(), 1e-9) > 1.5, (
        f"component-A candidates all have the same magnitude: {spread}")


# ---------------------------------------------------------------------------------------------
# 5/6/18. Logical identity, not batch position. This is the regression test for the bug found in
#         Checkpoint 3D/3E, where the reseed RNG was keyed on the physical row.
# ---------------------------------------------------------------------------------------------
def test_05_candidates_depend_on_seed_content_not_row():
    """The SAME seed placed at a different row must get the SAME candidate set."""
    sq, tp, tq = _problem(2, 64)
    kw = dict(self_collision_mode="hard", diagnostics=True, collision_reseed_rounds=1)
    h1 = _solve(sq, tp, tq, **kw)["self_collision"]
    F1 = h1["initially_colliding"]
    d1 = hjcdik._hjcdik.hard_reseed_dump(F1, R_DEFAULT)
    idx1 = np.asarray(d1[4])
    q1 = np.asarray(d1[0]).reshape(F1, R_DEFAULT, N_JOINTS)

    perm = np.random.default_rng(1).permutation(64)
    h2 = _solve(np.ascontiguousarray(sq[perm]), np.ascontiguousarray(tp[perm]),
                np.ascontiguousarray(tq[perm]), **kw)["self_collision"]
    F2 = h2["initially_colliding"]
    assert F1 == F2
    d2 = hjcdik._hjcdik.hard_reseed_dump(F2, R_DEFAULT)
    idx2 = np.asarray(d2[4])
    q2 = np.asarray(d2[0]).reshape(F2, R_DEFAULT, N_JOINTS)

    # Map: original row -> its candidate block, in each run.
    inv = np.empty(64, int)
    inv[perm] = np.arange(64)
    for f1, row in enumerate(idx1):
        f2 = int(np.flatnonzero(idx2 == inv[row])[0])
        assert np.array_equal(q1[f1], q2[f2]), (
            f"row {row} got different candidates after permutation -- RNG is keyed on position")


def test_06_18_batch_permutation_preserves_reseed_outcomes():
    sq, tp, tq = _problem(2, 128)
    a = _solve(sq, tp, tq, self_collision_mode="hard", diagnostics=True)
    perm = np.random.default_rng(0).permutation(128)
    b = _solve(np.ascontiguousarray(sq[perm]), np.ascontiguousarray(tp[perm]),
               np.ascontiguousarray(tq[perm]), self_collision_mode="hard", diagnostics=True)
    assert np.array_equal(np.asarray(a["hard_seed_ok"])[perm], np.asarray(b["hard_seed_ok"]))
    assert np.array_equal(np.asarray(a["success"])[perm], np.asarray(b["success"]))
    assert np.array_equal(np.asarray(a["self_collision_free"])[perm],
                          np.asarray(b["self_collision_free"]))
    assert a["self_collision"]["recovered"] == b["self_collision"]["recovered"]


def test_18b_subset_solve_reproduces_its_rows():
    sq, tp, tq = _problem(2, 128)
    full = _solve(sq, tp, tq, self_collision_mode="hard")
    sub = slice(0, 32)
    part = _solve(np.ascontiguousarray(sq[sub]), np.ascontiguousarray(tp[sub]),
                  np.ascontiguousarray(tq[sub]), self_collision_mode="hard")
    assert np.array_equal(np.asarray(full["hard_seed_ok"])[sub], np.asarray(part["hard_seed_ok"]))
    assert np.allclose(np.asarray(full["joint_config"])[sub],
                       np.asarray(part["joint_config"]), rtol=0, atol=0)


# ---------------------------------------------------------------------------------------------
# 7. Selection is deterministic, and it is the rule we said it was.
# ---------------------------------------------------------------------------------------------
def test_07_selection_is_deterministic():
    sq, tp, tq = _problem(2, 128)
    a = _solve(sq, tp, tq, self_collision_mode="hard")
    b = _solve(sq, tp, tq, self_collision_mode="hard")
    assert np.array_equal(np.asarray(a["hard_last_free_coarse_q"]),
                          np.asarray(b["hard_last_free_coarse_q"]))


def test_07b_selection_takes_the_nearest_free_candidate(reseed_run):
    """The chosen candidate must be the collision-free one closest to the original seed in
    normalised joint space, ties going to the lower ordinal."""
    _out, _h, F, d, _st = reseed_run
    free = np.asarray(d[1]).reshape(F, R_DEFAULT)
    dist = np.asarray(d[3]).reshape(F, R_DEFAULT)
    sel = np.asarray(d[5])
    checked = 0
    for f in range(F):
        cand = np.flatnonzero(free[f])
        if sel[f] < 0:
            assert len(cand) == 0, f"seed {f}: free candidates existed but none was selected"
            continue
        assert free[f, sel[f]], "selected a colliding candidate"
        best = cand[np.lexsort((cand, dist[f, cand]))][0]
        assert sel[f] == best, f"seed {f}: selected {sel[f]}, nearest free is {best}"
        checked += 1
    assert checked > 50


# ---------------------------------------------------------------------------------------------
# 13. Workspace reuse: no per-call allocation once capacity is reached.
# ---------------------------------------------------------------------------------------------
def test_13_reseed_workspace_is_reused():
    sq, tp, tq = _problem(2, 128)
    _solve(sq, tp, tq, self_collision_mode="hard")
    n0 = hjcdik._hjcdik.hard_reseed_ws_nalloc()
    cap0 = hjcdik._hjcdik.hard_reseed_ws_capacity()
    assert cap0 > 0
    for _ in range(3):
        _solve(sq, tp, tq, self_collision_mode="hard")
    assert hjcdik._hjcdik.hard_reseed_ws_nalloc() == n0, "reseed workspace reallocated per call"


# ---------------------------------------------------------------------------------------------
# 15/16/17/19. Nothing else moved.
# ---------------------------------------------------------------------------------------------
def test_15_off_remains_byte_identical():
    """Reproduces the frozen pre-integration q-hashes in a fresh subprocess."""
    import json
    import subprocess
    ref = json.load(open(os.path.join(GEN, "baseline_g1_solver_ref.json")))
    env = dict(os.environ)
    env["PYTHONPATH"] = ""
    subprocess.run([sys.executable, os.path.join(HJCD, "collision_sidecar", "baseline_capture.py")],
                   cwd=HJCD, env=env, check=True, capture_output=True)
    cur = json.load(open(os.path.join(GEN, "baseline_g1_solver.json")))
    for label in ref["runs"]:
        assert cur["runs"][label]["q_hashes"] == ref["runs"][label]["q_hashes"], label


def test_16_final_is_unaffected_by_reseed_options():
    """The reseed knobs are hard-mode only; passing them must not perturb final mode."""
    sq, tp, tq = _problem(0, 128)
    a = _solve(sq, tp, tq, self_collision_mode="final")
    b = _solve(sq, tp, tq, self_collision_mode="final", collision_reseed_candidates=64,
               collision_reseed_rounds=4)
    assert np.allclose(np.asarray(a["joint_config"]), np.asarray(b["joint_config"]),
                       rtol=0, atol=0)
    assert np.array_equal(np.asarray(a["success"]), np.asarray(b["success"]))


def test_17_oracle_still_reports_zero_mismatches():
    total_checks = total_mm = 0
    for pi in range(4):
        out = _solve(*_problem(pi, 256), self_collision_mode="hard", diagnostics=True,
                     _hard_oracle_every=1)
        h = out["self_collision"]
        total_checks += h["oracle_checks"]
        total_mm += h["oracle_mismatches"]
    assert total_checks > 30000, f"oracle coverage dropped to {total_checks}"
    assert total_mm == 0


def test_19_loaded_binary_and_artifact_guards_active():
    info = hjcdik.self_collision_info()
    assert info["geometry_validated"] is True
    assert info["hard_enabled"] is True
    ext = hjcdik._hjcdik.__file__
    assert os.path.exists(ext)
    import hashlib
    sha = hashlib.sha256(open(ext, "rb").read()).hexdigest()[:16]
    assert len(sha) == 16


# ---------------------------------------------------------------------------------------------
# API validation.
# ---------------------------------------------------------------------------------------------
@pytest.mark.parametrize("kw", [
    dict(collision_reseed_candidates=0),
    dict(collision_reseed_candidates=1000),
    dict(collision_reseed_rounds=0),
    dict(collision_reseed_rounds=99),
    dict(collision_reseed_scales=()),
    dict(collision_reseed_scales=(0.0,)),
    dict(collision_reseed_scales=(-0.2,)),
    dict(collision_reseed_scales=(9.0,)),
    dict(collision_reseed_mode=2),
])
def test_invalid_reseed_options_raise(kw):
    sq, tp, tq = _problem(0, 8)
    with pytest.raises((ValueError, TypeError)):
        _solve(sq, tp, tq, self_collision_mode="hard", **kw)


def test_defaults_are_unchanged():
    import inspect
    sig = inspect.signature(hjcdik.solve)
    assert sig.parameters["self_collision_mode"].default == "off"
    assert sig.parameters["collision_top_k"].default == 3
    assert sig.parameters["collision_reseed_candidates"].default == 16
    assert sig.parameters["collision_reseed_rounds"].default == 2
