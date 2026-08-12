"""The production default: fp32 + coarse 120 / LM 60 caps + Policy B (patience 2, rel 1e-3).

    precision="float32", coarse_iters=120, lm_iters=60, stag_patience=2, stag_rel=1e-3

Policy B stops a seed when it makes negligible progress in E_phys -- a TOLERANCE-NORMALISED PHYSICAL
error -- for `stag_patience` consecutive iterations. It is deliberately NOT driven by the row-scaled
cost: the row scales s_{k,r} = 1/||J_{k,r}|| are re-frozen every iteration, so C^(t) and C^(t-1) are
expressed in different units and a "relative improvement" between them is not a real quantity. A seed
whose scaled cost merely wobbles because its scaling changed must NOT be killed; a seed whose
physical error has genuinely flatlined must be.

stag_patience=0 is the explicit opt-out and exactly restores the previous fixed-cap behaviour. The
hard caps still bound the worst case.
"""
import numpy as np
import pytest

import hjcdik

N = hjcdik.num_joints()
K = hjcdik.num_targets()
LIM = hjcdik.joint_limits()
LO, HI = LIM[:, 0], LIM[:, 1]
PTOL, OTOL = 1e-4, 1e-3
SEED = 12345


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


@pytest.fixture(scope="module")
def mixed():
    """Half the seeds start next to the answer (they solve fast), half are random restarts (they
    stall). Both stop reasons are therefore exercised in one batch."""
    B = 256
    rng = np.random.default_rng(SEED)
    q_true = rng.uniform(LO, HI)
    T = hjcdik.target_transforms(q_true[None, :])[0]
    p = np.repeat(T[:, :3, 3][None], B, axis=0)
    q = np.repeat(np.stack([_quat(T[k][:3, :3]) for k in range(K)])[None], B, axis=0)
    seeds = rng.uniform(LO, HI, size=(B, N))
    seeds[1::2] = np.clip(q_true + rng.normal(scale=0.02, size=(B // 2, N)), LO, HI)
    m = np.full(B, (1 << K) - 1, dtype=np.uint32)
    return seeds, p, q, m


# --- 1/2/3. the default, the opt-out, and the fixed cap -------------------------------------------
def test_public_default_enables_policy_b():
    import inspect
    assert hjcdik.DEFAULT_STAG_PATIENCE == 2 and hjcdik.DEFAULT_STAG_REL == 1e-3
    for fn in (hjcdik.solve, hjcdik.refine):
        p = inspect.signature(fn).parameters
        assert p["precision"].default == "float32"
        assert p["stag_patience"].default == 2
        assert p["stag_rel"].default == 1e-3


def test_stag_patience_zero_exactly_restores_the_fixed_cap(mixed):
    """The opt-out must be EXACT, not approximately equivalent."""
    s, p, q, m = mixed
    kw = dict(active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL, max_iters=60,
              diagnostics=True)
    off = hjcdik.refine(s, p, q, stag_patience=0, **kw)
    # with the adaptive stop disabled, unsolved seeds must run the full cap
    unsolved = ~off["success"].astype(bool)
    assert unsolved.any()
    assert (off["lm_iterations"][unsolved] >= 60).all(), \
        "stag_patience=0 still stopped an unsolved seed early"


def test_explicit_fixed_cap_behaviour_is_still_reachable(mixed):
    s, p, q, m = mixed
    a = hjcdik.solve(s, p, q, active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
                     coarse_iters=120, lm_iters=60, stag_patience=0)
    b = hjcdik.solve(s, p, q, active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
                     coarse_iters=120, lm_iters=60, stag_patience=0)
    np.testing.assert_array_equal(a["joint_config"], b["joint_config"])   # deterministic
    d = hjcdik.solve(s, p, q, active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
                     coarse_iters=120, lm_iters=60)                       # default = Policy B
    assert d["lm_kernel_ms"] < a["lm_kernel_ms"], "the default is not actually adaptive"


# --- 4. the metric and the reported errors describe ONE state --------------------------------------
def test_e_phys_and_reported_errors_are_from_the_same_state(mixed):
    """E_phys (trace col 10) must be computed from the same s_pn/s_on that produced the reported
    max_pos_err / max_ori_err (cols 6/7). Computing it at a different point in the iteration would
    silently mix two states -- which is a bug I actually shipped once and this test now pins."""
    s, p, q, m = mixed
    out = hjcdik.refine(s, p, q, active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
                        max_iters=40, stag_patience=0, diagnostics=True, return_trace=True)
    tr = out["trace"]
    assert tr.shape[2] == 11
    checked = 0
    for b in range(0, len(s), 13):
        rows = tr[b][tr[b][:, 0] != 0]
        if not len(rows):
            continue
        ep = rows[:, 10]
        # the worst active target alone contributes this much; E_phys sums over all active targets
        lower = (rows[:, 6] / PTOL) ** 2 + (rows[:, 7] / OTOL) ** 2
        assert np.all(ep >= lower * (1 - 1e-3) - 1e-6), (
            "E_phys is below the worst-target contribution of the SAME row -- the metric and the "
            "reported errors are not from the same configuration state")
        assert np.all(ep <= K * lower * (1 + 1e-3) + 1e-6), "E_phys exceeds K x the worst target"
        checked += 1
    assert checked > 5


# --- 5/6. the two behaviours that matter -----------------------------------------------------------
def test_a_still_progressing_seed_is_not_stopped(mixed):
    """Policy B must not kill a seed that is still reducing its PHYSICAL error, however its
    row-scaled cost happens to move."""
    s, p, q, m = mixed
    out = hjcdik.refine(s, p, q, active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
                        max_iters=60, stag_patience=2, stag_rel=1e-3,
                        diagnostics=True, return_trace=True)
    tr = out["trace"]
    bad = []
    for b in range(len(s)):
        rows = tr[b][tr[b][:, 0] != 0]
        if len(rows) < 3 or bool(out["success"][b]):
            continue                              # solved seeds stop for a different reason
        if len(rows) >= 60:
            continue                              # ran to the cap: not stopped by stagnation
        ep = rows[:, 10]
        # it was stopped early -> the last `patience` steps must all have been negligible
        rel = (ep[-3:-1] - ep[-2:]) / np.maximum(ep[-3:-1], 1e-30)
        if np.any(rel > 1e-3 * 10):               # an order of magnitude above the threshold
            bad.append((b, rel))
    assert not bad, (
        f"{len(bad)} seeds were stopped by stagnation while still making real physical progress, "
        f"e.g. seed {bad[0][0]} with relative E_phys improvements {bad[0][1]}")


def test_a_genuinely_stagnant_seed_is_stopped_after_the_patience(mixed):
    """Stagnant seeds must stop. Note this fixture refines RAW random restarts with no coarse
    preconditioning, so many unsolved seeds are still making genuine (slow) progress and Policy B
    correctly lets them run -- mean ~36 of 60 here, against ~7 in the benchmark where coarse has
    already done its work. What must hold is that a substantial share stop strictly early, and that
    every one that does had actually flatlined (the converse is pinned by
    test_a_still_progressing_seed_is_not_stopped).
    """
    s, p, q, m = mixed
    kw = dict(active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL, max_iters=60,
              diagnostics=True)
    off = hjcdik.refine(s, p, q, stag_patience=0, **kw)
    on = hjcdik.refine(s, p, q, stag_patience=2, stag_rel=1e-3, return_trace=True, **kw)
    unsolved = ~off["success"].astype(bool)
    assert unsolved.any()
    assert (off["lm_iterations"][unsolved] >= 60).all(), "the fixed cap let an unsolved seed out"

    it_on = on["lm_iterations"][unsolved]
    stopped_early = it_on < 60
    assert stopped_early.mean() > 0.3, (
        f"only {100*stopped_early.mean():.0f}% of unsolved seeds stopped early -- Policy B is "
        f"barely firing")
    assert it_on.mean() < 55, "Policy B saved essentially nothing on unsolved seeds"

    # every seed it DID stop early had genuinely flatlined: the final `patience` relative
    # improvements in E_phys were all below the threshold
    tr = on["trace"]
    for b in np.where(unsolved & (on["lm_iterations"] < 60))[0][:40]:
        rows = tr[b][tr[b][:, 0] != 0]
        if len(rows) < 4:
            continue
        ep = rows[:, 10]
        rel = (ep[-3:-1] - ep[-2:]) / np.maximum(ep[-3:-1], 1e-30)
        assert np.all(rel < 1e-3 * 10), (
            f"seed {b} was stopped by stagnation but its last E_phys improvements were {rel}")


# --- 7. Policy A still stops a solved seed immediately ---------------------------------------------
def test_a_solved_seed_stops_immediately(mixed):
    s, p, q, m = mixed
    out = hjcdik.refine(s, p, q, active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
                        max_iters=60, diagnostics=True, return_trace=True)
    succ = out["success"].astype(bool)
    assert succ.any()
    tr = out["trace"]
    for b in np.where(succ)[0]:
        rows = tr[b][tr[b][:, 0] != 0]
        hit = np.where((rows[:, 6] <= PTOL) & (rows[:, 7] <= OTOL))[0]
        assert len(hit) and int(hit[0]) + 1 == len(rows), \
            f"seed {b} kept iterating after it had already succeeded"


# --- 8/9. diagnostics and precision ----------------------------------------------------------------
def test_diagnostics_do_not_change_the_default_solution(mixed):
    s, p, q, m = mixed
    kw = dict(active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
              coarse_iters=120, lm_iters=60)
    off = hjcdik.solve(s, p, q, diagnostics=False, **kw)
    on = hjcdik.solve(s, p, q, diagnostics=True, **kw)
    np.testing.assert_array_equal(off["joint_config"], on["joint_config"])
    np.testing.assert_array_equal(off["cost"], on["cost"])


@pytest.mark.parametrize("precision,dtype", [("float32", np.float32), ("float64", np.float64)])
def test_policy_b_works_in_both_precisions(mixed, precision, dtype):
    s, p, q, m = mixed
    kw = dict(active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
              coarse_iters=120, lm_iters=60, precision=precision)
    full = hjcdik.solve(s, p, q, stag_patience=0, **kw)
    stag = hjcdik.solve(s, p, q, **kw)
    assert stag["joint_config"].dtype == dtype
    assert stag["lm_kernel_ms"] < full["lm_kernel_ms"]
    assert np.isfinite(np.asarray(stag["joint_config"], np.float64)).all()
    pe = stag["position_errors"].max(axis=1)
    oe = stag["orientation_errors"].max(axis=1)
    ok_full = (full["position_errors"].max(axis=1) <= PTOL) & \
              (full["orientation_errors"].max(axis=1) <= OTOL)
    ok_stag = (pe <= PTOL) & (oe <= OTOL)
    if ok_full[int(np.argmin(full["cost"]))]:
        assert ok_stag[int(np.argmin(stag["cost"]))], \
            f"{precision}: Policy B lost a top-1 success the fixed cap found"


# --- top-N diversity -------------------------------------------------------------------------------
def test_top_n_solution_diversity_is_preserved(mixed):
    """Policy B trims the solved-seed POOL (it stops stalled seeds), so check the thing that
    actually gets consumed: the best-N candidates by cost."""
    s, p, q, m = mixed
    kw = dict(active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
              coarse_iters=120, lm_iters=60)
    full = hjcdik.solve(s, p, q, stag_patience=0, **kw)
    stag = hjcdik.solve(s, p, q, **kw)

    def top_n_solved(o, n):
        pe = o["position_errors"].max(axis=1); oe = o["orientation_errors"].max(axis=1)
        ok = (pe <= PTOL) & (oe <= OTOL)
        return int(ok[np.argsort(o["cost"])[:n]].sum())

    for n in (1, 5, 10):
        f, b = top_n_solved(full, n), top_n_solved(stag, n)
        assert b >= f - 1, f"top-{n}: Policy B returns {b} solved vs {f} for the fixed cap"
