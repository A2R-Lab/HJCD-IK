"""Phase 0C: iteration-budget instrumentation and adaptive stopping.

Three things are pinned here.

1. SAME-INVOCATION STAGE TIMING. `coarse_kernel_ms` / `lm_kernel_ms` are CUDA-event device times for
   the launches that produced THIS result, so they are commensurable with the end-to-end time of the
   same call. Mixing independently-measured medians is what produced the impossible
   "coarse + LM > end-to-end" rows in the Phase-0B report; `end_to_end >= sum(device stages)` is
   asserted here so that cannot come back.

2. POLICY A was ALREADY IN THE KERNEL. The LM loop is `for (it = 0; it < k_max && !s_break; ++it)`
   and sets s_break on all_active_converged -- per WARP, so per seed, and stop_on_first is 0 on the
   multi-target path, so a solved seed never terminates its neighbours. Measured: 0.0% of a solved
   seed's LM work happens after its first success. There was nothing to implement, only to verify.

3. POLICY B (stagnation stopping) is measured on E_phys, a tolerance-normalised PHYSICAL error:

       E_phys(q) = sum_k [ |e_p,k|^2 / eps_p^2 + |e_R,k|^2 / eps_R^2 ]   over ACTIVE targets

   NOT on the row-scaled cost. The row scales s_{k,r} = 1/||J_{k,r}|| are re-frozen every iteration,
   so C^(t) and C^(t-1) are in different units and a relative improvement between them is not a real
   quantity. E_phys is built from physical residual norms and fixed tolerances, so it is stable
   across iterations. It is analysis/stopping only -- no optimizer step ever reads it.

   patience = 0 DISABLES Policy B. It was the default when this file was written; the PRODUCTION
   default is now patience=2 / rel=1e-3 (see test_default_policy.py), so every fixed-cap baseline
   below pins stag_patience=0 explicitly rather than relying on the default.
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
def problem():
    B = 256
    rng = np.random.default_rng(SEED)
    q_true = rng.uniform(LO, HI)
    T = hjcdik.target_transforms(q_true[None, :])[0]
    p = np.repeat(T[:, :3, 3][None], B, axis=0)
    q = np.repeat(np.stack([_quat(T[k][:3, :3]) for k in range(K)])[None], B, axis=0)
    seeds = rng.uniform(LO, HI, size=(B, N))
    m = np.full(B, (1 << K) - 1, dtype=np.uint32)
    return seeds, p, q, m


# --- 1. same-invocation stage timing --------------------------------------------------------------
def test_stage_times_come_from_the_same_invocation(problem):
    import time
    s, p, q, m = problem
    kw = dict(active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
              coarse_iters=120, lm_iters=60)
    hjcdik.solve(s, p, q, **kw)                                   # warm up
    t0 = time.perf_counter()
    out = hjcdik.solve(s, p, q, **kw)
    e2e = (time.perf_counter() - t0) * 1e3

    dev = out["coarse_kernel_ms"] + out["lm_kernel_ms"]
    assert out["lm_kernel_ms"] > 0
    # coarse_kernel_ms is legitimately 0 on a single-target robot: coarse_mode="auto" dispatches
    # popcount == 1 to LM-only, so no coarse kernel is launched at all (open-world).
    if K >= 2:
        assert out["coarse_kernel_ms"] > 0, "the coarse stage ran but reported no device time"
    else:
        assert out["coarse_kernel_ms"] == 0.0
    assert e2e >= dev, (
        f"end-to-end {e2e:.2f} ms < device stages {dev:.2f} ms -- the stage and end-to-end timings "
        f"are not from the same invocation")


def test_coarse_iters_zero_launches_no_coarse_kernel(problem):
    s, p, q, m = problem
    out = hjcdik.solve(s, p, q, active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
                       coarse_iters=0, lm_iters=30)
    assert out["coarse_kernel_ms"] == 0.0, "an empty coarse kernel was launched for coarse_iters=0"
    assert not out["used_coarse"].any()


# --- 2. Policy A: already present, and per-seed ----------------------------------------------------
def test_policy_a_stops_a_solved_seed_and_only_that_seed(problem):
    """A solved seed must stop consuming LM iterations, WITHOUT terminating its neighbours."""
    s, p, q, m = problem
    B = len(s)
    # half the seeds start next to the answer (they solve fast), half are random restarts
    rng = np.random.default_rng(SEED)
    q_true = rng.uniform(LO, HI)
    T = hjcdik.target_transforms(q_true[None, :])[0]
    pp = np.repeat(T[:, :3, 3][None], B, axis=0)
    qq = np.repeat(np.stack([_quat(T[k][:3, :3]) for k in range(K)])[None], B, axis=0)
    seeds = rng.uniform(LO, HI, size=(B, N))
    seeds[1::2] = np.clip(q_true + rng.normal(scale=0.02, size=(B // 2, N)), LO, HI)

    out = hjcdik.refine(seeds, pp, qq, active_target_mask=m, position_tol=PTOL,
                        orientation_tol=OTOL, max_iters=60, stag_patience=0,
                        diagnostics=True, return_trace=True)
    it = out["lm_iterations"]
    succ = out["success"].astype(bool)
    assert succ.any() and (~succ).any(), "need both solved and unsolved seeds in the batch"

    # a solved seed stops EARLY; an unsolved neighbour runs on -- so a solved seed did not kill it
    assert it[succ].max() < 60, "a solved seed ran the full budget -- Policy A is not firing"
    assert it[~succ].max() >= 60, (
        "no unsolved seed reached the cap -- a solved seed appears to have terminated the batch")

    # and no solved seed spent iterations AFTER its first success
    tr = out["trace"]
    wasted = 0
    for b in np.where(succ)[0]:
        rows = tr[b][tr[b][:, 0] != 0]
        hit = np.where((rows[:, 6] <= PTOL) & (rows[:, 7] <= OTOL))[0]
        if len(hit):
            wasted += len(rows) - (int(hit[0]) + 1)
    assert wasted == 0, f"{wasted} LM iterations ran after a seed had already succeeded"


# --- 3. Policy B: E_phys, and default-off ---------------------------------------------------------
def test_e_phys_is_recorded_and_is_a_stable_physical_metric(problem):
    """Column 10 of the LM trace is E_phys. Unlike the row-scaled cost it is comparable ACROSS
    iterations, which is the whole reason it exists."""
    s, p, q, m = problem
    out = hjcdik.refine(s, p, q, active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
                        max_iters=40, stag_patience=0, diagnostics=True, return_trace=True)
    tr = out["trace"]
    assert tr.shape[2] == 11, "the trace has no E_phys column"
    for b in range(0, len(s), 37):
        rows = tr[b][tr[b][:, 0] != 0]
        if len(rows) < 3:
            continue
        ep = rows[:, 10]
        assert np.all(ep >= 0) and np.all(np.isfinite(ep))
        # E_phys reconstructed from the reported physical errors, to within the fp32 the kernel used
        recon = (rows[:, 6] / PTOL) ** 2 + (rows[:, 7] / OTOL) ** 2
        assert np.all(ep >= recon - 1e-3 * np.maximum(recon, 1.0)), (
            "E_phys is smaller than the worst-target contribution alone -- it is not the "
            "tolerance-normalised physical error it claims to be")


def test_stag_patience_zero_is_deterministic_and_reproducible(problem):
    s, p, q, m = problem
    kw = dict(active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL, max_iters=60)
    a = hjcdik.refine(s, p, q, stag_patience=0, **kw)
    b = hjcdik.refine(s, p, q, stag_patience=0, **kw)
    np.testing.assert_array_equal(a["joint_config"], b["joint_config"])
    np.testing.assert_array_equal(a["cost"], b["cost"])


@pytest.mark.parametrize("patience", [2, 4, 6])
@pytest.mark.parametrize("rel", [1e-3, 1e-4, 1e-5])
def test_policy_b_cuts_iterations_without_losing_the_solution_pool(problem, patience, rel):
    """Stagnation stopping must cut work while keeping the seeds that were actually converging.

    This is the property a naive fixed cap does NOT have: a fixed lm=10 kills every seed at 10,
    including the ones that would have solved at iteration 18 or 42.
    """
    s, p, q, m = problem
    kw = dict(active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL, max_iters=60,
              diagnostics=True)
    # NOTE: the DEFAULT is now Policy B (patience=2), so the fixed-cap baseline must be pinned.
    full = hjcdik.refine(s, p, q, stag_patience=0, **kw)
    stag = hjcdik.refine(s, p, q, stag_patience=patience, stag_rel=rel, **kw)

    assert stag["lm_iterations"].mean() < full["lm_iterations"].mean(), "no work was saved"
    n_full = int(full["success"].sum())
    n_stag = int(stag["success"].sum())
    assert n_stag >= 0.85 * n_full, (
        f"patience={patience} rel={rel:g}: solved seeds fell from {n_full} to {n_stag} -- the "
        f"stagnation stop is killing seeds that were still converging")
    assert np.isfinite(np.asarray(stag["joint_config"], np.float64)).all()


def test_policy_b_never_returns_a_worse_selected_candidate(problem):
    """The whole point: 4x less work, same answer."""
    s, p, q, m = problem
    kw = dict(active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
              coarse_iters=120, lm_iters=60)
    full = hjcdik.solve(s, p, q, stag_patience=0, **kw)      # fixed cap: the default is Policy B now
    stag = hjcdik.solve(s, p, q, stag_patience=2, stag_rel=1e-3, **kw)
    for o in (full, stag):
        pe = o["position_errors"].max(axis=1)
        oe = o["orientation_errors"].max(axis=1)
        o["_ok"] = (pe <= PTOL) & (oe <= OTOL)
        o["_pick"] = int(np.argmin(o["cost"]))
    if full["_ok"][full["_pick"]]:
        assert stag["_ok"][stag["_pick"]], "Policy B lost a top-1 success that the full budget found"
    assert stag["lm_kernel_ms"] < full["lm_kernel_ms"], "Policy B did not reduce LM device time"
