"""Phase 0E: persistent workspace, direct I/O paths, and the stable-metric best_x.

WORKSPACE. Every solve used to cudaMalloc ~10 device buffers, free them, and copy the B x N
configuration FOUR times on the way out (D2H -> widen to double -> narrow back to float -> pybind
copy into numpy). An HJCDSolver now owns a capacity-based device arena, so after warm-up a steady
stream of same-or-smaller solves performs ZERO cudaMalloc and ZERO cudaFree, and the configuration
makes ONE pass (D2H straight into the numpy buffer).

BEST_X. The LM and coarse searches both used to track their best-so-far state on the ROW-SCALED
cost. The row scales s_{k,r} = 1/||J_{k,r}|| are re-frozen every iteration, so C^(t) and C^(t-1) are
expressed in different units -- comparing them across iterations is not a real comparison. Measured
consequence: 41.8% of seeds returned a configuration PHYSICALLY WORSE than one they had already
visited (median 1.05x, worst 10.8x). best_x is now tracked on E_phys, the tolerance-normalised
physical error, which IS comparable across iterations. Nothing inside an iteration changed: the trial
acceptance, damping, line search and trust region all still use the row-scaled cost.
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


def _problem(B, seed=SEED):
    rng = np.random.default_rng(seed)
    q_true = rng.uniform(LO, HI)
    T = hjcdik.target_transforms(q_true[None, :])[0]
    p = np.repeat(T[:, :3, 3][None], B, axis=0)
    q = np.repeat(np.stack([_quat(T[k][:3, :3]) for k in range(K)])[None], B, axis=0)
    seeds = rng.uniform(LO, HI, size=(B, N))
    m = np.full(B, (1 << K) - 1, dtype=np.uint32)
    return seeds, p, q, m


# =============================== WORKSPACE =====================================================
def test_steady_state_solves_perform_zero_cuda_alloc():
    """THE requirement. After warm-up, same-or-smaller solves must allocate and free NOTHING."""
    sv = hjcdik.HJCDSolver()
    s, p, q, m = _problem(512)
    kw = dict(active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
              coarse_iters=60, lm_iters=30)
    sv.solve(s, p, q, **kw)                       # warm-up: this one MAY allocate
    st0 = sv.workspace_stats()
    for _ in range(10):
        sv.solve(s, p, q, **kw)
    st1 = sv.workspace_stats()
    assert st1["cuda_mallocs"] == st0["cuda_mallocs"], (
        f"{st1['cuda_mallocs'] - st0['cuda_mallocs']} cudaMalloc calls across 10 steady-state solves")
    assert st1["cuda_frees"] == st0["cuda_frees"] == 0, "the workspace freed and reallocated"
    assert st0["cuda_mallocs"] >= 1 and st0["bytes"] > 0


def test_a_smaller_batch_reuses_the_larger_workspace():
    sv = hjcdik.HJCDSolver()
    big = _problem(1024)
    kw = dict(position_tol=PTOL, orientation_tol=OTOL, coarse_iters=30, lm_iters=20)
    sv.solve(big[0], big[1], big[2], active_target_mask=big[3], **kw)
    st0 = sv.workspace_stats()
    for B in (512, 128, 16, 1):
        s, p, q, m = _problem(B)
        sv.solve(s, p, q, active_target_mask=m, **kw)
    st1 = sv.workspace_stats()
    assert st1["cuda_mallocs"] == st0["cuda_mallocs"], "a smaller batch reallocated the workspace"
    assert st1["bytes"] == st0["bytes"], "the workspace shrank"


def test_a_larger_batch_grows_the_workspace_once():
    sv = hjcdik.HJCDSolver()
    kw = dict(position_tol=PTOL, orientation_tol=OTOL, coarse_iters=30, lm_iters=20)
    s, p, q, m = _problem(64)
    sv.solve(s, p, q, active_target_mask=m, **kw)
    n0 = sv.workspace_stats()["cuda_mallocs"]
    s, p, q, m = _problem(2048)
    sv.solve(s, p, q, active_target_mask=m, **kw)
    n1 = sv.workspace_stats()["cuda_mallocs"]
    assert n1 == n0 + 1, f"growth took {n1 - n0} allocations, expected exactly 1"
    for _ in range(5):                              # and it is stable afterwards
        sv.solve(s, p, q, active_target_mask=m, **kw)
    assert sv.workspace_stats()["cuda_mallocs"] == n1


def test_diagnostics_disabled_allocates_no_trace_buffer():
    sv = hjcdik.HJCDSolver()
    s, p, q, m = _problem(256)
    kw = dict(active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
              coarse_iters=120, lm_iters=60)
    sv.solve(s, p, q, diagnostics=False, **kw)
    off_bytes = sv.workspace_stats()["bytes"]
    sv2 = hjcdik.HJCDSolver()
    sv2.solve(s, p, q, diagnostics=True, **kw)
    on_bytes = sv2.workspace_stats()["bytes"]
    assert on_bytes > off_bytes, (
        "a diagnostics-ON solve did not need more workspace than diagnostics-OFF -- the trace "
        "buffer is being allocated either always or never")


def test_solver_is_not_reentrant_and_says_so():
    """One active call per instance. A re-entrant call must RAISE, not race on the arena."""
    sv = hjcdik.HJCDSolver()
    sv._enter()
    try:
        with pytest.raises(RuntimeError, match="not thread-safe"):
            sv._enter()
    finally:
        sv._exit()
    s, p, q, m = _problem(8)                       # and it still works afterwards
    sv.solve(s, p, q, active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
             coarse_iters=10, lm_iters=10)


def test_free_functions_still_work_and_are_allocation_free():
    """The free-function API wraps a THREAD-LOCAL solver, so it stays allocation-free without ever
    sharing an arena across threads."""
    s, p, q, m = _problem(256)
    kw = dict(active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
              coarse_iters=60, lm_iters=30)
    hjcdik.solve(s, p, q, **kw)
    sv = hjcdik._default_solver()
    n0 = sv.workspace_stats()["cuda_mallocs"]
    for _ in range(8):
        hjcdik.solve(s, p, q, **kw)
    assert sv.workspace_stats()["cuda_mallocs"] == n0


# =============================== I/O DTYPE ====================================================
@pytest.mark.parametrize("in_dtype", [np.float64, np.float32])
@pytest.mark.parametrize("precision,out_dtype", [("float32", np.float32), ("float64", np.float64)])
def test_input_dtypes_are_both_accepted_and_output_matches_precision(in_dtype, precision, out_dtype):
    """float64 input is NEVER rejected. The returned config is in the REQUESTED compute precision;
    task metrics stay float64."""
    s, p, q, m = _problem(64)
    out = hjcdik.solve(s.astype(in_dtype), p.astype(in_dtype), q.astype(in_dtype),
                       active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
                       coarse_iters=30, lm_iters=20, precision=precision)
    assert out["joint_config"].dtype == out_dtype
    assert out["position_errors"].dtype == np.float64
    assert out["cost"].dtype == np.float64
    assert out["joint_config"].flags.c_contiguous
    assert 0 not in out["joint_config"].strides


def test_float32_and_float64_input_agree():
    s, p, q, m = _problem(256)
    kw = dict(active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
              coarse_iters=60, lm_iters=30)
    a = hjcdik.solve(s, p, q, **kw)
    b = hjcdik.solve(s.astype(np.float32), p.astype(np.float32), q.astype(np.float32), **kw)
    np.testing.assert_array_equal(a["joint_config"], b["joint_config"])


def test_strides_and_contiguity_survive_the_new_output_path(_=None):
    """The config is now D2H'd straight into the numpy buffer -- the Phase-0A stride invariant must
    still hold on it and on everything else."""
    s, p, q, m = _problem(128)
    out = hjcdik.solve(s, p, q, active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
                       coarse_iters=30, lm_iters=20, diagnostics=True)
    for name, v in sorted(out.items()):
        if not isinstance(v, np.ndarray) or v.ndim == 0:
            continue
        assert 0 not in v.strides, f"{name}: zero stride"
        assert v.flags.c_contiguous, f"{name}: not C-contiguous"


# =============================== BEST_X =======================================================
def _ephys(pe, oe):
    return ((pe / PTOL) ** 2 + (oe / OTOL) ** 2).sum(axis=1)


@pytest.mark.parametrize("stag_patience", [0, 2])
def test_returned_state_is_never_physically_worse_than_a_visited_one(stag_patience):
    """8. The headline: the 41.8% physically-worse return rate must fall to ~0.

    The LM trace records E_phys per iteration on the SAME state as the reported errors, so this is a
    direct replay check: E_phys(returned) <= min(E_phys over the seed's own trace).
    """
    B = 1024
    s, p, q, m = _problem(B)
    c = hjcdik.coarse_search(s, p, q, active_target_mask=m, position_tol=PTOL,
                             orientation_tol=OTOL, max_iters=120)
    out = hjcdik.refine(np.asarray(c["joint_config"], np.float64), p, q, active_target_mask=m,
                        position_tol=PTOL, orientation_tol=OTOL, max_iters=60,
                        stag_patience=stag_patience, diagnostics=True, return_trace=True)
    tr = out["trace"]
    ep_final = _ephys(out["position_errors"], out["orientation_errors"])
    worse = 0
    checked = 0
    for b in range(B):
        rows = tr[b][tr[b][:, 0] != 0]
        if len(rows) < 2:
            continue
        checked += 1
        # 1e-3 relative slack: the kernel computed E_phys in fp32, we recompute from fp64 errors
        if ep_final[b] > rows[:, 10].min() * (1 + 1e-3):
            worse += 1
    assert checked > 500
    assert worse == 0, (
        f"stag_patience={stag_patience}: {worse}/{checked} seeds returned a configuration "
        f"PHYSICALLY WORSE than one they had already visited (was 41.8% before the fix)")


def test_a_lower_row_scaled_cost_with_worse_physical_merit_does_not_win():
    """1/2. best_x must follow the PHYSICAL metric, not the row-scaled cost.

    Direct evidence: across the batch, the returned state's E_phys is the trace minimum, while the
    returned state's row-scaled cost is frequently NOT the trace minimum -- i.e. states with a lower
    scaled cost existed and were correctly passed over.
    """
    B = 512
    s, p, q, m = _problem(B)
    out = hjcdik.refine(s, p, q, active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
                        max_iters=60, stag_patience=0, diagnostics=True, return_trace=True)
    tr = out["trace"]
    ep_final = _ephys(out["position_errors"], out["orientation_errors"])
    passed_over = 0
    for b in range(B):
        rows = tr[b][tr[b][:, 0] != 0]
        if len(rows) < 3:
            continue
        assert ep_final[b] <= rows[:, 10].min() * (1 + 1e-3), "best_x is not the E_phys minimum"
        # col 5 is the row-scaled cost. Did a state with a LOWER scaled cost than the E_phys-optimal
        # one exist? If so, the old rule would have taken it and the new rule correctly did not.
        i_best = int(np.argmin(rows[:, 10]))
        if rows[:, 5].min() < rows[i_best, 5] * (1 - 1e-6):
            passed_over += 1
    assert passed_over > 20, (
        "no seed had a lower-scaled-cost state than its physically-best one, so this test cannot "
        "distinguish the two rules")
    print(f"\n  {passed_over}/{B} seeds had a lower row-scaled cost available at a physically WORSE "
          f"state; best_x correctly ignored it")


@pytest.mark.parametrize("stag_patience", [0, 2])
def test_a_solved_state_is_never_lost(stag_patience):
    """3. If a seed ever satisfied every tolerance, the call must report success."""
    B = 1024
    s, p, q, m = _problem(B)
    c = hjcdik.coarse_search(s, p, q, active_target_mask=m, position_tol=PTOL,
                             orientation_tol=OTOL, max_iters=120)
    out = hjcdik.refine(np.asarray(c["joint_config"], np.float64), p, q, active_target_mask=m,
                        position_tol=PTOL, orientation_tol=OTOL, max_iters=60,
                        stag_patience=stag_patience, diagnostics=True, return_trace=True)
    tr = out["trace"]
    succ = out["success"].astype(bool)
    lost = 0
    visited = 0
    for b in range(B):
        rows = tr[b][tr[b][:, 0] != 0]
        if len(rows) and np.any((rows[:, 6] <= PTOL) & (rows[:, 7] <= OTOL)):
            visited += 1
            if not succ[b]:
                lost += 1
    assert visited > 0
    assert lost == 0, f"{lost}/{visited} seeds reached a solving state and did not return it"


def test_diagnostics_do_not_change_the_solution():
    """5."""
    s, p, q, m = _problem(256)
    kw = dict(active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
              coarse_iters=120, lm_iters=60)
    a = hjcdik.solve(s, p, q, diagnostics=False, **kw)
    b = hjcdik.solve(s, p, q, diagnostics=True, **kw)
    np.testing.assert_array_equal(a["joint_config"], b["joint_config"])
    np.testing.assert_array_equal(a["cost"], b["cost"])


@pytest.mark.parametrize("stag_patience", [0, 2])
def test_top_n_quality_does_not_regress(stag_patience):
    """7. best_x now returns physically better states, so top-N must not get worse."""
    B = 1024
    s, p, q, m = _problem(B)
    out = hjcdik.solve(s, p, q, active_target_mask=m, position_tol=PTOL, orientation_tol=OTOL,
                       coarse_iters=120, lm_iters=60, stag_patience=stag_patience)
    pe = out["position_errors"].max(axis=1)
    oe = out["orientation_errors"].max(axis=1)
    ok = (pe <= PTOL) & (oe <= OTOL)
    order = np.argsort(out["cost"])
    assert ok[order[0]], "top-1 is not a solved candidate"
    for n in (5, 32):
        assert ok[order[:n]].sum() >= 1, f"top-{n} contains no solved candidate"
    assert np.isfinite(np.asarray(out["joint_config"], np.float64)).all()
