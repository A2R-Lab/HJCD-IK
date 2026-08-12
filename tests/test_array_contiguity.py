"""Phase 0A: every array crossing the pybind boundary must be C-contiguous with real strides.

pybind11's shape-only constructor -- py::array_t<T>(shape) -- yields a ZERO-STRIDED array when the
shape is ONE-dimensional. Every element then aliases element 0, so B distinct C++ values collapse
into a single repeated value. Multi-dimensional shapes are unaffected, which is why this hid: the
[B, K] error arrays were correct while the [B] `cost` and `success` arrays built beside them were
not.

The damage was not cosmetic. `cost` is the candidate-ranking signal, so argmin over a constant array
always returned candidate 0 and the returned solution was effectively arbitrary; `success` broadcast
one candidate's flag over the whole batch, so the benchmark's solved-rate was meaningless. These
tests pin the invariant so it cannot regress.
"""
import numpy as np
import pytest

import hjcdik

N = hjcdik.num_joints()
K = hjcdik.num_targets()
LIM = hjcdik.joint_limits()
LO, HI = LIM[:, 0], LIM[:, 1]
B = 64
SEED = 12345


def _quat(R):
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        elif i == 1:
            s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        else:
            s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.asarray(q)
    return q / np.linalg.norm(q)


@pytest.fixture(scope="module")
def problem():
    """A reachable target set (FK of a random config) plus random-restart seeds."""
    rng = np.random.default_rng(SEED)
    q_true = rng.uniform(LO, HI)
    T = hjcdik.target_transforms(q_true[None, :])[0]
    p = np.repeat(T[:, :3, 3][None], B, axis=0)
    q = np.repeat(np.stack([_quat(T[k][:3, :3]) for k in range(K)])[None], B, axis=0)
    seeds = rng.uniform(LO, HI, size=(B, N))
    mask = np.full(B, (1 << K) - 1, dtype=np.uint32)
    return seeds, p, q, mask


def _outputs(problem):
    """Every dict the public API returns, with diagnostics on."""
    seeds, p, q, mask = problem
    kw = dict(active_target_mask=mask, position_tol=1e-4, orientation_tol=1e-3)
    return {
        "target_residuals": hjcdik.target_residuals(seeds, p, q, active_target_mask=mask,
                                                    position_tol=1e-4, orientation_tol=1e-3),
        "refine": hjcdik.refine(seeds, p, q, max_iters=20, diagnostics=True,
                                return_trace=True, **kw),
        "coarse_search": hjcdik.coarse_search(seeds, p, q, max_iters=20, diagnostics=True,
                                              return_trace=True, **kw),
        "solve": hjcdik.solve(seeds, p, q, coarse_iters=20, lm_iters=20, diagnostics=True, **kw),
        "normal_equations": hjcdik.normal_equations(seeds, p, q, active_target_mask=mask),
        # updates = [B, M, 2] of (joint_index, new_value); one accepted step per candidate
        "incremental_probe": hjcdik.incremental_probe(
            seeds,
            np.stack([np.zeros(B), seeds[:, 0]], axis=1).reshape(B, 1, 2),
            np.ones((B, 1), dtype=bool),
            p, q, active_target_mask=mask),
    }


def test_every_returned_array_is_c_contiguous_with_real_strides(problem):
    """THE regression. A zero stride anywhere means values are silently aliased."""
    bad = []
    checked = 0
    for call, out in _outputs(problem).items():
        for name, v in sorted(out.items()):
            if not isinstance(v, np.ndarray) or v.ndim == 0:
                continue
            checked += 1
            if 0 in v.strides or not v.flags.c_contiguous:
                bad.append(f"{call}.{name}: shape={v.shape} strides={v.strides} "
                           f"c_contiguous={v.flags.c_contiguous}")
    assert not bad, "zero-strided / non-contiguous arrays returned:\n  " + "\n  ".join(bad)
    assert checked > 25, f"only {checked} arrays checked -- the sweep is not covering the API"
    print(f"\n  {checked} returned arrays: all C-contiguous, no zero strides")


def test_one_dimensional_arrays_have_itemsize_stride(problem):
    """The exact shape of the bug: a 1-D array whose stride is 0 instead of itemsize."""
    for call, out in _outputs(problem).items():
        for name, v in sorted(out.items()):
            if isinstance(v, np.ndarray) and v.ndim == 1:
                assert v.strides == (v.dtype.itemsize,), (
                    f"{call}.{name}: strides {v.strides} != ({v.dtype.itemsize},)")
                assert v.flags.c_contiguous


@pytest.fixture(scope="module")
def mixed_problem():
    """Half the seeds start next to the answer (they converge early, few iterations, success=True);
    half are random restarts (they run to the cap, success=False).

    The per-candidate values are therefore genuinely DIFFERENT in C++ -- which is what makes this a
    real test of the crossing. With uniformly-random seeds `success` is honestly all-False and
    `lm_iterations` honestly all-at-the-cap, so a constant array would prove nothing.
    """
    rng = np.random.default_rng(SEED)
    q_true = rng.uniform(LO, HI)
    T = hjcdik.target_transforms(q_true[None, :])[0]
    p = np.repeat(T[:, :3, 3][None], B, axis=0)
    q = np.repeat(np.stack([_quat(T[k][:3, :3]) for k in range(K)])[None], B, axis=0)

    seeds = rng.uniform(LO, HI, size=(B, N))                       # random restarts
    near = np.clip(q_true + rng.normal(scale=0.02, size=(B // 2, N)), LO, HI)
    seeds[1::2] = near                                             # ODD indices converge
    mask = np.full(B, (1 << K) - 1, dtype=np.uint32)
    return seeds, p, q, mask


@pytest.mark.parametrize("call,field,kind", [
    ("refine", "cost", "float64"),
    ("coarse_search", "cost", "float64"),
    ("solve", "cost", "float64"),
    ("target_residuals", "cost_raw", "float64"),
    ("refine", "lm_iterations", "int"),
    ("refine", "accepted_lm_steps", "int"),
    # NOT coarse_iterations: the coarse search is a seeder with no convergence exit, so it always
    # runs the full budget and that counter is honestly constant. accepted_coarse_steps varies.
    ("coarse_search", "accepted_coarse_steps", "int"),
    ("refine", "success", "bool"),
    ("solve", "success", "bool"),
])
def test_distinct_cpp_values_stay_distinct_in_python(mixed_problem, call, field, kind):
    """A zero-strided array is CONSTANT. Distinct per-candidate values must survive the crossing.

    Covers float64, int32 and bool [B] vectors -- the three element types the API returns.
    """
    seeds, p, q, mask = mixed_problem
    kw = dict(active_target_mask=mask, position_tol=1e-4, orientation_tol=1e-3)
    outs = {
        "refine": lambda: hjcdik.refine(seeds, p, q, max_iters=60, diagnostics=True, **kw),
        "coarse_search": lambda: hjcdik.coarse_search(seeds, p, q, max_iters=60,
                                                      diagnostics=True, **kw),
        "solve": lambda: hjcdik.solve(seeds, p, q, coarse_iters=120, lm_iters=60,
                                      diagnostics=True, **kw),
        "target_residuals": lambda: hjcdik.target_residuals(seeds, p, q, **kw),
    }
    v = outs[call]()[field]
    assert v.shape == (B,)
    assert v.strides == (v.dtype.itemsize,), f"{call}.{field}: zero/!= itemsize stride"
    assert v.flags.c_contiguous
    expected_kind = {"float64": "f", "int": "i", "bool": "b"}[kind]
    assert v.dtype.kind == expected_kind, f"{call}.{field}: dtype {v.dtype} is not {kind}"
    assert len(np.unique(v)) > 1, (
        f"{call}.{field} is CONSTANT across {B} candidates (value {v[0]!r}) -- the hallmark of the "
        f"zero-stride bug. Half these seeds start next to the answer and half are random restarts, "
        f"so the C++ values genuinely differ; collapsing them means they are being aliased.")


@pytest.mark.parametrize("precision,dtype", [("float64", np.float64), ("float32", np.float32)])
def test_float32_and_float64_round_trip_distinctly(problem, precision, dtype):
    """Both float widths cross the boundary intact: joint_config comes back in the REQUESTED
    precision (fp32 is now the default) and every row is distinct -- no aliasing."""
    seeds, p, q, mask = problem
    out = hjcdik.refine(seeds, p, q, active_target_mask=mask, position_tol=1e-4,
                        orientation_tol=1e-3, max_iters=20, precision=precision)
    qc = out["joint_config"]
    assert qc.dtype == dtype and qc.flags.c_contiguous
    assert len(np.unique(qc, axis=0)) == B, "joint_config rows collapsed -- values are aliased"


def test_boolean_vector_is_not_broadcast(problem):
    """success[] must be per-candidate, not one candidate's flag repeated."""
    seeds, p, q, mask = problem
    out = hjcdik.refine(seeds, p, q, active_target_mask=mask,
                        position_tol=1e-4, orientation_tol=1e-3, max_iters=60)
    succ = out["success"]
    pe = out["position_errors"].max(axis=1)
    oe = out["orientation_errors"].max(axis=1)
    truth = (pe <= 1e-4) & (oe <= 1e-3)
    np.testing.assert_array_equal(
        succ, truth,
        err_msg="the success flag disagrees with the errors the SAME call reported")


def test_success_flag_agrees_with_reported_errors(problem):
    """No false negatives and no false positives, on every entry point."""
    seeds, p, q, mask = problem
    kw = dict(active_target_mask=mask, position_tol=1e-4, orientation_tol=1e-3)
    for name, out in (("refine", hjcdik.refine(seeds, p, q, max_iters=60, **kw)),
                      ("solve", hjcdik.solve(seeds, p, q, coarse_iters=60, lm_iters=60, **kw))):
        pe = out["position_errors"].max(axis=1)
        oe = out["orientation_errors"].max(axis=1)
        truth = (pe <= 1e-4) & (oe <= 1e-3)
        succ = out["success"].astype(bool)
        assert int((truth & ~succ).sum()) == 0, f"{name}: false negatives (solutions dropped)"
        assert int((succ & ~truth).sum()) == 0, f"{name}: false positives"


# ---------------------------------------------------------------------------------------------
# Candidate selection. This is what the bug actually broke: ranking by argmin over a CONSTANT cost
# array always returned candidate 0, so the solver found good solutions and then returned a
# different one.
# ---------------------------------------------------------------------------------------------
def test_argmin_cost_selects_the_genuine_best_candidate(problem):
    """Deterministic selection regression: the best candidate must NOT be index 0, the cost array
    must be nonconstant, and argmin(cost) must land on a candidate that is actually good."""
    seeds, p, q, mask = problem
    out = hjcdik.solve(seeds, p, q, active_target_mask=mask,
                       position_tol=1e-4, orientation_tol=1e-3,
                       coarse_iters=120, lm_iters=60)
    cost = out["cost"]
    pe = out["position_errors"].max(axis=1)

    assert len(np.unique(cost)) > 1, "cost is constant -- zero-stride bug is back"

    best = int(np.argmin(pe))
    assert best != 0, ("this fixture must have its best candidate away from index 0, otherwise the "
                       "test cannot distinguish a correct argmin from the zero-stride failure mode")

    pick = int(np.argmin(cost))
    # cost must be a real ranking signal: its argmin lands in the top decile by true error
    rank = int((pe < pe[pick]).sum())
    assert rank < max(1, B // 10), (
        f"argmin(cost) chose candidate {pick}, ranked #{rank + 1}/{B} by true position error -- "
        f"cost is not tracking solution quality")

    # and the returned joint_config really is that candidate's. rtol is 1e-3, not 1e-9: the solve
    # runs in fp32 by default and reports its own fp32 error, while target_residuals recomputes in
    # fp64 -- they agree to ~1e-7 m, which is far tighter than the thing being checked (that this is
    # the RIGHT candidate, whose neighbours differ by whole millimetres).
    qsel = np.asarray(out["joint_config"][pick], dtype=np.float64)
    res = hjcdik.target_residuals(qsel[None, :], p[:1], q[:1],
                                  active_target_mask=mask[:1])
    np.testing.assert_allclose(res["position_errors"][0].max(), pe[pick], rtol=1e-3, atol=1e-6)


def test_cost_correlates_with_error(problem):
    """The ranking signal must be positively correlated with the thing it ranks."""
    seeds, p, q, mask = problem
    out = hjcdik.solve(seeds, p, q, active_target_mask=mask,
                       position_tol=1e-4, orientation_tol=1e-3,
                       coarse_iters=120, lm_iters=60)
    c = np.corrcoef(out["cost"], out["position_errors"].max(axis=1))[0, 1]
    assert c > 0.3, f"corr(cost, position error) = {c:.3f} -- cost is not a usable ranking signal"
    print(f"\n  corr(cost, position error) = {c:+.3f}")
