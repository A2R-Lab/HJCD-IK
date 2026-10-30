"""Predeclared rail gate: 32 targets, B=2000/S=4, >=31 accurate per cell.

Both precisions, self-only and obstacle scenes, cold and neighboring warm targets.
Every hard-mode result is independently checked, not just the best candidate.
No latency measurements or assertions; safe on a shared GPU.
"""
import hashlib
import json
import runpy

import numpy as np
import pytest
import hjcdik
from test_joint_fixtures import ROOT, SphereModel, fk, load_file
from test_general_joints import build_solver, diagnostic, targets_for, assert_poses, independent_quality

EXAMPLE = runpy.run_path(str(ROOT / "examples/rail_ur5e.py"))
TARGET = EXAMPLE["TARGET"]
DATA = json.loads((ROOT / "examples/rail_ur5e_targets.json").read_text())


@pytest.fixture(scope="module")
def rail(tmp_path_factory):
    urdf, spheres = EXAMPLE["assemble"](tmp_path_factory.mktemp("rail"))
    solver, module = build_solver(urdf, "test-rail-ur5e", spheres, TARGET)
    return solver, module, urdf, spheres, SphereModel(spheres, TARGET)


def test_assembly_and_targets_reproduce(tmp_path):
    generator = load_file("rail_targets", ROOT / "scripts/codegen/make_rail_targets.py")
    output = tmp_path / "targets.json"
    generator.generate(output)
    assert output.read_bytes() == (ROOT / "examples/rail_ur5e_targets.json").read_bytes()
    urdf, spheres = EXAMPLE["assemble"](tmp_path)
    assert hashlib.sha256(urdf.read_bytes()).hexdigest() == DATA["urdf_sha256"]
    assert hashlib.sha256(spheres.read_bytes()).hexdigest() == DATA["spheres_sha256"]
    model = SphereModel(spheres, TARGET)
    base = model.links == "base_link_inertia"
    assert base.sum() > 0 and (model.anchors[base] == 0).all()
    assert (model.anchors >= 0).all()  # no supplied arm spheres left at a skipped root
    for cell in DATA["sets"].values():
        for q, nearby in zip(cell["q"], cell["nearby_q"]):
            assert model.free(q, cell["world"].get("spheres", []), .003)
            assert model.free(nearby, cell["world"].get("spheres", []), .003)
        assert_poses(targets_for(urdf, cell["q"], TARGET), np.array(cell["targets"]), 1e-14)


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_rail_fk_jacobian_and_moving_geometry(rail, precision):
    solver, module, urdf, spheres, model = rail
    limits = np.array(solver.info.joint_limits)
    qs = np.random.default_rng(20261025).uniform(limits[:, 0], limits[:, 1], (200, 7))
    qs[:2, 0] = [1e-5, 1-1e-5]
    assert solver.info.joint_names[0] == "rail_joint"
    assert solver.info.joint_units == ("m",) + ("rad",)*6
    poses, jac, free = diagnostic(solver, module, qs, precision == "fp64")
    assert_poses(poses, targets_for(urdf, qs, TARGET), 3e-6 if precision == "fp32" else 1e-9)
    np.testing.assert_allclose(jac[:, :, 0], np.tile([1, 0, 0, 0, 0, 0], (200, 1)), atol=1e-9)
    np.testing.assert_array_equal(free, [model.free(q) for q in qs])
    source = hjcdik.presets()["ur5e"]["urdf"]
    source_model = SphereModel(hjcdik.sphere_presets()["ur5e"], TARGET)
    for q in qs:
        expected = fk(source, q[1:], TARGET); expected[0, 3] += q[0]
        np.testing.assert_allclose(fk(urdf, q, TARGET), expected, atol=2e-14)
        placed = source_model.placed(q[1:]); placed[:, 0] += q[0]
        np.testing.assert_allclose(model.placed(q), placed, atol=2e-14)
    for q, actual in zip(qs[:12], jac[:12]):
        center = fk(urdf, q, TARGET)
        for h in (1e-5, 3e-6):
            expected = np.empty((6, 7))
            for j in range(7):
                delta = np.eye(7)[j]*h
                plus, minus = fk(urdf, q+delta, TARGET), fk(urdf, q-delta, TARGET)
                expected[:3, j] = (plus[:3, 3]-minus[:3, 3])/(2*h)
                omega = ((plus[:3, :3]-minus[:3, :3])/(2*h)) @ center[:3, :3].T
                expected[3:, j] = [omega[2, 1], omega[0, 2], omega[1, 0]]
            np.testing.assert_allclose(actual, expected, atol=3e-6 if precision == "fp32" else 2e-9, rtol=0)
    # An obstacle at the now-moving base must be checked even with self checks off.
    q = np.array(DATA["sets"]["self_only"]["q"][0])
    center = model.placed(q)[model.links == "base_link_inertia"][0, :3]
    solver.set_world(spheres=[[*center, .001]])
    assert not diagnostic(solver, module, q, precision == "fp64", check_self=False)[2][0]
    solver.clear_world()


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
@pytest.mark.parametrize("scene", ["self_only", "obstacle"])
def test_public_rail_journey(rail, precision, scene, monkeypatch, tmp_path, record_property):
    solver, module, urdf, spheres, model = rail
    monkeypatch.chdir(tmp_path)
    cell = DATA["sets"][scene]
    targets, nearby = np.array(cell["targets"]), np.array(cell["nearby_targets"])
    results = EXAMPLE["journey"](solver, targets, nearby, cell["world"], precision)
    counts = {}
    for kind, poses in (("out", targets), ("warm", nearby[results["selected"]]), ("restored", targets[:1])):
        out = results[kind]
        assert out is not None
        good = independent_quality(solver, urdf, poses, out, TARGET)[0]
        assert good.sum() >= (1 if kind == "restored" else 31), (scene, precision, kind, good)
        counts[kind] = int(good.sum())
        for result in out:
            for q in result.q:
                assert model.free(q, cell["world"].get("spheres", []))
    assert results["blocked"].count.tolist() == [0]
    assert not results["blocked"].valid.any() and np.isnan(results["blocked"].q).all()
    assert solver.world_counts == dict(boxes=0, spheres=0, capsules=0)
    record_property("rail_quality", json.dumps(dict(scene=scene, precision=precision, **counts)))


def test_rail_example_does_not_warm_empty_results(monkeypatch):
    from test_examples import _empty_result
    solver = hjcdik.Solver()
    calls = []
    def empty(*args, **kwargs):
        calls.append(kwargs)
        return _empty_result()
    monkeypatch.setattr(solver, "solve", empty)
    cell = DATA["sets"]["self_only"]
    rows = EXAMPLE["journey"](solver, np.array(cell["targets"][:1]),
                              np.array(cell["nearby_targets"][:1]), {})
    assert rows["warm"] is None and len(calls) == 3
    assert all("q_init" not in call for call in calls)


@pytest.mark.parametrize("precision", ["fp32", "fp64"])
def test_rail_partial_warp_sanitizer_path(rail, precision, monkeypatch):
    solver, module, urdf, spheres, model = rail
    monkeypatch.setenv("HJCD_LM_WARPS", "3")
    cell = DATA["sets"]["obstacle"]
    q, target = np.array(cell["q"][0]), np.array(cell["targets"][0])
    solver.set_world(**cell["world"])
    out = solver.solve(target, batch_size=37, num_solutions=4, precision=precision,
                       collision="hard", q_init=q[None, :])
    assert independent_quality(solver, urdf, [target], out, TARGET)[0].all()
    assert all(model.free(row, cell["world"]["spheres"]) for row in out[0].q)
    solver.set_world(spheres=[[0, 0, 0, 100]])
    assert solver.solve(target, batch_size=37, num_solutions=4, precision=precision).count.tolist() == [0]
    solver.clear_world()
