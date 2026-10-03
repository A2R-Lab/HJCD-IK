"""CPU checks of the fairness tooling: MotionBenchMaker mesh export, clearance ladder, sphere-model loaders."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))

trimesh = pytest.importorskip("trimesh")
import mbm_export as me  # noqa: E402
from make_clearance_ladder import grow_cuboids  # noqa: E402
from panda_collision import panda_spheres_world  # noqa: E402
from panda_model import _link_spheres_to_joint_frames, load_hjcd_spheres  # noqa: E402


def _report_kinds(report):
    return [kind.split(" ")[0] for _, kind, _ in report]


def test_box_mesh_exports_as_one_exact_cuboid():
    box = trimesh.creation.box(extents=[0.3, 0.2, 0.1])
    T = me._T([1.0, 2.0, 3.0], [0.7071068, 0.0, 0.0, 0.7071068])
    report = []
    prims = me.mesh_to_primitives(box.vertices, box.faces, T, "b", report)
    assert _report_kinds(report) == ["box"] and len(prims) == 1
    kind, p = prims[0]
    assert kind == "cuboid"
    assert sorted(np.round(p["dims"], 6)) == [0.1, 0.2, 0.3]
    np.testing.assert_allclose(p["pose"][:3], [1.0, 2.0, 3.0], atol=1e-9)


def test_regular_prism_exports_as_cylinder_with_circumradius():
    prism = trimesh.creation.cylinder(radius=0.036, height=0.9, sections=32)
    T = me._T([0.5, 0.1, 0.2], [1.0, 0.0, 0.0, 0.0])
    report = []
    prims = me.mesh_to_primitives(prism.vertices, prism.faces, T, "bar", report)
    assert _report_kinds(report) == ["cylinder"] and prims[0][0] == "cylinder"
    cyl = prims[0][1]
    assert cyl["radius"] == pytest.approx(0.036, abs=1e-6) and cyl["height"] == pytest.approx(0.9, abs=1e-6)
    np.testing.assert_allclose(cyl["pose"][:3], [0.5, 0.1, 0.2], atol=1e-9)
    axis = me._quat_to_R(cyl["pose"][3:7])[:, 2]
    assert abs(abs(axis[2]) - 1.0) < 1e-6              # prism axis preserved


def test_rectilinear_l_shape_decomposes_exactly_and_keeps_the_notch_free():
    # L prism built by hand (no boolean backend needed): the union of [0,.4]x[0,.1] and [0,.1]x[0,.4], height .1
    ring = [(0, 0), (0.4, 0), (0.4, 0.1), (0.1, 0.1), (0.1, 0.4), (0, 0.4), (0, 0.1)]
    v = [(x, y, 0.0) for x, y in ring] + [(x, y, 0.1) for x, y in ring]
    cap = [(0, 1, 2), (0, 2, 3), (0, 3, 6), (6, 3, 4), (6, 4, 5)]
    faces = [(c, b_, a_) for a_, b_, c in cap] + [(a_ + 7, b_ + 7, c + 7) for a_, b_, c in cap]
    for i in range(7):
        j = (i + 1) % 7
        faces += [(i, j, j + 7), (i, j + 7, i + 7)]
    l_shape = trimesh.Trimesh(np.array(v), np.array(faces), process=True)
    assert l_shape.is_watertight and l_shape.volume == pytest.approx(0.007, rel=1e-9)
    report = []
    prims = me.mesh_to_primitives(l_shape.vertices, l_shape.faces, np.eye(4), "L", report)
    assert _report_kinds(report) == ["rectilinear->2"] or _report_kinds(report) == ["rectilinear->3"]
    vol = sum(np.prod(p["dims"]) for _, p in prims)
    assert vol == pytest.approx(0.4 * 0.1 * 0.1 + 0.3 * 0.1 * 0.1, rel=1e-6)
    # the notch (0.3, 0.3, 0.05) is outside every box
    for _, p in prims:
        local = me._quat_to_R(p["pose"][3:7]).T @ (np.array([0.3, 0.3, 0.05]) - np.array(p["pose"][:3]))
        assert np.any(np.abs(local) > np.array(p["dims"]) / 2 + 1e-9)


def test_grow_cuboids_shrinks_clearance_by_delta_per_face():
    inst = {"obstacles": {"cuboid": {"c": {"dims": [0.5, 0.2, 0.1], "pose": [0, 0, 0, 1, 0, 0, 0]}},
                          "cylinder": {"k": {"radius": 0.01, "height": 0.1, "pose": [0, 0, 0, 1, 0, 0, 0]}}},
            "goal_ik": [[0.0] * 7]}
    grown = grow_cuboids(inst, 0.01)
    np.testing.assert_allclose(grown["obstacles"]["cuboid"]["c"]["dims"], [0.52, 0.22, 0.12])
    assert grown["obstacles"]["cylinder"] == inst["obstacles"]["cylinder"]      # cylinders untouched
    assert inst["obstacles"]["cuboid"]["c"]["dims"] == [0.5, 0.2, 0.1]            # source not mutated


def test_link_sphere_loader_reproduces_the_foam_model():
    import xml.etree.ElementTree as ET
    from panda_model import _SPHERE_URDF
    link_spheres = {}
    for link in ET.parse(_SPHERE_URDF).getroot().findall("link"):
        out = []
        for col in link.findall("collision"):
            xyz = np.fromstring(col.find("origin").get("xyz"), sep=" ")
            out.append((*xyz, float(col.find("geometry/sphere").get("radius"))))
        link_spheres[link.get("name")] = out
    spheres, anchors = _link_spheres_to_joint_frames(link_spheres)
    ref_s, ref_a = load_hjcd_spheres()
    np.testing.assert_allclose(spheres, ref_s)
    np.testing.assert_array_equal(anchors, ref_a)


def test_curobo_sphere_model_when_curobo_is_installed():
    pytest.importorskip("curobo")
    from panda_model import panda_sphere_model
    spheres, anchors = panda_sphere_model("curobo")
    assert len(spheres) == 61 and (anchors != 0).sum() == 59
    # fingers locked at 0.04 like ours: finger spheres sit |y| >= 0.04 - radius from the hand axis at q = 0
    world = panda_spheres_world(np.zeros(7), model="curobo")
    assert world[:, 3].max() <= 0.06 + 1e-9


def test_exported_problem_schema_matches_dataset(tmp_path):
    problems = json.load(open(ROOT / "tests/mb_problems.json"))["problems"]["cage_panda"][0]
    scene = {"world": {"collision_objects": [
        {"id": "t", "primitives": [{"type": "box", "dimensions": [0.7, 0.7, 0.04]}],
         "primitive_poses": [{"position": [0.7, -0.1, 0.2], "orientation": [0, 0, -0.04, 0.999]}]},
        {"id": "c", "primitives": [{"type": "cylinder", "dimensions": [0.14, 0.01]}],
         "primitive_poses": [{"position": [0.6, 0.0, 0.3], "orientation": [0, 0, 0, 1]}]}]}}
    request = {"goal_constraints": [{"joint_constraints": [{"joint_name": f"panda_joint{i}", "position": v}
                                                           for i, v in enumerate(problems["goal_ik"][0], 1)]}],
               "start_state": {"joint_state": {"name": [f"panda_joint{i}" for i in range(1, 8)],
                                               "position": problems["start"]}}}
    import yaml
    (tmp_path / "scene0001.yaml").write_text(yaml.safe_dump(scene))
    (tmp_path / "request0001.yaml").write_text(yaml.safe_dump(request))
    from gen_targets import _parse_joints, _chain_to_target
    chain = _chain_to_target(_parse_joints(ROOT / "csrc/urdf/panda.urdf"), "panda_hand_joint")
    out = me.export_problem(tmp_path / "scene0001.yaml", tmp_path / "request0001.yaml", chain, [])
    assert set(out) == set(problems)                                   # same top-level schema
    assert out["goal_pose"]["frame"] == "panda_hand"
    # the dataset's goal_ik reaches its goal_pose: FK(goal_ik) must reproduce the dataset pose
    np.testing.assert_allclose(out["goal_pose"]["position_xyz"], problems["goal_pose"]["position_xyz"], atol=1e-6)
    assert out["obstacles"]["cylinder"]["cylinder0"] == {"radius": 0.01, "height": 0.14, "pose": [0.6, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]}
    assert out["obstacles"]["cuboid"]["cube0"]["pose"][3:] == [0.999, 0.0, 0.0, -0.04]       # xyzw -> wxyz
    assert "cube_robot_stand" in out["obstacles"]["cuboid"]
