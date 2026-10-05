"""CPU geometry identity checks; never infer equality from easy IK scenes."""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmark"))
import gen_targets as gt
from panda_collision import panda_config_collision_free, panda_link_transforms, panda_spheres_world
from panda_model import (_array_body, _floats, _ints, SPHERES, SPHERE_TO_JOINT,
                         collision_model_metadata, load_hjcd_spheres, panda_sphere_model)


def test_compiled_spheres_match_independent_urdf_geometry():
    # Check all baked rows, not just the four known finger differences.
    header = (ROOT / "csrc/generated/grid.cuh").read_text()
    spheres, anchors = load_hjcd_spheres()
    keep = anchors != 0
    np.testing.assert_array_equal(_ints(_array_body(header, "mt_anchor")), anchors[keep] - 1)
    offsets = np.array(_floats(_array_body(header, "mt_offset"))).reshape(-1, 3)
    np.testing.assert_allclose(offsets, spheres[keep, :3], atol=1e-12, rtol=0)
    radii = _floats(_array_body(header, "g_collision_sphere_r"))
    np.testing.assert_allclose(radii, spheres[keep, 3], atol=1e-8, rtol=0)
    assert len(radii) == 58


def test_all_reference_joint_frames_match_kinematic_urdf():
    joints = gt._parse_joints(ROOT / "csrc/urdf/panda.urdf")
    chains = [gt._chain_to_target(joints, f"panda_joint{i}") for i in range(1, 8)]
    for q in np.random.default_rng(73).uniform(-2.5, 2.5, (8, 7)):
        reference = panda_link_transforms(q)
        for i, chain in enumerate(chains, 1):
            # URDF quarter turns are decimal-rounded; the frozen reference uses exact 0/1.
            np.testing.assert_allclose(reference[i], gt._fk(chain, q[:i]), atol=1e-10, rtol=0)


def test_paper_default_is_preserved_and_openings_are_distinct():
    assert panda_sphere_model()[0] is SPHERES
    assert panda_sphere_model()[1] is SPHERE_TO_JOINT
    q = np.zeros(7)
    paper = panda_spheres_world(q)
    hjcd = panda_spheres_world(q, model="hjcd")
    np.testing.assert_allclose(paper[:55], hjcd[:55], atol=1e-6, rtol=0)
    np.testing.assert_allclose(np.linalg.norm(paper[55:, :3] - hjcd[55:, :3], axis=1),
                               .025, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(paper[:, 3], hjcd[:, 3])


@pytest.mark.parametrize("sphere_index", [55, 56, 57, 58])
def test_obstacle_at_each_compiled_finger_distinguishes_reference(sphere_index):
    q = np.zeros(7)
    center = panda_spheres_world(q, model="hjcd")[sphere_index, :3]
    world = {"cuboid": {"finger": {"dims": [.001] * 3, "pose": [*center, 1, 0, 0, 0]}}}
    assert not panda_config_collision_free(q, world, model="hjcd")
    assert panda_config_collision_free(q, world, model="paper")


@pytest.mark.parametrize("model,opening", [("paper", .065), ("hjcd", .04)])
def test_model_metadata_identifies_opening_and_sources(model, opening):
    meta = collision_model_metadata(model)
    assert meta["model"] == model
    assert meta["finger_joint_origin_y_m"] == [opening, -opening]
    assert meta["scope"] == "environment-only; non-base spheres"
    assert all(re.fullmatch("[0-9a-f]{64}", h) for h in meta["source_sha256"].values())


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError, match="collision model"):
        panda_spheres_world(np.zeros(7), model="typo")


def test_result_sidecar_records_validation_model_and_compiled_header(tmp_path):
    from panda_model import write_collision_model_metadata
    result = tmp_path / "results.csv"
    result.write_text("unchanged CSV schema\n")
    build = {"grid_header_sha256": "a" * 64}
    write_collision_model_metadata(result, "hjcd", build=build)
    metadata = json.loads((tmp_path / "results.csv.metadata.json").read_text())
    assert result.read_text() == "unchanged CSV schema\n"
    assert metadata["solver_build"] == build
    assert metadata["collision_validation"]["model"] == "hjcd"
    assert metadata["collision_validation"]["finger_joint_origin_y_m"] == [.04, -.04]


def test_grid_warp_tables_match_the_baked_collision_batch():
    """Upstream warp tables match the block FK extractor and independent URDF checks above."""
    header = (ROOT / "csrc/generated/grid.cuh").read_text()
    side = header
    n = 58
    anchors = _ints(_array_body(side, "sphere_anchor"))
    offsets = np.array(_floats(_array_body(side, "sphere_offset"))).reshape(-1, 3)
    radii = _floats(_array_body(side, "sphere_radius"))
    assert len(anchors) == len(radii) == len(offsets) == n == 58
    np.testing.assert_array_equal(anchors, _ints(_array_body(header, "mt_anchor")))
    np.testing.assert_allclose(offsets, np.array(_floats(_array_body(header, "mt_offset"))).reshape(-1, 3), atol=1e-6, rtol=0)
    np.testing.assert_allclose(radii, _floats(_array_body(header, "g_collision_sphere_r")), atol=1e-7, rtol=0)
