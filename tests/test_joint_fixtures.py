"""Independent CPU geometry checks before the GPU solver consumes these fixtures."""
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gt = load_file("fixture_fk", ROOT / "benchmark/gen_targets.py")
variants = load_file("fixture_generator", ROOT / "scripts/codegen/make_panda_variants.py")


def fk(path, q, target="panda_grasptarget_hand"):
    return gt._fk(gt._chain_to_target(gt._parse_joints(path), target), q)


def configs(n=200):
    chain = gt._chain_to_target(gt._parse_joints(ROOT / "csrc/urdf/panda.urdf"), "panda_grasptarget_hand")
    lo, hi = gt._actuated_limits(chain)
    return np.random.default_rng(20261025).uniform(lo, hi, (n, 7))


def test_fixture_generation_is_reproducible(tmp_path):
    variants.generate(tmp_path)
    for path in tmp_path.glob("*.urdf"):
        assert path.read_bytes() == (ROOT / "tests/robots" / path.name).read_bytes()


@pytest.mark.parametrize("name,sign", [("panda_axes", 1), ("panda_negz", -1)])
def test_reexpressed_fk_matches_at_mapped_configurations(name, sign):
    for q in configs():
        np.testing.assert_allclose(fk(ROOT / f"tests/robots/{name}.urdf", sign*q),
                                   fk(ROOT / "csrc/urdf/panda.urdf", q), atol=2e-14, rtol=0)


def sphere_centers(path, q):
    root = ET.parse(path).getroot()
    joints = gt._parse_joints(path)
    by_child = {v["child"]: k for k, v in joints.items()}
    # Full arm coordinates by name; each sphere's ancestor chain may end earlier.
    coords = {f"panda_joint{i+1}": q[i] for i in range(7)}
    placed = []
    for link in root.findall("link"):
        parent = by_child.get(link.get("name"))
        if parent is None:
            transform = np.eye(4)
        else:
            chain = gt._chain_to_target(joints, parent)
            names = {id(v): k for k, v in joints.items()}
            chain_q = [coords[names[id(v)]] for v in chain if v["type"] != "fixed"]
            transform = gt._fk(chain, chain_q)
        for collision in link.findall("collision"):
            sphere = collision.find("geometry/sphere")
            if sphere is not None:
                center = variants.vector(collision.find("origin").get("xyz"))
                placed.append([*(transform[:3, :3] @ center + transform[:3, 3]), float(sphere.get("radius"))])
    return np.array(placed)


def test_reexpressed_spheres_preserve_physical_placement():
    for q in configs():
        np.testing.assert_allclose(sphere_centers(ROOT / "tests/robots/panda_axes_spheres.urdf", q),
                                   sphere_centers(ROOT / "csrc/collision/panda_foam_rebound.urdf", q),
                                   atol=2e-14, rtol=0)


def test_slider_fk_is_a_translation():
    for displacement, q in zip(np.linspace(0, 0.5, 20), configs(20)):
        reference = fk(ROOT / "csrc/urdf/panda.urdf", q)
        reference[2, 3] += displacement
        np.testing.assert_allclose(fk(ROOT / "tests/robots/panda_slider.urdf", np.r_[displacement, q]),
                                   reference, atol=2e-14, rtol=0)
