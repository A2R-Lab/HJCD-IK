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


class SphereModel:
    """Independent serial-URDF sphere placement and the declared adjacency policy.

    Uses NumPy FK, not GRiD transforms/tables. Same and adjacent movable-anchor
    pairs are excluded, matching the documented model policy, not hardware safety.
    """
    def __init__(self, path, target="panda_grasptarget_hand"):
        joints = gt._parse_joints(path)
        chain = gt._chain_to_target(joints, target)
        ordered = [id(v) for v in chain if v["type"] != "fixed"]
        by_child = {v["child"]: k for k, v in joints.items()}
        self.groups, anchors, links = [], [], []
        for link in ET.parse(path).getroot().findall("link"):
            name = link.get("name")
            parent = by_child.get(name)
            chain = [] if parent is None else gt._chain_to_target(joints, parent)
            indices = [ordered.index(id(v)) for v in chain if v["type"] != "fixed"]
            rows = []
            for collision in link.findall("collision"):
                sphere = collision.find("geometry/sphere")
                if sphere is not None:
                    origin = collision.find("origin")
                    center = variants.vector("0 0 0" if origin is None else origin.get("xyz", "0 0 0"))
                    rows.append([*center, float(sphere.get("radius"))])
            if rows:
                self.groups.append((chain, indices, np.array(rows)))
                anchors.extend([indices[-1] if indices else -1] * len(rows))
                links.extend([name] * len(rows))
        self.anchors = np.array(anchors)
        self.links = np.array(links)
        active = (self.anchors[:, None] >= 0) & (self.anchors[None, :] >= 0)
        self.pairs = np.where(np.triu(active & (np.abs(self.anchors[:, None]-self.anchors[None, :]) > 1), 1))

    def placed(self, q):
        rows = []
        for chain, indices, local in self.groups:
            transform = gt._fk(chain, np.asarray(q)[indices])
            rows.append(np.column_stack([local[:, :3] @ transform[:3, :3].T + transform[:3, 3], local[:, 3]]))
        return np.concatenate(rows)

    def free(self, q, spheres=(), clearance=0):
        placed = self.placed(q)
        a, b = self.pairs
        if np.any(np.linalg.norm(placed[a, :3]-placed[b, :3], axis=1) < placed[a, 3]+placed[b, 3]+clearance):
            return False
        moving = placed[self.anchors >= 0]
        for obstacle in spheres:
            if np.any(np.linalg.norm(moving[:, :3]-obstacle[:3], axis=1) < moving[:, 3]+obstacle[3]+clearance):
                return False
        return True


def sphere_centers(path, q):
    return SphereModel(path).placed(q)


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
