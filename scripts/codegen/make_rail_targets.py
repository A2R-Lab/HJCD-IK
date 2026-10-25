#!/usr/bin/env python3
"""Freeze rail-example targets using CPU FK/collision only, never solver outcomes.

Run with the development environment. The test oracle lives outside the package.
The assembly and witnesses are illustrative, not physical-geometry certification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))
from test_joint_fixtures import SphereModel, fk, gt


def generate(output):
    example = runpy.run_path(str(ROOT / "examples/rail_ur5e.py"))
    with tempfile.TemporaryDirectory(prefix="rail-targets-") as temporary:
        urdf, spheres = example["assemble"](temporary)
        target = example["TARGET"]
        model = SphereModel(spheres, target)
        limits = np.array(gt._actuated_limits(gt._chain_to_target(gt._parse_joints(urdf), target)))
        lo, hi = limits
        def pose(q):
            transform = fk(urdf, q, target)
            return np.r_[transform[:3, 3], gt._quat_from_R(transform[:3, :3])].tolist()
        data = dict(seed_base=20261025, selection="first 32 CPU sphere-free witnesses and neighbors with 3 mm clearance; no IK selection",
                    urdf_sha256=hashlib.sha256(urdf.read_bytes()).hexdigest(),
                    spheres_sha256=hashlib.sha256(spheres.read_bytes()).hexdigest(), sets={})
        for index, (name, world) in enumerate((
            ("self_only", {}), ("obstacle", {"spheres": [[.5, .25, .35, .12]]}))):
            rng = np.random.default_rng(20261025 + index)
            qs, nearby, attempts = [], [], 0
            while len(qs) < 32 and attempts < 10000:
                attempts += 1
                q = rng.uniform(lo + .02*(hi-lo), hi - .02*(hi-lo))
                neighbor = q + np.array([.002, .003, -.003, .002, -.002, .003, -.003])
                if model.free(q, world.get("spheres", []), .003) and model.free(neighbor, world.get("spheres", []), .003):
                    qs.append(q.tolist()); nearby.append(neighbor.tolist())
            if len(qs) != 32:
                raise RuntimeError(f"only {len(qs)} independent feasible witnesses for {name}")
            data["sets"][name] = dict(world=world, attempted=attempts, q=qs, nearby_q=nearby,
                                      targets=[pose(q) for q in qs], nearby_targets=[pose(q) for q in nearby])
        Path(output).write_text(json.dumps(data, indent=2) + "\n")
        print({name: cell["attempted"] for name, cell in data["sets"].items()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "examples/rail_ur5e_targets.json")
    generate(parser.parse_args().output)
