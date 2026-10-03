"""Panda collision model as Python data, parsed from the frozen reference header
benchmark/reference/panda_collision_model.cuh (the paper's 59-sphere model). This is the INDEPENDENT
oracle for the Table II collision-free column, not the compiled model: the paper's fixed finger origins
are +/-65 mm, whereas HJCD's kinematic URDF uses +/-40 mm. load_hjcd_spheres() binds the foam
sphere geometry to those kinematic frames independently of GRiD for implementation checks.

Historically this parsed csrc/robots/panda.cuh; that header was removed when the kernel migrated to
grid_collision, so the data arrays it needs were vendored into benchmark/reference/ (see that file's
banner). Pure stdlib + numpy (no GPU). Parsing constant arrays with regex is robust for this fixed header.
"""
from __future__ import annotations

import re
import hashlib
import json
import xml.etree.ElementTree as ET
from functools import lru_cache
from pathlib import Path

import numpy as np

_CUH = Path(__file__).resolve().parent / "reference" / "panda_collision_model.cuh"


def _strip_comments(s: str) -> str:
    return re.sub(r"//[^\n]*", "", s)


def _array_body(text: str, name: str) -> str:
    m = re.search(re.escape(name) + r"\s*\[[^\]]*\]\s*=\s*\{(.*?)\}\s*;", text, re.S)
    if not m:
        raise ValueError(f"array {name!r} not found in {_CUH}")
    return m.group(1)


def _floats(body: str) -> list[float]:
    return [float(t[:-1] if t.endswith("f") else t)
            for t in re.findall(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?f?", body)]


def _ints(body: str) -> list[int]:
    return [int(t) for t in re.findall(r"-?\d+", body)]


_text = _strip_comments(_CUH.read_text())

#: (59, 4) collision spheres in link-local frames: [x, y, z, radius]
SPHERES = np.array(_floats(_array_body(_text, "panda_spheres_array")), dtype=float).reshape(-1, 4)
#: (59,) index of the actuated joint each sphere rigidly moves with (0 = base link)
SPHERE_TO_JOINT = np.array(_ints(_array_body(_text, "panda_sphere_to_joint")), dtype=int)
#: (8, 4, 4) per-joint fixed (origin) transforms, row-major; joint 0 = identity base
FIXED_TRANSFORMS = np.array(_floats(_array_body(_text, "panda_fixed_transforms")),
                            dtype=float).reshape(-1, 4, 4)
#: (8,) pRRTC joint type per joint (0/1/2 = prism x/y/z, 3/4/5 = rot x/y/z); joint 0 unused by FK
JOINT_TYPES = np.array(_ints(_array_body(_text, "panda_joint_types")), dtype=int)

N_JOINTS = len(FIXED_TRANSFORMS)          # 8 (index 0 = base, 1..7 = the 7 actuated joints)
N_SPHERES = len(SPHERES)                  # 59
N_ACTUATED = N_JOINTS - 1                 # 7

# Sanity: the header's own PANDA_SPHERE_COUNT / PANDA_JOINT_COUNT must agree with what we parsed.
_decl = dict(re.findall(r"#define\s+(PANDA_\w+)\s+(\d+)", _text))
assert N_SPHERES == int(_decl["PANDA_SPHERE_COUNT"]), (N_SPHERES, _decl.get("PANDA_SPHERE_COUNT"))
assert len(SPHERE_TO_JOINT) == N_SPHERES, (len(SPHERE_TO_JOINT), N_SPHERES)
assert N_JOINTS == int(_decl["PANDA_JOINT_COUNT"]), (N_JOINTS, _decl.get("PANDA_JOINT_COUNT"))
assert len(JOINT_TYPES) == N_JOINTS, (len(JOINT_TYPES), N_JOINTS)

_ROOT = Path(__file__).resolve().parents[1]
_KINEMATIC_URDF = _ROOT / "csrc/urdf/panda.urdf"
_SPHERE_URDF = _ROOT / "external/foam/assets/panda/smaller_panda_spherized.urdf"


def _link_spheres_to_joint_frames(link_spheres: dict):
    """{link: [(x, y, z, r), ...]} in URDF link frames -> (spheres (S,4), anchors (S,)) in the actuated-joint
    frames the numpy FK uses (fixed suffix transforms folded in). Kinematics from the kinematic URDF, so any
    sphere source (foam, cuRobo, ...) is placed on the same chain."""
    from gen_targets import _parse_joints, _chain_to_target, _fk

    joints = _parse_joints(_KINEMATIC_URDF)
    by_child = {joint["child"]: name for name, joint in joints.items()}
    root_links = {j["parent"] for j in joints.values()} - set(by_child)
    spheres, anchors = [], []
    for name, link_list in link_spheres.items():
        if not link_list:
            continue
        if name in root_links:
            chain = []
        elif name in by_child:
            chain = _chain_to_target(joints, by_child[name])
        else:
            raise ValueError(f"sphere link {name!r} is not in {_KINEMATIC_URDF.name}")
        active = [i for i, j in enumerate(chain) if j["type"] != "fixed"]
        suffix = chain[active[-1] + 1:] if active else chain
        fixed = _fk(suffix, [])
        for x, y, z, r in link_list:
            center = fixed @ np.array([x, y, z, 1.0])
            spheres.append([*center[:3], float(r)])
            anchors.append(len(active))
    data, mapping = np.asarray(spheres), np.asarray(anchors, dtype=int)
    data.setflags(write=False)
    mapping.setflags(write=False)
    return data, mapping


@lru_cache(maxsize=1)
def load_hjcd_spheres():
    """Sphere offsets in actuated Panda frames, from URDF inputs, never generated code.

    Fixed-link transforms come from the kinematic URDF; collision-local centers and
    radii come from foam. This deliberately retains HJCD's existing gripper opening.
    Includes the base sphere so callers can choose their base-contact policy.
    """
    link_spheres = {}
    for link in ET.parse(_SPHERE_URDF).getroot().findall("link"):
        out = []
        for collision in link.findall("collision"):
            sphere = collision.find("geometry/sphere")
            if sphere is None:
                raise ValueError(f"expected pre-spherized geometry on {link.get('name')}")
            origin = collision.find("origin")
            xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
            out.append((*xyz, float(sphere.get("radius"))))
        link_spheres[link.get("name")] = out
    return _link_spheres_to_joint_frames(link_spheres)


def curobo_franka_yml_path() -> Path:
    """cuRobo's bundled Panda robot config (its collision spheres live in it). Needs cuRobo installed."""
    from curobo.content import get_robot_configs_path
    return Path(get_robot_configs_path()) / "franka.yml"


@lru_cache(maxsize=1)
def load_curobo_spheres():
    """cuRobo's bundled Panda sphere model (61 spheres, `franka.yml`, fingers locked at 0.04 like ours),
    placed on OUR kinematic chain. Lets a cuRobo-model oracle judge any solver's configuration."""
    import yaml
    cfg = yaml.safe_load(open(curobo_franka_yml_path()))["robot_cfg"]["kinematics"]
    link_spheres = {link: [(*[float(c) for c in sp["center"]], float(sp["radius"])) for sp in sps]
                    for link, sps in cfg["collision_spheres"].items() if link != "attached_object"}
    return _link_spheres_to_joint_frames(link_spheres)


def panda_sphere_model(model="paper"):
    """Select explicit geometry; preserve the historical paper benchmark default."""
    if model == "paper":
        return SPHERES, SPHERE_TO_JOINT
    if model == "hjcd":
        return load_hjcd_spheres()
    if model == "curobo":
        return load_curobo_spheres()
    raise ValueError("collision model must be 'paper', 'hjcd' or 'curobo'")


def collision_model_metadata(model="paper"):
    """Identify the chosen validation geometry in benchmark result sidecars."""
    panda_sphere_model(model)  # reject typos before writing metadata
    paths = {"paper": lambda: [_CUH], "hjcd": lambda: [_KINEMATIC_URDF, _SPHERE_URDF],
             "curobo": lambda: [_KINEMATIC_URDF, curobo_franka_yml_path()]}[model]()
    return {
        "model": model,
        "scope": "environment-only; non-base spheres",
        "finger_joint_origin_y_m": [0.065, -0.065] if model == "paper" else
            [float(j.find("origin").get("xyz").split()[1])
             for j in ET.parse(_KINEMATIC_URDF).getroot().findall("joint")
             if j.get("name") in ("panda_finger_joint1", "panda_finger_joint2")],
        "source_sha256": {(str(p.relative_to(_ROOT)) if p.is_relative_to(_ROOT) else p.name):
                          hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    }


def write_collision_model_metadata(result_path, model, *, build=None):
    """Keep CSV/YAML schemas stable; identify validation geometry in a sidecar."""
    path = Path(str(result_path) + ".metadata.json")
    metadata = {"collision_validation": collision_model_metadata(model)}
    if build is not None:
        metadata["solver_build"] = build
    path.write_text(json.dumps(metadata, indent=2) + "\n")
