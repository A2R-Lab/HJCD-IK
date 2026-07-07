"""Panda collision model as Python data, parsed directly from csrc/robots/panda.cuh (the single source of
truth — no transcription). Exposes the 59-sphere model + the pRRTC fixed transforms/joint types so the
shared-sphere-model collision validator (benchmark/panda_collision.py) can place spheres in the world frame
at any q with the SAME geometry HJCD's own kernel uses.

Pure stdlib + numpy (no GPU). Parsing constant arrays with regex is robust for this fixed local header.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_CUH = Path(__file__).resolve().parents[1] / "csrc" / "robots" / "panda.cuh"


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
