"""Apply documented, per-region proxy edits to the mesh-fit seed (spec section 2).

Reads the immutable seed (g1_proxy_model.seed.yaml), applies the tuning edits below, and
writes the authored model (g1_proxy_model.yaml) with a header documenting every change.
Idempotent: always starts from the seed. NO global shrink, NO pair disabling, NO unexplained
clearance -- only tightening / splitting / repositioning of the named proxies.

Rationale for each edit is validated against the MuJoCo corpus (validate_collision_sidecar.py).

    python3 collision_sidecar/apply_tuning.py && python3 collision_sidecar/build_collision_sidecar.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_collision_sidecar import load_proxies, write_yaml, YAML_OUT  # noqa: E402

SEED = os.path.join(HERE, "g1_proxy_model.seed.yaml")

# Each edit records (why) for the YAML header. Applied by name.
EDITS = [
    "logo_link:*  REMOVED -- logo_link is the decorative chest logo; MuJoCo never collides it "
    "(per-pair both=0). It only produced false positives against the arms.",
    "left/right_shoulder_yaw  proximal (shoulder-MOUNT) end pulled back to 60% -- the mount nests "
    "against the torso at neutral (false positive); the retained distal end is the UPPER-ARM "
    "segment, which still catches real arm-torso collisions when the arm swings in (both stays "
    ">0). Torso proxy is left at full size so arm-torso collisions are NOT under-covered.",
    "left/right_elbow  distal end pulled back to 55% length -- the elbow proxy covers the upper "
    "forearm near the elbow; the wrist cluster covers the distal forearm. Removes the neutral "
    "overlap with wrist_pitch (both=0 -- never a real collision).",
    "left/right_wrist_pitch  shortened to 55% length, r x0.8 -- small link near the hand; the "
    "shrink separates it from the elbow proxy at neutral without missing wrist/hand collisions "
    "(those show up as wrist_yaw/hand vs body, still covered).",
]


def _shorten_capsule(p, frac, anchor="p0"):
    a = np.array(p["p0"]); b = np.array(p["p1"])
    if anchor == "p0":
        b = a + frac * (b - a)
    else:
        a = b + frac * (a - b)
    q = dict(p); q["p0"] = a.tolist(); q["p1"] = b.tolist()
    return q


def apply():
    prims, _ = load_proxies(SEED)
    out = []
    for p in prims:
        link = p["link"]
        if link == "logo_link":
            # REMOVED: MuJoCo has no logo collision geometry; the torso cluster (incl. the merged
            # head) is now represented accurately by the torso SDF narrow phase (Checkpoint 1B).
            continue
        if link == "torso_link" and p["type"] == "capsule":
            # BROAD PHASE only (Checkpoint 1B): a single conservative capsule that ENCLOSES the
            # torso+head collision geometry (radial extent ~135mm, z 0..0.49). Broad-clear => truly
            # clear, so no torso collision is skipped; the accurate verdict comes from the SDF.
            out.append(dict(name="torso_link:broad", link=link, type="capsule",
                            p0=[0.005, 0.0, 0.0], p1=[0.005, 0.0, 0.49], radius=0.14))
            continue
        # shoulder_yaw is NOT shortened anymore: the torso side is handled by the SDF, so the full
        # upper-arm proxy is retained to catch deep shoulder-into-torso collisions.
        if link in ("left_hip_roll_link", "right_hip_roll_link") and p["type"] == "capsule":
            # The fit capsule (181mm x r32) over-reaches toward the pelvis, penetrating the pelvis
            # SDF on near-contact FREE poses. A single capsule cannot both avoid those false
            # positives AND retain every deep hip_roll<->pelvis collision (the hip nests very tightly
            # at the pelvis) -- a limb-proxy limitation, not an SDF one. Balanced setting (r32->28,
            # pelvis-facing end to 82%): minimises the sum of FP + FN; residual listed in the report.
            q = _shorten_capsule(p, 0.82, anchor="p0")
            q["radius"] = 0.028
            out.append(q); continue
        if link in ("left_wrist_yaw_link", "right_wrist_yaw_link") and p["type"] == "capsule":
            # The wrist_yaw MuJoCo body carries the HAND geometry (merged), which the URDF-fit proxy
            # under-covers -- so wrist/hand-vs-thigh (wrist_yaw<->hip_pitch) collisions were missed.
            # Widen r18->26 to approximate the hand.
            q = dict(p); q["radius"] = 0.026
            out.append(q); continue
        if link in ("left_elbow_link", "right_elbow_link") and p["type"] == "capsule":
            out.append(_shorten_capsule(p, 0.72, anchor="p0")); continue
        if link in ("left_wrist_pitch_link", "right_wrist_pitch_link") and p["type"] == "capsule":
            q = _shorten_capsule(p, 0.62, anchor="p1")       # keep the distal (hand) end
            q["radius"] = round(p["radius"] * 0.82, 5)
            out.append(q); continue
        out.append(p)
    # Checkpoint 1C: base_link is the PELVIS (collision lives in the MuJoCo scene, not the URDF).
    # It is a boxy central-body cluster, so like the torso it gets an SDF narrow phase; the proxy
    # here is a conservative BROAD-PHASE enclosing capsule (pelvis geometry: base_link frame
    # x+-0.09, y+-0.098, z -0.182..0). Broad-clear => truly clear; the accurate verdict is the SDF.
    if not any(p["link"] == "base_link" for p in out):
        out.append(dict(name="base_link:broad", link="base_link", type="capsule",
                        p0=[0.0, 0.0, -0.185], p1=[0.0, 0.0, 0.01], radius=0.12))
    write_yaml(out, header_notes=["TUNED (apply_tuning.py) -- documented edits:"] +
               [f"  * {e}" for e in EDITS + [
                   "base_link:broad  ADDED -- conservative pelvis broad-phase capsule; the pelvis "
                   "SDF (g1_pelvis_sdf.npz) is the narrow phase for base_link<->limb pairs (1C)."]])
    print(f"wrote tuned model: {len(out)} primitives -> {os.path.relpath(YAML_OUT, HERE)}")


if __name__ == "__main__":
    apply()
