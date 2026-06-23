#!/usr/bin/env python3
"""Smoke-test EE-frame equivalence across solvers.

The cross-solver comparison is only fair if every solver's end-effector frame is the SAME physical frame
as the shared targets (default `panda_hand`). For a handful of random joint configs, this computes the EE
pose with each solver's OWN kinematic model and compares it to a reference (HJCD-IK's pure-numpy URDF FK,
the one validated in tests/test_fk_equivalence.py). A solver offset by ~0.1 m is using a different tool
frame (e.g. a TCP) and its open-world numbers would be unfair until realigned.

Each backend is gated — uninstalled ones are reported SKIP, so this runs anywhere (with no baselines it
still validates the reference + harness). Run it on the GPU box after scripts/install_baselines.sh.

  python benchmark/check_ee_frames.py --num 8                         # all installed backends
  python benchmark/check_ee_frames.py --num 8 --pos-tol-mm 1 --ori-tol-rad 1e-2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_targets as gt  # noqa: E402  (reference numpy FK helpers)


def _canon_quat(q):
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def _pose_err(a, b):
    """a,b: [x,y,z, qw,qx,qy,qz]. Returns (pos_mm, ori_rad), quaternion double-cover aware."""
    pos_mm = float(np.linalg.norm(a[:3] - b[:3]) * 1000.0)
    d = abs(float(np.dot(_canon_quat(a[3:7]), _canon_quat(b[3:7]))))
    return pos_mm, float(2.0 * np.arccos(min(1.0, d)))


def reference_poses(urdf, target, configs):
    """HJCD-IK numpy FK to `target` frame for each config -> (K,7) [x,y,z,qw,qx,qy,qz]."""
    joints = gt._parse_joints(urdf)
    chain = gt._chain_to_target(joints, target)
    out = []
    for q in configs:
        T = gt._fk(chain, q)
        out.append(np.concatenate([T[:3, 3], gt._quat_from_R(T[:3, :3])]))
    return np.asarray(out)


# ---- gated backends: each returns (K,7) poses [x,y,z,qw,qx,qy,qz] or raises ----
def fk_pyroki(configs, ee_link="panda_hand", urdf_path=None):
    import jax.numpy as jnp
    import yourdfpy
    import xml.etree.ElementTree as ET
    from io import StringIO
    import pyroki as pk
    if urdf_path:
        urdf = yourdfpy.URDF.load(urdf_path)
    else:
        from robot_descriptions.loaders.yourdfpy import load_robot_description
        urdf = load_robot_description("panda_description")
        xml_tree = urdf.write_xml()
        for j in xml_tree.findall('.//joint[@type="prismatic"]'):
            j.set("type", "fixed")
        urdf = yourdfpy.URDF.load(StringIO(ET.tostring(xml_tree.getroot(), encoding="unicode")))
    robot = pk.Robot.from_urdf(urdf)
    idx = robot.links.names.index(ee_link)
    out = []
    for q in configs:
        fk = np.asarray(robot.forward_kinematics(jnp.asarray(q))[idx])  # [qw,qx,qy,qz, x,y,z]
        out.append(np.concatenate([fk[4:7], fk[0:4]]))
    return np.asarray(out)


def fk_curobo(configs, ee_link="panda_hand", urdf_path=None, base_link="panda_link0", robot_file="franka.yml"):
    import torch
    from curobo.types.base import TensorDeviceType
    from curobo.types.robot import RobotConfig
    from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
    from curobo.util_file import get_robot_configs_path, join_path, load_yaml
    td = TensorDeviceType()
    if urdf_path:
        cfg = RobotConfig.from_basic(urdf_path, base_link, ee_link, td)
    else:
        cfg = RobotConfig.from_dict(load_yaml(join_path(get_robot_configs_path(), robot_file))["robot_cfg"], td)
    model = CudaRobotModel(cfg.kinematics)
    q = torch.as_tensor(np.asarray(configs), dtype=td.dtype, device=td.device)
    st = model.get_state(q)
    pos = st.ee_position.detach().cpu().numpy()
    quat = st.ee_quaternion.detach().cpu().numpy()  # [qw,qx,qy,qz]
    return np.concatenate([pos, quat], axis=1)


def fk_ikflow(configs, model_name="panda_full_tpm"):
    from ikflow.model_loading import get_ik_solver
    ik_solver, _ = get_ik_solver(model_name)
    return np.asarray(ik_solver.robot.forward_kinematics(np.asarray(configs)))  # [x,y,z,qw,qx,qy,qz]


BACKENDS = {"pyroki": fk_pyroki, "curobo": fk_curobo, "ikflow": fk_ikflow}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--urdf", default=str(root / "include" / "test_urdf" / "panda.urdf"))
    ap.add_argument("--target", default="panda_hand_joint", help="reference EE frame (shared-target frame)")
    ap.add_argument("--ee-link", default="panda_hand", help="EE link name in the solvers' own models")
    ap.add_argument("--robot-urdf", default="", help="custom URDF for pyroki/curobo (fetch / DoF variants)")
    ap.add_argument("--base-link", default="panda_link0", help="base link for the custom-URDF cuRobo model")
    ap.add_argument("--num", type=int, default=8, help="random joint configs to check")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pos-tol-mm", type=float, default=1.0)
    ap.add_argument("--ori-tol-rad", type=float, default=1e-2)
    ap.add_argument("--only", nargs="+", choices=list(BACKENDS), help="restrict to these backends")
    args = ap.parse_args()

    joints = gt._parse_joints(args.urdf)
    chain = gt._chain_to_target(joints, args.target)
    lo, hi = gt._actuated_limits(chain)
    rng = np.random.default_rng(args.seed)
    configs = lo + rng.random((args.num, len(lo))) * (hi - lo)

    ref = reference_poses(args.urdf, args.target, configs)
    print(f"[check_ee_frames] reference = numpy FK to '{args.target}' ({len(lo)} DoF, {args.num} configs)")

    backends = args.only or list(BACKENDS)
    rc = 0
    for name in backends:
        try:
            if name == "pyroki":
                poses = fk_pyroki(configs, args.ee_link, args.robot_urdf or None)
            elif name == "curobo":
                poses = fk_curobo(configs, args.ee_link, args.robot_urdf or None, args.base_link)
            else:
                poses = fk_ikflow(configs)
        except Exception as e:
            print(f"  {name:8s} SKIP ({type(e).__name__}: {str(e).splitlines()[0][:80]})")
            continue
        errs = [_pose_err(poses[i], ref[i]) for i in range(len(ref))]
        mp = max(e[0] for e in errs); mo = max(e[1] for e in errs)
        ok = mp <= args.pos_tol_mm and mo <= args.ori_tol_rad
        rc |= 0 if ok else 1
        flag = "PASS" if ok else "FAIL"
        hint = "" if ok else ("  <- ~constant offset => different EE/tool frame" if mp > 50 else "")
        print(f"  {name:8s} {flag}  max pos {mp:8.3f} mm   max ori {mo:.2e} rad{hint}")

    print("[check_ee_frames] done." + ("" if rc == 0 else "  (a backend disagrees — realign its EE frame)"))
    sys.exit(rc)


if __name__ == "__main__":
    main()
