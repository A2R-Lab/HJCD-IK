"""Reusable rigid-cluster signed-distance-field builder (Checkpoint 1B/1C, spec section 1).

Generalizes the torso SDF builder: ONE implementation builds an SDF for any rigid body
cluster from the canonical MuJoCo collision geometry attached to a body (URDF-fixed children
are merged into the MuJoCo body, so a single body id carries the whole cluster). Used for:

    torso_link  -> generated/g1_torso_sdf.npz   (cluster id TORSO)
    base_link   -> generated/g1_pelvis_sdf.npz  (cluster id PELVIS)

Mesh vertices are placed via MuJoCo's own FK and expressed in the cluster (body) frame, then
convex-hulled (MuJoCo uses the convex hull for mesh collision). Occupancy -> signed distance
via an exact Euclidean distance transform, stored int16-quantized. Same geometry semantics,
hashing, quantization and margin for every cluster (the torso SDF is byte-identical to 1B).

    MUJOCO_GL=egl python3 collision_sidecar/build_cluster_sdf.py [--spacing 0.0035]
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
PARENT = os.path.dirname(os.path.dirname(HJCD))
sys.path.insert(0, os.path.join(PARENT, "src"))
sys.path.insert(0, os.path.join(PARENT, "production"))
sys.path.insert(0, HERE)

MARGIN = 0.03           # m of free space padded around the geometry (positive SDF band)
QUANT = 1e4             # int16 quantization: 0.1 mm resolution

CLUSTERS = [            # (body/link name, cluster id, output npz)
    ("torso_link", "TORSO", "g1_torso_sdf.npz"),
    ("base_link", "PELVIS", "g1_pelvis_sdf.npz"),
]


def _cluster_hull_points(m, d, link):
    """Convex-hull vertex clouds (cluster frame) + geom descriptors for a body's collision geoms."""
    import mujoco
    bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, link)
    Rb = d.xmat[bid].reshape(3, 3); pb = d.xpos[bid]
    hulls, geom_desc = [], []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] != bid:
            continue
        if not (m.geom_contype[g] or m.geom_conaffinity[g]):
            continue
        if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = m.geom_dataid[g]
        va, vn = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
        verts = np.array(m.mesh_vert[va:va + vn]).reshape(-1, 3)
        world = d.geom_xpos[g] + (d.geom_xmat[g].reshape(3, 3) @ verts.T).T
        hulls.append((world - pb) @ Rb)                  # world -> cluster frame
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
        geom_desc.append(nm if nm else f"geom{g}:mesh{mid}")
    return hulls, geom_desc


def build_cluster_sdf(link, cluster_id, out_name, spacing):
    import mujoco
    from scipy.spatial import ConvexHull
    from scipy.ndimage import distance_transform_edt
    import route_qs_lsq as RQ

    m = RQ.load_inputs().model
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)                               # neutral; rigid attachment is config-free
    hulls, geom_desc = _cluster_hull_points(m, d, link)
    if not hulls:
        raise SystemExit(f"[cluster-sdf] no collidable mesh geoms on {link}")

    allpts = np.vstack(hulls)
    lo = allpts.min(0) - MARGIN
    hi = allpts.max(0) + MARGIN
    dims = np.ceil((hi - lo) / spacing).astype(int) + 1
    xs = [lo[i] + spacing * np.arange(dims[i]) for i in range(3)]
    eqs = [ConvexHull(h).equations for h in hulls]

    occ = np.zeros(tuple(int(v) for v in dims), bool)     # slice-by-slice, memory-frugal
    YY, ZZ = np.meshgrid(xs[1], xs[2], indexing="ij")
    yz = np.stack([YY.ravel(), ZZ.ravel()], 1)
    for ix, x in enumerate(xs[0]):
        P = np.column_stack([np.full(yz.shape[0], x), yz])
        inside = np.zeros(P.shape[0], bool)
        for E in eqs:
            inside |= np.all(P @ E[:, :3].T + E[:, 3] <= 1e-9, axis=1)
        occ[ix] = inside.reshape(int(dims[1]), int(dims[2]))

    sdf = (distance_transform_edt(~occ) * spacing - distance_transform_edt(occ) * spacing).astype(np.float32)
    q = np.clip(np.round(sdf * QUANT), -32767, 32767).astype(np.int16)

    urdf = os.path.join(HJCD, "csrc", "urdf", "g1_29dof_rev_1_0.urdf")
    urdf_hash = hashlib.sha1(open(urdf, "rb").read()).hexdigest()[:16]
    geom_hash = hashlib.sha1(np.round(allpts, 6).tobytes()).hexdigest()[:16]
    out = os.path.join(HJCD, "generated", out_name)
    np.savez_compressed(
        out, sdf_i16=q, sdf_scale=np.float32(QUANT),
        origin=lo.astype(np.float32), spacing=np.float32(spacing),
        dims=dims.astype(np.int32), torso_link=link,          # 'torso_link' key kept for loader compat
        cluster_link=link, cluster_id=cluster_id,
        geom_names=np.array(geom_desc, object),
        urdf_hash=urdf_hash, geom_hash=geom_hash, n_geoms=len(hulls))
    mb = os.path.getsize(out) / 1e6
    print(f"[{cluster_id}] {link}: dims={tuple(int(v) for v in dims)} spacing={spacing*1000:.1f}mm "
          f"cells={int(np.prod(dims)):,} int16 file={mb:.2f}MB")
    print(f"    bbox={np.round(lo,3)}..{np.round(hi,3)} geoms={geom_desc} "
          f"sdf={sdf.min()*1000:.0f}..{sdf.max()*1000:.0f}mm geom_hash={geom_hash}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--spacing", type=float, default=0.0035)
    a = ap.parse_args()
    for link, cid, out in CLUSTERS:
        build_cluster_sdf(link, cid, out, a.spacing)
