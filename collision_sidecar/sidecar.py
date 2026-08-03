"""Standalone GPU collision sidecar -- Python wrapper (Checkpoint 2, Stage 6).

Loads the _sidecar pybind extension, uploads the validated SDF grids + convex vertices once,
and exposes batched full/incremental self-collision checks over the G1's 29-joint config.
Behaviorally isolated from HJCD: importing this does not touch the _hjcdik solver.

    from sidecar import Sidecar
    sc = Sidecar()                       # builds/loads the module, uploads model data
    verdict = sc.full_check(q_batch)     # [B, n_pairs] uint8 colliding flags
    free = sc.collision_free(q_batch)    # [B] bool
"""
from __future__ import annotations
import json, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
GEN = os.path.join(HJCD, "generated")


def _ensure_module():
    import importlib.util, sysconfig
    ext = sysconfig.get_config_var("EXT_SUFFIX")
    so = os.path.join(GEN, f"_sidecar{ext}")
    if not os.path.exists(so):
        subprocess.run(["bash", os.path.join(HERE, "build_sidecar_module.sh")], check=True)
    spec = importlib.util.spec_from_file_location("_sidecar", so)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


class Sidecar:
    CID = {"TORSO": 0, "PELVIS": 1}

    def __init__(self):
        self.m = _ensure_module()
        self.info = self.m.model_info()
        self.n_pairs = self.info["n_checked_pairs"]
        art = json.load(open(os.path.join(GEN, "g1_collision_sidecar.json")))
        self.checked_link_pairs = art["checked_link_pairs"]
        # upload SDF grids (C-order int16)
        for name, cid in self.CID.items():
            fn = "g1_torso_sdf.npz" if cid == 0 else "g1_pelvis_sdf.npz"
            z = np.load(os.path.join(GEN, fn), allow_pickle=True)
            self.m.upload_sdf(cid, np.ascontiguousarray(z["sdf_i16"].astype(np.int16).ravel(order="C")))
        # upload convex verts (canonical piece order == PIECE_VERT_OFF, from emit_cuda_header.py)
        self.m.upload_convex(np.ascontiguousarray(np.load(os.path.join(GEN, "g1_convex_verts.npy")).astype(np.float64)))

    def full_check(self, q, margin=0.0):
        q = np.ascontiguousarray(np.atleast_2d(q).astype(np.float32))
        return self.m.full_check(q, float(margin))

    def collision_free(self, q, margin=0.0):
        return ~self.full_check(q, margin).any(axis=1)

    def incr_check(self, qbase, base, jidx, newval, margin=0.0):
        qbase = np.ascontiguousarray(np.atleast_2d(qbase).astype(np.float32))
        base = np.ascontiguousarray(np.atleast_2d(base).astype(np.uint8))
        jidx = np.ascontiguousarray(np.atleast_1d(jidx).astype(np.int32))
        newval = np.ascontiguousarray(np.atleast_1d(newval).astype(np.float32))
        return self.m.incr_check(qbase, base, jidx, newval, float(margin))

    # diagnostics
    def prim_gaps(self, q):    return self.m.prim_gaps(np.ascontiguousarray(np.atleast_2d(q).astype(np.float32)))
    def cluster_gaps(self, q): return self.m.cluster_gaps(np.ascontiguousarray(np.atleast_2d(q).astype(np.float32)))
    def gjk_gaps(self, q):     return self.m.gjk_gaps(np.ascontiguousarray(np.atleast_2d(q).astype(np.float32)))
