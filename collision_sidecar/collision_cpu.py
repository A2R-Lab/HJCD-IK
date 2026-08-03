"""CPU reference for the G1 collision sidecar -- HYBRID, multi-cluster (Checkpoint 1B/1C).

    non-cluster pairs        : sphere/capsule signed-distance checks
    limb-vs-CLUSTER pairs     : broad phase (compact cluster capsule) -> narrow phase (cluster SDF)

Rigid clusters (each with its own local SDF): TORSO (torso_link) and PELVIS (base_link). The
narrow phase is pair-generic -- a pair carries which cluster SDF it uses; there is ONE
sphere/capsule-vs-SDF algorithm. Cluster SDF pairs participate in incremental checking normally.

This is the ORACLE the GPU sidecar (Checkpoint 2) must match. Distances are true signed proxy
clearances (negative => penetrating).
"""
from __future__ import annotations

import json
import os

import numpy as np

from urdf_model import parse_urdf, HJCD_JOINT_ORDER

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
GEN = os.path.join(REPO, "generated")
BROAD_MARGIN = 0.02          # m: skip the SDF when the limb clears the enclosing cluster capsule


def seg_seg_dist(p1, q1, p2, q2):
    """Minimum distance between segments [p1,q1] and [p2,q2] (Ericson, Real-Time Collision)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    EPS = 1e-12
    if a <= EPS and e <= EPS:
        return np.linalg.norm(p1 - p2)
    if a <= EPS:
        s, t = 0.0, np.clip(f / e, 0, 1)
    else:
        c = d1 @ r
        if e <= EPS:
            t, s = 0.0, np.clip(-c / a, 0, 1)
        else:
            b = d1 @ d2
            denom = a * e - b * b
            s = np.clip((b * f - c * e) / denom, 0, 1) if denom > EPS else 0.0
            t = (b * s + f) / e
            if t < 0:
                t, s = 0.0, np.clip(-c / a, 0, 1)
            elif t > 1:
                t, s = 1.0, np.clip((b - c) / a, 0, 1)
    c1, c2 = p1 + d1 * s, p2 + d2 * t
    return float(np.linalg.norm(c1 - c2))


def pt_seg_dist(p, a, b):
    ab = b - a
    t = np.clip((p - a) @ ab / (ab @ ab + 1e-12), 0, 1)
    return float(np.linalg.norm(p - (a + ab * t)))


class TorsoSDF:
    """Trilinear cluster-local SDF + sphere/capsule queries (spec section 2/3). Named for
    back-compat; used for any rigid cluster (torso, pelvis)."""

    def __init__(self, npz_path):
        z = np.load(npz_path, allow_pickle=True)
        self.sdf = z["sdf_i16"].astype(np.float32) / float(z["sdf_scale"])
        self.origin = z["origin"].astype(float)
        self.spacing = float(z["spacing"])
        self.dims = np.asarray(z["dims"], int)
        self.torso_link = str(z["torso_link"])
        self.cluster_id = str(z["cluster_id"]) if "cluster_id" in z else "TORSO"
        self.hashes = {"urdf": str(z["urdf_hash"]), "geom": str(z["geom_hash"])}
        self.n_geoms = int(z["n_geoms"])

    @classmethod
    def from_grid(cls, sdf, origin, spacing):
        self = cls.__new__(cls)
        self.sdf = np.asarray(sdf, np.float32)
        self.origin = np.asarray(origin, float)
        self.spacing = float(spacing)
        self.dims = np.asarray(self.sdf.shape, int)
        self.torso_link = "torso_link"; self.cluster_id = "TORSO"
        self.hashes = {"urdf": "", "geom": ""}; self.n_geoms = 0
        return self

    def trilinear_sdf(self, p):
        g = (np.asarray(p, float) - self.origin) / self.spacing
        gc = np.clip(g, 0.0, self.dims - 1.0001)
        i = np.floor(gc).astype(int)
        f = gc - i
        s = self.sdf
        val = 0.0
        for dx in (0, 1):
            wx = f[0] if dx else 1 - f[0]
            for dy in (0, 1):
                wy = f[1] if dy else 1 - f[1]
                for dz in (0, 1):
                    wz = f[2] if dz else 1 - f[2]
                    val += wx * wy * wz * s[i[0] + dx, i[1] + dy, i[2] + dz]
        outside = float(np.linalg.norm((g - gc) * self.spacing))
        return float(val) + outside

    def sphere_torso_sdf_distance(self, center, radius):
        return self.trilinear_sdf(center) - radius, np.asarray(center, float), 1

    def capsule_torso_sdf_distance(self, p0, p1, radius, tol=1e-3, max_evals=48):
        p0 = np.asarray(p0, float); p1 = np.asarray(p1, float)
        seg = p1 - p0
        L = float(np.linalg.norm(seg))

        def f(s):
            return self.trilinear_sdf(p0 + s * seg)
        fa, fb = f(0.0), f(1.0)
        evals = 2
        best, bs = (fa, 0.0) if fa <= fb else (fb, 1.0)
        stack = [(0.0, 1.0, fa, fb)]
        while stack and evals < max_evals:
            sa, sb, fsa, fsb = stack.pop()
            if min(fsa, fsb) - 0.5 * (sb - sa) * L >= best - tol:
                continue
            sm = 0.5 * (sa + sb)
            fm = f(sm); evals += 1
            if fm < best:
                best, bs = fm, sm
            stack.append((sa, sm, fsa, fm))
            stack.append((sm, sb, fm, fsb))
        return best - radius, p0 + bs * seg, evals


class SidecarCPU:
    def __init__(self, json_path=None, urdf_path=None):
        json_path = json_path or os.path.join(GEN, "g1_collision_sidecar.json")
        self.art = json.load(open(json_path))
        self.model = parse_urdf(urdf_path or os.path.join(REPO, "csrc", "urdf",
                                                          "g1_29dof_rev_1_0.urdf"))
        self.prims = self.art["primitives"]
        pp = self.art["primitive_pairs"]
        self.nontorso_pairs = pp["sphere_sphere"] + pp["sphere_capsule"] + pp["capsule_capsule"]
        # rigid-cluster SDFs (torso, pelvis, ...)
        self.clusters = {}
        for cid, c in self.art.get("clusters", {}).items():
            sdf_path = os.path.join(GEN, c["sdf_file"])
            self.clusters[cid] = {
                "link": c["link"], "limb_prims": c["limb_prims"],
                "broad": self.prims[c["broad_prim"]],
                "sdf": TorsoSDF(sdf_path) if os.path.exists(sdf_path) else None}
        self.cluster_link_to_id = {c["link"]: cid for cid, c in self.clusters.items()}
        self.torso_link = self.art["torso_link"]        # back-compat
        self.torso_prims = self.clusters.get("TORSO", {}).get("limb_prims", [])
        # convex/GJK narrow phase (Checkpoint 1D)
        self.gjk_pairs = [tuple(p) for p in self.art.get("gjk_pairs", [])]
        self.gjk_pair_set = {frozenset(p) for p in self.gjk_pairs}
        self.convex = {}
        cpf = self.art.get("convex_pieces_file")
        if cpf and os.path.exists(os.path.join(GEN, cpf)):
            cj = json.load(open(os.path.join(GEN, cpf)))
            for link, v in cj["links"].items():
                for p in v["pieces"]:
                    if p["type"] == "hull":
                        p["_verts"] = np.asarray(p["verts"], float)
                self.convex[link] = v
            self.convex_hashes = {"urdf": cj["urdf_hash"], "geom": cj["geom_hash"]}
        self._classify_linkpairs()

    def _classify_linkpairs(self):
        link_prims = self.art["link_primitives"]
        self.lp_class = []          # ("prim",..) | ("cluster",cid,..) | ("gjk",a,b)
        for a, b in self.art["checked_link_pairs"]:
            if frozenset((a, b)) in self.gjk_pair_set:
                self.lp_class.append(("gjk", a, b)); continue
            ca = self.cluster_link_to_id.get(a); cb = self.cluster_link_to_id.get(b)
            if ca or cb:
                cid = ca or cb
                limb = b if ca else a
                self.lp_class.append(("cluster", cid, list(link_prims[limb])))
            else:
                self.lp_class.append(("prim", [(ia, ib) for ia in link_prims[a] for ib in link_prims[b]]))

    def world_primitives(self, q):
        T = self.model.fk(self.model.q_vector_to_names(np.asarray(q, float)))
        out = []
        for p in self.prims:
            Tl = T[p["link"]]; R, t = Tl[:3, :3], Tl[:3, 3]
            if p["type"] == "sphere":
                out.append(("sphere", R @ np.array(p["center"]) + t, None, p["radius"]))
            else:
                out.append(("capsule", R @ np.array(p["p0"]) + t, R @ np.array(p["p1"]) + t,
                            p["radius"]))
        return out, T

    def _pair_gap(self, W, i, j):
        ti, a0, a1, ri = W[i]; tj, b0, b1, rj = W[j]
        if ti == "sphere" and tj == "sphere":
            d = np.linalg.norm(a0 - b0)
        elif ti == "sphere":
            d = pt_seg_dist(a0, b0, b1)
        elif tj == "sphere":
            d = pt_seg_dist(b0, a0, a1)
        else:
            d = seg_seg_dist(a0, a1, b0, b1)
        return d - (ri + rj)

    def _cluster_gap(self, Wi, Tc, cid):
        """Broad phase (limb vs enclosing cluster capsule) -> narrow phase (limb vs cluster SDF).
        Returns (signed_gap, n_sdf_evals, diag)."""
        c = self.clusters[cid]
        typ, a0, a1, r = Wi
        R, t = Tc[:3, :3], Tc[:3, 3]
        broad = c["broad"]
        b0 = t + R @ np.array(broad["p0"]); b1 = t + R @ np.array(broad["p1"]); br = broad["radius"]
        bgap = (pt_seg_dist(a0, b0, b1) if typ == "sphere" else seg_seg_dist(a0, a1, b0, b1)) - (r + br)
        if bgap > BROAD_MARGIN or c["sdf"] is None:
            return bgap, 0, {"sdf_id": cid, "broad_rejected": True, "seg_s": None}
        if typ == "sphere":
            cc = R.T @ (a0 - t)
            gap, closest, ev = c["sdf"].sphere_torso_sdf_distance(cc, r)
            return gap, ev, {"sdf_id": cid, "broad_rejected": False, "seg_s": 0.0}
        p0 = R.T @ (a0 - t); p1 = R.T @ (a1 - t)
        gap, closest, ev = c["sdf"].capsule_torso_sdf_distance(p0, p1, r)
        L = np.linalg.norm(p1 - p0) + 1e-12
        return gap, ev, {"sdf_id": cid, "broad_rejected": False,
                         "seg_s": float(np.dot(closest - p0, p1 - p0) / (L * L))}

    def _gjk_gap(self, a, b, T, margin=0.0):
        """Exact convex narrow phase for link pair (a,b). Conservative bounding-sphere broad
        phase; then union-of-convex GJK. Returns (gap, gjk_iters, diag)."""
        from gjk import link_pieces_collide
        ca, cb = self.convex.get(a), self.convex.get(b)
        Ta, Tb = T[a], T[b]
        Ra, ta = Ta[:3, :3], Ta[:3, 3]; Rb, tb = Tb[:3, :3], Tb[:3, 3]
        # broad phase: enclosing bounding spheres (conservative -> never suppresses a true hit)
        cAw = Ra @ np.array(ca["bound_center"]) + ta
        cBw = Rb @ np.array(cb["bound_center"]) + tb
        bgap = float(np.linalg.norm(cAw - cBw)) - (ca["bound_radius"] + cb["bound_radius"])
        if bgap > margin:
            return bgap, 0, {"narrow": "GJK", "broad_rejected": True, "pieces": (0, 0)}
        colliding, gap, iters, npairs = link_pieces_collide(
            ca["pieces"], Ra, ta, cb["pieces"], Rb, tb, margin=margin)
        return gap, iters, {"narrow": "GJK", "broad_rejected": False,
                            "colliding": colliding, "piece_pairs": npairs}

    def check(self, q, margin=0.0):
        W, T = self.world_primitives(q)
        colliding, min_gap, self._last_evals, self._gjk_iters = [], np.inf, 0, 0
        for i, j in self.nontorso_pairs:
            g = self._pair_gap(W, i, j); min_gap = min(min_gap, g)
            if g < margin:
                colliding.append((i, j, g))
        for cid, c in self.clusters.items():
            Tc = T[c["link"]]
            for il in c["limb_prims"]:
                g, ev, _ = self._cluster_gap(W[il], Tc, cid)
                self._last_evals += ev; min_gap = min(min_gap, g)
                if g < margin:
                    colliding.append((il, cid, g))
        for a, b in self.gjk_pairs:
            g, it, _ = self._gjk_gap(a, b, T, margin)
            self._gjk_iters += it; min_gap = min(min_gap, g)
            if g < margin:
                colliding.append((("gjk", a), ("gjk", b), g))
        return (len(colliding) == 0), colliding, float(min_gap)

    def collision_free(self, q, margin=0.0):
        return self.check(q, margin)[0]

    def colliding_link_pairs(self, q, margin=0.0):
        _, cp, _ = self.check(q, margin)
        out = {}
        for i, j, gap in cp:
            if isinstance(i, tuple) and i[0] == "gjk":     # GJK pair -> (a,b)
                key = tuple(sorted((i[1], j[1])))
            else:
                b = self.clusters[j]["link"] if j in self.clusters else self.prims[j]["link"]
                key = tuple(sorted((self.prims[i]["link"], b)))
            out[key] = min(out.get(key, np.inf), gap)
        return out

    def _linkpair_colliding(self, idx, W, T, margin):
        kind = self.lp_class[idx]
        if kind[0] == "gjk":
            return self._gjk_gap(kind[1], kind[2], T, margin)[0] < margin
        if kind[0] == "cluster":
            _, cid, limbs = kind
            Tc = T[self.clusters[cid]["link"]]
            return any(self._cluster_gap(W[il], Tc, cid)[0] < margin for il in limbs)
        return any(self._pair_gap(W, i, j) < margin for (i, j) in kind[1])

    def full_linkpair_verdict(self, q, margin=0.0):
        W, T = self.world_primitives(q)
        return {idx for idx in range(len(self.lp_class)) if self._linkpair_colliding(idx, W, T, margin)}

    def incremental_linkpair_verdict(self, q_base, joint_idx, new_value, margin=0.0):
        Wb, Tb = self.world_primitives(q_base)
        base = {idx for idx in range(len(self.lp_class)) if self._linkpair_colliding(idx, Wb, Tb, margin)}
        q_new = np.asarray(q_base, float).copy(); q_new[joint_idx] = new_value
        W, T = self.world_primitives(q_new)
        jn = self.art["hjcd_joint_order"][joint_idx]
        affected = set(self.art["joint_affected_pairs"][jn])
        out = set()
        for idx in range(len(self.lp_class)):
            if idx in affected:
                if self._linkpair_colliding(idx, W, T, margin):
                    out.add(idx)
            elif idx in base:
                out.add(idx)
        return out
