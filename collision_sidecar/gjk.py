"""Deterministic CPU GJK for the convex narrow phase (Checkpoint 1D, spec section 3).

Support mappings for point / sphere / capsule / box / convex-hull pieces, and a GJK that
returns intersection (colliding / free) plus the separation distance when disjoint. Double
precision, bounded iterations, conservative failure. EPA is intentionally omitted -- the
mandatory runtime result is colliding/free + separated distance; penetration depth is a
MuJoCo-side diagnostic.

A "piece" is a dict in a link-local frame; `world_support(piece, R, t, d)` returns the support
point in world after the rigid link transform (R, t). A LINK is a union of pieces; two links
collide iff any piece pair collides (broad phase = enclosing bounding spheres).
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


# --------------------------------------------------------------------------- support mappings
def _support_local(piece, d):
    """Support point of a piece in its OWN local frame, direction d (local)."""
    t = piece["type"]
    if t == "sphere":
        c = np.asarray(piece["center"]); n = np.linalg.norm(d)
        return c + piece["radius"] * (d / n if n > EPS else np.zeros(3))
    if t == "capsule":
        p0 = np.asarray(piece["p0"]); p1 = np.asarray(piece["p1"])
        base = p0 if d @ p0 >= d @ p1 else p1
        n = np.linalg.norm(d)
        return base + piece["radius"] * (d / n if n > EPS else np.zeros(3))
    if t == "box":
        c = np.asarray(piece["center"]); R = np.asarray(piece["R"]); h = np.asarray(piece["half"])
        loc = R.T @ d
        corner = np.sign(loc) * h
        return c + R @ corner
    if t == "hull":
        V = np.asarray(piece["_verts"] if "_verts" in piece else piece["verts"])
        return V[np.argmax(V @ d)]
    if t == "point":
        return np.asarray(piece["p"])
    raise ValueError(t)


def world_support(piece, R, t, d):
    """Support in WORLD after rigid transform (R, t): argmax_x in piece  x . d."""
    return R @ _support_local(piece, R.T @ np.asarray(d, float)) + t


# --------------------------------------------------------------------------- closest-on-simplex
def _closest_seg(a, b):
    ab = b - a
    tt = -(a @ ab) / (ab @ ab + EPS)
    tt = min(max(tt, 0.0), 1.0)
    return a + tt * ab, ([0] if tt <= 0 else [1] if tt >= 1 else [0, 1])


def _closest_tri(a, b, c):
    # closest point on triangle abc to origin (Ericson 5.1.5), returns point + used idxs
    ab, ac, ao = b - a, c - a, -a
    d1, d2 = ab @ ao, ac @ ao
    if d1 <= 0 and d2 <= 0:
        return a, [0]
    bo = -b; d3, d4 = ab @ bo, ac @ bo
    if d3 >= 0 and d4 <= d3:
        return b, [1]
    vc = d1 * d4 - d3 * d2
    if vc <= 0 and d1 >= 0 and d3 <= 0:
        v = d1 / (d1 - d3); return a + v * ab, [0, 1]
    co = -c; d5, d6 = ab @ co, ac @ co
    if d6 >= 0 and d5 <= d6:
        return c, [2]
    vb = d5 * d2 - d1 * d6
    if vb <= 0 and d2 >= 0 and d6 <= 0:
        w = d2 / (d2 - d6); return a + w * ac, [0, 2]
    va = d3 * d6 - d5 * d4
    if va <= 0 and (d4 - d3) >= 0 and (d5 - d6) >= 0:
        w = (d4 - d3) / ((d4 - d3) + (d5 - d6)); return b + w * (c - b), [1, 2]
    denom = 1.0 / (va + vb + vc)
    v = vb * denom; w = vc * denom
    return a + ab * v + ac * w, [0, 1, 2]


def _closest_simplex(W):
    """Closest point on the simplex (1..4 points) to the origin. Returns (point, kept idxs,
    origin_enclosed)."""
    n = len(W)
    if n == 1:
        return W[0], [0], np.dot(W[0], W[0]) < EPS
    if n == 2:
        p, idx = _closest_seg(W[0], W[1]); return p, idx, np.dot(p, p) < EPS
    if n == 3:
        p, idx = _closest_tri(W[0], W[1], W[2]); return p, idx, np.dot(p, p) < EPS
    # tetrahedron: closest over the 4 faces; if origin inside all -> enclosed
    a, b, c, dd = W
    best_p, best_idx, best_d2 = None, None, np.inf
    inside = True
    for face in ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)):
        pa, pb, pc = W[face[0]], W[face[1]], W[face[2]]
        # outward test: is origin on the far side of this face from the 4th vertex?
        other = W[[i for i in range(4) if i not in face][0]]
        nrm = np.cross(pb - pa, pc - pa)
        if (nrm @ (other - pa)) * (nrm @ (-pa)) < 0:      # origin outside this face
            inside = False
            p, idx = _closest_tri(pa, pb, pc)
            d2 = p @ p
            if d2 < best_d2:
                best_p, best_idx, best_d2 = p, [face[i] for i in idx], d2
    if inside:
        return np.zeros(3), [0, 1, 2, 3], True
    return best_p, best_idx, False


# --------------------------------------------------------------------------- GJK
def gjk(supportA, supportB, max_iter=64, tol=1e-9):
    """supportA/supportB: d -> world support point. Returns dict:
        colliding (bool), distance (m, 0 if colliding), iterations, reason."""
    def sup(d):
        return supportA(d) - supportB(-d)
    W = [sup(np.array([1.0, 0.0, 0.0]))]
    closest = W[0]
    for it in range(1, max_iter + 1):
        d = -closest
        if d @ d < tol:
            return {"colliding": True, "distance": 0.0, "iterations": it, "reason": "origin_in_hull"}
        a = sup(d)
        # no progress toward the origin along d => shapes are disjoint
        if (a @ d) - (closest @ d) < tol * (1.0 + abs(closest @ d)):
            return {"colliding": False, "distance": float(np.linalg.norm(closest)),
                    "iterations": it, "reason": "converged"}
        W.append(a)
        closest, keep, enclosed = _closest_simplex(W)
        W = [W[i] for i in keep]
        if enclosed:
            return {"colliding": True, "distance": 0.0, "iterations": it, "reason": "origin_enclosed"}
    return {"colliding": float(np.linalg.norm(closest)) < 1e-4, "distance": float(np.linalg.norm(closest)),
            "iterations": max_iter, "reason": "max_iter"}


def link_pieces_collide(piecesA, RA, tA, piecesB, RB, tB, margin=0.0):
    """Union-of-convex vs union-of-convex. Returns (colliding, min_distance, total_iters,
    n_piece_pairs). A link pair collides iff any piece pair does (gap < margin)."""
    colliding, min_gap, iters, npairs = False, np.inf, 0, 0
    for pa in piecesA:
        for pb in piecesB:
            npairs += 1
            res = gjk(lambda d, p=pa: world_support(p, RA, tA, d),
                      lambda d, p=pb: world_support(p, RB, tB, d))
            iters += res["iterations"]
            gap = -1e-9 if res["colliding"] else res["distance"]
            min_gap = min(min_gap, gap)
            if gap < margin:
                colliding = True
    return colliding, float(min_gap), iters, npairs
