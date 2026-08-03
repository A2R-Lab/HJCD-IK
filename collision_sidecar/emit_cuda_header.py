"""Emit the generated CUDA header from the accepted Checkpoint 1 artifacts (Checkpoint 2, s2).

Reads the validated sidecar artifacts (nothing is hand-transcribed) and writes
generated/g1_collision_sidecar.cuh: a structure-of-arrays, FP32 header the CUDA sidecar
consumes. The BULK data (SDF grids, convex vertices) stay in their .npz/.json artifacts and
are uploaded at host init; the header carries the compact immutable metadata + all hashes.

Emitted (SoA, spec section 3):
    * hashes tying the header to URDF / joint order / proxy YAML / torso SDF / pelvis SDF /
      convex pieces / checked-pair policy  (a mismatch must be rejected at load)
    * link FK topology (parent link, fixed parent->joint transform, joint axis, q index)
    * primitives (type, link, params) + primitive->link
    * checked pairs grouped by narrow phase: PRIMITIVE / CLUSTER_SDF / CONVEX_GJK
    * per-joint descendant-link and affected-pair lists (for incremental checking)
    * cluster SDF metadata (origin, spacing, dims, quant) for torso + pelvis
    * convex piece ranges + per-link bounding spheres

    python3 collision_sidecar/emit_cuda_header.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
GEN = os.path.join(HJCD, "generated")
sys.path.insert(0, HERE)

from urdf_model import parse_urdf, HJCD_JOINT_ORDER  # noqa: E402

OUT = os.path.join(GEN, "g1_collision_sidecar.cuh")
NP_TYPE = {"PRIMITIVE": 0, "CLUSTER_SDF": 1, "CONVEX_GJK": 2}
PRIM_TYPE = {"sphere": 0, "capsule": 1}
CLUSTER_ID = {"TORSO": 0, "PELVIS": 1}


def _mat_colmajor(T):
    return [float(T[r, c]) for c in range(4) for r in range(4)]


def _arr(name, ctype, vals, per_line=12):
    # __device__ __constant__ so the arrays are readable in device code (namespace-scope
    # constexpr arrays are host-only). The header is included by exactly ONE .cu (the sidecar),
    # so these are single definitions.
    n = len(vals)
    s = [f"__device__ __constant__ {ctype} {name}[{n if n else 1}] = {{"]
    line = "    "
    for i, v in enumerate(vals):
        if ctype == "float":
            s0 = f"{float(v):.7g}"
            if "." not in s0 and "e" not in s0 and "n" not in s0:
                s0 += ".0"
            tok = s0 + "f,"
        else:
            tok = str(int(v)) + ","
        if len(line) + len(tok) > 100:
            s.append(line); line = "    "
        line += tok
    s.append(line); s.append("};")
    return "\n".join(s)


def emit():
    art = json.load(open(os.path.join(GEN, "g1_collision_sidecar.json")))
    model = parse_urdf(os.path.join(HJCD, "csrc", "urdf", "g1_29dof_rev_1_0.urdf"))
    torso = np.load(os.path.join(GEN, "g1_torso_sdf.npz"), allow_pickle=True)
    pelvis = np.load(os.path.join(GEN, "g1_pelvis_sdf.npz"), allow_pickle=True)
    convex = json.load(open(os.path.join(GEN, "g1_convex_pieces.json")))

    # -- link FK topology (BFS from root) ---------------------------------
    order = []
    frontier = [model.root_link]
    while frontier:
        lk = frontier.pop(0); order.append(lk)
        for jn in model.children_joints.get(lk, ()):
            frontier.append(model.joint_by_name[jn].child)
    link_idx = {lk: i for i, lk in enumerate(order)}
    parent_link, Torigin, axis, qindex = [], [], [], []
    for lk in order:
        jn = model.parent_of_link.get(lk)
        if jn is None:
            parent_link.append(-1); Torigin += _mat_colmajor(np.eye(4)); axis += [0, 0, 1]; qindex.append(-1)
            continue
        j = model.joint_by_name[jn]
        parent_link.append(link_idx[j.parent])
        Torigin += _mat_colmajor(model.joint_origin(j))
        axis += [float(x) for x in j.axis]
        qindex.append(HJCD_JOINT_ORDER.index(j.name) if j.movable else -1)

    # -- primitives -------------------------------------------------------
    prims = art["primitives"]
    p_type, p_link, p_par = [], [], []
    for p in prims:
        p_type.append(PRIM_TYPE[p["type"]]); p_link.append(link_idx[p["link"]])
        if p["type"] == "sphere":
            p_par += [*p["center"], p["radius"], 0, 0, 0]
        else:
            p_par += [*p["p0"], p["radius"], *p["p1"]]

    # -- link -> primitives CSR (feet carry up to 4 prims; the checker needs the FULL set) --
    lp = art["link_primitives"]
    link_prim_off, link_prim = [0], []
    for lk in order:
        link_prim += list(lp.get(lk, []))
        link_prim_off.append(len(link_prim))

    # -- clusters (rigid SDF): id order TORSO(0), PELVIS(1) --------------------
    clusters = art["clusters"]; cl_by_link = {c["link"]: cid for cid, c in clusters.items()}
    link_cluster = [CLUSTER_ID[cl_by_link[lk]] if lk in cl_by_link else -1 for lk in order]
    cl_ids = sorted(clusters, key=lambda c: CLUSTER_ID[c])
    cl_broad = [clusters[cid]["broad_prim"] for cid in cl_ids]
    cl_npz = {"TORSO": torso, "PELVIS": pelvis}
    cl_origin, cl_spacing, cl_dims, cl_scale = [], [], [], []
    for cid in cl_ids:
        z = cl_npz[cid]
        cl_origin += [float(x) for x in z["origin"]]
        cl_spacing.append(float(z["spacing"]))
        cl_dims += [int(x) for x in z["dims"]]
        cl_scale.append(float(z["sdf_scale"]))

    # -- checked pairs: link-based, faithful to _linkpair_colliding (aligned w/ checked_link_pairs) --
    #    PAIR_TYPE priority: GJK > CLUSTER > PRIMITIVE. Link ids only; checker derives prims/cluster side.
    gjk_set = {frozenset(p) for p in art["gjk_pairs"]}
    pair_type, pair_a, pair_b = [], [], []
    n_prim = n_cluster = n_gjk = 0
    for (a, b) in art["checked_link_pairs"]:
        key = frozenset((a, b))
        if key in gjk_set:
            pair_type.append(NP_TYPE["CONVEX_GJK"]); n_gjk += 1
        elif a in cl_by_link or b in cl_by_link:
            pair_type.append(NP_TYPE["CLUSTER_SDF"]); n_cluster += 1
        else:
            pair_type.append(NP_TYPE["PRIMITIVE"]); n_prim += 1
        pair_a.append(link_idx[a]); pair_b.append(link_idx[b])

    # -- link -> convex slot (GJK narrow phase) ---------------------------
    link_convex = [-1] * len(order)

    # -- joint descendant + affected-pair CSR -----------------------------
    desc_flat, desc_off = [], [0]
    aff_flat, aff_off = [], [0]
    for jn in HJCD_JOINT_ORDER:
        desc_flat += [link_idx[l] for l in art["joint_descendant_links"][jn] if l in link_idx]
        desc_off.append(len(desc_flat))
        aff_flat += art["joint_affected_pairs"][jn]
        aff_off.append(len(aff_flat))

    # -- convex pieces for the GJK narrow phase (MULTI-piece links, e.g. torso = 3 hulls). --
    #    Canonical PIECE ORDER = ascending link index (FK/BFS `order`); within a link, JSON piece
    #    order; hull pieces only. g_cverts MUST be uploaded in this same order (see sidecar upload).
    #    Per-link enclosing sphere kept for a cheap link-level broad phase; per-piece bounding sphere
    #    for the piece-level broad phase before each exact GJK.
    # TYPED PIECES (native-completion checkpoint, Task A). A link is a union of typed pieces --
    # HULL (0) or SPHERE (1) -- so a sphere-only link (the four-sphere foot) routes through exact
    # GJK correctly, matching the CPU oracle (gjk.py already supported both). Every piece emits:
    #   PIECE_TYPE   : 0 hull, 1 sphere
    #   PIECE_VERT_OFF: g_cverts range (EMPTY for a sphere -- off[i+1]==off[i])
    #   PIECE_SPHERE : link-local center(3)+radius; hull pieces store their bound-sphere here too so
    #                  the array is always meaningful, but the narrow phase only reads it for spheres
    #   PIECE_BOUND  : per-piece conservative bounding sphere (broad phase) -- for a sphere, itself
    PIECE_HULL, PIECE_SPHERE_T = 0, 1
    gjk_link_set = {l for pr in art["gjk_pairs"] for l in pr}
    conv_link_idx, conv_bound = [], []          # slot -> link, per-link enclosing sphere
    link_piece_off = [0]                          # CSR link -> pieces (len N_LINKS+1)
    piece_vert_off = [0]                          # CSR piece -> g_cverts range
    piece_bound = []                              # 4 per piece: center + radius (link-local)
    piece_type = []                               # 0 hull / 1 sphere
    piece_sphere = []                             # 4 per piece: center + radius (spheres only used)
    verts_canonical = []                          # concatenated piece verts IN THIS ORDER (saved -> npy)
    nverts = npieces = 0
    for L, lk in enumerate(order):
        if lk in gjk_link_set:
            v = convex["links"][lk]
            link_convex[L] = len(conv_link_idx); conv_link_idx.append(L)
            conv_bound += [*v["bound_center"], v["bound_radius"]]
            for p in v["pieces"]:
                if p["type"] == "hull":
                    verts = np.asarray(p["verts"], float)
                    verts_canonical.append(verts)
                    nverts += len(verts)
                    c = 0.5 * (verts.min(0) + verts.max(0))
                    r = float(np.linalg.norm(verts - c, axis=1).max())
                    piece_type.append(PIECE_HULL)
                    piece_sphere += [float(c[0]), float(c[1]), float(c[2]), 0.0]
                    piece_bound += [float(c[0]), float(c[1]), float(c[2]), r]
                elif p["type"] == "sphere":
                    c = np.asarray(p["center"], float); r = float(p["radius"])
                    piece_type.append(PIECE_SPHERE_T)
                    piece_sphere += [float(c[0]), float(c[1]), float(c[2]), r]
                    piece_bound += [float(c[0]), float(c[1]), float(c[2]), r]
                else:
                    # CAPSULE etc. are not yet supported by the exact GPU path -- fail loudly rather
                    # than silently drop the piece (spec section 3: no silent skips).
                    raise SystemExit(f"emit_cuda_header: unsupported convex piece type "
                                     f"{p['type']!r} on link {lk}; only hull/sphere are emitted")
                npieces += 1
                piece_vert_off.append(nverts)       # spheres do not advance nverts -> empty range
        link_piece_off.append(npieces)
    # save the canonical vertex array so every uploader (Python wrapper, parity harnesses) is
    # order-consistent with PIECE_VERT_OFF without re-deriving the ordering.
    np.save(os.path.join(GEN, "g1_convex_verts.npy"),
            np.ascontiguousarray(np.concatenate(verts_canonical, axis=0), dtype=np.float64))

    # -- hashes -----------------------------------------------------------
    def sd(*xs):
        return hashlib.sha1("|".join(str(x) for x in xs).encode()).hexdigest()[:16]
    hashes = {
        "URDF": model.urdf_hash(),
        "JOINT_ORDER": model.joint_order_hash(),
        "PROXY_YAML": art["proxy_yaml_hash"],
        "TORSO_SDF": str(torso["geom_hash"]),
        "PELVIS_SDF": str(pelvis["geom_hash"]),
        "CONVEX": convex["geom_hash"],
        "PAIR_POLICY": sd(art["checked_link_pairs"], art["gjk_pairs"], list(clusters)),
        # typed-piece schema hash (Task A): covers the per-piece type + sphere params, so a change
        # in the typed representation is detectable independently of the raw vertex hash.
        "TYPED_PIECE": sd("v1", piece_type, [round(x, 9) for x in piece_sphere]),
    }

    def sdf_meta(z, cid):
        o = z["origin"].astype(float); dm = z["dims"]
        return (f"// cluster {cid}: origin=({o[0]:.5f},{o[1]:.5f},{o[2]:.5f}) spacing={float(z['spacing']):.6f} "
                f"dims=({int(dm[0])},{int(dm[1])},{int(dm[2])}) quant={float(z['sdf_scale']):.1f}")

    L = ["// GENERATED by collision_sidecar/emit_cuda_header.py -- DO NOT HAND-EDIT.",
         "// Structure-of-arrays metadata for the CUDA collision sidecar (Checkpoint 2).",
         "// Bulk data (SDF grids, convex vertices) are uploaded at host init from the .npz/.json",
         "// artifacts; this header carries compact immutable metadata + hashes only.",
         "#pragma once", "#include <cstdint>", "", "namespace g1_sidecar {", ""]
    L += [f'static constexpr const char* HASH_{k} = "{v}";' for k, v in hashes.items()]
    L += ["", f"static constexpr int N_JOINTS = {len(HJCD_JOINT_ORDER)};",
          f"static constexpr int N_LINKS = {len(order)};",
          f"static constexpr int N_PRIMITIVES = {len(prims)};",
          f"static constexpr int N_CHECKED_PAIRS = {len(pair_type)};",
          f"static constexpr int N_PRIM_PAIRS = {n_prim};",
          f"static constexpr int N_CLUSTER_PAIRS = {n_cluster};",
          f"static constexpr int N_GJK_PAIRS = {n_gjk};",
          f"static constexpr int N_CLUSTERS = {len(cl_ids)};",
          f"static constexpr int N_CONVEX_LINKS = {len(conv_link_idx)};",
          f"static constexpr int N_CONVEX_PIECES = {npieces};",
          f"static constexpr int N_CONVEX_VERTS = {nverts};   // uploaded from g1_convex_pieces.json",
          f"static constexpr int PAIR_PRIMITIVE = {NP_TYPE['PRIMITIVE']};",
          f"static constexpr int PAIR_CLUSTER_SDF = {NP_TYPE['CLUSTER_SDF']};",
          f"static constexpr int PAIR_CONVEX_GJK = {NP_TYPE['CONVEX_GJK']};",
          f"static constexpr float BROAD_MARGIN = {0.02:.7g}f;   // m, skip SDF beyond enclosing capsule",
          "static constexpr float SDF_TOL = 0.001f;   // capsule-SDF adaptive tolerance",
          "static constexpr int SDF_MAX_EVALS = 48;   // capsule-SDF eval cap",
          "", sdf_meta(torso, "TORSO(0)"), sdf_meta(pelvis, "PELVIS(1)"), "",
          "// ---- link FK topology (BFS order) ----",
          _arr("LINK_PARENT", "int", parent_link),
          _arr("LINK_T_ORIGIN", "float", Torigin),        # 16 per link, column-major
          _arr("LINK_AXIS", "float", axis),               # 3 per link
          _arr("LINK_QINDEX", "int", qindex),             # movable q index or -1
          "", "// ---- primitives ----",
          _arr("PRIM_TYPE", "int", p_type),
          _arr("PRIM_LINK", "int", p_link),
          _arr("PRIM_PARAM", "float", p_par),             # 7 per prim: sphere[c,r,-,-,-] capsule[p0,r,p1]
          "", "// ---- link -> primitives CSR ----",
          _arr("LINK_PRIM_OFF", "int", link_prim_off),    # len N_LINKS+1
          _arr("LINK_PRIM", "int", link_prim),
          "", "// ---- clusters (rigid SDF): id 0=TORSO 1=PELVIS ----",
          _arr("LINK_CLUSTER", "int", link_cluster),      # cluster id per link or -1
          _arr("CLUSTER_BROAD_PRIM", "int", cl_broad),    # enclosing-capsule primitive per cluster
          _arr("CLUSTER_ORIGIN", "float", cl_origin),     # 3 per cluster
          _arr("CLUSTER_SPACING", "float", cl_spacing),
          _arr("CLUSTER_DIMS", "int", cl_dims),           # 3 per cluster (nx,ny,nz)
          _arr("CLUSTER_SCALE", "float", cl_scale),       # int16 quant scale (npz sdf_scale)
          "", "// ---- checked pairs (link-based; PAIR_TYPE priority GJK>CLUSTER>PRIM) ----",
          _arr("PAIR_TYPE", "int", pair_type),            # aligned with checked_link_pairs
          _arr("PAIR_LINK_A", "int", pair_a),
          _arr("PAIR_LINK_B", "int", pair_b),
          "", "// ---- joint descendant links / affected pairs (CSR) ----",
          _arr("JOINT_DESC_OFF", "int", desc_off),
          _arr("JOINT_DESC", "int", desc_flat),
          _arr("JOINT_AFFPAIR_OFF", "int", aff_off),
          _arr("JOINT_AFFPAIR", "int", aff_flat),
          "", "// ---- convex piece ranges + bounding spheres ----",
          _arr("LINK_CONVEX", "int", link_convex),        # convex slot per link or -1 (link-level bound)
          _arr("CONVEX_LINK", "int", conv_link_idx),
          _arr("CONVEX_BOUND", "float", conv_bound),      # 4 per link: enclosing center + radius
          _arr("LINK_PIECE_OFF", "int", link_piece_off),  # CSR: link -> convex pieces (len N_LINKS+1)
          _arr("PIECE_VERT_OFF", "int", piece_vert_off),  # CSR: piece -> g_cverts range (empty=sphere)
          _arr("PIECE_BOUND", "float", piece_bound),      # 4 per piece: center + radius (link-local)
          _arr("PIECE_TYPE", "int", piece_type),          # 0 hull, 1 sphere (typed pieces, Task A)
          _arr("PIECE_SPHERE", "float", piece_sphere),    # 4 per piece: center + radius (spheres)
          "", "}  // namespace g1_sidecar", ""]
    open(OUT, "w").write("\n".join(L))
    print(f"wrote {os.path.relpath(OUT, HJCD)}: {len(order)} links, {len(prims)} prims, "
          f"{len(link_prim)} link-prim entries, {len(pair_type)} pairs "
          f"(prim {n_prim}/cluster {n_cluster}/gjk {n_gjk}), {len(cl_ids)} clusters, "
          f"{npieces} convex pieces / {nverts} verts")
    print("  hashes:", {k: v for k, v in hashes.items()})


if __name__ == "__main__":
    emit()
