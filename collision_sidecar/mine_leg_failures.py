"""Checkpoint 3C.2 sec.1 -- enumerate every deep native false negative in the saved 3,000-candidate
held-out corpus, grouped by EXACT unordered MuJoCo link pair (from saved geom/link data, not names).

For each FN records the MuJoCo link pair + depth, the current native pair type + signed gap for that
pair, and whether another native collision masked it. Writes generated/g1_leg_hard_negatives.json.
"""
from __future__ import annotations
import json, os, sys, collections
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
GEN = os.path.join(HJCD, "generated")
sys.path.insert(0, HERE)
from collision_cpu import SidecarCPU  # noqa: E402

DEEP = -5.0  # mm (penetration deeper than 5mm => "deep")


def main():
    cpu = SidecarCPU()
    art = json.load(open(os.path.join(GEN, "g1_collision_sidecar.json")))
    checked = [tuple(sorted(p)) for p in art["checked_link_pairs"]]
    checked_set = {p: i for i, p in enumerate(checked)}
    prims = art["primitives"]; lp = art["link_primitives"]

    def pair_type(a, b):
        key = tuple(sorted((a, b)))
        if key not in checked_set:
            return "UNCHECKED"
        gjk = {frozenset(p) for p in art["gjk_pairs"]}
        if frozenset((a, b)) in gjk:
            return "CONVEX_GJK"
        cl = {c["link"] for c in art["clusters"].values()}
        if a in cl or b in cl:
            return "CLUSTER_SDF"
        return "PRIMITIVE"

    def prim_repr(link):
        return sorted({prims[i]["type"] for i in lp.get(link, [])})

    def native_pair_gap(q, a, b):
        """Signed sidecar gap for the specific link pair (min over its narrow-phase sub-checks)."""
        W, T = cpu.world_primitives(q)
        key = tuple(sorted((a, b)))
        if key not in checked_set:
            return None
        idx = None
        for i, (x, y) in enumerate(cpu.art["checked_link_pairs"]):
            if tuple(sorted((x, y))) == key:
                idx = i; break
        kind = cpu.lp_class[idx]
        if kind[0] == "prim":
            return float(min(cpu._pair_gap(W, i, j) for (i, j) in kind[1]))
        if kind[0] == "cluster":
            _, cid, limbs = kind; Tc = T[cpu.clusters[cid]["link"]]
            return float(min(cpu._cluster_gap(W[il], Tc, cid)[0] for il in limbs))
        return float(cpu._gjk_gap(kind[1], kind[2], T, 0.0)[0])

    R = json.load(open(os.path.join(GEN, "g1_heldout_corpus.json")))["rows"]
    fails = []
    for r in R:
        if not (r["mujoco_colliding"] and not r["native_colliding"]):
            continue
        deep_pairs = [(tuple(sorted(p)), r["mujoco_min_depth_mm"]) for p in r["mujoco_pairs"]]
        if r["mujoco_min_depth_mm"] >= DEEP:
            continue
        fails.append(dict(spec=r["spec"], candidate=r["candidate"], success=r["success"],
                          q=r.get("q"), depth_mm=r["mujoco_min_depth_mm"],
                          mujoco_link_pairs=[list(p) for p, _ in deep_pairs],
                          native_masked=r["native_colliding"]))   # False => fully native-free (unmasked)

    # the held-out corpus did NOT store q; re-mine q by reproducing the corpus specs would be needed
    # for exact gaps. Instead group by link pair from the saved MuJoCo data (authoritative).
    groups = collections.defaultdict(lambda: dict(count=0, depths=[], left=0, right=0, specs=set()))
    for f in fails:
        for lp_ in f["mujoco_link_pairs"]:
            key = tuple(sorted(lp_))
            g = groups[key]; g["count"] += 1; g["depths"].append(f["depth_mm"]); g["specs"].add(f["spec"])
            if any(l.startswith("left") for l in lp_): g["left"] += 1
            if any(l.startswith("right") for l in lp_): g["right"] += 1

    report = {"deep_threshold_mm": DEEP, "sign_convention": "native gap<0 = collision; MuJoCo depth<0 = penetration",
              "n_deep_fn_candidates": len(fails), "pairs": []}
    print(f"deep leg-leg false-negative candidates: {len(fails)}")
    print(f"{'pair':>48} {'count':>6} {'min':>7} {'mean':>7} {'max':>7} {'type':>12} {'prim_repr'}")
    for key, g in sorted(groups.items(), key=lambda kv: -kv[1]["count"]):
        a, b = key
        d = np.array(g["depths"])
        typ = pair_type(a, b)
        pr = f"{prim_repr(a)}|{prim_repr(b)}"
        report["pairs"].append(dict(link_pair=list(key), count=g["count"],
                                    depth_min_mm=float(d.min()), depth_mean_mm=float(d.mean()),
                                    depth_max_mm=float(d.max()), current_pair_type=typ,
                                    prim_repr=pr, left=g["left"], right=g["right"]))
        print(f"{a+'|'+b:>48} {g['count']:>6} {d.max():>7.1f} {d.mean():>7.1f} {d.min():>7.1f} {typ:>12} {pr}")
    json.dump({"report": report, "candidates": fails}, open(os.path.join(GEN, "g1_leg_hard_negatives.json"), "w"), indent=2)
    print("wrote generated/g1_leg_hard_negatives.json")


if __name__ == "__main__":
    main()
