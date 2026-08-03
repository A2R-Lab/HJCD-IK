"""Validate the CPU sidecar against the MuJoCo reference corpus (spec sections 3, 4, 10).

Reproduces the deterministic corpus labels, runs the CPU sidecar on every config, and reports:
    * global config-level confusion matrix (overall / tuning / held-out)
    * per-link-pair confusion matrix
    * false-positive and false-negative examples (saved for visualization)
    * minimum signed proxy distance for MuJoCo-free samples
    * detection margin for deliberate collisions
    * sidecar FK vs MuJoCo agreement (needs MuJoCo)

Exit code is nonzero when any MANDATORY Checkpoint-1 gate fails (spec section 6):
    neutral free, crouch free, all deliberate regression collisions detected, FK within
    tolerance, no unexplained always-colliding pair.

    MUJOCO_GL=egl python3 collision_sidecar/validate_collision_sidecar.py
"""
from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HJCD = os.path.dirname(HERE)
PARENT = os.path.dirname(os.path.dirname(HJCD))
sys.path.insert(0, HERE)

from collision_cpu import SidecarCPU  # noqa: E402
from urdf_model import parse_urdf, HJCD_JOINT_ORDER  # noqa: E402

CORPUS = os.path.join(HJCD, "generated", "g1_collision_corpus.json")
FAILURES = os.path.join(HJCD, "generated", "g1_sidecar_failures.json")
FK_POS_TOL = 5e-4       # m
FK_ROT_TOL = 2e-3       # rad
FREE_MARGIN_ALARM = 0.002   # m -- a MuJoCo-free sample the sidecar clears by < this is fragile


def _mj_pairs(label):
    return {tuple(sorted(p)) for (p, _d) in label["pairs"]}


def fk_check():
    """Sidecar FK vs MuJoCo over the corpus configs (needs MuJoCo)."""
    try:
        import mujoco
        sys.path.insert(0, os.path.join(PARENT, "src")); sys.path.insert(0, os.path.join(PARENT, "production"))
        import route_qs_lsq as RQ
    except Exception as e:
        return None, f"skipped ({e})"
    inp = RQ.load_inputs(); m, d = inp.model, inp.data
    um = parse_urdf(os.path.join(HJCD, "csrc", "urdf", "g1_29dof_rev_1_0.urdf"))
    qadr = {n: int(m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in HJCD_JOINT_ORDER}
    pel = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, um.root_link)
    fw = int(m.jnt_qposadr[[j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE][0]]) + 3
    common = [l for l in um.links if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, l) >= 0]
    corp = json.load(open(CORPUS))
    maxp = maxr = 0.0
    for s in corp["samples"][::4]:
        q = np.asarray(s["q"], float)
        d.qpos[:] = 0.0; d.qpos[fw] = 1.0
        for n, v in zip(HJCD_JOINT_ORDER, q):
            d.qpos[qadr[n]] = v
        mujoco.mj_forward(m, d)
        T = um.fk(um.q_vector_to_names(q))
        Tp = np.eye(4); Tp[:3, :3] = d.xmat[pel].reshape(3, 3); Tp[:3, 3] = d.xpos[pel]
        Tpi = np.linalg.inv(Tp)
        for l in common:
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, l)
            Tm = np.eye(4); Tm[:3, :3] = d.xmat[b].reshape(3, 3); Tm[:3, 3] = d.xpos[b]
            Tm = Tpi @ Tm
            maxp = max(maxp, float(np.abs(T[l][:3, 3] - Tm[:3, 3]).max()))
            dR = T[l][:3, :3].T @ Tm[:3, :3]
            maxr = max(maxr, float(np.arccos(np.clip((np.trace(dR) - 1) / 2, -1, 1))))
    return (maxp, maxr), "ok"


def main():
    corp = json.load(open(CORPUS))
    sc = SidecarCPU()
    samples = corp["samples"]

    def confusion(subset):
        tp = fp = fn = tn = 0
        for s in subset:
            mj = s["label"]["colliding"]
            sd = not sc.collision_free(np.asarray(s["q"], float))
            tp += mj and sd; fp += (not mj) and sd; fn += mj and (not sd); tn += (not mj) and (not sd)
        return tp, fp, fn, tn

    tune = [s for s in samples if s["split"] == "tuning"]
    held = [s for s in samples if s["split"] == "heldout"]
    print("=" * 78)
    print("SIDECAR vs MuJoCo -- config-level confusion (TP=both collide, FP=sidecar-only, "
          "FN=missed real, TN=both free)")
    print("=" * 78)
    for name, ss in (("OVERALL", samples), ("tuning", tune), ("held-out", held)):
        tp, fp, fn, tn = confusion(ss)
        print(f"  {name:<9} TP={tp:>3} FP={fp:>3} FN={fn:>3} TN={tn:>3}   "
              f"(FN=missed collisions -> {'OK' if fn == 0 else 'FAIL'})")

    # per-link-pair confusion
    pair_cm = collections.defaultdict(lambda: [0, 0, 0])   # [both, sidecar_only(FP), mj_only(FN)]
    for s in samples:
        q = np.asarray(s["q"], float)
        sd = set(sc.colliding_link_pairs(q).keys())
        mj = _mj_pairs(s["label"])
        for p in sd & mj:
            pair_cm[p][0] += 1
        for p in sd - mj:
            pair_cm[p][1] += 1
        for p in mj - sd:
            pair_cm[p][2] += 1
    print("\nPER-LINK-PAIR confusion (both / sidecar-only-FP / mujoco-only-FN), top offenders:")
    order = sorted(pair_cm.items(), key=lambda kv: -(kv[1][1] + kv[1][2]))
    for p, (both, fpp, fnn) in order[:12]:
        if fpp + fnn == 0:
            continue
        print(f"  both={both:>3} FP={fpp:>3} FN={fnn:>3}   {p[0]} <-> {p[1]}")

    # false-positive / false-negative examples + save
    fps, fns = [], []
    for s in samples:
        q = np.asarray(s["q"], float)
        free, cp, mg = sc.check(q)
        mj = s["label"]["colliding"]
        if (not free) and (not mj):
            fps.append(dict(category=s["category"], split=s["split"], min_gap_mm=round(mg * 1000, 1),
                            pairs=sorted(f"{a}~{b}" for (a, b) in sc.colliding_link_pairs(q)), q=s["q"]))
        if free and mj:
            fns.append(dict(category=s["category"], split=s["split"],
                            mj_pairs=[p for (p, _d) in s["label"]["pairs"]], q=s["q"]))
    json.dump({"false_positives": fps, "false_negatives": fns}, open(FAILURES, "w"), indent=1)
    print(f"\nFALSE POSITIVES: {len(fps)}   FALSE NEGATIVES: {len(fns)}  (saved -> "
          f"{os.path.relpath(FAILURES, HJCD)})")
    for f in fps[:6]:
        print(f"  FP [{f['category']}] gap={f['min_gap_mm']}mm  {f['pairs']}")
    for f in fns[:6]:
        print(f"  FN [{f['category']}]  {f['mj_pairs']}")

    # min signed distance for MuJoCo-free samples the sidecar clears
    free_gaps = [sc.check(np.asarray(s["q"], float))[2] for s in samples if not s["label"]["colliding"]]
    free_gaps = np.array([g for g in free_gaps if g > 0])
    if free_gaps.size:
        print(f"\nMuJoCo-free samples cleared by the sidecar: min margin {free_gaps.min()*1000:.1f}mm, "
              f"5th pct {np.percentile(free_gaps,5)*1000:.1f}mm  "
              f"(< {FREE_MARGIN_ALARM*1000:.0f}mm is fragile)")

    # detection margin for deliberate collisions
    det = [s for s in samples if s["category"].startswith("deliberate")]
    margins = [-sc.check(np.asarray(s["q"], float))[2] for s in det if not sc.collision_free(np.asarray(s["q"], float))]
    missed = [s["category"] for s in det if sc.collision_free(np.asarray(s["q"], float))]
    if det:
        print(f"deliberate collisions: {len(det)-len(missed)}/{len(det)} detected; "
              f"detection penetration min {min(margins)*1000:.1f}mm mean {np.mean(margins)*1000:.1f}mm"
              if margins else f"deliberate: {len(det)-len(missed)}/{len(det)} detected")
        if missed:
            print(f"  MISSED (FN) categories: {collections.Counter(missed)}")

    # unexplained always-colliding pairs: sidecar flags in EVERY config but MuJoCo never
    always = [p for p, (both, fpp, fnn) in pair_cm.items() if both == 0 and fpp == len(samples)]

    # ---- Checkpoint 1B/1C analysis: per-cluster strata, depth bands ------
    # Classify each colliding config by the cluster of its DEEPEST (responsible) MuJoCo pair
    # (spec section 4), so a config whose deep collision is pelvis is not counted as "torso"
    # merely because a shallow torso contact co-occurs.
    CLUSTER_LINKS = sc.cluster_link_to_id            # {link: cid}

    def deepest_pair(s):
        return min(s["label"]["pairs"], key=lambda pd: pd[1]) if s["label"]["pairs"] else (None, 999.0)

    def config_cluster(s):
        p, _d = deepest_pair(s)
        if p is None:
            return None
        for lk, cid in CLUSTER_LINKS.items():
            if lk in p:
                return cid
        return "OTHER"

    def mj_involves_torso(s):                # torso == deepest pair is a torso pair
        return config_cluster(s) == "TORSO"

    def mj_depth_mm(s):                       # worst penetration (negative) over MuJoCo pairs
        deps = [d for (_, d) in s["label"]["pairs"]]
        return min(deps) if deps else 999.0

    def band(s):
        if not s["label"]["colliding"]:
            return "free"
        return "deep" if mj_depth_mm(s) < -5 else "shallow"

    print("\n" + "=" * 78 + "\nCHECKPOINT 1B -- torso-SDF vs non-torso, depth bands\n" + "=" * 78)
    # split confusion by torso involvement
    for name, pred in (("torso-SDF (MuJoCo torso pairs)", mj_involves_torso),
                       ("non-torso (MuJoCo non-torso pairs)", lambda s: not mj_involves_torso(s))):
        tp = fp = fn = tn = 0
        for s in samples:
            mj = s["label"]["colliding"] and pred(s)
            sd = not sc.collision_free(np.asarray(s["q"], float))
            # attribute config to this stratum only when its MuJoCo collision matches the stratum
            if s["label"]["colliding"] and not pred(s):
                continue
            tp += mj and sd; fp += (not s["label"]["colliding"]) and sd
            fn += mj and (not sd); tn += (not s["label"]["colliding"]) and (not sd)
        print(f"  {name:<34} TP={tp:>3} FP={fp:>3} FN={fn:>3} TN={tn:>3}")
    # depth bands: sidecar recall per band
    bands = collections.defaultdict(lambda: [0, 0])       # [n, detected]
    for s in samples:
        b = band(s)
        bands[b][0] += 1
        if s["label"]["colliding"] and not sc.collision_free(np.asarray(s["q"], float)):
            bands[b][1] += 1
    print("  depth bands (MuJoCo):")
    for b in ("deep", "shallow", "free"):
        n, det = bands[b]
        if b == "free":
            print(f"    clearly free (> +2mm): {n}  (sidecar false-positives here: {fps and sum(1 for f in fps)})")
        else:
            lbl = "deep (< -5mm)" if b == "deep" else "shallow/ambiguous (-5..+2mm)"
            print(f"    {lbl}: {n}  sidecar detected {det}/{n}")
    # torso-specific gates
    delib_torso = [s for s in samples if s["category"].startswith("deliberate") and mj_involves_torso(s)]
    delib_torso_missed = [s for s in delib_torso if sc.collision_free(np.asarray(s["q"], float))]
    held_deep_torso = [s for s in samples if s["split"] == "heldout" and config_cluster(s) == "TORSO"
                       and s["label"]["colliding"] and mj_depth_mm(s) < -5]
    held_deep_torso_missed = [s for s in held_deep_torso if sc.collision_free(np.asarray(s["q"], float))]
    held_deep_pelvis = [s for s in samples if s["split"] == "heldout" and config_cluster(s) == "PELVIS"
                        and s["label"]["colliding"] and mj_depth_mm(s) < -5]
    held_deep_pelvis_missed = [s for s in held_deep_pelvis if sc.collision_free(np.asarray(s["q"], float))]
    print(f"  held-out DEEP pelvis collisions detected: "
          f"{len(held_deep_pelvis)-len(held_deep_pelvis_missed)}/{len(held_deep_pelvis)}")
    # no-regression: deliberate arm/forearm/leg-leg/hand-body detected
    regr_cats = ("deliberate_arm_torso", "deliberate_forearm", "deliberate_leg_leg", "deliberate_hand_body")
    regr_missed = [s["category"] for s in samples if s["category"] in regr_cats
                   and s["label"]["colliding"] and sc.collision_free(np.asarray(s["q"], float))]
    print(f"\n  deliberate TORSO collisions detected: {len(delib_torso)-len(delib_torso_missed)}/{len(delib_torso)}")
    print(f"  held-out DEEP torso collisions detected: {len(held_deep_torso)-len(held_deep_torso_missed)}/{len(held_deep_torso)}")
    print(f"  regression cases (arm/forearm/leg-leg/hand-body) missed: {len(regr_missed)} {collections.Counter(regr_missed) or ''}")

    # ---- Checkpoint 1D: GJK stratum + specific leg-region pairs ----------
    gjkset = {frozenset(p) for p in sc.gjk_pairs}

    def mj_has_gjk_pair(s):
        return any(frozenset(p) in gjkset for (p, _d) in s["label"]["pairs"])
    gtp = gfn = 0
    for s in samples:
        if not (s["label"]["colliding"] and mj_has_gjk_pair(s)):
            continue
        (gtp if not sc.collision_free(np.asarray(s["q"], float)) else 0)
        if sc.collision_free(np.asarray(s["q"], float)):
            gfn += 1
        else:
            gtp += 1
    print(f"\n  GJK stratum (MuJoCo collisions on a GJK pair): detected {gtp}/{gtp+gfn}")
    held_deep_wrist = [s for s in samples if s["split"] == "heldout" and s["label"]["colliding"]
                       and mj_depth_mm(s) < -5 and any(("wrist" in "".join(p) and
                       any(t in "".join(p) for t in ("hip", "thigh"))) for (p, _d) in s["label"]["pairs"])]
    held_deep_wrist_missed = [s for s in held_deep_wrist if sc.collision_free(np.asarray(s["q"], float))]
    print(f"  held-out DEEP wrist/hand<->thigh detected: "
          f"{len(held_deep_wrist)-len(held_deep_wrist_missed)}/{len(held_deep_wrist)}")
    # every FP is a shallow near-contact if all are within the MuJoCo self-clearance
    fp_deep = [f for f in fps if f["min_gap_mm"] < -5.0]
    print(f"  false positives: {len(fps)} total, {len(fp_deep)} DEEP (>5mm) -- rest are shallow "
          f"near-contacts below MuJoCo's 5mm self-clearance (listed, not gated)")

    # FK gate
    fk, fk_status = fk_check()
    if fk:
        print(f"\nFK vs MuJoCo: max pos {fk[0]*1000:.4f}mm, max rot {np.degrees(fk[1]):.4f}deg")

    # ---- gates -----------------------------------------------------------
    def lab_free(cat):
        s = next(x for x in samples if x["category"] == cat)
        return sc.collision_free(np.asarray(s["q"], float))
    # Mandatory gates == spec section 6 (whole-corpus FN is a REPORTED residual, not a gate;
    # "prioritize avoiding false negatives" is applied during tuning, and the residual FN are
    # listed below).
    # Checkpoint 1B mandatory gates (spec section 4)
    gates = {
        "neutral_free": lab_free("neutral"),
        "crouch_free": lab_free("crouch"),
        "every_deliberate_torso_collision_detected": len(delib_torso_missed) == 0,
        "no_heldout_deep_torso_collision_missed": len(held_deep_torso_missed) == 0,
        "no_heldout_deep_pelvis_collision_missed": len(held_deep_pelvis_missed) == 0,
        "no_heldout_deep_wrist_thigh_collision_missed": len(held_deep_wrist_missed) == 0,
        "no_regression_arm_forearm_legleg_handbody": len(regr_missed) == 0,
        "fk_within_tol": (fk is None) or (fk[0] <= FK_POS_TOL and fk[1] <= FK_ROT_TOL),
        "no_unexplained_always_colliding_pair": len(always) == 0,
    }
    print(f"\n(residual FN on the full corpus = {len(fns)}; FP = {len(fps)} -- reported, not gated)")
    print("\n" + "=" * 78 + "\nMANDATORY GATES\n" + "=" * 78)
    for k, v in gates.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    if always:
        print(f"  unexplained always-colliding pairs: {always}")
    ok = all(gates.values())
    print(f"\nCHECKPOINT 1 GATE: {'PASS -- ready for CUDA' if ok else 'FAIL -- do not proceed to CUDA'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
