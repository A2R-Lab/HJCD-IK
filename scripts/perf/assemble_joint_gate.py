#!/usr/bin/env python3
"""Freeze workload hashes and matched API/native/runtime/24-DoF P2b gate configs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"benchmark"))
from gen_targets import _parse_joints,_chain_to_target,_actuated_limits,_fk,_quat_from_R


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",type=Path,required=True)
    p.add_argument("--targets",type=Path,required=True)
    a=p.parse_args(); stage=a.stage.resolve(); targets=a.targets.resolve()
    endpoints={label:json.loads((stage/(label+"-endpoint.json")).read_text()) for label in ("main","branch")}
    urdf=ROOT/"csrc/urdf/panda_ext_24dof.urdf"
    chain=_chain_to_target(_parse_joints(urdf),"panda_hand_joint")
    lo,hi=_actuated_limits(chain)
    qs=np.random.default_rng(20261025).uniform(lo,hi,(100,len(lo)))
    high_targets={"targets":[np.r_[t[:3,3],_quat_from_R(t[:3,:3])].tolist() for t in (_fk(chain,q) for q in qs)]}
    high_path=stage/"highdof-targets.json"
    if high_path.exists() or any((stage/(n+".json")).exists() for n in ("api","native","runtime","highdof")):
        raise SystemExit("configs already exist; preserve them and stage in a new directory")
    high_path.write_text(json.dumps(high_targets,indent=2)+"\n")
    for name in ("api","native","runtime","highdof"):
        kind="highdof" if name=="highdof" else "runtime" if name=="runtime" else "packaged"
        eps={label:copy.deepcopy(entries[kind]) for label,entries in endpoints.items()}
        for ep in eps.values():
            ep["env"]["HJCDIK_TIMING_ENTRY"]="native" if name=="native" else "api"
            if name=="highdof": ep["env"]["HJCDIK_TIMING_URDF"]=str(urdf)
        selected=high_path if name=="highdof" else targets
        problems=ROOT/"tests/mb_problems.json"
        batches={"table1":[1,2000,8000,32000]}
        if name!="highdof": batches["table2"]=[100,2000]
        config=dict(rounds=4,solutions=[1,4],batches=batches,target_counts={leg:100 for leg in batches},
                    targets_json=str(selected),targets_json_sha=digest(selected),
                    problems_json=str(problems),problems_json_sha=digest(problems),
                    problem_set="box_panda",endpoints=eps)
        if name=="highdof": config.update(reference_urdf=str(urdf),reference_urdf_sha=digest(urdf))
        (stage/(name+".json")).write_text(json.dumps(config,indent=2)+"\n")
    print(f"Frozen four gate configurations in {stage}")


if __name__=="__main__": main()
