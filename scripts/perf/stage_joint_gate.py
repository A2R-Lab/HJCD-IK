#!/usr/bin/env python3
"""Prepare one isolated P2b timing endpoint; never deletes or overwrites a checkout.

Run with the development venv. This compiles and performs correctness checks, never timings.
The output endpoint.json is consumed when assembling the quiet-window gate configs.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--commit",required=True)
    p.add_argument("--label",choices=("main","branch"),required=True)
    p.add_argument("--directory",required=True,type=Path)
    a=p.parse_args()
    dest=a.directory.resolve()
    if dest.exists(): raise SystemExit("destination already exists; choose a new endpoint directory")
    revision=subprocess.check_output(["git","rev-parse",a.commit+"^{commit}"],cwd=ROOT,text=True).strip()
    env=dict(os.environ,GIT_CONFIG_COUNT="1",GIT_CONFIG_KEY_0="protocol.file.allow",GIT_CONFIG_VALUE_0="always")
    def run(args,cwd=dest,**kwargs):
        return subprocess.run(list(map(str,args)),cwd=cwd,env=env,check=True,**kwargs)
    dest.parent.mkdir(parents=True,exist_ok=True)
    run(["git","clone","--quiet",ROOT,dest],cwd=ROOT)
    run(["git","checkout","--quiet",revision])
    run(["git","submodule","init"])
    for sub in ("GRiD","GLASS","foam"):
        run(["git","config",f"submodule.external/{sub}.url",ROOT/"external"/sub])
    run(["git","submodule","update","--init"])
    nested=dest/"external/GRiD"
    run(["git","submodule","init"],cwd=nested)
    for sub in ("GLASS","URDFParser","RBDReference"):
        run(["git","config",f"submodule.external/{sub}.url",ROOT/"external/GRiD/external"/sub],cwd=nested)
    run(["git","submodule","update","--init","external/GLASS","external/URDFParser","external/RBDReference"],cwd=nested)
    run([sys.executable,"-m","venv",dest/".venv"])
    python=dest/".venv/bin/python"
    dependencies=[f"{name}=={importlib.metadata.version(name)}" for name in
                  ("numpy","sympy","beautifulsoup4","lxml","pybind11","scikit-build-core","grid-rbd")]
    run([python,"-m","pip","install",*dependencies])
    env["PATH"]=str(dest/".venv/bin")+os.pathsep+env["PATH"]
    env["GRID_RBD_CACHE_DIR"]=str(dest/".venv/robot-cache")
    run([python,"scripts/codegen/generate_grid.py","csrc/urdf/panda.urdf","-t","panda_hand_joint",
         "--collision","--spherized-urdf","external/foam/assets/panda/smaller_panda_spherized.urdf"])
    run([python,"-m","pip","install","--no-build-isolation","--no-deps","-e","."])
    headers=["csrc/generated/grid.cuh"]
    if (dest/"csrc/generated/hjcd_joint_model.h").exists(): headers.append("csrc/generated/hjcd_joint_model.h")
    run(["git","add",*headers])
    run(["git","commit","-m","Prepare hand-frame timing endpoint"])
    probe='''import hashlib,json,sys,numpy,hjcdik,hjcdik._hjcdik as native
from pathlib import Path
label=sys.argv[1]
entries={}
for kind,urdf,target,spheres in (("packaged",None,None,None),
    ("runtime","csrc/urdf/panda.urdf","panda_hand_joint","panda_foam"),
    ("highdof","csrc/urdf/panda_ext_24dof.urdf","panda_hand_joint",None)):
    name="p2b-"+label+"-"+kind
    if urdf: hjcdik.build(urdf,ee_link=target,spheres=spheres,name=name)
    solver=hjcdik.Solver(name if urdf else None,allow_build=False)
    result=solver.solve(solver.sample_targets(4,seed=17),batch_size=2000,num_solutions=4,collision=None)
    assert result.ok(5,.05).any(axis=1).all()
    binary=solver.info.artifact or native.__file__
    entries[kind]=dict(header_sha=solver.info.grid_header_sha256,binary_sha=hashlib.sha256(Path(binary).read_bytes()).hexdigest(),
                       binary=binary,robot=name if urdf else None,ee_frame=solver.info.ee_frame)
Path(".venv/endpoint-builds.json").write_text(json.dumps(entries,indent=2)+"\\n")
'''
    run([python,"-c",probe,a.label])
    head=run(["git","rev-parse","HEAD"],capture_output=True,text=True).stdout.strip()
    entries=json.loads((dest/".venv/endpoint-builds.json").read_text())
    data={}
    for kind,entry in entries.items():
        extra=dict(GRID_RBD_CACHE_DIR=env["GRID_RBD_CACHE_DIR"])
        if entry["robot"]: extra["HJCDIK_TIMING_ROBOT"]=entry["robot"]
        data[kind]=dict(repo=str(dest),python=str(python),commit=head,source_commit=revision,
                        header_sha=entry["header_sha"],binary_sha=entry["binary_sha"],env=extra)
    result=dest.parent/(a.label+"-endpoint.json")
    if result.exists(): raise SystemExit(f"refusing to overwrite endpoint manifest {result}")
    result.write_text(json.dumps(data,indent=2)+"\n")
    print(f"Precompiled and correctness-checked: {result}")


if __name__=="__main__": main()
