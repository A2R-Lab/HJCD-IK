#!/usr/bin/env python3
"""Quiet-CPU-window cold runtime build and cache-hit costs; not packaged-wheel build costs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import hjcdik


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--quiet-window",action="store_true",required=True)
    a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False)
    for robot,filename in (("panda","panda.urdf"),("24dof","panda_ext_24dof.urdf")):
        source=Path.cwd()/"csrc/urdf"/filename
        cache=a.out/robot
        t=time.perf_counter()
        cold=hjcdik.build(source,ee_link="panda_hand_joint",name="compile-cost-"+robot,cache_dir=cache)
        cold_s=time.perf_counter()-t
        t=time.perf_counter()
        hit=hjcdik.build(source,ee_link="panda_hand_joint",name="compile-cost-"+robot,cache_dir=cache)
        hit_s=time.perf_counter()-t
        if not cold.compiled or hit.compiled or cold.content_key!=hit.content_key:
            raise RuntimeError("cold/cache-hit protocol was not satisfied")
        data=dict(robot=robot,cold_seconds=cold_s,cache_hit_seconds=hit_s,
                  source_sha=hashlib.sha256(source.read_bytes()).hexdigest(),
                  content_key=cold.content_key,header_sha=cold.grid_header_sha256,
                  cuda_compiler_version=cold.cuda_compiler_version,
                  binary=str(cold.path),pid=os.getpid())
        (a.out/(robot+".json")).write_text(json.dumps(data,indent=2)+"\n")
        print(json.dumps(data),flush=True)


if __name__=="__main__": main()
