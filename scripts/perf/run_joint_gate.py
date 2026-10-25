#!/usr/bin/env python3
"""Run all precompiled P2b timing legs sequentially; requires explicit quiet-window approval."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from run_audit_gate import run_worker


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    modes=p.add_mutually_exclusive_group(required=True)
    modes.add_argument("--quiet-window",action="store_true")
    modes.add_argument("--correctness-only",action="store_true")
    modes.add_argument("--check",action="store_true")
    a=p.parse_args()
    mode="--check" if a.check else "--correctness-only" if a.correctness_only else "--quiet-window"
    if not a.check and a.out.exists(): raise SystemExit("choose a new output directory; partial runs are preserved")
    for name in ("api","native","runtime","highdof"):
        subprocess.run([sys.executable,str(Path(__file__).with_name("run_audit_gate.py")),
                        "--config",str(a.stage/(name+".json")),"--out",str(a.out/name),mode],check=True)
    # Cold compilation is intentionally after every precompiled GPU leg. The user-announced
    # window must also have a quiet CPU; foreign GPU checks alone cannot establish that.
    if a.quiet_window:
        config=json.loads((a.stage/"api.json").read_text())
        parent=a.out.resolve()/"compile"
        parent.mkdir()
        env={k:v for k,v in os.environ.items() if not k.startswith(("HJCD_","HJCDIK_")) and k!="PYTHONPATH"}
        env.update(OMP_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",PYTHONSAFEPATH="1")
        for repeat in (1,2):
            for label in (("main","branch") if repeat==1 else ("branch","main")):
                endpoint=config["endpoints"][label]
                cell=f"{repeat}-{label}"
                cmd=[endpoint["python"],str(Path(__file__).with_name("compile_cost_driver.py")),
                     "--quiet-window","--out",str(parent/cell)]
                run_worker(cmd,endpoint["repo"],{**env,**endpoint.get("env",{})},parent/(cell+".log"),True)


if __name__=="__main__": main()
