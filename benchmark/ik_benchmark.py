import os, sys, subprocess, yaml
from pathlib import Path

ROOT   = Path(__file__).resolve().parents[1]
BUILD  = ROOT / "build"
EXE    = BUILD / ("Release/globeik.exe" if os.name == "nt" else "globeik")
OUTDIR = ROOT / "benchmark" / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"[run] {EXE}")
subprocess.run([str(EXE)], cwd=OUTDIR, check=True)

yml_path = OUTDIR / "globeik_ik_per_batch.yml"
data = yaml.safe_load(yml_path.read_text())

def mean(xs): return sum(xs)/len(xs) if xs else float("nan")

print("=== GLOBE_IK summary ===")
print("samples       :", len(data.get("IK-time(ms)", [])))
print("mean time (ms):", mean(data.get("IK-time(ms)", [])))
print("mean pos err  :", mean(data.get("Pos-Error", [])))
print("mean ori err  :", mean(data.get("Ori-Error", [])))
print(f"\nYAML at: {yml_path}")
