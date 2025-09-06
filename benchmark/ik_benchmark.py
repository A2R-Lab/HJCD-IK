import os, subprocess, yaml
from pathlib import Path

SEEDS_SPEC = os.environ.get("HJCD_SEEDS", "1,10,100,1000,2000,10000")

ROOT   = Path(__file__).resolve().parents[1]
BUILD  = ROOT / "build"
EXE    = BUILD / ("Release/hjcdik.exe" if os.name == "nt" else "hjcdik")
OUTDIR = ROOT / "benchmark" / "outputs"
OUTDIR.mkdir(parents=True, exist_ok=True)

print(f"[run] {EXE} {SEEDS_SPEC}")
subprocess.run([str(EXE), SEEDS_SPEC, "0"], cwd=OUTDIR, check=True)

yml_path = OUTDIR / "hjcd_ik_per_batch.yml"
data = yaml.safe_load(yml_path.read_text())

def mean(xs): return sum(xs)/len(xs) if xs else float("nan")

times = data.get("IK-time(ms)", [])
batches = data.get("Batch-Size", [])
pos_errs = data.get("Pos-Error", [])
ori_errs = data.get("Ori-Error", [])

print("=== HJCD-IK summary ===")
print("rows          :", len(times))
print("unique batches:", sorted(set(int(b) for b in batches)))

from collections import defaultdict
by_batch = defaultdict(lambda: {"t": [], "pe": [], "oe": []})
for b, t, pe, oe in zip(batches, times, pos_errs, ori_errs):
    b = int(b)
    by_batch[b]["t"].append(t)
    by_batch[b]["pe"].append(pe)
    by_batch[b]["oe"].append(oe)

print("\nPer-batch means:")
for b in sorted(by_batch):
    print(f"  {b:>6} : time(ms)={mean(by_batch[b]['t']):.3f}  "
          f"pos={mean(by_batch[b]['pe']):.6g}  ori={mean(by_batch[b]['oe']):.6g}")

print(f"\nYAML at: {yml_path}")
