import sys, time, os
from pathlib import Path
import argparse
from collections import defaultdict 
import math

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external"))

def run_grid_codegen(urdf: Path, skip: bool) -> bool:
    if skip:
        print("[GRiD] skipping URDF parse/codegen")
        return False

    from GRiD.URDFParser import URDFParser
    from GRiD.GRiDCodeGenerator import GRiDCodeGenerator

    urdf = urdf if urdf.is_absolute() else (ROOT / urdf).resolve()
    print(f"[GRiD] parsing {urdf}")
    robot = URDFParser().parse(str(urdf))
    codegen = GRiDCodeGenerator(robot, False, True)

    out_dir = ROOT / "external" / "GRiD"
    out_dir.mkdir(parents=True, exist_ok=True)
    cwd = os.getcwd()
    os.chdir(out_dir)
    try:
        codegen.gen_all_code(include_homogenous_transforms=True)
    finally:
        os.chdir(cwd)
    print("[GRiD] codegen done")
    return True

# Rebuild extension so changes reflect grid.cuh
def rebuild_against_current_header():
    import subprocess
    env = os.environ.copy()
    
    env.pop("CMAKE_ARGS", None)
    print("[build] Rebuilding _hjcdik against external/GRiD/grid.cuh")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-e", "."], cwd=ROOT, env=env)

def write_yaml_flat(path: Path, batch_sizes, time_ms, pos_err, ori_err):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as y:
        y.write("Batch-Size:\n")
        for v in batch_sizes:
            y.write(f"  - {v}\n")
        y.write("IK-time(ms):\n")
        for v in time_ms:
            y.write(f"  - {v:.9f}\n")
        y.write("Pos-Error:\n")
        for v in pos_err:
            y.write(f"  - {v:.17g}\n")
        y.write("Ori-Error:\n")
        for v in ori_err:
            y.write(f"  - {v:.17g}\n")

def _parse_batches(s: str):
    parts = [p.strip() for p in s.replace(",", " ").split()]
    vals = [int(p) for p in parts if p]
    if not vals:
        raise argparse.ArgumentTypeError("batches list is empty")
    return vals

# Print summary
def print_batch_summary(batches, y_batch, y_time_ms, y_pos, y_ori):
    g_time = defaultdict(list)
    g_pos = defaultdict(list)
    g_ori = defaultdict(list)

    for B, t, p, o in zip(y_batch, y_time_ms, y_pos, y_ori):
        g_time[B].append(t)
        g_pos[B].append(p)
        g_ori[B].append(o)

    print("\n==== Batch Summary (averages) ====")
    for B in sorted(g_time.keys()):
        print(f"Batch Size {B}:")
        print(f"  Time (ms): {sum(g_time[B]) / len(g_time[B]):.6f}")
        print(f"  Position Error: {sum(g_pos[B]) / len(g_pos[B]):12.6e}")
        print(f"  Orientation Error: {sum(g_ori[B]) / len(g_ori[B]):12.6e}")

def main():
    ap = argparse.ArgumentParser(description="HJCD-IK benchmark")
    ap.add_argument("--num-targets", type=int, default=100)
    ap.add_argument("--batches", type=_parse_batches, default=_parse_batches("1,10,100,1000,2000"))
    ap.add_argument("--num-solutions", type=int, default=1)
    ap.add_argument("--yaml-out", type=str, default="results.yml")
    ap.add_argument("--urdf", type=str, default=str(ROOT / "include" / "test_urdf" / "panda.urdf"))
    ap.add_argument("--skip-grid-codegen", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # Generate grid.cuh
    did_codegen = run_grid_codegen(Path(args.urdf), args.skip_grid_codegen)

    # Rebuild if needed
    if did_codegen:
        rebuild_against_current_header()
    else:
        print("[build] skipping rebuild because --skip-grid-codegen was set")

    import importlib
    hjcdik = importlib.import_module("hjcdik")

    try:
        print("[build info]", hjcdik.build_info())
    except Exception:
        pass

    # Sample Targets
    N = hjcdik.num_joints()
    print(f"[info] robot with {N} joints")
    T, batches, S = int(args.num_targets), list(args.batches), int(args.num_solutions)
    targets = hjcdik.sample_targets(T, seed=args.seed)

    # Warm-up
    warmup_target = hjcdik.sample_targets(1, seed=args.seed + 12345)[0]
    _ = hjcdik.generate_solutions(warmup_target, batch_size=max(batches), num_solutions=1)

    y_batch, y_time_ms, y_pos, y_ori = [], [], [], []

    # Benchmark
    print(f"[info] running {T} targets, batches={batches}, num_solutions={S}")
    
    for i, target in enumerate(targets, 1):
        for B in batches:
            t0 = time.perf_counter()
            res = hjcdik.generate_solutions(target, batch_size=B, num_solutions=S)
            dt_ms = (time.perf_counter() - t0) * 1e3
            per_sample_ms = dt_ms / max(1, S)

            pos_err = res["pos_errors"]
            ori_err = res["ori_errors"]
            for r in range(S):
                y_batch.append(B)
                y_time_ms.append(per_sample_ms)
                y_pos.append(float(pos_err[r]))
                y_ori.append(float(ori_err[r]))

        if (i % 50) == 0 or i == T:
            print(f"[info] processed {i}/{T} targets")

    print_batch_summary(batches, y_batch, y_time_ms, y_pos, y_ori)
    out_path = (Path(args.yaml_out) if Path(args.yaml_out).is_absolute()
                else (ROOT / args.yaml_out).resolve())
    write_yaml_flat(out_path, y_batch, y_time_ms, y_pos, y_ori)
    print(f"\n[OK] wrote {out_path} with {T*S*len(batches)} entries "
          f"({T} targets × {len(batches)} batches × {S} solutions each).")


if __name__ == "__main__":
    main()
