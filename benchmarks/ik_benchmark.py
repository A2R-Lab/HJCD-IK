# benchmarks/ik_benchmark.py
import sys, time, os
from pathlib import Path
import argparse

# Make local GRiD importable (GRiD/__init__.py lives in external/GRiD)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "external"))

import hjcdik


def write_yaml_flat(path: Path, batch_sizes, time_ms, pos_err, ori_err):
    """Emit the same flat YAML lists as main.cpp."""
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


def maybe_run_grid_codegen(urdf: Path, skip: bool):
    if skip:
        print("[GRiD] skipping URDF parse/codegen")
        return
    from GRiD.URDFParser import URDFParser
    from GRiD.GRiDCodeGenerator import GRiDCodeGenerator

    if not urdf.is_absolute():
        urdf = (ROOT / urdf).resolve()
    print(f"[GRiD] parsing {urdf}")
    robot = URDFParser().parse(str(urdf))
    codegen = GRiDCodeGenerator(robot, False, True)

    out_dir = ROOT / "external" / "GRiD"   # .../hjcdik/external/GRiD
    out_dir.mkdir(parents=True, exist_ok=True)

    cwd = os.getcwd()
    os.chdir(out_dir)
    try:
        codegen.gen_all_code(include_homogenous_transforms=True)
    finally:
        os.chdir(cwd)
    print("[GRiD] codegen done")


def _parse_batches(s: str):
    # accepts "1,10,1000" or "1 10 1000"
    parts = [p.strip() for p in s.replace(",", " ").split()]
    vals = [int(p) for p in parts if p]
    if not vals:
        raise argparse.ArgumentTypeError("batches list is empty")
    return vals


def main():
    ap = argparse.ArgumentParser(description="HJCD-IK benchmark (YAML output like main.cpp)")
    ap.add_argument("--num-targets", type=int, default=100, help="How many target poses to sample")
    ap.add_argument("--batches", type=_parse_batches, default=_parse_batches("1,10,100,1000,2000"),
                    help='Batch sizes, e.g. "1,10,100,1000,2000"')
    ap.add_argument("--num-solutions", type=int, default=1, help="Solutions to return per call")
    ap.add_argument("--yaml-out", type=str, default="results.yml", help="Output YAML file")
    ap.add_argument("--urdf", type=str, default=str(ROOT / "include" / "test_urdf" / "panda.urdf"),
                    help="URDF path for GRiD codegen")
    ap.add_argument("--skip-grid-codegen", action="store_true",
                    help="Skip GRiD URDF parse/codegen step")
    ap.add_argument("--seed", type=int, default=0, help="Seed for target sampling")
    args = ap.parse_args()

    # Optional GRiD step
    maybe_run_grid_codegen(Path(args.urdf), args.skip_grid_codegen)

    N = hjcdik.num_joints()
    print(f"[info] robot with {N} joints")
    T, batches, S = int(args.num_targets), list(args.batches), int(args.num_solutions)
    targets = hjcdik.sample_targets(T, seed=args.seed)

    # Accumulators like main.cpp
    y_batch = []        # list[int]
    y_time_ms = []      # list[float] (per-solution time, same semantics as main.cpp)
    y_pos = []          # list[float]
    y_ori = []          # list[float]

    _ = hjcdik.generate_solutions(targets[0], batch_size=batches[-1], num_solutions=S)

    print(f"[info] running {T} targets, batches={batches}, num_solutions={S}")
    for i, target in enumerate(targets, 1):
        for B in batches:
            t0 = time.perf_counter()
            res = hjcdik.generate_solutions(target, batch_size=B, num_solutions=S)
            dt_ms = (time.perf_counter() - t0) * 1e3

            # Per-solution accounting (main.cpp divides total call time across S)
            per_sample_ms = dt_ms / max(1, S)
            pos_err = res["pos_errors"]   # shape (S,)
            ori_err = res["ori_errors"]   # shape (S,)

            for r in range(S):
                y_batch.append(B)
                y_time_ms.append(per_sample_ms)
                y_pos.append(float(pos_err[r]))
                y_ori.append(float(ori_err[r]))

        if (i % 50) == 0 or i == T:
            print(f"[info] processed {i}/{T} targets")

    out_path = (Path(args.yaml_out) if Path(args.yaml_out).is_absolute()
                else (ROOT / args.yaml_out).resolve())
    write_yaml_flat(out_path, y_batch, y_time_ms, y_pos, y_ori)
    print(f"[OK] wrote {out_path} with {T*S*len(batches)} entries "
          f"({T} targets × {len(batches)} batches × {S} solutions each).")


if __name__ == "__main__":
    main()
