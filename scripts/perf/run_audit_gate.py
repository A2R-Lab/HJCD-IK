#!/usr/bin/env python3
"""Fail-closed, precompiled Panda A/B gate. See docs/development/timing_gate.md."""
import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

DRIVER = Path(__file__).with_name("timing_driver.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def foreign_gpu_pids(allowed=()):
    text = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                    "--format=csv,noheader,nounits"], text=True)
    return {int(line.strip()) for line in text.splitlines() if line.strip()} - set(allowed)


def cases(config):
    return list(itertools.product(range(1, config["rounds"] + 1), config["solutions"],
                                  config["batches"], config["endpoints"]))


def analyze(config, rows, correctness_only=False):
    expected = {(f"{ep}-S{s}", leg, int(b), r)
                for r, s, leg, ep in cases(config) for b in config["batches"][leg]}
    actual = {}
    for row in rows:
        key = row["label"], row["leg"], int(row["batch"]), int(row["round"])
        if key in actual:
            raise ValueError(f"duplicate timing cell: {key}")
        actual[key] = row
        ep = row["label"].rsplit("-S", 1)[0]
        if row["header_sha"] != config["endpoints"][ep]["header_sha"]:
            raise ValueError("mixed compiled headers")
        if row["correctness_only"] != str(correctness_only) or int(row["refine_fp64"]) != -1:
            raise ValueError("mixed precision or timing protocols")
        n = int(row["n_targets"])
        if n != config["target_counts"][row["leg"]]:
            raise ValueError("wrong query denominator")
        if not 0 <= int(row["solved"]) <= n or not 0 <= int(row["empty"]) <= n:
            raise ValueError("invalid quality counts")
    if set(actual) != expected:
        raise ValueError(f"incomplete gate: missing {expected-set(actual)}, extra {set(actual)-expected}")
    findings = []
    for s, leg in itertools.product(config["solutions"], config["batches"]):
        for b in config["batches"][leg]:
            base = [actual[(f"main-S{s}", leg, b, r)] for r in range(1, config["rounds"]+1)]
            for ep in config["endpoints"]:
                if ep == "main":
                    continue
                test = [actual[(f"{ep}-S{s}", leg, b, r)] for r in range(1, config["rounds"]+1)]
                # Report stochastic quality differences for review, never silently excuse them.
                delta = sum(int(t["solved"])-int(a["solved"]) for a, t in zip(base, test))
                item = dict(endpoint=ep, solutions=s, leg=leg, batch=b, solved_delta=delta,
                            quality_review_required=delta < 0)
                if not correctness_only:
                    ratios = [float(t["median_ms"])/float(a["median_ms"]) for a, t in zip(base, test)]
                    if not all(0 < v < float("inf") for v in ratios):
                        raise ValueError("invalid timing measurement")
                    item.update(paired_ratio=statistics.median(ratios), ratio_min=min(ratios), ratio_max=max(ratios))
                findings.append(item)
    return findings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="New directory; never overwrite an earlier run")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--quiet-window", action="store_true")
    mode.add_argument("--correctness-only", action="store_true")
    mode.add_argument("--check", action="store_true", help="Verify endpoint/input provenance only; no GPU work or output directory")
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    for key in ("targets_json", "problems_json"):
        if digest(config[key]) != config[key+"_sha"]:
            raise ValueError(f"changed workload: {key}")
    if "main" not in config["endpoints"]:
        raise ValueError("main comparison endpoint required")
    # Read-only preflight: provenance includes the installed binary, not just the checkout.
    provenance = {}
    env = {k: v for k, v in os.environ.items() if not k.startswith("HJCD_") and k != "PYTHONPATH"}
    env.update(PYTHONSAFEPATH="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    for label, ep in config["endpoints"].items():
        sha = subprocess.check_output(["git", "-C", ep["repo"], "rev-parse", "HEAD"], text=True).strip()
        if sha != ep["commit"]:
            raise ValueError(f"changed checkout: {label}")
        info = json.loads(subprocess.check_output([ep["python"], "-c",
            "import json,sys,numpy,hjcdik,hjcdik._hjcdik as m; print(json.dumps(dict(build=hjcdik.build_info(),binary=m.__file__,python=sys.version,numpy=numpy.__version__)))"],
            cwd="/tmp", env=env, text=True))
        if info["build"]["grid_header_sha256"] != ep["header_sha"] or digest(info["binary"]) != ep["binary_sha"]:
            raise ValueError(f"changed installed build: {label}")
        if info["build"].get("ee_target", "panda_hand_joint") != "panda_hand_joint":
            raise ValueError("gate requires hand-frame builds")
        provenance[label] = info
    if len({(p["python"], p["numpy"]) for p in provenance.values()}) != 1:
        raise ValueError("endpoint Python/NumPy runtimes differ")
    if args.check:
        print("Preflight passed: endpoint commits, installed builds, runtimes and inputs match. No GPU work started.")
        return
    if args.quiet_window and foreign_gpu_pids():
        raise RuntimeError("GPU is occupied; no timing started")
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "manifest.json").write_text(json.dumps(dict(config=config, provenance=provenance,
        driver_sha=digest(DRIVER), correctness_only=args.correctness_only,
        helpers={name: digest(DRIVER.parents[2] / "benchmark" / name) for name in
                 ("query_results.py", "hjcd_ik_bench.py", "panda_collision.py", "panda_model.py", "collision_check.py", "gen_targets.py")}), indent=2)+"\n")
    output = args.out.resolve() / "results.csv"
    endpoints = list(config["endpoints"])
    for r in range(1, config["rounds"]+1):
        order = endpoints[(r-1) % len(endpoints):] + endpoints[:(r-1) % len(endpoints)]
        for s, ep_name, leg in itertools.product(config["solutions"], order, config["batches"]):
            ep = config["endpoints"][ep_name]
            cmd = [ep["python"], str(DRIVER), "--label", f"{ep_name}-S{s}", "--leg", leg,
                   "--round", str(r), "--targets-json", config["targets_json"],
                   "--problems-json", config["problems_json"], "--problem-set", config["problem_set"],
                   "--num-solutions", str(s), "--batches", ",".join(map(str, config["batches"][leg])),
                   "--expected-header-sha", ep["header_sha"], "--out", str(output)]
            if args.correctness_only:
                cmd.append("--correctness-only")
            if args.quiet_window and foreign_gpu_pids():
                raise RuntimeError("foreign GPU process appeared; partial run is invalid")
            print(f"round {r}: {ep_name} S={s} {leg}", flush=True)
            with (args.out / f"{r}-{ep_name}-S{s}-{leg}.log").open("w") as log:
                worker = subprocess.Popen(cmd, cwd="/tmp", env={**env, **ep.get("env", {})}, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while True:
                        try:
                            code = worker.wait(timeout=2)
                            break
                        except subprocess.TimeoutExpired:
                            if args.quiet_window and foreign_gpu_pids({worker.pid}):
                                raise RuntimeError("GPU contention detected; partial run is invalid")
                    if code:
                        raise subprocess.CalledProcessError(code, cmd)
                finally:
                    if worker.poll() is None:
                        worker.terminate()  # only this launcher's child, never another agent's process
                        try:
                            worker.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            worker.kill()
                            worker.wait()
    with output.open(newline="") as stream:
        findings = analyze(config, list(csv.DictReader(stream)), args.correctness_only)
    (args.out / "analysis.json").write_text(json.dumps(findings, indent=2)+"\n")
    print("Complete; inspect analysis.json for quality regressions and paired latency ratios. No automatic performance approval.")


if __name__ == "__main__":
    main()
