#!/usr/bin/env python3
"""Capture HJCD-IK baseline metrics for regression testing.

Run this on the **current `main`** (pre-integration) build, then commit the output to
`tests/baseline_metrics.json`. `tests/test_regression.py` asserts the integration branch does not regress
against these recorded numbers (with a small slack band).

Requires a GPU + a built `hjcdik` extension.
"""
import json
import math
from pathlib import Path

import numpy as np
import hjcdik

OUT = Path(__file__).resolve().parent.parent / "tests" / "baseline_metrics.json"


def run_suite(num_targets=64, seed=0, batch_size=2000, num_solutions=1):
    targets = hjcdik.sample_targets(num_targets=num_targets, seed=seed)
    solved, pos_errs, ori_errs = 0, [], []
    for t in targets:
        out = hjcdik.generate_solutions(t, batch_size=batch_size, num_solutions=num_solutions)
        if out["count"] > 0:
            solved += 1
            pos_errs.append(float(np.min(out["pos_errors"])))
            ori_errs.append(float(np.min(out["ori_errors"])))
    return {
        "num_targets": num_targets,
        "batch_size": batch_size,
        "solved_rate": solved / num_targets,
        "mean_pos_err": float(np.mean(pos_errs)) if pos_errs else math.inf,
        "mean_ori_err": float(np.mean(ori_errs)) if ori_errs else math.inf,
    }


def main():
    data = {"sampled_unconstrained": run_suite(num_targets=64, seed=0)}
    OUT.write_text(json.dumps(data, indent=2) + "\n")
    print(f"wrote {OUT}:")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
