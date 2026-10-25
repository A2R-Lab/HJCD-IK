"""Native-binding A/B: time the extension entry point directly (no Python Solver layer) on either API."""
import json, statistics, sys, time
import numpy as np
import hjcdik, hjcdik._hjcdik as m
label, rnd = sys.argv[1], int(sys.argv[2])
targets = json.load(open(sys.argv[3]))["targets"]
targets = [t if isinstance(t, list) else t["target"] for t in targets][:100]
new = hasattr(m, "solve_one")
def solve(t, B, S):
    if new:
        return m.solve_one(tuple(map(float, t)), B, S, 0, S == 1, False, None, 0, 0.0, True, 1337, None)
    return m.generate_solutions(t, batch_size=B, num_solutions=S)
out = []
for S in (1, 4):
    for B in (1, 2000, 32000):
        for t in targets[:5]: solve(t, B, S)
        times = []
        for t in targets:
            t0 = time.perf_counter(); r = solve(t, B, S); times.append((time.perf_counter() - t0) * 1e3)
            assert r["count"] >= 0
        out.append(dict(label=label, round=rnd, S=S, B=B, median_ms=statistics.median(times), mean_ms=statistics.fmean(times)))
print(json.dumps(out))
