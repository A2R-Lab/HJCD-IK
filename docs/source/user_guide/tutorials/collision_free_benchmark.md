# Collision-free benchmark

HJCD-IK ships MotionBenchMaker-style problem sets under `tests/`:
- `tests/mb_problems.json` — collision-free scenes (e.g. bookshelf variants).
- `tests/wall_problems.json` — a wall scenario with cuboid/cylinder obstacles.

## Run
```bash
python benchmark/hjcd_ik_bench.py \
    --collision-free \
    --problems-json tests/mb_problems.json \
    --batches "1,10,100,1000"
```

The harness reports, per batch size: **solved-rate**, mean **position error**, mean **orientation error**, and
**timing**. These are the metrics the regression tests assert against a committed baseline — see
`tests/test_regression.py`. When comparing runs, isolate timing (no concurrent GPU load).
