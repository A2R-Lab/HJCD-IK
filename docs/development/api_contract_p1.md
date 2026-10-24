# Proposed user API contract (P1, Stage 0) — not yet the installed API

Written for: the HJCD-IK maintainers and reviewers. This document is the design contract that Stage 1
implements. Until Stage 1 lands, the installed package still exposes `generate_solutions`; the API
reference, README, examples and upgrading guide describe that API and change together with the Stage 1
code. When Stage 1 lands, this file is folded into `docs/source/user_guide/upgrading.md` and the Python
API reference and is deleted here.

## Surface

```python
import hjcdik

solver = hjcdik.Solver()                  # the robot the installed extension was compiled for
solver.info                               # robot, dof, ee_frame, collision_enabled, header sha256, ...
solver.set_world(boxes=..., capsules=..., spheres=..., cylinders=...)
solver.clear_world()
targets = solver.sample_targets(10, seed=0)             # (10, 7) float64, Halton over joint limits
r = solver.solve(targets, num_solutions=4)              # Result, (T, S, ...) NaN-padded
r[0].q, r.count, r.valid, r.ok(pos_mm=1.0, ori_rad=1e-3)
```

`Solver(robot=None)` wraps the installed, statically compiled robot. `robot` may name it
(`Solver("panda")`) and must then match `hjcdik.build_info()` (robot name, DoF, end-effector frame,
collision flag) or the constructor raises. Runtime robot selection and builds are Stage 2.

`hjcdik.__all__` after Stage 1: `Solver`, `Result`, `TargetResult`, `Stats`, `SolveError`, `build_info`,
`worlds`. `generate_solutions`, `sample_targets`, `num_joints`, `collision_enabled` and `write_stats`
are removed (clean break, no shim).

## Collision semantics

A build has collision geometry when its header was generated with `--collision`
(`Solver.info.collision_enabled`). An omitted `collision` argument is a build-dependent default:
`"hard"` on a collision build, `None` on an open-world build. Explicit `None` is open-world on any build.
"No world installed" and an explicitly empty world are the same state: environment empty.

| `collision=` | self-collision checked | environment | notes |
| --- | --- | --- | --- |
| `None` (explicit) | no | no | open-world; the result is not a safety statement |
| `"hard"` | yes | filtered; empty environment never hits | default on a collision build; `clear_world()` never disables self-collision |
| `"soft"` | no | penetration cost ranks candidates; raises if the environment is empty | environment-only ranking; not a collision-free solve |
| `"both"` | yes | ranks by environment cost and filters self and environment; empty environment behaves as `"hard"` | |
| anything but `None` on an open-world build | | | raises |

So the ways to skip self-collision are explicit `None` and `"soft"`; `hard` and `both` always check self.
Margins and a self-collision toggle are Stage 3 and will apply to `hard` and `both` only.

The default changes relative to `generate_solutions`, whose default was collision checking off. On a
collision build `Solver.solve(pose)` defaults to `"hard"`, so counts, candidates and latency can differ from
the old default call. Comparisons between the two APIs are made with matched settings.

## Geometry and frames

`set_world(boxes=None, capsules=None, spheres=None, cylinders=None)`. Capsules are the native primitive;
cylinders are converted to a capsule of the same axis and length (a conservative approximation whose
rounded ends extend `radius` beyond each flat face). Every array is float64, one row per primitive,
expressed in the robot base frame, metres, quaternions scalar-first (`w x y z`):

| array | shape | row |
| --- | --- | --- |
| `boxes` | (M, 10) | `cx cy cz, qw qx qy qz, sx sy sz` (full side lengths) |
| `capsules` | (M, 7) | `x0 y0 z0, x1 y1 z1, r` (segment endpoints, radius) |
| `cylinders` | (M, 9) | `cx cy cz, qw qx qy qz, r, L` (axis = local +z, full length) |
| `spheres` | (M, 4) | `cx cy cz, r` |

Validation: finite values, positive sizes and radii, quaternions that normalise. `set_world` builds and
uploads the new world first and swaps only on success; a failed call leaves the previous world intact.
`clear_world()` installs the empty world. Convenience loaders for MotionBenchMaker problem JSON live in
`hjcdik.worlds` (`from_mbm_problem(problem_dict)`, `from_mbm_file(path, set_name, index)`); they return
the keyword arguments for `set_world` and are not part of the solver API.

## Results

```python
class Result:            # immutable, host numpy (writeable=False); survives later solves and solver destruction
    q:          (T, S, N) float64, NaN-padded     # S = num_solutions requested (fixed capacity)
    pose:       (T, S, 7) float64, NaN-padded
    pos_err_mm: (T, S)    float64, NaN-padded
    ori_err_rad:(T, S)    float64, NaN-padded
    count:      (T,)      int64                    # valid rows per target, 0 allowed
    valid:      (T, S)    bool                     # == arange(S) < count[:, None]
    elapsed_ms: (T,)      float64                  # host-observed per-target solve time; diagnostic only
    stats:      Stats | None                       # per-target diagnostics when solve(..., stats=True)
    def ok(self, pos_mm, ori_rad) -> (T, S) bool   # pose accuracy only, never physical safety
    def __getitem__(self, i) -> TargetResult       # un-padded views for target i
```

A single pose `(7,)` gives `T == 1` with the leading dimension kept; `r[0]` is the un-padded view. `T == 0`
is legal and returns an empty `Result`. `TargetResult` exposes `q (count, N)`, `pose (count, 7)`,
`pos_err_mm`, `ori_err_rad`, `count`, `elapsed_ms` and `stats` for one target.

`Stats` carries, per target, the fields of the former `ik_stats.csv` row: `batch_size`, `n_refined`,
`n_ik_accurate`, `n_coll_free_refined`, `n_feasible`, `n_ik_lost`, `n_coll_in_refined`, `n_coll_in_coarse`,
`env_cost_min_mm`, `env_cost_max_mm`, `env_cost_mean_mm`, `n_returned`, `n_returned_ik_accurate`,
`n_returned_coll_free`, `pct_returned_coll_free`, `n_returned_feasible`. Unmeasured fields are `-1`. The
solver never opens files; CSV serialisation belongs to the benchmark.

## Batch and resource contract

- The whole batch is validated before any GPU work: shapes, finite values, quaternion norms,
  `1 <= num_solutions <= batch_size`, `batch_size <= Solver.info.max_batch_size`.
- `solve` is all-or-nothing: a CUDA failure at target `i` raises `SolveError` with `target_index` and
  `completed`; no partial `Result` is returned.
- One world per call: the solver's lock is held for the entire multi-target `solve`; `set_world` and
  `clear_world` from another thread wait for it. Calls from several threads are safe and serialised; they
  are not concurrent on the GPU.
- A `Solver` binds to the CUDA device current at construction. Every operation runs on that device and
  restores the caller's current device before returning.
- Two solvers on the same robot hold independent worlds. A `Result` never references device memory.
- `precision="auto" | "fp32" | "fp64"`: auto is fp64 for `num_solutions == 1`, fp32 otherwise. I/O is
  float64 in every case.
- `seed=` and `q_init=` are not in Stage 1.

## Errors

`ValueError` for shapes, non-finite values, geometry and unsupported collision requests; `TypeError` for
argument conversion; `SolveError` (a `RuntimeError`) for CUDA and launch failures, after best-effort
cleanup; a context-invalidating CUDA error may still require a process restart. `BuildError` arrives with
Stage 2.

## Native layer

`generate_ik_solutions` loses the JSON triple and `write_stats`; it takes an optional collision world
handle built from host arrays, a `CollisionMode` (`none`, `hard`, `soft`, `both`) and a `collect_stats`
flag, and returns the diagnostics inside `Result<T>`. The MotionBenchMaker JSON parser leaves the native
layer. `sample_random_target_poses` is unchanged. The CLI and the native test move with it.

## Validation plan for Stage 1

Contract tests for every table row and every bullet above; single versus batched equivalence
(statistical); stats parity with the stored campaign rows; fresh-venv install smoke; CTest; strict
Sphinx; receipt. The latency comparison against `generate_solutions` runs with matched collision settings
in a user-announced window; the new default is measured as its own line.
