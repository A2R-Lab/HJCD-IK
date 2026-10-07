# Nullspace repair experiment (C2) — negative result, 2026-10-07

Question: does replacing the LM repair round's deterministic joint-space kick with a principled step —
push penetrating spheres out along their environment normals (`grid_collision::warp::collision_distance`),
projected onto the nullspace of the unscaled EE Jacobian (6×6 `glass::warp::posv`), scaled so the deepest
sphere clears by a margin, then LM re-projection — recover more of the clearance-ladder misses?

Answer: **no.** Same build (`dd4c350`, foam spheres, `panda_hand_joint`), same ladder (`mb_tight.json`, the
16 levels with ≥ 20 kept problems), B = 100 and 2000, 2776 queries each, shared GPU (correctness only):

| variant | pose ok | pose & own spheres | pose & visual mesh (1 mm) | empty |
| --- | ---: | ---: | ---: | ---: |
| kick only (`HJCD_REPAIR_MODE=0`, the shipped behaviour) | 2599 | 2599 | 2547 | 0 |
| nullspace first, kick fallback (`HJCD_REPAIR_MODE=1`) | 2601 | 2601 | 2545 | 0 |

Per cell the two differ by at most ±2 queries with no direction; identical on every cage level; identical on
the three dataset hard sets at B = 100/1000/2000/8000 and on kitchen/table_bars. `scores_ladder.json` has the
per-cell numbers (`hjcdik_kick` / `hjcdik_c2`); the JSONL dumps are the stored configurations.

Why there is nothing to gain: in every cell `pose_ok == pose & own spheres`, i.e. the solver never returns an
accurate configuration that its own sphere model rejects — the hard filter and the band fallback already turn
every accurate-but-colliding candidate into either a free one or a filtered one. The remaining misses are
(a) pose misses where no free accurate configuration exists under the sphere model (the model ceiling shown
in the fairness evidence: HJCD on cuRobo's spheres matches cuRobo), and (b) ~2 % of returned configurations
the foam spheres accept but the true mesh rejects (model under-coverage). Neither is a search problem.

Decision: not shipped. The kernel is unchanged (`nullspace_repair.patch` is the full implementation against
`dd4c350` for the record); no timing A/B is needed. A conservative, cascaded sphere model (GRiD G3) is the
remaining route to the model-limited cells, not a better repair step.
