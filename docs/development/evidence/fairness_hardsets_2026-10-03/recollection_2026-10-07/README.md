# b10 / b15 conservative-model recollection — 2026-10-07

The October 3 dumps for the bounded-bulge models omitted empty queries (2775 / 2771 of 2776 ladder records),
so their rates were mis-denominated. Recollected on the shipped kernel (`dd4c350`, `panda_hand_joint`, hard
mode) with the empties-aware driver (`collect_hjcd.sh`, `query_results` records), shared GPU, correctness
only: the 8 dataset sets, the 16 ladder levels with ≥ 20 kept problems, and kitchen/table_bars; B = 100 and
2000; 4776 queries per model. Scorer: `score_collision_oracles.py --mesh-geometries visual --mesh-tols-mm 1`.

| model | spheres | queries | pose ok | pose & visual mesh (1 mm) | cage −0.5 / −1 / −1.5 cm (B=2000) |
| --- | ---: | ---: | ---: | ---: | --- |
| bounded bulge 10 mm (`panda_visual_b10`) | 377 | 4776 | 4196 | 4196 | 52 / 4 / 0 % |
| bounded bulge 15 mm (`panda_visual_b15`) | 200 | 4776 | 4026 | 4026 | 44 / 0 / 0 % |

The cage numbers match the October 3 table exactly. New fact: with a model that fully contains the robot,
`pose ok == pose & visual mesh` on every one of the 4776 queries — zero sphere/mesh disagreement — whereas
foam's spheres produce ~2 % of returned configurations the true mesh rejects (fairness evidence, ladder).
The price remains lower success in tight scenes (and the sphere-count latency cost, unmeasured here).
Files: `hjcdik_b10*.jsonl`, `hjcdik_b15*.jsonl` (stored configurations incl. empties), `scores_*.json`.
