# C1 — success vs batch size on the hard sets, 2026-10-07 (closed)

The October 3 collections showed HJCD's hard-set success non-monotone in B (cage 99/90/97 % at
B = 100/1000/2000). Hypothesis: the collision-blind early stop — more blocks → an earlier first accurate
(colliding) candidate stops the batch sooner. Re-measured on the shipped kernel (`dd4c350`, foam spheres,
`panda_hand_joint`, hard mode, shared GPU, correctness only), 100 problems per set, judge = visual mesh 1 mm:

| set | B=100 | B=1000 | B=2000 | B=8000 |
| --- | ---: | ---: | ---: | ---: |
| cage | 99 | 100 | 100 | 100 |
| table_pick | 98 | 100 | 100 | 100 |
| table_under_pick | 99 | 99 | 100 | 100 |

Monotone (within one query) with the collision-aware stop. `hjcdik_main.jsonl` = stored configurations,
`scores_main.json` = scorer rows, `hard3.json` = the problem subset used. Item closed.
