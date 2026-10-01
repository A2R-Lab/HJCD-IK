# HJCD-IK targeted timing

Campaign status: **complete**.

Environment-only collision oracle; no independent self-collision claim.
Positive delta means latest is slower. Review quality and variability, not just latency.

| Case | Paired rounds | Median delta range | Baseline/latest median ms | Baseline/latest p95 ms | Solved calls |
| --- | --- | --- | --- | --- | --- |
| both-compact-repeat | 3 | -1.34% … -0.69% | 2.1720 / 2.1492 | 3.0674 / 2.4927 | 384/384 / 384/384 |
| both-compact-switch | 3 | -28.82% … -28.47% | 3.0220 / 2.1561 | 3.3897 / 2.4946 | 384/384 / 384/384 |
| both-full-repeat | 3 | -14.37% … -13.74% | 2.8928 / 2.4821 | 34.9564 / 2.8143 | 384/384 / 384/384 |
| both-full-switch | 3 | -92.82% … -92.62% | 34.3185 / 2.4929 | 35.7069 / 2.8264 | 384/384 / 384/384 |
| hard-compact-repeat | 3 | -1.51% … -1.09% | 2.1395 / 2.1106 | 3.0210 / 2.4548 | 384/384 / 384/384 |
| hard-compact-switch | 3 | -29.29% … -28.96% | 2.9849 / 2.1170 | 3.3287 / 2.4619 | 384/384 / 384/384 |
| hard-full-repeat | 3 | -14.93% … -13.71% | 2.8542 / 2.4419 | 34.7256 / 2.7970 | 384/384 / 384/384 |
| hard-full-switch | 3 | -92.92% … -92.82% | 34.3388 / 2.4531 | 35.5363 / 2.7801 | 384/384 / 384/384 |
| open-s1-fp64 | 3 | -0.28% … +0.89% | 1.2869 / 1.2857 | 9.0112 / 9.0127 | 384/384 / 384/384 |
| open-s4-fp32 | 3 | -0.08% … +0.52% | 1.4649 / 1.4684 | 1.8414 / 1.8467 | 384/384 / 384/384 |

Quality-check failures: none in completed outputs.
A stopped/contaminated campaign is not acceptance evidence; see run.json and telemetry.jsonl.
Raw outputs retain scene-cache-hit tags, zero counts, FK errors and every returned configuration.
