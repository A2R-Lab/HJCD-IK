## Panda collision-free, box_panda (dataset protocol: panda_hand, goal_pose)

| Batch | curobo Time(ms) | curobo Pos(mm) | curobo Ori(rad) | hjcdik Time(ms) | hjcdik Pos(mm) | hjcdik Ori(rad) | pyroki Time(ms) | pyroki Pos(mm) | pyroki Ori(rad) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.765 | 0.006069 | 3.129e-07 | 6.485 | 22.15 | 0.03437 | 3.363 | 522.7 | 0.2789 |
| 10 | 1.835 | 0.006069 | 3.129e-07 | 2.427 | 5.263e-06 | 7.833e-10 | 5.851 | 0.4027 | 2.374e-04 |
| 100 | 1.928 | 0.004938 | 1.628e-07 | 2.226 | 4.971e-06 | 7.924e-10 | 6.584 | 1.172e-04 | 1.943e-07 |
| 1000 | 2.08 | 0.004717 | 4.048e-08 | 2.298 | 4.808e-06 | 7.800e-10 | 7.567 | 1.129e-04 | 1.963e-07 |
| 2000 | 2.208 | 0.002101 | 1.451e-07 | 2.553 | 4.702e-06 | 8.112e-10 | 7.73 | 1.143e-04 | 1.831e-07 |

_solvers: curobo: n=100/batch, hjcdik: n=1/batch, pyroki: n=100/batch_
