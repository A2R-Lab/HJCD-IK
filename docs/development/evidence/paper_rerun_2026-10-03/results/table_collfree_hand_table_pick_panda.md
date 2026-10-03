## Panda collision-free, table_pick_panda (dataset protocol: panda_hand, goal_pose)

| Batch | curobo Time(ms) | curobo Pos(mm) | curobo Ori(rad) | hjcdik Time(ms) | hjcdik Pos(mm) | hjcdik Ori(rad) | pyroki Time(ms) | pyroki Pos(mm) | pyroki Ori(rad) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1.876 | 0.04065 | 9.589e-06 | 3.983 | 20.57 | 0.07436 | 3.602 | 714.6 | 0.384 |
| 10 | 1.89 | 0.04065 | 9.589e-06 | 2.501 | 1.519 | 0.007368 | 5.745 | 1.177 | 0.002196 |
| 100 | 1.981 | 3.493e-04 | 1.056e-07 | 2.437 | 11.31 | 0.01095 | 5.56 | 0.02373 | 1.863e-05 |
| 1000 | 2.063 | 8.471e-05 | 5.223e-08 | 2.361 | 14.06 | 0.008891 | 6.702 | 0.001454 | 1.585e-06 |
| 2000 | 2.184 | 7.267e-05 | 5.040e-08 | 2.605 | 11.78 | 0.003344 | 6.807 | 2.948e-04 | 3.184e-07 |

_solvers: curobo: n=100/batch, hjcdik: n=1/batch, pyroki: n=100/batch_
