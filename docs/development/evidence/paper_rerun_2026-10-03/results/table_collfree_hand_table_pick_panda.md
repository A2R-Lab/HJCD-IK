## Panda collision-free, table_pick_panda (dataset protocol: panda_hand, goal_pose)

| Batch | curobo Time(ms) | curobo Pos(mm) | curobo Ori(rad) | hjcdik Time(ms) | hjcdik Pos(mm) | hjcdik Ori(rad) | pyroki Time(ms) | pyroki Pos(mm) | pyroki Ori(rad) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 5.124 | 12.76 | 0.03221 | 3.983 | 20.57 | 0.07436 | 3.602 | 714.6 | 0.384 |
| 10 | 6.447 | 1.717 | 0.003665 | 2.501 | 1.519 | 0.007368 | 5.745 | 1.177 | 0.002196 |
| 100 | 2.119 | 0.005463 | 3.481e-06 | 2.437 | 11.31 | 0.01095 | 5.56 | 0.02373 | 1.863e-05 |
| 1000 | 2.307 | 1.839e-04 | 7.439e-08 | 2.361 | 14.06 | 0.008891 | 6.702 | 0.001454 | 1.585e-06 |
| 2000 | 2.436 | 1.718e-04 | 6.805e-08 | 2.605 | 11.78 | 0.003344 | 6.807 | 2.948e-04 | 3.184e-07 |

_solvers: curobo: n=100/batch, hjcdik: n=1/batch, pyroki: n=100/batch_
