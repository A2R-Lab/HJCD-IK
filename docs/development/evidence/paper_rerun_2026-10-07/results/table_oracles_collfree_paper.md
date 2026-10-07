## Pose-success AND collision-free (%), per oracle

| set | solver | B | n | pose ok | pose & hjcd | pose & paper | pose & curobo | pose & hull<=1mm | pose & hull<=5mm | pose & visual<=1mm | pose & visual<=5mm | all oracles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| box_panda | curobo | 1 | 100 | 98 | 96 | 96 | 98 | 98 | 98 | 98 | 98 | 96 |
| box_panda | curobo | 10 | 100 | 100 | 98 | 98 | 100 | 100 | 100 | 100 | 100 | 98 |
| box_panda | curobo | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| box_panda | curobo | 1000 | 100 | 100 | 99 | 99 | 100 | 100 | 100 | 100 | 100 | 99 |
| box_panda | curobo | 2000 | 100 | 100 | 99 | 99 | 100 | 100 | 100 | 100 | 100 | 99 |
| box_panda | hjcdik | 1 | 100 | 89 | 89 | 89 | 89 | 89 | 89 | 89 | 89 | 89 |
| box_panda | hjcdik | 10 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| box_panda | hjcdik | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| box_panda | hjcdik | 1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| box_panda | hjcdik | 2000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| box_panda | pyroki | 1 | 100 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| box_panda | pyroki | 10 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| box_panda | pyroki | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| box_panda | pyroki | 1000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| box_panda | pyroki | 2000 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |

_pose ok = pos < 5 mm and ori < 0.05 rad (independent URDF FK for new dumps; legacy solver reports only when explicitly permitted); oracles ignore the base link and check environment obstacles only; sphere columns permit touching, mesh columns shrink every obstacle by the stated tolerance (hull = franka collision hulls, MoveIt's geometry; visual = visual-link mesh approximation); `all oracles` = every column agrees free._
