## Pose-success AND collision-free (%), per oracle

| set | solver | B | n | pose ok | pose & hjcd | pose & paper | pose & curobo | pose & hull<=1mm | pose & hull<=5mm | pose & visual<=1mm | pose & visual<=5mm | all oracles |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| kitchen_panda | curobo | 100 | 100 | 100 | 99 | 99 | 100 | 100 | 100 | 100 | 100 | 99 |
| kitchen_panda | curobo | 2000 | 100 | 100 | 97 | 97 | 100 | 99 | 100 | 99 | 100 | 97 |
| kitchen_panda | hjcdik | 100 | 100 | 99 | 99 | 99 | 99 | 99 | 99 | 99 | 99 | 99 |
| kitchen_panda | hjcdik | 2000 | 100 | 99 | 99 | 99 | 99 | 99 | 99 | 99 | 99 | 99 |
| kitchen_panda | hjcdik_cusph | 100 | 100 | 100 | 97 | 97 | 100 | 99 | 100 | 100 | 100 | 97 |
| kitchen_panda | hjcdik_cusph | 2000 | 100 | 100 | 98 | 98 | 100 | 100 | 100 | 100 | 100 | 98 |
| kitchen_panda | pyroki | 100 | 100 | 99 | 99 | 99 | 99 | 99 | 99 | 99 | 99 | 99 |
| kitchen_panda | pyroki | 2000 | 100 | 99 | 99 | 99 | 99 | 99 | 99 | 99 | 99 | 99 |
| table_bars_panda | curobo | 100 | 100 | 100 | 97 | 97 | 100 | 34 | 48 | 97 | 100 | 34 |
| table_bars_panda | curobo | 2000 | 100 | 100 | 100 | 100 | 100 | 33 | 49 | 100 | 100 | 33 |
| table_bars_panda | hjcdik | 100 | 100 | 100 | 100 | 100 | 100 | 70 | 80 | 100 | 100 | 70 |
| table_bars_panda | hjcdik | 2000 | 100 | 100 | 100 | 100 | 100 | 72 | 95 | 100 | 100 | 72 |
| table_bars_panda | hjcdik_cusph | 100 | 100 | 100 | 100 | 100 | 100 | 70 | 82 | 100 | 100 | 70 |
| table_bars_panda | hjcdik_cusph | 2000 | 100 | 100 | 96 | 96 | 100 | 69 | 91 | 96 | 100 | 69 |
| table_bars_panda | pyroki | 100 | 100 | 100 | 100 | 100 | 100 | 63 | 64 | 100 | 100 | 63 |
| table_bars_panda | pyroki | 2000 | 100 | 100 | 100 | 100 | 100 | 71 | 96 | 100 | 100 | 71 |

_pose ok = pos < 5 mm and ori < 0.05 rad on the solver's own report; oracles ignore the base link and check environment obstacles only; sphere columns permit touching, mesh columns shrink every obstacle by the stated tolerance (hull = franka collision hulls, MoveIt's geometry; visual = true link shape); `all oracles` = every column agrees free._
