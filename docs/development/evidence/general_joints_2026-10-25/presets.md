# Preset decisions and boundary findings

UR5e is included. Source is Universal Robots ROS2 Description at
`22f055da2fa7e2158254426107d1f257fd56aebb`, expanded with xacro 2.1.1, ur_type/name=ur5e,
empty prefix, safety limits disabled and stock nominal calibration. Target link `tool0`
is selected through fixed joint `flange-tool0`. The generator, source/license attribution,
mesh hashes and complete collision/visual fidelity reports ship with the model.
The generated representation uses local +Z revolute joints; the independent re-expressed
Panda fixtures, not this UR description, provide the arbitrary-axis kernel validation.

Independent FK matched at 200 configurations. Of these, 121 were self-free under the
sphere model; all 32 selected self-free targets solved in the exploratory hard-mode run.
The signed regression tests repeat independent FK, collision verdicts and >=31/32 solve
quality in both precisions. These rates describe the selected known-feasible set, not an
unfiltered benchmark. The visual bulge reaches 60.2 mm, so the model can reject narrow gaps.

Z1 is withheld. The Unitree source at `267182b8521c8d6a631bab1fe63836873237a525` omits
collision primitives on link01 and link05. A complete fit to its supplied collision STLs
produced zero self-free poses in 200 samples. Independent CPU sphere placement confirmed
two consistently overlapping non-adjacent pairs: link03/link05, minimum clearance -11.56
to -5.61 mm; link04/link06, -9.10 to -6.63 mm. This is a collision-model/policy limitation,
not evidence that the physical robot is immovable. Do not drop those links or disable self
checking implicitly. The generator can reproduce the candidate for investigation; it is
not in `presets()`. Tighter fitting or validated exclusions need a separate decision.

Fetch-with-torso is withheld from distribution. The authoritative `fetch_description`
package at fetch_ros `d22b98f6c21577895f4d2e3a8bd54e329083732b` declares
CreativeCommons-Attribution-NonCommercial-ShareAlike-4.0-International in package.xml.
No new Fetch assets were bundled. Resolve distribution terms before adding this preset.
The older kinematics-only Fetch smoke fixture is not a substitute for the full description.

Boundary checks found GRiD 0.6.0 calls `.atoms` on a NumPy fixed homogeneous transform
when generating an all-prismatic chain. Mixed chains short-circuit that scan on a revolute
transform. HJCD rejects all-prismatic models explicitly until an upstream release fixes
and verifies the no-trig path. The mixed-chain scale gate remains valid; it did not test
all-prismatic code generation. GRiD ask G13 records the reproducer.

The hyphenated UR5e tool name also exposed an HJCD frame lookup bug. Both static and runtime
paths now share GRiD's identifier-escaping rule when reading the generated fixed-frame index.
A regression test uses `flange-tool0`; no generated header is hand-edited.
