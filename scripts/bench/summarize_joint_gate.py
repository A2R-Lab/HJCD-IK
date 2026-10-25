#!/usr/bin/env python3
"""Extract the declared mixed-joint correctness cells from a pytest JUnit report.

Use pytest -o junit_family=legacy --junitxml=gate.xml tests/test_general_joints.py.
This records solve quality only; pytest durations are deliberately not performance data.
"""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("junit", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = ET.parse(args.junit)
    if root.findall(".//failure") or root.findall(".//error"):
        raise SystemExit("refusing to summarize a failing correctness run as a passing gate")
    cells = [cell for prop in root.findall('.//property[@name="scale_cells"]')
             for cell in json.loads(prop.get("value"))]
    keys = {(c["travel_m"], c["slider_first"], c["precision"], c["solutions"], c["seed"], c["warm"])
            for c in cells}
    expected = {(t,f,p,s,r,w) for t in (.01,.5,5.) for f in (False,True)
                for p in ("fp32","fp64") for s in (1,4) for r in (13,29) for w in (False,True)}
    if keys != expected or len(cells) != len(expected):
        raise SystemExit("incomplete or duplicated scale matrix")
    if not all(c["attempted"] == 32 and c["accurate"] >= 31 for c in cells):
        raise SystemExit("a declared >=31/32 scale cell failed")
    result = dict(schema=1, batch_size=512, position_tolerance_m=.005,
                  orientation_tolerance_rad=.001, minimum_accurate_per_cell=31,
                  attempted=sum(c["attempted"] for c in cells),
                  accurate=sum(c["accurate"] for c in cells), cells=cells)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(f'{result["accurate"]}/{result["attempted"]} accurate targets across {len(cells)} cells')


if __name__ == "__main__":
    main()
