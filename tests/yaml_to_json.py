# yaml_to_problems_json.py
import yaml, json, sys

in_path  = sys.argv[1]          # mb_set.yml
out_path = sys.argv[2]          # problems.json

with open(in_path, "r") as f:
    y = yaml.safe_load(f)

# Wrap in the key your C++ expects:
j = {"problems": y}

with open(out_path, "w") as f:
    json.dump(j, f)

print("Wrote", out_path)