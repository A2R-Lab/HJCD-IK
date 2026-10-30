import argparse
import csv
import hashlib
import importlib.util
import json
import statistics as stats
from pathlib import Path

p = argparse.ArgumentParser(description='Validate both complete boundary gates and summarize paired measurements.')
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--run-a', type=Path, required=True)
p.add_argument('--run-b', type=Path, required=True)
p.add_argument('--summary', type=Path, required=True)
args = p.parse_args()
ROOT = args.repo.resolve()
spec = importlib.util.spec_from_file_location('gate', ROOT / 'scripts/perf/run_audit_gate.py')
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
runs = {'a': args.run_a, 'b': args.run_b}
cells = {}
totals = {}
compiles = []
for run, root in runs.items():
    totals[run] = dict(queries=0, solved=0, empty=0, returned=0, mismatches=[])
    for leg in ('api', 'native', 'runtime', 'highdof', 'warm'):
        manifest = json.loads((root / leg / 'manifest.json').read_text())
        rows = list(csv.DictReader((root / leg / 'results.csv').open()))
        assert not manifest['correctness_only']
        assert manifest['driver_sha'] == gate.digest(gate.DRIVER)
        findings = gate.analyze(manifest['config'], rows)
        assert findings == json.loads((root / leg / 'analysis.json').read_text())
        for r in rows:
            for source, dest in (('n_targets','queries'), ('solved','solved'), ('empty','empty'), ('returned','returned')):
                totals[run][dest] += int(r[source])
        index = {(r['label'], r['leg'], int(r['batch']), int(r['round'])): r for r in rows}
        for f in findings:
            key = (leg, f['solutions'], f['leg'], f['batch'])
            cell = cells.setdefault(key, dict(leg=leg, solutions=f['solutions'], scene=f['leg'], batch=f['batch'], rounds=[], run_median_percent={}))
            cell['run_median_percent'][run] = 100 * (f['paired_ratio'] - 1)
            for round_ in range(1, 5):
                a = index[(f"main-S{f['solutions']}", f['leg'], f['batch'], round_)]
                b = index[(f"branch-S{f['solutions']}", f['leg'], f['batch'], round_)]
                for field in ('solved', 'empty', 'returned'):
                    if a[field] != b[field]:
                        totals[run]['mismatches'].append([*key, round_, field, a[field], b[field]])
                base, new = float(a['median_ms']), float(b['median_ms'])
                cell['rounds'].append(dict(run=run, round=round_, base_ms=base, candidate_ms=new,
                                          percent=100*(new/base-1), delta_us=1000*(new-base)))
    for round_ in (1, 2):
        for endpoint in ('main', 'branch'):
            for robot in ('panda', '24dof'):
                data = json.loads((root / 'compile' / f'{round_}-{endpoint}' / f'{robot}.json').read_text())
                compiles.append(dict(run=run, round=round_, endpoint=endpoint, **data))

for c in cells.values():
    for field in ('percent', 'delta_us', 'base_ms', 'candidate_ms'):
        c['median_' + field] = stats.median(r[field] for r in c['rounds'])
    c['min_percent'] = min(r['percent'] for r in c['rounds'])
    c['max_percent'] = max(r['percent'] for r in c['rounds'])
    c['slower_rounds'] = sum(r['percent'] > 0 for r in c['rounds'])
summary = dict(totals=totals, cells=list(cells.values()), compiles=compiles)
args.summary.write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(totals, indent=2))
print('| Path | S | Scene | B | A % | B % | Pooled % | Delta us | Slower rounds |')
print('|---|---:|---|---:|---:|---:|---:|---:|---:|')
for c in cells.values():
    print(f"| {c['leg']} | {c['solutions']} | {c['scene']} | {c['batch']} | {c['run_median_percent']['a']:+.3f} | {c['run_median_percent']['b']:+.3f} | {c['median_percent']:+.3f} | {c['median_delta_us']:+.2f} | {c['slower_rounds']}/8 |")
for robot in ('panda', '24dof'):
    for endpoint in ('main', 'branch'):
        rows = [c for c in compiles if c['robot']==robot and c['endpoint']==endpoint]
        print(robot, endpoint, {field: dict(median=stats.median(c[field] for c in rows), min=min(c[field] for c in rows), max=max(c[field] for c in rows)) for field in ('cold_seconds', 'cache_hit_seconds')})
