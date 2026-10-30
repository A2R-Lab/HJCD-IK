import argparse, csv, importlib.util, json, statistics as stats
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--repo', type=Path, required=True)
p.add_argument('--runs', type=Path, nargs='+', required=True)
p.add_argument('--summary', type=Path, required=True)
a = p.parse_args(); ROOT = a.repo.resolve()
spec = importlib.util.spec_from_file_location('gate', ROOT / 'scripts/perf/run_audit_gate.py')
gate = importlib.util.module_from_spec(spec); spec.loader.exec_module(gate)
runs = {chr(97+i): r for i, r in enumerate(a.runs)}
cells, totals, compiles = {}, {}, []
for run, root in runs.items():
    totals[run] = dict(queries=0, solved=0, empty=0, returned=0, mismatches=[], quality_review=[])
    for leg in ('api','native','runtime','highdof','warm'):
        manifest = json.loads((root/leg/'manifest.json').read_text())
        rows = list(csv.DictReader((root/leg/'results.csv').open()))
        assert not manifest['correctness_only']
        assert manifest['driver_sha'] == gate.digest(gate.DRIVER)
        findings = gate.analyze(manifest['config'], rows)
        assert findings == json.loads((root/leg/'analysis.json').read_text())
        for r in rows:
            for s, d in (('n_targets','queries'),('solved','solved'),('empty','empty'),('returned','returned')):
                totals[run][d] += int(r[s])
        index = {(r['label'], r['leg'], int(r['batch']), int(r['round'])): r for r in rows}
        for f in findings:
            if f.get('quality_review_required'): totals[run]['quality_review'].append([leg, f['solutions'], f['leg'], f['batch']])
            key = (leg, f['solutions'], f['leg'], f['batch'])
            cell = cells.setdefault(key, dict(leg=leg, solutions=f['solutions'], scene=f['leg'], batch=f['batch'], rounds=[], run_median_percent={}))
            cell['run_median_percent'][run] = 100*(f['paired_ratio']-1)
            for rd in range(1,5):
                m = index[(f"main-S{f['solutions']}", f['leg'], f['batch'], rd)]
                b = index[(f"branch-S{f['solutions']}", f['leg'], f['batch'], rd)]
                for field in ('solved','empty','returned'):
                    if m[field] != b[field]: totals[run]['mismatches'].append([*key, rd, field, m[field], b[field]])
                base, new = float(m['median_ms']), float(b['median_ms'])
                cell['rounds'].append(dict(run=run, round=rd, base_ms=base, candidate_ms=new, percent=100*(new/base-1), delta_us=1000*(new-base)))
    for rd in (1,2):
        for ep in ('main','branch'):
            for robot in ('panda','24dof'):
                compiles.append(dict(run=run, round=rd, endpoint=ep, **json.loads((root/'compile'/f'{rd}-{ep}'/f'{robot}.json').read_text())))
n = 4*len(runs)
for c in cells.values():
    for field in ('percent','delta_us','base_ms','candidate_ms'): c['median_'+field] = stats.median(r[field] for r in c['rounds'])
    c['min_percent'] = min(r['percent'] for r in c['rounds']); c['max_percent'] = max(r['percent'] for r in c['rounds'])
    c['slower_rounds'] = sum(r['percent'] > 0 for r in c['rounds'])
a.summary.write_text(json.dumps(dict(totals=totals, cells=list(cells.values()), compiles=compiles), indent=2)+'\n')
print(json.dumps(totals, indent=2))
hdr = ' | '.join(f'{r} %' for r in runs)
print(f'| Path | S | Scene | B | {hdr} | Pooled % | Delta us | Base ms | Slower rounds |'); print('|---|---:|---|---:|' + '---:|'*len(runs) + '---:|---:|---:|---:|')
for c in cells.values():
    per = ' | '.join(f"{c['run_median_percent'][r]:+.3f}" for r in runs)
    print(f"| {c['leg']} | {c['solutions']} | {c['scene']} | {c['batch']} | {per} | {c['median_percent']:+.3f} | {c['median_delta_us']:+.2f} | {c['median_base_ms']:.3f} | {c['slower_rounds']}/{n} |")
for robot in ('panda','24dof'):
    for ep in ('main','branch'):
        rows = [c for c in compiles if c['robot']==robot and c['endpoint']==ep]
        print(robot, ep, {f: dict(median=round(stats.median(c[f] for c in rows),3), min=round(min(c[f] for c in rows),3), max=round(max(c[f] for c in rows),3)) for f in ('cold_seconds','cache_hit_seconds')})
