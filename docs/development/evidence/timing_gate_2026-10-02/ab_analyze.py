import csv, statistics, sys
from collections import defaultdict
rows = list(csv.DictReader(open(sys.argv[1])))
d = defaultdict(dict)   # (leg,S,batch,round) -> {endpoint: median}
for r in rows:
    ep, S = r["label"].split("-S")
    d[(r["leg"], int(S), int(r["batch"]), int(r["round"]))][ep] = (float(r["median_ms"]), float(r["min_ms"]), float(r["p95_ms"]))
cases = sorted({k[:3] for k in d})
print(f"{'leg':7s} {'S':>2s} {'batch':>6s} | {'main med ms':>11s} {'branch med ms':>13s} | {'paired ratio (branch/main)':>27s} | {'round ratios min..max':>22s} | {'min-of-min ratio':>16s}")
for c in cases:
    rr = [d[c + (r,)] for r in sorted({k[3] for k in d if k[:3] == c})]
    ratios = [x["branch"][0] / x["main"][0] for x in rr]
    m = statistics.median(x["main"][0] for x in rr); b = statistics.median(x["branch"][0] for x in rr)
    mm = min(x["main"][1] for x in rr); bm = min(x["branch"][1] for x in rr)
    med = statistics.median(ratios)
    print(f"{c[0]:7s} {c[1]:2d} {c[2]:6d} | {m:11.4f} {b:13.4f} | {100*(med-1):+26.2f}% | {100*(min(ratios)-1):+9.2f}%..{100*(max(ratios)-1):+8.2f}% | {100*(bm/mm-1):+15.2f}%")
