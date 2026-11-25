#!/usr/bin/env python3
"""A3 table from 1P1D runs, plus B2's uneven-step share and the per-rank rows for the A1 rerun.

Usage: report.py RUN_DIR... (runs named pd-<ARM>-c<C>-r<k>-<stamp>, each with report.json from pd_report.py)
Per arm and point: mean of the runs and their spread (max-min as % of the mean) for output tok/s per GPU
(8 GPUs), TPOT p50/p90, TTFT p50/p90, decode running batch mean, and each run's limit verdict.
DP arms: steps/dp.<pid> (run tree's ATOM_SPIKE_STEP_LOG, one line per step: every rank's scheduled tokens,
any-prefill, padded running_bs, running_tokens). On all-decode steps (8 verify rows a request) a step is
"uneven" when the padded group rows (4 x running_tokens) reach 256, the MXFP4 threshold MegaMoE runs at,
while the true group rows (sum of scheduled tokens) stay below it.
"""
import glob
import json
import os
import re
import statistics
import sys

ROWS = 8  # DSpark k=7: 8 verify rows a request
THRESH = 256
KEYS = [("out_tok_s_per_gpu", "client"), ("tpot_p50_ms", "client"), ("tpot_p90_ms", "client"),
        ("ttft_p50_s", "client"), ("ttft_p90_s", "client"), ("batch_mean", "decode")]


def steps(run):
    files = sorted(glob.glob(os.path.join(run, "steps", "dp.*")), key=os.path.getsize)
    if not files:
        return None
    dec = []
    for line in open(files[-1]):  # every rank logs the same table; take the longest
        p = line.split()
        if len(p) == 4 and p[1] == "0":
            dec.append(([int(t) for t in p[0].split(",")], int(p[3])))
    if not dec:
        return None
    uneven = sum(1 for t, rt in dec if len(t) * rt >= THRESH and sum(t) < THRESH)
    mega = sum(1 for t, rt in dec if len(t) * rt >= THRESH)
    req = [sorted((x + ROWS - 1) // ROWS for x in t)[::-1] for t, _ in dec]
    n = len(req[0])

    def pos(q):
        return [sorted(r[i] for r in req)[int(q * (len(req) - 1))] for i in range(n)]

    return {"decode_steps": len(dec), "mega_steps": mega, "uneven_steps": uneven,
            "uneven_share": uneven / len(dec),
            "req_per_rank_mean": [round(statistics.mean(r[i] for r in req)) for i in range(n)],
            "req_per_rank_p50": pos(0.5), "req_per_rank_p90": pos(0.9)}


def main(runs):
    groups = {}
    for run in runs:
        m = re.match(r"pd-(A|B1|B2)-c(\d+)-r\d+", os.path.basename(run.rstrip("/")))
        f = os.path.join(run, "report.json")
        if m and os.path.exists(f):
            groups.setdefault((m.group(1), int(m.group(2))), []).append(run)
    out = {}
    for (arm, c), rs in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0])):
        reps = [json.load(open(os.path.join(r, "report.json"))) for r in rs]
        row = {"runs": [os.path.basename(r.rstrip("/")) for r in rs]}
        for k, sec in KEYS:
            v = [x[sec].get(k) for x in reps]
            if None in v:
                row[k] = {"runs": v}
                continue
            mu = statistics.mean(v)
            row[k] = {"mean": round(mu, 3), "spread_pct": round(100 * (max(v) - min(v)) / mu, 1), "runs": v}
        row["limit"] = [x.get("limit") for x in reps]
        st = [s for s in (steps(r) for r in rs) if s]
        if st:
            row["steps"] = st
        out[f"{arm}@c{c}"] = row
    json.dump(out, sys.stdout, indent=1)
    print()
    print(f"\n{'arm@point':10} {'n':>2} {'out tok/s/GPU':>16} {'TPOT p50/p90 ms':>17} {'TTFT p50/p90 s':>17} {'batch':>13}")
    for name, row in out.items():
        def f(k, d=0):
            x = row[k]
            return f"{x['mean']:.{d}f}" if "mean" in x else "n/a"
        o = row["out_tok_s_per_gpu"]
        print(f"{name:10} {len(row['runs']):>2} {f('out_tok_s_per_gpu'):>9} ({o.get('spread_pct', '-')}%) "
              f"{f('tpot_p50_ms', 2):>8}/{f('tpot_p90_ms', 2):<8} {f('ttft_p50_s', 1):>8}/{f('ttft_p90_s', 1):<8} "
              f"{f('batch_mean', 1):>6} ({row['batch_mean'].get('spread_pct', '-')}%)")


if __name__ == "__main__":
    main(sys.argv[1:])
