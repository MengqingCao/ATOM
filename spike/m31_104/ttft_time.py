#!/usr/bin/env python3
"""TTFT>5s requests by 30-s start-time bin, and TTFT percentiles: usage: ttft_time.py RUN_DIR..."""
import glob, json, sys
for run in sys.argv[1:]:
    f = glob.glob(run + "/slurm_job-*/benchmark_results/aiperf*/profile_export.jsonl")[0]
    R = [(d["metadata"]["request_start_ns"]/1e9, d["metrics"]["time_to_first_token"]["value"]/1e3, d["metadata"]["turn_index"], d["metadata"]["conversation_id"])
         for d in map(json.loads, open(f)) if d["metadata"]["benchmark_phase"] == "profiling"]
    t0 = min(r[0] for r in R); bins = [0]*21
    for s, t, *_ in R:
        if t > 5: bins[min(int((s-t0)//30), 20)] += 1
    tt = sorted(r[1] for r in R); q = lambda p: tt[int(p*(len(tt)-1))]
    print(run.split('/')[-1][:12], "p50/90/95/99", [round(q(p),2) for p in (.5,.9,.95,.99)], "slow/30s:", bins)
