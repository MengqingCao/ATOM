#!/usr/bin/env python3
"""Per-request reconciliation of aiperf profiling-phase records: usage: reqs.py RUN_DIR..."""
import glob, json, statistics as S, sys
for run in sys.argv[1:]:
    f = glob.glob(run + "/slurm_job-*/benchmark_results/aiperf*/profile_export.jsonl")[0]
    R = []
    for l in open(f):
        d = json.loads(l); m = d["metadata"]; x = d["metrics"]
        if m.get("benchmark_phase") != "profiling": continue
        R.append((m["request_start_ns"]/1e9, m["request_end_ns"]/1e9, x["time_to_first_token"]["value"]/1e3,
                  x.get("full_decode_duration", {}).get("value", 0)/1e3, x["output_sequence_length"]["value"],
                  x.get("usage_prompt_cache_read_tokens", {}).get("value", 0), m.get("was_cancelled")))
    t0 = min(r[0] for r in R); t1 = max(r[1] for r in R)
    ttft = [r[2] for r in R]; dec = [r[3] for r in R]; lat = [r[1]-r[0] for r in R]
    slow = [r for r in R if r[2] > 5]
    miss = sum(1 for r in R if r[5] < 100000)
    print(f"{run.split('/')[-1][:14]:14} n={len(R)} span={t1-t0:.0f}s ttft_mean={S.mean(ttft):.2f} ttft>5s={len(slow)} "
          f"(sum {sum(r[2] for r in slow):.0f}s) cache_miss={miss} dec_mean={S.mean(dec):.2f} "
          f"lat_mean={S.mean(lat):.2f} 45or64/lat={len(R)/(t1-t0):.3f}req/s osl={S.mean(r[4] for r in R):.0f} cancel={sum(1 for r in R if r[6])}")
