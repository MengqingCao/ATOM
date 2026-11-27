#!/usr/bin/env python3
"""Token-time client throughput: output tokens emitted inside [t0+A, t0+B] (tokens placed linearly between first token
and request end), per GPU over 8 GPUs; immune to completion-wave quantization. Also mean in-flight requests in the window.
usage: tokrate.py A B RUN_DIR..."""
import glob, json, sys
A, B = float(sys.argv[1]), float(sys.argv[2])
for run in sys.argv[3:]:
    f = glob.glob(run + "/slurm_job-*/benchmark_results/aiperf*/profile_export.jsonl")[0]
    R = [(d["metadata"]["request_start_ns"]/1e9, d["metadata"]["request_end_ns"]/1e9, d["metrics"]["time_to_first_token"]["value"]/1e3,
          d["metrics"]["output_sequence_length"]["value"]) for d in map(json.loads, open(f))
         if d["metadata"]["benchmark_phase"] == "profiling"]
    t0 = min(r[0] for r in R); a, b = t0 + A, t0 + B
    tok = 0.; busy = 0.; dec = 0.
    for s, e, t, n in R:
        f1 = s + t  # first token
        busy += max(0., min(e, b) - max(s, a))
        dec += max(0., min(e, b) - max(f1, a))
        if e > f1:
            tok += (n - 1) * max(0., min(e, b) - max(f1, a)) / (e - f1)
        tok += 1 if a <= f1 < b else 0
    W = B - A
    print(f"{run.split('/')[-1][:12]:12} win={A:.0f}-{B:.0f}s out_tok/s/GPU={tok/W/8:.0f} inflight={busy/W:.2f} in_decode={dec/W:.2f}")
