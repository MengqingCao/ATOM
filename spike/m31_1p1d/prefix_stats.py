"""ATOM#81 shared-prefix cell: what one run did with the 128k prefix. Usage: prefix_stats.py RUN_DIR
One JSON line: prefill prefix / Store hit % (pd_report), the decode side's prefix hit over its received prompts and
the tokens a request's KV transfer carried (full - locally cached), decode KV usage p90 and running batch, and the
Mooncake Store's final usage (the masters' metrics the launcher saves at stop)."""

import glob
import json
import os
import sys

run = sys.argv[1]
sys.path.insert(0, "/share_nfs/mengqing/code/tmp/m31-decode-dep4/81/harness/bin")
import pd_report as p  # noqa: E402

rep = json.load(open(os.path.join(run, "report.json")))
pre, dec = rep.get("prefill") or {}, rep.get("decode") or {}
s0, s1, _ = p.phase_window(run)
rows = p.load_scrapes(os.path.join(run, "scrape_decode.jsonl"))
w0 = p.to_epoch(s0, rows[0]["t"]) if s0 and rows else None
w1 = p.to_epoch(s1, rows[0]["t"]) if s1 and rows else None
rows = [x for x in rows if (w0 is None or x["t"] >= w0) and (w1 is None or x["t"] <= w1)]
d = {}
if len(rows) >= 2:
    def delta(name):
        a, b = p.by(rows[0]["m"], name), p.by(rows[-1]["m"], name)
        return sum(b.values()) - sum(a.get(k, 0.0) for k in b)
    full, cached = delta("atom:prefix_cache_full_tokens_total"), delta("atom:prefix_cache_cached_tokens_total")
    n = delta("atom:pd_kv_transfer_seconds_count")
    d = {
        "prefix_hit_pct": round(100 * cached / full, 2) if full else None,
        "transfers": int(n),
        "recv_tokens_per_req": round((full - cached) / n) if n else None,
        "prompt_tokens_per_req": round(full / n) if n else None,
    }
store = {}
for f in glob.glob(os.path.join(run, "slurm_job-*/logs/mooncake-master-*.metrics")):
    for line in open(f):
        k, _, v = line.partition(" ")
        if k in ("master_allocated_bytes", "master_total_capacity_bytes", "master_key_count"):
            store[k] = store.get(k, 0) + float(v)
print(json.dumps({
    "prefill_prefix_hit_pct": pre.get("prefix_hit_pct"), "prefill_store_hit_pct": pre.get("offload_hit_pct"),
    "prefill_computed_tok_s": pre.get("computed_tok_s"),
    "decode": d, "decode_kv_usage_p90": dec.get("kv_usage_p90"),
    "decode_batch_mean": dec.get("batch_mean_all", dec.get("batch_mean")), "decode_batch_p90": dec.get("batch_p90"),
    "req_s": (rep.get("client") or {}).get("req_s"),
    "store_used_GiB": round(store["master_allocated_bytes"] / 2**30, 1) if store else None,
    "store_cap_GiB": round(store["master_total_capacity_bytes"] / 2**30, 1) if store else None,
    "store_keys": int(store["master_key_count"]) if store else None,
}))
