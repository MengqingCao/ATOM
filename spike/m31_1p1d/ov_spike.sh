# OVERRIDES for bin/run_pd.sh (ATOM#35): DP arms record every decode step's per-rank scheduled
# tokens (run tree's ATOM_SPIKE_STEP_LOG) under <run>/slurm_job-*/steps/; TRACE=1 adds the decode profiler window.
if [[ "${ARM}" == B1 || "${ARM}" == B2 ]]; then
  # Relative: the decode container mounts only <run>/slurm_job-*/ (as RUN_DIR), so the files land there.
  export ATOMESH_DECODE_ENV_ATOM_SPIKE_STEP_LOG="steps/dp"
fi
if [[ -n "${TRACE:-}" ]]; then
  source "${H}/cells/ov_trace_decode.sh"
fi
# Owner's ruling (#35, issuecomment-7164687062), the same for A, B1 and B2:
# (1) no decode DP session affinity: the launcher's DPA branch (dp_sticky + --dp-aware + X-Session-ID) is off,
#     so B1/B2 get the A arm's router (policy random, one decode endpoint); works around the deadlock fixed in #66.
# (2) AgentX warmup 1 request per lane (cell: 10); the 1200 s window and the 1800 s grace period stay.
python3 - .github/scripts/atomesh/pd_server_atom.sh <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
a = "is_agentic_dpa() {\n"
assert s.count(a) == 1, "is_agentic_dpa in pd_server_atom.sh changed"
open(p, "w").write(s.replace(a, a + "  return 1  # ATOM#35: decode DP session affinity off\n"))
PY
export AIPERF_WARMUP_REQUESTS_PER_LANE=1
# Owner's ruling (#35, issuecomment-7167528921): a 2 TB (decimal, 1863 GiB) Mooncake Store CPU (host DRAM) KV tier for the prefill,
# through the launcher's MOONCAKE_STORE=1 (ATOM_KV_OFFLOAD=mooncake_store behind the P/D connector). The decode
# keeps no tier (ATOM's Store is the prefill's offload tier). 4 owners x 465 GiB, one per NIC: an owner above
# 832 GiB may not register on an ionic NIC; 930 GiB a NUMA node stays under the ~1017 GiB a node compacts to huge pages
# (2 TiB = 1024 GiB a node failed rc=134 on g48 and g40).
export ATOMESH_PREFILL_ENV_MOONCAKE_STORE=1
export ATOMESH_PREFILL_ENV_MOONCAKE_STORE_OWNERS="0:465:rdma4;1:465:rdma6"
# Ports: another process on the node can hold a launcher default (29600 on g53, run 20261126035137). batch.sh picks
# ATOMESH_PORT_SHIFT so every default port (and its eval-phase +1000 twin) is free; the launcher shifts all of them
# (servers, router, Prometheus, handshake, P and D torch dist master/base, Mooncake master/metadata/metrics/owners)
# through ATOMESH_SERVICE_PORT_OFFSET, so P and D stay wired to each other.
python3 - .github/scripts/atomesh/pd_slurm_job.sh <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
for a, b in (("printf '%s\\n' \"${ATOMESH_RESTART_PORT_OFFSET}\"",
              "printf '%s\\n' \"$((ATOMESH_RESTART_PORT_OFFSET + ${ATOMESH_PORT_SHIFT:-0}))\"  # ATOM#35"),
             ("printf '0\\n'", "printf '%s\\n' \"${ATOMESH_PORT_SHIFT:-0}\"  # ATOM#35")):
    assert s.count(a) == 1, "execution_phase_port_offset in pd_slurm_job.sh changed"
    s = s.replace(a, b)
open(p, "w").write(s)
PY
# ATOM#81 fixed-length cell (run_pd.sh KIND=fixed, the AgentX cell's P/D topology), owner's ruling 7184230573:
# every request = one shared FIXED_PREFIX (130048) token prefix + a unique FIXED_TAIL (1024) token synthetic tail
# (128k context), OSL FIXED_OSL (2048), stddev 0, ignore_eos; prefix caching ON on P and D (the cell's default).
# aiperf --prefix-prompt-pool-size 1 gives every request the same prefix; dataset entries = 1 + measured requests,
# so each tail is unique. Warmup: 1 request at concurrency 1 computes the prefix once (aiperf waits for it before the
# profiling phase); then a closed loop at CONC with CONC x FIXED_REQ_MULT (5) measured requests.
# FIXED_WINDOW=1 (owner's ruling 7195240774, the r2 runs): time-based instead, aiperf --benchmark-duration
# (the run's DURATION_S, AIPERF_BENCHMARK_DURATION) with aiperf's own end of window: no new request after it, in-flight
# ones get aiperf's default 30 s grace period and count if they finish in it. CONC x 60 dataset entries (unique tails;
# c64 ran 2.5 req/s, ~1500 in 600 s) so the closed loop never runs short or reuses a tail.
if [[ "${KIND}" == fixed ]]; then
  python3 - .github/scripts/atomesh/pd_server_atom.sh "${FIXED_PREFIX:-130048}" "${FIXED_TAIL:-1024}" "${FIXED_OSL:-2048}" "${FIXED_REQ_MULT:-5}" "${FIXED_WINDOW:-}" <<'PY'
import sys
p, pre, isl, osl, mult, window = sys.argv[1:]
s = open(p).read()
a = '      "${AGENTX_VENV}/bin/agentx" profile \\\n'
b = '      --public-dataset "${AIPERF_PUBLIC_DATASET}" \\\n'
i, j = s.find(a), s.find(b)
assert s.count(a) == 1 and s.count(b) == 1 and i < j, "agentx profile call in pd_server_atom.sh changed"
k = '      "${server_metrics_args[@]}" \\\n      --output-artifact-dir "${out_dir}" \\\n'
assert k in s[i:j], "aiperf output args in pd_server_atom.sh changed"
n = f"$(( conc * {mult} ))"
stop = f'--request-count "{n}"'
if window:
    n, stop = "$(( conc * 60 ))", '--benchmark-duration "${AIPERF_BENCHMARK_DURATION}"'
new = ('      "${AGENTX_VENV}/bin/aiperf" profile \\\n'
       '      --url "http://127.0.0.1:${ROUTER_PORT}" --endpoint /v1/chat/completions --endpoint-type chat --streaming \\\n'
       '      --model "${MODEL_PATH}" --tokenizer "${MODEL_PATH}" --tokenizer-trust-remote-code \\\n'
       f'      --concurrency "${{conc}}" --warmup-request-count 1 --warmup-concurrency 1 {stop} \\\n'
       f'      --num-dataset-entries "$(( 1 + {n} ))" --random-seed 42 \\\n'
       f'      --prefix-prompt-pool-size 1 --prefix-prompt-length {pre} \\\n'
       f'      --isl {isl} --isl-stddev 0 --osl {osl} --osl-stddev 0 --extra-inputs ignore_eos:true \\\n'
       '      --use-server-token-count --no-gpu-telemetry --stats-interval 30 \\\n'
       '      --failed-request-threshold "${AIPERF_FAILED_REQUEST_THRESHOLD}" --slice-duration "${AIPERF_SLICE_DURATION}" \\\n'
       + k)
open(p, "w").write(s[:i] + new + s[j + len(b):])
PY
fi
