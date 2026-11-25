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
