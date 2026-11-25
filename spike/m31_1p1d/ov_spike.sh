# OVERRIDES for bin/run_pd.sh (ATOM#35): DP arms record every decode step's per-rank scheduled
# tokens (run tree's ATOM_SPIKE_STEP_LOG) under <run>/steps/; TRACE=1 adds the decode profiler window.
if [[ "${ARM}" == B1 || "${ARM}" == B2 ]]; then
  mkdir -p "${LOG_ROOT}/steps" && chmod 777 "${LOG_ROOT}/steps"
  export ATOMESH_DECODE_ENV_ATOM_SPIKE_STEP_LOG="${LOG_ROOT}/steps/dp"
fi
if [[ -n "${TRACE:-}" ]]; then
  source "${H}/cells/ov_trace_decode.sh"
fi
