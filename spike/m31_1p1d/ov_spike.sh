# OVERRIDES for bin/run_pd.sh (ATOM#35): DP arms record every decode step's per-rank scheduled
# tokens (run tree's ATOM_SPIKE_STEP_LOG) under <run>/slurm_job-*/steps/; TRACE=1 adds the decode profiler window.
if [[ "${ARM}" == B1 || "${ARM}" == B2 ]]; then
  # Relative: the decode container mounts only <run>/slurm_job-*/ (as RUN_DIR), so the files land there.
  export ATOMESH_DECODE_ENV_ATOM_SPIKE_STEP_LOG="steps/dp"
fi
if [[ -n "${TRACE:-}" ]]; then
  source "${H}/cells/ov_trace_decode.sh"
fi
