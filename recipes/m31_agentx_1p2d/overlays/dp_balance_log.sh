# MengqingCao/ATOM#113: per-step DP load-balance log of every decode DP rank (atom/utils/dp_balance_log.py),
# one jsonl per process in the job's log dir (mounted at /run_logs in the containers). SPUR_JOB_ID is rank-env.sh's.
export ATOMESH_DECODE_ENV_ATOM_DP_BALANCE_LOG=/run_logs/slurm_job-${SPUR_JOB_ID}/dpb
