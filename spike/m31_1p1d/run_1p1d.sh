#!/usr/bin/env bash
# One 1P1D run on THIS node (whole node, gpool -g 8): P = PP4 x TP1 optimized prefill stack, D = ARM (A | B1 | B2),
# both on the run tree and the one image.  Run ON the node.  Usage: run_1p1d.sh TAG ARM agentx|gsm8k|fixed [CONC] [DURATION_S]
#   TREE  d53cff270 + #32's atom_prefill.patch + step_log.patch, committed locally (setup_tree.sh)
#   H     #34's harness copy; run_pd.sh snapshots TREE's HEAD (no stash) and sources cells/ov_spike.sh
#   TRACE=1 adds a 1-s decode profiler window 300 s into AgentX's profiling phase (the run then gives the trace only)
set -euo pipefail
S=/share_nfs/mengqing/code/tmp/m31-decode-dep4/81
export TREE=${TREE:-$S/../35/run-tree/atom} IMAGE=${IMAGE:-mengqing-dep4-35/all:e4858b08e}
export PREFILL_ARM=P OVERRIDES=$S/harness/cells/ov_spike.sh
exec bash "$S/harness/bin/run_pd.sh" "$1" "$2" "${4:-140}" "${5:-1200}" "$3"
