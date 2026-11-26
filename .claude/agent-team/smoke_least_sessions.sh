#!/bin/bash
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
# Live DP=2 smoke for least-session placement: four new sessions one after another, then a
# second turn of each, checking atom:dp_sessions and the affinity counters on /metrics.
# Run inside the container from anywhere in the tree, with only the two leased GPUs visible
# (ROCR_VISIBLE_DEVICES=GPU-<hex>,GPU-<hex>). Exit 0 is a pass.
set -uo pipefail
TREE=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-python3}
MODEL=${MODEL:-/share_nfs/models/Qwen/Qwen3-0.6B}
PORT=${PORT:-${GPOOL_PORT_BASE:-8000}}
LOG=${LOG:-/tmp/smoke_least_sessions.$$.log}
URL=http://127.0.0.1:$PORT
export PYTHONPATH=$TREE${PYTHONPATH:+:$PYTHONPATH}

ATOM_DIR=$("$PY" -c "import atom, os; print(os.path.dirname(os.path.abspath(atom.__file__)))")
echo "atom: $ATOM_DIR"
[ "$ATOM_DIR" = "$TREE/atom" ] || { echo "FAIL: atom does not resolve inside $TREE"; exit 1; }

echo "server log: $LOG"
AITER_LOG_LEVEL=WARNING ATOM_DP_SESSION_AFFINITY=1 ATOM_DP_LEAST_SESSIONS=1 ATOM_DP_MASTER_PORT=$((PORT + 2)) \
    setsid "$PY" -m atom.entrypoints.openai_server --model "$MODEL" -dp 2 \
    --server-port "$PORT" --port $((PORT + 1)) >"$LOG" 2>&1 &
SERVER=$!
# setsid makes the server a group leader, so this also stops its engine processes.
trap 'kill -- -$SERVER 2>/dev/null; wait $SERVER 2>/dev/null' EXIT

for _ in $(seq 600); do
    curl -sf "$URL/health" >/dev/null && break
    kill -0 $SERVER 2>/dev/null || { echo "FAIL: server exited"; tail -50 "$LOG"; exit 1; }
    sleep 1
done
curl -sf "$URL/health" >/dev/null || { echo "FAIL: server not healthy in 600 s"; exit 1; }

fail=0
turn() { # session, turn number
    curl -sf "$URL/v1/chat/completions" -H 'Content-Type: application/json' -H "X-Session-ID: $1" \
        -d "{\"model\": \"$MODEL\", \"max_tokens\": 8, \"messages\": [{\"role\": \"user\", \"content\": \"Turn $2 of $1: say hi.\"}]}" \
        >/dev/null || { echo "FAIL: $1 turn $2 request failed"; fail=1; }
}
metric() { # series with labels as printed, e.g. 'atom:dp_sessions{rank="0"}'
    awk -v s="$1" '$1 == s {printf "%d", $2; found = 1} END {if (!found) printf "missing"}' <<<"$METRICS"
}
expect() { # series, wanted value
    local got
    got=$(metric "$1")
    echo "  $1 = $got (want $2)"
    [ "$got" = "$2" ] || fail=1
}
check() { # round name
    sleep 3  # the exporter refreshes every ATOM_METRICS_UPDATE_INTERVAL_S (1 s)
    METRICS=$(curl -sf "$URL/metrics")
    echo "after $1:"
    expect 'atom:dp_sessions{rank="0"}' 2
    expect 'atom:dp_sessions{rank="1"}' 2
    expect 'atom:dp_affinity_owner_hit_total' "$2"
    expect 'atom:dp_affinity_spill_total' 0
}

# Under the default placement on an idle server, the hash tie-break puts all four on rank 1.
SESSIONS="smoke-s3 smoke-s4 smoke-s5 smoke-s6"
for s in $SESSIONS; do turn "$s" 1; done
check "first turns" 0
for s in $SESSIONS; do turn "$s" 2; done
check "second turns" 4

[ $fail = 0 ] && echo PASS || echo FAIL
exit $fail
