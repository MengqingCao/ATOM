#!/usr/bin/env bash
# On the node during a 1P1D run: keep the decode server's last /debug/mtp_stats (PORT, default 9020: the harness decode port) in OUT.
# Exits when the decode server stays down for 5 minutes after having answered.  Usage: poll_mtp.sh OUT
OUT=$1; seen=0; miss=0
while :; do
  if r=$(curl -s -m 10 http://127.0.0.1:${PORT:-9020}/debug/mtp_stats) && [[ "$r" == *total_draft_tokens* ]]; then
    printf '%s\n' "$r" > "$OUT.tmp" && mv "$OUT.tmp" "$OUT"; seen=1; miss=0
  elif ((seen)); then miss=$((miss+1)); ((miss > 15)) && exit 0
  fi
  sleep 20
done
