#!/usr/bin/env bash
# On the node, in the background: 1P1D runs one after another.  Usage: batch.sh "TAG ARM KIND [CONC] [DUR]" ...
# TRACE=1 in a spec's env (prefix "TRACE=1:TAG ...") traces it. A gsm8k run keeps the decode server's last
# /debug/mtp_stats (real acceptance) in <S>/acc/mtp_<TAG>.json; a traced gsm8k run also takes a 2-s decode
# profiler window once 20,000 draft tokens are in (the launcher opens windows only in AgentX runs).
S=/share_nfs/mengqing/code/tmp/m31-decode-dep4/35; D=$(dirname "$(readlink -f "$0")")
mkdir -p $S/acc $S/logs
for spec; do
  tr=; [[ "$spec" == TRACE=1:* ]] && { tr=1; spec=${spec#TRACE=1:}; }
  set -- $spec; tag=$1 kind=$3; pids=()
  # The first shift (0, 100, ... 1800) whose launcher ports, default base + 0..31 and the eval phase's +1000 twin,
  # no TCP socket on the node uses (stays below the 32768+ ephemeral range); run_pd.sh / ov_spike.sh apply it.
  export ATOMESH_PORT_SHIFT=$(ss -Htan | python3 -c '
import sys
used = {int(l.split()[3].rsplit(":", 1)[1]) for l in sys.stdin if l.strip()}
bases = (8000, 8010, 8020, 6301, 29100, 29500, 29600, 29700, 29800, 26051, 26080, 26090)
print(next((s for s in range(0, 1801, 100)
            if not any(b + s + e + k in used for b in bases for e in (0, 1000) for k in range(32))), ""))')
  [[ -n $ATOMESH_PORT_SHIFT ]] || { echo "$(date -u +%T) $tag rc=2 no free launcher port block"; continue; }
  mp=$((9020 + ATOMESH_PORT_SHIFT))  # poll_mtp.sh / gsm_profile.sh: the eval-phase decode port
  if [[ $kind == gsm8k ]]; then
    PORT=$mp bash $D/poll_mtp.sh $S/acc/mtp_$tag.json & pids+=($!)
    [[ -n $tr ]] && { PORT=$mp bash $D/gsm_profile.sh & pids+=($!); }
  fi
  TRACE=$tr EVAL_LIMIT=200 bash $D/run_1p1d.sh "$@" > $S/logs/$tag.log 2>&1
  echo "$(date -u +%T) $tag rc=$?"
  kill "${pids[@]}" 2>/dev/null; wait 2>/dev/null
done
