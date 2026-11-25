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
  if [[ $kind == gsm8k ]]; then
    bash $D/poll_mtp.sh $S/acc/mtp_$tag.json & pids+=($!)
    [[ -n $tr ]] && { bash $D/gsm_profile.sh & pids+=($!); }
  fi
  TRACE=$tr EVAL_LIMIT=200 bash $D/run_1p1d.sh "$@" > $S/logs/$tag.log 2>&1
  echo "$(date -u +%T) $tag rc=$?"
  kill "${pids[@]}" 2>/dev/null; wait 2>/dev/null
done
