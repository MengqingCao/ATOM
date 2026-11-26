#!/usr/bin/env bash
# Unattended #35 queue on ONE whole node, run detached on the dev host:
#   setsid nohup bash driver.sh > $S/driver.log 2>&1 < /dev/null &
# acquire (-g 8 --wait) -> smoke B2-c150-r1 -> AgentX matrix (A/B1/B2 x c140/c150 x r1,r2, 1200 s) -> traced runs
# -> GSM8K-200 (c32, 3 a arm; gsmB2r1 traced) -> A1 rerun (a1.sh, POINTS_EXTRA = B2's per-rank rows).
# One line a run in $S/driver_status.txt. Stops on the first AgentX run without client data (a run past
# duration + WD_MARGIN s, 1800 by default, or a gsm8k run past 3600 s, is stopped by the watchdog and counts as invalid), or on YIELD;
# always cleans its own containers and releases the lease (trap). A node whose GPUs a foreign (unleased) process
# uses is released with --force and skipped; the queue re-acquires elsewhere, at most MAXTRIES grants in all.
set -uo pipefail
S=/share_nfs/mengqing/code/tmp/m31-decode-dep4/35; D=$(dirname "$(readlink -f "$0")")
G=/share_nfs/yajizhan/gpu-pool/bin/gpool; ST=$S/driver_status.txt; R=$S/harness/runs
IMG=mengqing-dep4-35/all:e4858b08e
LEASE=; HOST=; NODE=; JOB=; done_n=0; why="queue finished"
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$ST"; }

clean() {  # this driver's containers only: the current run's launcher job, and a1.sh's container
  [[ -n $JOB ]] && ssh "$HOST" "bash $S/harness/bin/stop_pd.sh $JOB" 2>&1 | tail -1
  ssh "$HOST" "pkill -u \$(id -u) -f 'm31_1p1d/[b]atch.sh|m31_1p1d/[r]un_1p1d.sh|m31_1p1d/[p]oll_mtp.sh|m31_1p1d/[g]sm_profile.sh'; docker rm -f mengqing-dep4-35-a1" >/dev/null 2>&1
  JOB=
}
finish() {
  trap - EXIT INT TERM HUP
  kill $TOUCHER 2>/dev/null
  [[ -z $LEASE ]] && { log "END no lease: $why"; exit; }
  clean
  for i in 1 2 3 4 5 6 7 8 9 10; do  # release refuses (rc 3) while GPU memory drains
    $G release "$LEASE" --note "#35 driver: $done_n runs done; $why" && break; sleep 30
  done
  : > $S/driver_lease; log "END released $LEASE on $NODE: $why"
}
# CPU-only phases (image prep, Store compaction, reports) leave the GPUs empty: reset the idle clock every 10 min.
( while sleep 600; do l=$(cat $S/driver_lease 2>/dev/null); [[ -n $l ]] && $G touch "$l" >/dev/null 2>&1; done ) &
TOUCHER=$!
trap finish EXIT; trap 'why="driver killed"; exit 1' INT TERM HUP

# Foreign GPU users on $HOST, one line each ("container NAME (ID)" or "pid P uid U"); empty = clean.
# Ours: mengqing-dep4-35-* containers, atomesh-*-<job>-<rank> containers of our launcher jobs, our uid on the host.
foreign() {
  local jobs; jobs=$(cat $R/*/run.txt $S/harness/runs-aborted/*/run.txt 2>/dev/null | grep -o 'job=[0-9]*' | cut -d= -f2 | paste -sd'|')
  ssh "$HOST" 'for p in $(rocm-smi --showpids --json 2>/dev/null | grep -o "PID[0-9]*" | cut -c4-); do
    c=$(grep -o "[0-9a-f]\{64\}" /proc/$p/cgroup 2>/dev/null | head -1)
    if [ -n "$c" ]; then echo "container $(docker inspect -f "{{.Name}}" $c 2>/dev/null | tr -d /) (${c:0:12})"
    else u=$(ps -o uid= -p $p | tr -d " "); [ -n "$u" ] && [ "$u" != "$(id -u)" ] && echo "pid $p uid $u"; fi
  done' | sort -u | grep -vE "^container (mengqing-dep4-35-|atomesh-.*-(${jobs:-NONE})-[0-9]+)"
}
bad=; tries=0; MAXTRIES=4
# Take a whole node (never one in $bad), get the image on it, and check no one else is on its GPUs.
grant() {
  local nodes out who
  while :; do
    tries=$((tries + 1)); ((tries > MAXTRIES)) && { why="no clean node after $MAXTRIES grants (busy: ${bad:-none})"; exit 1; }
    nodes=$($G status --json | python3 -c "import json,sys;b='$bad'.split();print(','.join(n for n in json.load(sys.stdin)['free'] if n not in b))")
    [[ -n $nodes ]] || { why="no pool node left outside ${bad}"; exit 1; }
    log "START acquiring a whole node (attempt $tries, nodes $nodes)"
    local arc stale=0
    while :; do  # an NFS "Stale file handle" (errno 116) in gpool is transient: retry it, at most 5 times
      out=$($G acquire -g 8 --node "$nodes" -p "MengqingCao/ATOM#35 spike: M3.1 1P1D A/B1/B2 AgentX c140/c150 + GSM8K-200 + A1 rerun, prefill 2 TB Mooncake CPU KV tier (~10h)" --wait --timeout 3h 2>&1); arc=$?
      ((arc == 0)) && break
      echo "$out"
      grep -qiE 'stale file handle|errno 116' <<<"$out" && ((++stale <= 5)) || { why="acquire rc=$arc (timed out or refused)"; exit 1; }
      log "acquire hit a stale NFS file handle (rc=$arc): retry $stale/5 in 60 s"; sleep 60
    done
    echo "$out"
    LEASE=$(grep -o 'L[0-9a-f]\{6\}' <<<"$out" | head -1); HOST=$(grep -o 'pit2-p03-g[0-9]*' <<<"$out" | head -1); NODE=${HOST#pit2-p03-}
    [[ -n $LEASE && -n $NODE ]] || { why="cannot parse grant: $out"; exit 1; }
    echo "$LEASE" > $S/driver_lease; log "GRANTED $LEASE $HOST"
    if ! ssh "$HOST" docker image inspect "$IMG" >/dev/null 2>&1; then
      $G touch "$LEASE"; log "prep_image.sh $NODE"; bash "$D/prep_image.sh" "$NODE" || { why="prep_image failed"; exit 1; }
    fi
    # The prefill's 2 TB (1863 GiB) Store pins 930 GiB a NUMA node (+128 GiB launcher reserve): MemFree + page cache must hold it.
    mem=$(ssh "$HOST" 'for n in 0 1; do awk "/MemFree|FilePages/{s+=\$4} END{print int(s/1048576)}" /sys/devices/system/node/node$n/meminfo; done' | paste -sd' ')
    log "MEM $NODE free+cache GiB per NUMA node: ${mem:-?} (Store needs 1058 each)"
    for m in ${mem:-0}; do ((m >= 1058)) || { drop "host memory short for the 2 TB Store (free+cache GiB per NUMA: $mem)"; continue 2; }; done
    who=$(foreign | paste -sd';'); [[ -z $who ]] && return 0
    drop "$who"
  done
}
# Our lease's GPUs are used by someone else: hand the node back (--force: that memory is provably not ours) and skip it.
drop() {
  log "BUSY $NODE $LEASE: gpus busy by foreign process: $1"
  clean
  $G release "$LEASE" --force --note "#35 driver: GPUs busy with a foreign unleased process ($1), nothing of ours ran; re-queuing elsewhere"
  : > $S/driver_lease; log "END released $LEASE on $NODE: foreign process"; bad="$bad $NODE"; LEASE=
}
log "CONFIG #35 ruling 7167528921, all arms: prefill KV CPU tier = Mooncake Store 2 TB = 1863 GiB host DRAM (MOONCAKE_STORE=1, owners 0:465:rdma4;0:465:rdma5;1:465:rdma6;1:465:rdma7, 930 GiB a NUMA node under the ~1017 GiB huge-page ceiling; ATOM_KV_OFFLOAD=mooncake_store via launcher; decode no tier); ruling 7164687062: decode DP session affinity off (no dp_sticky/--dp-aware/X-Session-ID; router policy random), AgentX warmup_requests_per_lane=1 (was 10), window 1200 s, warmup_grace_period 1800 s; watchdog agentx duration+${WD_MARGIN:-1800} s, gsm8k 3600 s"
grant

yield() { $G mine | grep "$LEASE" | grep -q YIELD; }

# run SPEC: one batch.sh spec; sets rc, dir; writes the status line. Returns 1 if the run is invalid.
# A run whose GPUs are (or became) used by a foreign process is skipped, its dir moved to runs-aborted/, and redone on another node.
run() {
  local spec=$1 t0 t1 tag kind valid who late=
  yield && { why="lease YIELD before '$spec'"; exit 0; }
  set -- ${spec#TRACE=1:}; tag=$1 kind=$3
  # Watchdog: server start (~8 min) + 1-per-lane warmup + the window + drain must fit in duration + WD_MARGIN s, else the run is stuck.
  local lim=$((SECONDS + 3600)); [[ $kind == agentx ]] && lim=$((SECONDS + $5 + ${WD_MARGIN:-1800}))
  who=$(foreign | paste -sd';')
  [[ -n $who ]] && { log "RUN $tag skipped: gpus busy by foreign process: $who"; drop "$who"; grant; run "$spec"; return; }
  t0=$(date -u +%FT%TZ)
  ssh "$HOST" "bash $D/batch.sh '$spec'" &
  local p=$!
  for _ in $(seq 60); do  # learn the launcher job id so clean() can stop it if we are killed
    sleep 10; dir=$(ls -dt $R/pd-$tag-* 2>/dev/null | head -1)
    [[ -n $dir && -f $dir/run.txt ]] && { JOB=$(grep -o 'job=[0-9]*' "$dir/run.txt" | cut -d= -f2); break; }
  done
  while kill -0 $p 2>/dev/null; do
    (( SECONDS > lim )) && { late=" watchdog-stopped"; log "RUN $tag past its watchdog limit: stopping it"; clean; kill $p 2>/dev/null; break; }
    sleep 30
  done
  wait $p; t1=$(date -u +%FT%TZ)
  dir=$(ls -dt $R/pd-$tag-* 2>/dev/null | head -1); rc=$(cat "$dir/local-rc" 2>/dev/null | cut -d= -f2)
  if [[ $spec == TRACE=1:* ]]; then  # a traced run gives the trace only
    [[ -n $(find "$dir" -path '*torch_traces*' -type f 2>/dev/null | head -1) ]] && valid=yes || valid=no
  elif [[ $kind == agentx ]]; then
    python3 -c "import json,sys;sys.exit(not json.load(open('$dir/report.json'))['client'])" 2>/dev/null && valid=yes || valid=no
  else
    [[ $rc == 0 && -s $S/acc/mtp_$tag.json ]] && valid=yes || valid=no
  fi
  [[ -n $late ]] && valid=no
  clean
  if [[ -z $late && $valid == no ]] && who=$(foreign | paste -sd';') && [[ -n $who ]]; then  # not our failure: someone took the GPUs
    [[ -n $dir ]] && mv "$dir" $S/harness/runs-aborted/
    log "RUN $tag skipped: gpus busy by foreign process: $who (rc=${rc:-?} start=$t0 end=$t1, dir moved to runs-aborted)"
    drop "$who"; grant; run "$spec"; return
  fi
  done_n=$((done_n + 1))
  log "RUN $tag rc=${rc:-?}$late valid=$valid start=$t0 end=$t1 dir=${dir:-none}"
  [[ $valid == yes ]]
}

run 'B2-c150-r1 B2 agentx 150 1200' || { why="smoke B2-c150-r1 invalid: diagnose"; exit 1; }
# The smoke must show the CPU tier in use and a higher prefix hit than the no-tier smoke (41.46 %, pd-B2-c150-r1-20261125141241).
hit=$(python3 -c "import json;p=json.load(open('$dir/report.json'))['prefill'];print(p.get('offload_hit_pct'),p.get('total_hit_pct'))" 2>/dev/null)
log "SMOKE prefill hit: offload(CPU tier) total = ${hit:-none} %"
python3 -c "import sys;o,t=map(float,'$hit'.split());sys.exit(not (o > 0 and t > 41.46))" 2>/dev/null \
  || { why="smoke B2-c150-r1: CPU tier not in use or prefix hit not above 41.46 % ($hit): diagnose"; exit 1; }
for r in 1 2; do for c in 140 150; do for a in A B1 B2; do
  [[ $a$c$r == B21501 ]] && continue
  run "$a-c$c-r$r $a agentx $c 1200" || { why="$a-c$c-r$r invalid: diagnose"; exit 1; }
done; done; done
# Traced AgentX (600 s: the 1-s window opens 300 s in): B2 at both points, plus A and B1 at c150 for the top-two breakdown.
for s in 'tB2-c140 B2 agentx 140 600' 'tB2-c150 B2 agentx 150 600' 'tB1-c150 B1 agentx 150 600' 'tA-c150 A agentx 150 600'; do
  run "TRACE=1:$s"
done
for r in 1 2 3; do for a in A B1 B2; do
  if [[ $a$r == B21 ]]; then run "TRACE=1:gsm${a}r$r $a gsm8k 32"; else run "gsm${a}r$r $a gsm8k 32"; fi
done; done

yield && { why="lease YIELD before A1"; exit 0; }
python3 "$D/report.py" $R/pd-B2-c1[45]0-r* > $S/report_b2.txt 2>&1
export POINTS_EXTRA=$(python3 - "$S/report_b2.txt" <<'PY'
import json, sys
d, _ = json.JSONDecoder().raw_decode(open(sys.argv[1]).read())
pts = {}
for name, row in d.items():
    for k, st in enumerate(row.get("steps", [])):
        for q in ("p50", "p90"):
            v = st[f"req_per_rank_{q}"]
            if v not in pts.values():
                pts[f"B2{name.split('@')[1]}r{k + 1}{q}"] = v
print(json.dumps(pts))
PY
)
log "A1 POINTS_EXTRA=$POINTS_EXTRA"
t0=$(date -u +%FT%TZ); bash "$D/a1.sh" "$NODE" rerun > $S/a1/rerun.log 2>&1; rc=$?
done_n=$((done_n + 1)); log "RUN a1-rerun rc=$rc valid=$([[ $rc == 0 ]] && echo yes || echo no) start=$t0 end=$(date -u +%FT%TZ) dir=$S/a1/rerun"
