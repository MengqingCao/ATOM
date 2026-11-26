#!/usr/bin/env bash
# Unattended #81 queue on ONE whole node, run detached on the dev host:
#   setsid nohup bash driver.sh > $S/driver.log 2>&1 < /dev/null &
# acquire (-g 8 --wait) -> A/B1/B2 x c45/c64 x r1,r2 minus B2-c45-r1 (done), no smoke gate (ruling 7185214449),
# all KIND=fixed (aiperf: one shared 127k prefix + unique 1k tail = 128k context, OSL 2k, prefix caching on; ov_spike.sh). One line a run in $S/driver_status.txt. Stops on the first
# run without client data (a run past 1200 + WD_MARGIN s is stopped by the watchdog and counts as invalid), or on YIELD;
# always cleans its own containers and releases the lease (trap). A node whose GPUs a foreign (unleased) process
# uses is released with --force and skipped; the queue re-acquires elsewhere, at most MAXTRIES grants in all.
set -uo pipefail
S=/share_nfs/mengqing/code/tmp/m31-decode-dep4/81; D=$(dirname "$(readlink -f "$0")")
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
    $G release "$LEASE" --note "#81 driver: $done_n runs done; $why" && break; sleep 30
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
bad=; tries=0; MAXTRIES=8; PREF=${PREF:-g53 g27}  # most cache-heavy nodes fail the MemFree check below
# Take a whole node (never one in $bad), get the image on it, and check no one else is on its GPUs.
grant() {
  local nodes out who
  while :; do
    tries=$((tries + 1)); ((tries > MAXTRIES)) && { why="no clean node after $MAXTRIES grants (busy: ${bad:-none})"; exit 1; }
    # g27 and g53 pass the per-NUMA MemFree check below: ask for them (queueing) first, any free node after that.
    nodes=$(for n in $PREF; do [[ " $bad " == *" $n "* ]] || echo $n; done | paste -sd,); local to=3h; [[ -n $nodes ]] && to=1h
    [[ -n $nodes ]] || nodes=$($G status --json | python3 -c "import json,sys;b='$bad'.split();print(','.join(n for n in json.load(sys.stdin)['free'] if n not in b))")
    [[ -n $nodes ]] || { why="no pool node left outside ${bad}"; exit 1; }
    log "START acquiring a whole node (attempt $tries, nodes $nodes)"
    local arc stale=0
    while :; do  # an NFS "Stale file handle" (errno 116) in gpool is transient: retry it, at most 5 times
      out=$($G acquire -g 8 --node "$nodes" -p "MengqingCao/ATOM#81 spike: M3.1 1P1D A/B1/B2 aiperf shared 128k prefix + unique tail, prefix caching on, OSL 2k, c45/c64 x2, prefill 2 TB Mooncake CPU KV tier (~5h)" --wait --timeout $to 2>&1); arc=$?
      ((arc == 0)) && break
      echo "$out"
      [[ $arc == 2 && -n $PREF ]] && { log "no grant on preferred $nodes in $to: any free node next"; PREF=; tries=$((tries - 1)); continue 2; }
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
    # The prefill's 2 TB (1863 GiB) Store pins 930 GiB a NUMA node; the launcher compacts 930 + min(128, 930/16) = 988 GiB
    # of each into huge pages. Page cache does not count: reclaiming 1200 GiB of it through compaction overran 600 s on g40
    # (run 20261126032634), and dropping the node's shared cache would hurt other users. Need MemFree >= 988 on each node.
    mem=$(ssh "$HOST" 'for n in 0 1; do awk "/MemFree/{print int(\$4/1048576)}" /sys/devices/system/node/node$n/meminfo; done' | paste -sd' ')
    log "MEM $NODE free GiB per NUMA node: ${mem:-?} (Store compacts 988 each)"
    for m in ${mem:-0}; do ((m >= 988)) || { drop "host memory short for the 2 TB Store (MemFree GiB per NUMA: $mem, page cache not counted)"; continue 2; }; done
    who=$(foreign | paste -sd';'); [[ -z $who ]] && return 0
    drop "$who"
  done
}
# Our lease's GPUs are used by someone else: hand the node back (--force: that memory is provably not ours) and skip it.
drop() {
  log "BUSY $NODE $LEASE: gpus busy by foreign process: $1"
  clean
  $G release "$LEASE" --force --note "#81 driver: GPUs busy with a foreign unleased process ($1), nothing of ours ran; re-queuing elsewhere"
  : > $S/driver_lease; log "END released $LEASE on $NODE: foreign process"; bad="$bad $NODE"; LEASE=
}
log "CONFIG #81: #35 setup (NVFP4 KV + NVFP4 index on P and D, DSpark draft KV fp8, 2 TB Mooncake Store prefill tier, decode DP session affinity off, forced AL 4.04) with aiperf: one shared 130048-token prefix (pool size 1) + a unique 1024-token tail a request (128k context), OSL 2048 (stddev 0, ignore_eos), prefix caching ON on P and D (owner's ruling 7184230573); 1 warmup request at concurrency 1 computes the prefix once, then 5 x CONC measured requests; watchdog duration+${WD_MARGIN:-1800} s"
grant

yield() { $G mine | grep "$LEASE" | grep -q YIELD; }

# run SPEC: one batch.sh spec; sets rc, dir; writes the status line. Returns 1 if the run is invalid.
# A run whose GPUs are (or became) used by a foreign process is skipped, its dir moved to runs-aborted/, and redone on another node.
run() {
  local spec=$1 t0 t1 tag kind valid who late=
  yield && { why="lease YIELD before '$spec'"; exit 0; }
  set -- ${spec#TRACE=1:}; tag=$1 kind=$3
  # Watchdog: server start (~8 min) + 1-per-lane warmup + the window + drain must fit in duration + WD_MARGIN s, else the run is stuck.
  local lim=$((SECONDS + 3600)); [[ $kind == agentx || $kind == fixed ]] && lim=$((SECONDS + $5 + ${WD_MARGIN:-1800}))
  who=$(foreign | paste -sd';')
  [[ -n $who ]] && { log "RUN $tag skipped: gpus busy by foreign process: $who"; drop "$who"; grant; run "$spec"; return; }
  t0=$(date -u +%FT%TZ)
  # This run's dir is the one pd-$tag-* that was not there before it (the newest may be an earlier run's: 20261126035137).
  local before; before=$(ls -d $R/pd-$tag-* 2>/dev/null)
  newdir() { comm -13 <(sort <<<"$before") <(ls -d $R/pd-$tag-* 2>/dev/null | sort) | tail -1; }
  dir=
  ssh "$HOST" "bash $D/batch.sh '$spec'" &
  local p=$!
  for _ in $(seq 60); do  # learn the launcher job id so clean() can stop it if we are killed
    sleep 10; dir=$(newdir)
    [[ -n $dir && -f $dir/run.txt ]] && { JOB=$(grep -o 'job=[0-9]*' "$dir/run.txt" | cut -d= -f2); break; }
  done
  while kill -0 $p 2>/dev/null; do
    (( SECONDS > lim )) && { late=" watchdog-stopped"; log "RUN $tag past its watchdog limit: stopping it"; clean; kill $p 2>/dev/null; break; }
    sleep 30
  done
  wait $p; t1=$(date -u +%FT%TZ)
  dir=$(newdir); rc=; [[ -n $dir ]] && rc=$(cut -d= -f2 "$dir/local-rc" 2>/dev/null)
  [[ -z $JOB && -f $dir/run.txt ]] && JOB=$(grep -o 'job=[0-9]*' "$dir/run.txt" | cut -d= -f2)
  if [[ $spec == TRACE=1:* ]]; then  # a traced run gives the trace only
    [[ -n $(find "$dir" -path '*torch_traces*' -type f 2>/dev/null | head -1) ]] && valid=yes || valid=no
  elif [[ $kind == agentx || $kind == fixed ]]; then
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
  [[ $kind == fixed && -n $dir ]] && log "PREFIX $tag $(python3 $D/prefix_stats.py "$dir" 2>&1 | tail -1)"
  [[ -n $dir ]] && log "METRICS $tag decode batch mean/p90, TPOT p50/p90 ms, TTFT p50/p90 s, out tok/s/GPU = $(python3 -c "import json;d=json.load(open('$dir/report.json'));e=d['decode'];c=d['client'];print(e.get('batch_mean'), e.get('batch_p90'), c.get('tpot_p50_ms'), c.get('tpot_p90_ms'), c.get('ttft_p50_s'), c.get('ttft_p90_s'), c.get('out_tok_s_per_gpu'))" 2>&1 | tail -1)"
  [[ $valid == yes ]]
}

# No smoke gate (owner's ruling 7185214449): B2-c45-r1 (run 20261126062333) is done and valid; run the rest of the matrix.
for r in 1 2; do for c in 45 64; do for a in A B1 B2; do
  [[ $a$c$r == B2451 ]] && continue
  run "$a-c$c-r$r $a fixed $c 1200" || { why="$a-c$c-r$r invalid: diagnose"; exit 1; }
done; done; done
