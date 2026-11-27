#!/usr/bin/env bash
# ATOM#104 spike: B2 (MegaMoE a8w4) vs C2 (ATOM_MEGA_A4W4=1) on ONE node, SAME image/tree (#73's: aiter a40489a9e,
# run tree 73), #81 r2 settings (fixed cell, 600 s aiperf window), 2 runs each at c45 and c64 (order B2,C2,C2,B2),
# then one 1-s decode torch-profiler window (TRACE=1, 300 s into profiling) per arm and concurrency.
# Adapted from 73/launch/c2_parallel.sh: the A4W4 env goes only to specs whose tag has -C2-; finished run dirs move
# to 104/runs.  setsid nohup bash driver.sh > 104/c/driver.log 2>&1 < /dev/null &
set -uo pipefail
S=/share_nfs/mengqing/code/tmp/m31-decode-dep4/81
G=/share_nfs/yajizhan/gpu-pool/bin/gpool; C=$S/../104/c; W=$S/../104/runs; ST=$C/driver_status.txt; R=$S/harness/runs
D=/share_nfs/mengqing/code/ATOM-worktrees/81/spike/m31_1p1d  # #81's batch.sh / run_1p1d.sh / prefix_stats.py
IMG=mengqing-dep4-73/all:a40489a9e
TREE=$S/../73/run-tree/atom IMAGE=$IMG  # passed to batch.sh on the node; C2 specs add ATOMESH_DECODE_ENV_ATOM_MEGA_A4W4=1
[[ -e $S/driver_stop ]] && { echo "$S/driver_stop present: #81's batch.sh would start nothing"; exit 1; }
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
    $G release "$LEASE" --note "#104 driver: $done_n runs done; $why" && break; sleep 30
  done
  : > $C/driver_lease; log "END released $LEASE on $NODE: $why"
}
# CPU-only phases (image prep, Store compaction, reports) leave the GPUs empty: reset the idle clock every 10 min.
( while sleep 600; do l=$(cat $C/driver_lease 2>/dev/null); [[ -n $l ]] && $G touch "$l" >/dev/null 2>&1; done ) &
TOUCHER=$!
trap finish EXIT; trap 'why="driver killed"; exit 1' INT TERM HUP

# Foreign GPU users on $HOST, one line each ("container NAME (ID)" or "pid P uid U"); empty = clean.
# Ours: mengqing-dep4-35-* containers, atomesh-*-<job>-<rank> containers of our launcher jobs, our uid on the host.
foreign() {
  local jobs; jobs=$(cat $R/*/run.txt $W/*/run.txt $S/harness/runs-aborted/*/run.txt 2>/dev/null | grep -o 'job=[0-9]*' | cut -d= -f2 | paste -sd'|')
  ssh "$HOST" 'for p in $(rocm-smi --showpids --json 2>/dev/null | grep -o "PID[0-9]*" | cut -c4-); do
    c=$(grep -o "[0-9a-f]\{64\}" /proc/$p/cgroup 2>/dev/null | head -1)
    if [ -n "$c" ]; then echo "container $(docker inspect -f "{{.Name}}" $c 2>/dev/null | tr -d /) (${c:0:12})"
    else u=$(ps -o uid= -p $p | tr -d " "); [ -n "$u" ] && [ "$u" != "$(id -u)" ] && echo "pid $p uid $u"; fi
  done' | sort -u | grep -vE "^container (mengqing-dep4-35-|atomesh-.*-(${jobs:-NONE})-[0-9]+)"
}
bad=; tries=0; MAXTRIES=8; PREF=${PREF:-g48}  # one node for the whole A/B1/B2 comparison
# Take a whole node (never one in $bad), get the image on it, and check no one else is on its GPUs.
grant() {
  local nodes out who
  while :; do
    tries=$((tries + 1)); ((tries > MAXTRIES)) && { why="no clean node after $MAXTRIES grants (busy: ${bad:-none})"; exit 1; }
    # g27 and g53 pass the per-NUMA MemFree check below: ask for them (queueing) first, any free node after that.
    nodes=$(for n in $PREF; do [[ " $bad " == *" $n "* ]] || echo $n; done | paste -sd,); local to=3h
    [[ -n $nodes ]] || { why="no pool node left outside ${bad}"; exit 1; }
    log "START acquiring a whole node (attempt $tries, nodes $nodes)"
    local arc stale=0
    while :; do  # an NFS "Stale file handle" (errno 116) in gpool is transient: retry it, at most 5 times
      out=$($G acquire -g 8 --node "$nodes" -p "MengqingCao/ATOM#104 spike: M3.1 1P1D B2 vs C2 (MegaMoE a8w4 vs a4w4) same node+image, c45/c64 x2 + 4 decode traces (~6h)" --wait --timeout $to 2>&1); arc=$?
      ((arc == 0)) && break
      echo "$out"
      grep -qiE 'stale file handle|errno 116' <<<"$out" && ((++stale <= 5)) || { why="acquire rc=$arc (timed out or refused)"; exit 1; }
      log "acquire hit a stale NFS file handle (rc=$arc): retry $stale/5 in 60 s"; sleep 60
    done
    echo "$out"
    LEASE=$(grep -o 'L[0-9a-f]\{6\}' <<<"$out" | head -1); HOST=$(grep -o 'pit2-p03-g[0-9]*' <<<"$out" | head -1); NODE=${HOST#pit2-p03-}
    [[ -n $LEASE && -n $NODE ]] || { why="cannot parse grant: $out"; exit 1; }
    echo "$LEASE" > $C/driver_lease; log "GRANTED $LEASE $HOST"
    if ! ssh "$HOST" docker image inspect "$IMG" >/dev/null 2>&1; then
      $G touch "$LEASE"; log "prep_image.sh $NODE"; bash "$S/../73/launch/prep_image.sh" "$NODE" || { why="prep_image failed"; exit 1; }
    fi
    # The prefill's 2 TB (1863 GiB) Store pins 930 GiB a NUMA node; the launcher compacts 930 + min(128, 930/16) = 988 GiB
    # of each into huge pages. Page cache does not count: reclaiming 1200 GiB of it through compaction overran 600 s on g40
    # (run 20261126032634), and dropping the node's shared cache would hurt other users. Need MemFree >= 988 on each node.
    mem=$(ssh "$HOST" 'for n in 0 1; do awk "/MemFree/{print int(\$4/1048576)}" /sys/devices/system/node/node$n/meminfo; done' | paste -sd' ')
    log "MEM $NODE free GiB per NUMA node: ${mem:-?} (Store compacts 494 each)"
    for m in ${mem:-0}; do ((m >= 494)) || { drop "host memory short for the 930 GiB Store (MemFree GiB per NUMA: $mem, page cache not counted)"; continue 2; }; done
    who=$(foreign | paste -sd';'); [[ -z $who ]] && return 0
    drop "$who"
  done
}
# Our lease's GPUs are used by someone else: hand the node back (--force: that memory is provably not ours) and skip it.
drop() {
  log "BUSY $NODE $LEASE: gpus busy by foreign process: $1"
  clean
  $G release "$LEASE" --force --note "#104 driver: GPUs busy with a foreign unleased process ($1), nothing of ours ran; re-queuing elsewhere"
  : > $C/driver_lease; log "END released $LEASE on $NODE: foreign process"; bad="$bad $NODE"; LEASE=
}
log "CONFIG #104 B2 vs C2 (C2 = ATOM_MEGA_A4W4=1; tree $TREE, image $IMG, one node) on #81's r2 settings: #35 setup (NVFP4 KV + NVFP4 index on P and D, DSpark draft KV fp8, 2 TB Mooncake Store prefill tier, decode DP session affinity off, forced AL 4.04) with aiperf: one shared 130048-token prefix (pool size 1) + a unique 1024-token tail a request (128k context), OSL 2048 (stddev 0, ignore_eos), prefix caching ON on P and D (owner's ruling 7184230573); 1 warmup request at concurrency 1 computes the prefix once, then 5 x CONC measured requests; watchdog 75 min to aiperf start, then duration+${WD_MARGIN:-1800} s"
grant
ssh "$HOST" "docker image inspect $IMG" >/dev/null 2>&1 || { log "PREP $IMG on $NODE"; $G touch "$LEASE" >/dev/null; bash $S/../73/launch/prep_image.sh "$NODE" >> $C/prep.log 2>&1 || { why="prep_image failed on $NODE"; exit 1; }; log "PREP done"; }

yield() { $G mine | grep "$LEASE" | grep -q YIELD; }

# run SPEC: one batch.sh spec; sets rc, dir; writes the status line. Returns 1 if the run is invalid.
# A run whose GPUs are (or became) used by a foreign process is skipped, its dir moved to runs-aborted/, and redone on another node.
run() {
  local spec=$1 t0 t1 tag kind valid who late=
  yield && { why="lease YIELD before '$spec'"; exit 0; }
  set -- ${spec#TRACE=1:}; tag=$1 kind=$3
  # Watchdog: an aiperf run gets 75 min to start (NFS weight load took 36 min on g52, run 20261126082103), then
  # duration + WD_MARGIN s from aiperf's start (.aiperf_pid) for warmup + the window + drain, else the run is stuck.
  local lim=$((SECONDS + 3600)) aip=; [[ $kind == agentx || $kind == fixed ]] && { lim=$((SECONDS + 4500)); aip=1; }
  who=$(foreign | paste -sd';')
  [[ -n $who ]] && { log "RUN $tag skipped: gpus busy by foreign process: $who"; drop "$who"; grant; run "$spec"; return; }
  t0=$(date -u +%FT%TZ)
  # This run's dir is the one pd-$tag-* that was not there before it (the newest may be an earlier run's: 20261126035137).
  local before; before=$(ls -d $R/pd-$tag-* 2>/dev/null)
  newdir() { comm -13 <(sort <<<"$before") <(ls -d $R/pd-$tag-* 2>/dev/null | sort) | tail -1; }
  dir=
  local a4=; [[ $tag == *-C2-* ]] && a4=ATOMESH_DECODE_ENV_ATOM_MEGA_A4W4=1
  ssh "$HOST" "TREE=$TREE IMAGE=$IMAGE $a4 bash $D/batch.sh '$spec'" &
  local p=$!
  for _ in $(seq 60); do  # learn the launcher job id so clean() can stop it if we are killed
    sleep 10; dir=$(newdir)
    [[ -n $dir && -f $dir/run.txt ]] && { JOB=$(grep -o 'job=[0-9]*' "$dir/run.txt" | cut -d= -f2); break; }
  done
  while kill -0 $p 2>/dev/null; do
    [[ -n $aip && -z $dir ]] && dir=$(newdir)
    [[ -n $aip && -n $dir ]] && compgen -G "$dir/slurm_job-*/benchmark_results/*/.aiperf_pid" >/dev/null &&
      { aip=; lim=$((SECONDS + $5 + ${WD_MARGIN:-1800})); log "RUN $tag aiperf started: watchdog in $(($5 + ${WD_MARGIN:-1800})) s"; }
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
  [[ -n $dir ]] && mv "$dir" $W/ && log "MOVED $(basename $dir) to $W"
  [[ $valid == yes ]]
}

for spec in x104-B2-c45-1 x104-C2-c45-1 x104-C2-c45-2 x104-B2-c45-2; do
  run "$spec B2 fixed 45 600 window" || { why="$spec invalid: diagnose"; exit 1; }
done
for spec in x104-B2-c64-1 x104-C2-c64-1 x104-C2-c64-2 x104-B2-c64-2; do
  run "$spec B2 fixed 64 600 window" || { why="$spec invalid: diagnose"; exit 1; }
done
for spec in "x104-B2-c45-t B2 fixed 45" "x104-C2-c45-t B2 fixed 45" "x104-B2-c64-t B2 fixed 64" "x104-C2-c64-t B2 fixed 64"; do
  run "TRACE=1:$spec 600 window" || { why="${spec%% *} trace invalid: diagnose"; exit 1; }
done
for d in $(ls -d $W/pd-x104-* 2>/dev/null); do log "MEGA $(basename $d) $(grep -h -m1 'MEGA-CAPACITY' $d/slurm_job-*/logs/decode-rank-0.log 2>/dev/null | grep -o 'capacity=.*')"; done
