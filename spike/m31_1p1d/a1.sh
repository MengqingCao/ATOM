#!/usr/bin/env bash
# A1 rerun: #33's bench (aiter e4858b08e op_tests/multigpu_tests/bench_mega_moe_m31.py), baseline = the
# pristine nightly_202611240650 kernels (aiter 82b6b4f19, flydsl as 62cb8decf), tuned = e4858b08e's mega_moe/
# over them, at #33's points plus POINTS_EXTRA (JSON {name: [requests per rank]}, B2's measured rows).
# Run from the dev host holding the node's lease; uses GPUs 0-3 pinned by UUID.  Usage: a1.sh NODE OUT_TAG [PORT]
set -euo pipefail
NODE=$1; TAG=$2; PORT=${3:-29571}; HOST=pit2-p03-$NODE; C=mengqing-dep4-35-a1
S=/share_nfs/mengqing/code/tmp/m31-decode-dep4/35/a1; IMAGE=rocm/atom-dev:nightly_202611240650
U=$(ssh $HOST rocm-smi --showuniqueid | awk '/Unique ID/ {match($0,/GPU\[[0-9]+\]/); i=substr($0,RSTART+4,RLENGTH-5); if (i<4) {u=$0; sub(/.*0x/,"",u); o=o (o?",":"") "GPU-" u}} END{print o}')
ssh $HOST "docker rm -f $C >/dev/null 2>&1; docker run -d --init --name $C --network host --ipc host --device /dev/kfd --device /dev/dri \
  --group-add video --group-add render --cap-add SYS_PTRACE --security-opt seccomp=unconfined --shm-size 64g --ulimit memlock=-1:-1 \
  -v /share_nfs:/share_nfs -e ROCR_VISIBLE_DEVICES=$U -e PYTHONUNBUFFERED=1 -e AITER_LOG_LEVEL=WARNING --entrypoint sleep $IMAGE infinity >/dev/null"
ssh $HOST "docker exec -e PORT=$PORT -e TAG=$TAG -e POINTS_EXTRA='${POINTS_EXTRA:-}' $C bash $(dirname "$(readlink -f "$0")")/a1_inner.sh"
ssh $HOST "docker rm -f $C >/dev/null"
