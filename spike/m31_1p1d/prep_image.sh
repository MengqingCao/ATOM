#!/usr/bin/env bash
# One image for every arm, P and D alike: rocm/atom-dev:nightly_202611240650 (aiter 82b6b4f19)
# + the optimized prefill stack's aiter side (merge-e2e/aiter/prep_aiter_merged.sh, TUNED=integ-layer/tuned_cfg)
# + aiter fork main e4858b08e's fused_moe.py and mega_moe/ (#33's tuned fused small-input route).
# #34's prep_image.sh with REV=e4858b08e.  Run from the dev host holding the node's lease.  Usage: prep_image.sh NODE
set -euo pipefail
NODE=$1; HOST=pit2-p03-${NODE}; REV=e4858b08e
A=/share_nfs/mengqing/code/tmp/m31-decode-dep4/34/aiter/$REV   # git archive of aiter $REV: aiter/fused_moe.py, aiter/ops/flydsl/kernels/mega_moe/
BASE_IMAGE=rocm/atom-dev:nightly_202611240650
IMAGE=${IMAGE:-mengqing-dep4-35/all:$REV}
T=/share_nfs/yajizhan/tmp/m31-tp1-tune
C=mengqing-dep4-35-prep
ssh "$HOST" "docker rm -f $C >/dev/null 2>&1; docker run -d --name $C --network host --ipc host \
  --device /dev/kfd --device /dev/dri --group-add video -v /share_nfs:/share_nfs \
  --entrypoint sleep $BASE_IMAGE infinity >/dev/null"
ssh "$HOST" "docker exec $C bash -c 'R=\$(python -c \"import aiter,os;print(os.path.dirname(os.path.dirname(aiter.__file__)))\" | tail -1); \
  AITER_ROOT=\$R TUNED=$T/integ-layer/tuned_cfg bash $T/merge-e2e/aiter/prep_aiter_merged.sh && cp -r $A/aiter/. \$R/aiter/ && rm -rf /tmp/aiter_configs*'" 2>&1 | tail -2
ssh "$HOST" "docker commit $C $IMAGE >/dev/null && docker rm -f $C >/dev/null"
printf 'FROM %s\nENTRYPOINT []\nCMD ["/bin/bash"]\n' "$IMAGE" | ssh "$HOST" "docker build -q -t $IMAGE -"
