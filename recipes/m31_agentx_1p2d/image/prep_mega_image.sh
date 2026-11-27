#!/usr/bin/env bash
# MengqingCao/ATOM#108: the hour's image (aiter c2f06e01) with aiter a40489a9e's mega_moe/ (C2's MegaMoE:
# fused small-input a4w4 for the M3.1 EP4 layout, SwiGLU-OAI), as #73 built C2's image. c2f06e01..a40489a9e
# outside mega_moe/ only removes code mega_moe does not import (tensor_shim: c2f06e01 is a superset).
# Usage (dev host, holding the node's lease): prep_mega_image.sh NODE [MEGA_DIR]
#   MEGA_DIR: `git archive a40489a9e aiter/ops/flydsl/kernels/mega_moe` unpacked, on storage the node mounts.
set -euo pipefail
NODE=$1; HOST=${AGX_HOST_PREFIX-pit2-p03-}${NODE}
MEGA=${2:-/share_nfs/mengqing/code/tmp/m31-decode-dep4/108/aiter/aiter/ops/flydsl/kernels/mega_moe}
BASE=rocm/atom-dev:m31-agx-local-c2f06e01; TAR=/share_nfs/yajizhan/m31agx-images/c2f06e01/image-c2f06e01.tar
IMAGE=${IMAGE:-mengqing-108/agx:c2f06e01-mega-a40489a9e}
ssh -o BatchMode=yes "$HOST" bash -s -- "$BASE" "$TAR" "$MEGA" "$IMAGE" <<'REMOTE'
set -euo pipefail
BASE=$1 TAR=$2 MEGA=$3 IMAGE=$4 C=mengqing-108-prep
docker image inspect "$BASE" >/dev/null 2>&1 || docker load -i "$TAR"
docker rm -f $C >/dev/null 2>&1 || true
docker run -d --name $C --network none -v "$MEGA":/mega:ro --entrypoint sleep "$BASE" infinity >/dev/null
docker exec -u 0 $C bash -c 'R=/app/aiter-test/aiter/ops/flydsl/kernels; rm -rf $R/mega_moe && cp -r /mega $R/mega_moe'
# docker commit keeps the container's `sleep infinity`: restore the base's entrypoint and cmd.
EP=$(docker image inspect -f '{{json .Config.Entrypoint}}' "$BASE"); CMD=$(docker image inspect -f '{{json .Config.Cmd}}' "$BASE")
[[ $EP == null ]] && EP='[]'; [[ $CMD == null ]] && CMD='[]'
docker commit -c "ENTRYPOINT $EP" -c "CMD $CMD" -c "LABEL com.amd.m31agx.mega_moe=a40489a9e" $C "$IMAGE" >/dev/null
docker rm -f $C >/dev/null
docker run --rm --network none --entrypoint python3 "$IMAGE" -c \
  'import aiter.ops.flydsl.kernels.mega_moe.mega_moe_config as m; print("mega a4w4 fused:", hasattr(m, "A4W4_FUSED_LAYOUTS"))' 2>&1 | tail -1
docker image inspect -f '{{.Id}} {{index .Config.Labels "com.amd.m31agx.aiter.commit"}}' "$IMAGE"
REMOTE
