#!/usr/bin/env bash
# Inside a1.sh's container.
set -euo pipefail
S=/share_nfs/mengqing/code/tmp/m31-decode-dep4/35/a1; W=$S/aiter; D=$S/$TAG; mkdir -p $D
CAP=/share_nfs/mengqing/code/tmp/m31-decode-dep4/15/runs/capture/parts
A=$(python -c "import aiter,os;print(os.path.dirname(os.path.dirname(aiter.__file__)))" | tail -1)
cp $W/op_tests/multigpu_tests/{test_mega_moe_v2.py,bench_mega_moe_m31.py} $A/op_tests/multigpu_tests/
cd $A/op_tests/multigpu_tests
P=$(python - <<'PY'
import json, os, re, ast
src = open("bench_mega_moe_m31.py").read()
pts = ast.literal_eval(re.search(r"POINTS = (\{.*?\n\})", src, re.S).group(1))
pts.update(json.loads(os.environ.get("POINTS_EXTRA") or "{}"))
print(json.dumps(pts))
PY
)
echo "points $P" | tee $D/points.txt
TR="torchrun --nproc-per-node 4 --master-port $PORT"
$TR bench_mega_moe_m31.py --capture $CAP --act silu --points "$P" --out $D/base.json > $D/base.log 2>&1; echo "base rc=$?"
cp $W/aiter/ops/flydsl/kernels/mega_moe/*.py $A/aiter/ops/flydsl/kernels/mega_moe/
$TR bench_mega_moe_m31.py --capture $CAP --act swigluoai --routed --points "$P" --baseline $D/base.json --out $D/tuned.json > $D/tuned.log 2>&1; echo "tuned rc=$?"
