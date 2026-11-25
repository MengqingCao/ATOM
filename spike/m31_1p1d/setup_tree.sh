#!/usr/bin/env bash
# The run tree P and D share, and the harness: run from the repository root on the dev host.
set -euo pipefail
S=/share_nfs/mengqing/code/tmp/m31-decode-dep4; D=$(dirname "$0")
git worktree add --detach $S/35/run-tree/atom d53cff270
git -C $S/35/run-tree/atom apply $S/32/run-tree/atom_prefill.patch
git -C $S/35/run-tree/atom add -A && git -C $S/35/run-tree/atom commit -qm "run tree: d53cff270 + atom_prefill.patch (local only)"
git -C $S/35/run-tree/atom am -q "$PWD/$D/step_log.patch"
mkdir -p $S/35/harness && cp -r $S/34/harness/bin $S/34/harness/cells $S/35/harness/
sed -i 's#dep4/34/#dep4/35/#g' $S/35/harness/bin/env.sh
patch -s -d $S/35/harness -p1 < "$D/harness.patch"
cp "$D/ov_spike.sh" $S/35/harness/cells/
