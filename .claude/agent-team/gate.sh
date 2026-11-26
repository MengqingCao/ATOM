#!/bin/bash
# agent-team gate for pd-admission-livelock: lint the changed Python files, run the tests they reach, and atom/mesh fmt + lib tests when it changed. CPU only.
# Run from a tree root; exit 0 is a pass.
set -uo pipefail
PY=${ATOM_GATE_PY:-/share_nfs/yajizhan/venv-cpu-test/bin/python}
BLACK=${ATOM_GATE_BLACK:-/share_nfs/yajizhan/venv-black26/bin/black}
RUFF=${ATOM_GATE_RUFF:-/share_nfs/mengqing/rv/bin/ruff}
BRANCH=pd-admission-livelock

FORK=$(git remote -v | awk '$2 ~ /[:\/]MengqingCao\/ATOM(\.git)?$/ && $3 == "(fetch)" {print $1; exit}')
[ -n "$FORK" ] || { echo "gate: no remote for MengqingCao/ATOM"; exit 2; }
# Before the line exists on the fork (the baseline), the tree itself is the base.
BASE=$(git merge-base HEAD "$FORK/$BRANCH" 2>/dev/null || git rev-parse HEAD)

mapfile -t PYFILES < <({ git diff --name-only --diff-filter=ACMR "$BASE" -- '*.py'; git ls-files --others --exclude-standard -- '*.py'; } | sort -u)

fail=0
if ((${#PYFILES[@]})); then
    echo "== ruff + black (${#PYFILES[@]} changed files)"
    "$RUFF" check "${PYFILES[@]}" || fail=1
    "$BLACK" --check "${PYFILES[@]}" || fail=1
fi

# Tests the change reaches: changed test files, plus test files that import a changed atom module.
mapfile -t TESTS < <(
    printf '%s\n' "${PYFILES[@]}" | grep -E '^tests/(.*/)?test_[^/]*\.py$' | grep -v '^tests/plugin/'
    for f in "${PYFILES[@]}"; do
        case "$f" in atom/*.py) m=${f%.py}; m=${m%/__init__}; m=${m//\//.}
            grep -rlE --include='test_*.py' "(from|import) ${m//./\\.}([ .]|$)" tests 2>/dev/null | grep -v '^tests/plugin/'
            # `from <parent> import <leaf>` (also parenthesized): any file with both; over-selection is safe.
            p=${m%.*}; grep -rlE --include='test_*.py' "from ${p//./\\.} import" tests 2>/dev/null | grep -v '^tests/plugin/' | xargs -r grep -lw "${m##*.}" ;;
        esac
    done | sort -u
)
TESTS=($(printf '%s\n' "${TESTS[@]}" | sort -u | grep -v '^tests/entrypoints/test_openai_server.py$' | while read -r t; do [ -f "$t" ] && echo "$t"; done))
if ((${#TESTS[@]})); then
    SCR=$(mktemp -d /dev/shm/pal-gate.XXXXXX)
    echo "== pytest ${TESTS[*]}"
    TMPDIR=$SCR PYTHONPATH=. "$PY" -m pytest -q -p no:cacheprovider --basetemp="$SCR/bt" "${TESTS[@]}" || fail=1
    rm -rf "$SCR"
else
    echo "== no tests reached"
fi
# Rust router: fmt and lib tests when atom/mesh changed.
if { git diff --name-only "$BASE" -- atom/mesh; git ls-files --others --exclude-standard -- atom/mesh; } | grep -q .; then
    echo "== atom/mesh: cargo fmt + lib tests"
    (source /share_nfs/mengqing/tools/rust/env.sh && cd atom/mesh && cargo fmt --check && timeout 2400 cargo test --lib -- --skip test_load_counter_performance) || fail=1
fi
exit $fail
