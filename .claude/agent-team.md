---
# Fork line for wuhuikx/LLM-Booster#709; ROCm/ATOM is the upstream.
repo: MengqingCao/ATOM
integration_branch: mesh-vllm-load
never_touch: ROCm/ATOM
# CLAUDE.md has no mesh command. fmt is atom/mesh/.pre-commit-config.yaml's; lib tests are CI's minus --features atom-engine; the test compile is the gate's own.
gate_task: cd atom/mesh && cargo fmt --check && cargo test --lib && cargo test --tests --no-run
design_entry: atom/mesh/README.md
new_tests_dir: atom/mesh/tests
publish_language: English
max_tasks: 2
# Rust unit tests live in tests.rs and tests/ dirs; Rulings on #5 and #6 counted them as code by mistake.
test_files: */tests/*,*/tests.rs,*/test_*,*_test.*
---
Scope is `atom/mesh` (Rust). The line stacks on ROCm/ATOM#3567 -> #3566 -> #3406 -> #3481;
when that chain changes upstream, rebasing this line is an escalation, never done silently.

Any run that uses GPUs on a pit2-p03 node (an E2E, benchmark or server) holds a gpool lease
(`/share_nfs/yajizhan/gpu-pool/bin/gpool`, rules in /share_nfs/yajizhan/gpu-pool/skill/SKILL.md)
and releases it the moment the run ends. Unit tests and the gate need no GPU.
