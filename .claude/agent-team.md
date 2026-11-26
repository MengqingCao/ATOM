---
# Fork line for wuhuikx/LLM-Booster#653; ROCm/ATOM is the upstream.
repo: MengqingCao/ATOM
integration_branch: least-session-routing
never_touch: ROCm/ATOM, wuhuikx/LLM-Booster
# Same gate as the m31 line (agent-team/m31-overlay): lint changed Python files, run the tests they reach. CPU only.
gate_task: bash .claude/agent-team/gate.sh
design_entry: docs/distributed_guide.md
new_tests_dir: tests
import_check: PYTHONPATH=. /share_nfs/yajizhan/venv-cpu-test/bin/python -c "import atom, os; print(os.path.dirname(os.path.abspath(atom.__file__)))"
publish_language: English
max_tasks: 2
---
# Least-session placement for ATOM DP session affinity

Requirement: the auto umbrella in MengqingCao/ATOM. Source: wuhuikx/LLM-Booster#653.
Agents never write to LLM-Booster or ROCm/ATOM; the owner upstreams by hand.

## Project rules

- Function first: this line delivers the placement policy and its tests. No benchmark or
  performance claim is made or required here.
- The new policy is opt-in; with it unset, DP routing is unchanged.
- Any run that uses GPUs on a pit2-p03 node invokes the `gpu-pool` skill first and holds a gpool
  lease for exactly the GPUs it uses, released the moment the run ends. The gate and unit tests
  need no GPU.
