---
# Fork line for wuhuikx/LLM-Booster#617; ROCm/ATOM is the upstream.
repo: MengqingCao/ATOM
integration_branch: pd-admission-livelock
never_touch: ROCm/ATOM, wuhuikx/LLM-Booster
# Same gate as least-session-routing, plus fmt and lib tests for atom/mesh when it changes. CPU only.
gate_task: bash .claude/agent-team/gate.sh
design_entry: docs/distributed_guide.md
new_tests_dir: tests
import_check: PYTHONPATH=. /share_nfs/yajizhan/venv-cpu-test/bin/python -c "import atom, os; print(os.path.dirname(os.path.abspath(atom.__file__)))"
publish_language: English
max_tasks: 2
---
# End chunked P/D admission timeouts on both sides at once

Requirement: the auto umbrella in MengqingCao/ATOM. Source: wuhuikx/LLM-Booster#617 (fixes 3 and 4).
Agents never write to LLM-Booster or ROCm/ATOM; the owner upstreams by hand.

## Project rules

- Fix 2 of #617 is ROCm/ATOM#3536 (Jasen2201). This line starts from ROCm/ATOM main, does not
  stack on #3536, and never edits what #3536 changes beyond what its own goal needs.
- Only the chunked P/D admission path changes behaviour; non-chunked P/D and non-P/D serving
  are unchanged.
- Any run that uses GPUs on a pit2-p03 node invokes the `gpu-pool` skill first and holds a gpool
  lease for exactly the GPUs it uses, released the moment the run ends. The gate and unit tests
  need no GPU.
