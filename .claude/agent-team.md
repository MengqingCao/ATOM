---
repo: MengqingCao/ATOM
integration_branch: mengqing/m31-decode-dep4
never_touch: ROCm/ATOM, ROCm/aiter, wuhuikx/LLM-Booster
gate_task: bash .claude/agent-team/gate.sh
design_entry: docs/architecture_guide.md
new_tests_dir: tests
import_check: PYTHONPATH=. /share_nfs/yajizhan/venv-cpu-test/bin/python -c "import atom, os; print(os.path.dirname(os.path.abspath(atom.__file__)))"
publish_language: English
max_tasks: 3
---

# MiniMax-M3.1 P/D decode at high concurrency: TP4 + indexer CP vs DEP4 (+MegaMoE)

Requirement: the auto umbrella in MengqingCao/ATOM. Source: wuhuikx/LLM-Booster#757 (part of #310).
Agents never write to LLM-Booster or ROCm/ATOM; the owner posts the final summary to #757 and splits
upstream PRs by hand.

## Project rules

- P/D disaggregation only. Every serving verdict and end-to-end number comes from 1P1D P/D runs
  (AgentX, c140/c150 load); the operator microbenchmark is the MegaMoE operator's own acceptance.
  Any other 4-GPU decode-only window is a step microbenchmark for traces and tuning, never a verdict.
- Numerics (InferenceX M3.1 compliance; anything needing an accuracy proof is out): native MXFP8
  linears, no PTPC; FP8 KV and FP8 index cache; index top-k on every layer; bf16 all-reduce and bf16
  MegaMoE combine wire; shared expert as its own MXFP8 MLP; DSpark k=7 as shipped; MoE on the
  checkpoint's MXFP4 path, never moved to a lower-precision path. An optimization is bit-identical
  or no worse versus fp32 than the code it replaces.
- Performance runs use forced acceptance `--spec-decode-acceptance-length 4.04`; accuracy runs
  (GSM8K-200, DSpark acceptance) use real acceptance.
- Fast-loop prompts are real AgentX text, one conversation per request, prefix cache off; random
  token ids only for smoke tests.
- P and D always run the same tree. The prefill instance runs yajizhan's optimized prefill stack
  (/share_nfs/yajizhan/tmp/m31-tp1-tune: prs/ the patch stacks, server_pp4.sh the settings, BASELINE.md,
  RESULTS.md); read it, never write there.
- MegaMoE kernel work is in aiter (aiter/ops/flydsl/kernels/mega_moe/, MegaMoEV2): its PRs go to
  MengqingCao/aiter (fork of ROCm/aiter), base `main` unless the task names a branch; issues stay in
  MengqingCao/ATOM. An ATOM PR that needs an aiter change names the aiter commit it was tested with.
  The operator baseline is aiter main as of the umbrella's start; the target is >= 1.6x at M3.1 decode
  shapes, then stop tuning.
- Prior material (read-only; copy what you use): /share_nfs/yajizhan/tmp/m31-decode/ (CONTEXT.md,
  CONTROL.md, prior-art/, dep-code/worktree_changes_at_stop.patch, harness/, cost-model/).
- Measurement data, scripts and reports go under /share_nfs/mengqing/code/tmp/m31-decode-dep4/<issue>/
  and are summarized in the PR or issue; large raw data is not committed.

## Environment (dev host)

- Invoke the `gpu-pool` skill before any GPU step. GPU work goes through the team pool, `/share_nfs/yajizhan/gpu-pool/bin/gpool`; read
  /share_nfs/yajizhan/gpu-pool/skill/SKILL.md first. One lease per GPU step, released the moment it
  ends; partial node by `ROCR_VISIBLE_DEVICES=GPU-<hex>` UUIDs, never HIP_VISIBLE_DEVICES. 4-GPU
  window: `-g 4`; 1P1D: `-g 8` on one node. Honour YIELD at once. Never P0.
- Python tests: `/share_nfs/yajizhan/venv-cpu-test/bin/python -m pytest` (CPU torch); set TMPDIR
  under /dev/shm (/share_nfs rejects `:` in names). black 26.10.1 at
  /share_nfs/yajizhan/venv-black26/bin/black, ruff at /share_nfs/mengqing/rv/bin/ruff. Whole-repo
  ruff is not required; the gate lints changed files only.
- The gate runs on the dev host, CPU only, and needs no GPU.
- Scratch (git-ignored, outside the repo): /share_nfs/mengqing/code/tmp/m31-decode-dep4/.
