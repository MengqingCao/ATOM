# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""CPU tests of the per-step DP load-balance log (`atom/utils/dp_balance_log.py`)."""

import json
import time
from types import SimpleNamespace

import numpy as np
import torch

from atom.utils import dp_balance_log as dpb


def _read(path):
    lines = open(path).read().splitlines()
    return json.loads(lines[0]), [json.loads(x) for x in lines[1:]]


def _config(dp):
    return SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=dp))


def test_off_by_default_and_without_dp(tmp_path, monkeypatch):
    monkeypatch.setattr(dpb, "LOG_DIR", "")
    assert dpb.open_log("engine", dpb.ENGINE_FIELDS, _config(4), 0) is None
    monkeypatch.setattr(dpb, "LOG_DIR", str(tmp_path))
    assert dpb.open_log("engine", dpb.ENGINE_FIELDS, _config(1), 0) is None
    assert dpb.open_log("engine", dpb.ENGINE_FIELDS, _config(4), 0) is not None


def test_engine_rows(tmp_path):
    log = dpb.JsonlLog("engine", dpb.ENGINE_FIELDS, 2, log_dir=str(tmp_path))
    eng = dpb.EngineStepLog(log)
    sched = SimpleNamespace(
        running=[1, 2, 3], waiting=[4], _num_parked_remote_kv=1,
        total_generation_tokens=12, block_manager=SimpleNamespace(kv=SimpleNamespace(num_used=7)),
    )  # fmt: skip
    eng.fwd = (0.02, 12, 3)
    eng.step(10.0, 10.025, True, 0.001, sched)
    sched.total_generation_tokens = 12
    eng.step(10.03, 10.05, False, None, sched)
    log.close()
    head, rows = _read(log.path)
    assert head["dp_rank"] == 2 and head["fields"] == list(dpb.ENGINE_FIELDS)
    real = dict(zip(head["fields"], rows[0]))
    assert real["real"] == 1 and real["sched_tok"] == 12 and real["running"] == 3
    assert real["parked_kv"] == 1 and real["kv_used"] == 7 and real["gen_tok"] == 12
    dummy = dict(zip(head["fields"], rows[1]))
    assert dummy["real"] == 0 and dummy["fwd_dt"] is None and dummy["gen_tok"] == 0


class _Ev:
    def __init__(self):
        self.ready = False

    def record(self):
        self.ready = False

    def query(self):
        return self.ready

    def elapsed_time(self, other):
        return 5.0


def test_runner_rows_wait_for_gpu_events(tmp_path):
    log = dpb.JsonlLog("runner", dpb.RUNNER_FIELDS, 0, log_dir=str(tmp_path))
    run = dpb.RunnerStepLog(log, _Ev)
    sync = SimpleNamespace(
        scheduled_tokens_per_dp_rank=np.array([8, 0, 16, 4]), max_bs_across_dp=4
    )
    fm = SimpleNamespace(sync=sync, running_bs=4, running_tokens=32, use_cudagraph=True)
    batch = SimpleNamespace(
        total_seqs_num=2, has_prefill=False, context_lens=np.array([100, 50])
    )
    run.meta = (0.002, fm)
    run.launch()
    run.done(1.0, 1.03, batch)
    assert len(run._pending) == 1  # written once its GPU time is known
    for _, start, end in run._pending:
        end.ready = True
    dummy = SimpleNamespace(
        total_seqs_num=0, has_prefill=False, context_lens=np.array([])
    )
    run.launch()
    run.done(1.04, 1.06, dummy)
    log.close()
    head, rows = _read(log.path)
    first = dict(zip(head["fields"], rows[0]))
    assert first["tok_per_rank"] == [8, 0, 16, 4] and first["ctx_tok"] == 150
    assert first["gpu_ms"] == 5.0 and first["dummy"] == 0 and first["graph"] == 1
    assert len(rows) == 1  # the dummy's events are still pending


def test_ep_counter_counts_valid_routed_rows():
    ep = dpb.EpCounter(torch, 4, "cpu")
    ids = torch.tensor(
        [[0, 5], [9, 15], [3, 16]]
    )  # 4 experts per rank; 16 = shared column
    ep.record("model.layers.0.mlp.experts", 1, ids, 4, torch.tensor([2]))
    snap = ep.snapshot(0.0, _Ev())[2]
    assert snap.tolist() == [[1, 1, 1, 1]]  # row 2 is padding
    ep.record("model.layers.0.mlp.experts", 1, ids, 4, None)
    assert ep.buf[0].tolist() == [3, 2, 2, 2]


def test_row_cost(tmp_path):
    log = dpb.JsonlLog("runner", dpb.RUNNER_FIELDS, 0, log_dir=str(tmp_path))
    row = [
        1.0,
        1,
        0.02,
        0.001,
        0,
        0,
        32,
        120000,
        [128, 128, 120, 96],
        32,
        32,
        256,
        1,
        4.2,
    ]
    t = time.perf_counter()
    for _ in range(10000):
        log.row(row)
    per_row = (time.perf_counter() - t) / 10000
    log.close()
    assert per_row < 50e-6, per_row  # steps are ~25 ms apart
