# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors

"""Per-step DP load-balance log (``ATOM_DP_BALANCE_LOG=<dir>``, off by default).

Each process of a DP (data_parallel_size > 1) decode writes one jsonl file in
<dir>: the engine core one row per busy-loop step, the model runner one row per
forward, the engine-core manager one row per request it dispatches to a rank.
A file starts with a header object (role, host, pid, dp rank, field names,
wall<->monotonic offset); every other line is a JSON array in that field order.
Timestamps are ``time.monotonic()``, shared by every process on a host.

CPU-side only, with one exception: ``EP_STRIDE`` > 0 adds a captured
``index_add_`` per sampled MoE layer that counts the rows each rank sends to
each EP rank. GPU times come from device events read with ``query()`` once
done, so nothing here waits on the device. Any error disables the log of that
process with one warning; the run itself never fails on it.
"""

from __future__ import annotations

import collections
import json
import logging
import os
import re
import socket
import time

import numpy as np

logger = logging.getLogger("atom")

LOG_DIR = os.getenv("ATOM_DP_BALANCE_LOG", "")
# Every Nth MoE layer of the target model is counted, read every EP_EVERY forwards.
EP_STRIDE = int(os.getenv("ATOM_DP_BALANCE_EP_STRIDE", "4"))
EP_EVERY = 64
_FLUSH_S = 1.0

ENGINE_FIELDS = (
    "t", "dt", "real", "sync_dt", "fwd_dt", "sched_tok", "sched_seqs",
    "running", "waiting", "parked_kv", "kv_used", "gen_tok",
)  # fmt: skip
RUNNER_FIELDS = (
    "t", "n", "dt", "meta_dt", "dummy", "prefill", "bs", "ctx_tok",
    "tok_per_rank", "max_bs", "run_bs", "run_tok", "graph", "gpu_ms",
)  # fmt: skip
MGR_FIELDS = ("t", "seq", "rank", "prompt_tok", "inflight")


class JsonlLog:
    """Buffered jsonl writer; flushes at most once per ``_FLUSH_S``."""

    def __init__(self, role: str, fields, dp_rank=None, extra=None, log_dir=None):
        d = log_dir or LOG_DIR
        os.makedirs(d, exist_ok=True)
        host = socket.gethostname()
        self.path = os.path.join(
            d,
            f"{role}-{host}-dp{'' if dp_rank is None else dp_rank}-{os.getpid()}.jsonl",
        )
        self._f = open(self.path, "a", buffering=1 << 16)
        self._next_flush = time.monotonic() + _FLUSH_S
        self.ok = True
        head = {
            "role": role,
            "host": host,
            "pid": os.getpid(),
            "dp_rank": dp_rank,
            "fields": list(fields),
            "wall_minus_mono": time.time() - time.monotonic(),
            **(extra or {}),
        }
        self._f.write(json.dumps(head) + "\n")

    def row(self, values) -> None:
        if not self.ok:
            return
        try:
            self._f.write(json.dumps(values, separators=(",", ":")) + "\n")
            now = time.monotonic()
            if now >= self._next_flush:
                self._next_flush = now + _FLUSH_S
                self._f.flush()
        except Exception:
            self.fail("write")

    def fail(self, where: str) -> None:
        if self.ok:
            self.ok = False
            logger.warning(
                "dp_balance_log %s: disabled after an error in %s",
                self.path,
                where,
                exc_info=True,
            )

    def close(self) -> None:
        try:
            self._f.close()
        except Exception:
            pass


def open_log(role, fields, config, dp_rank=None):
    """A JsonlLog when the log is on and ``config`` runs DP (> 1 rank), else None."""
    if not LOG_DIR:
        return None
    try:
        if config.parallel_config.data_parallel_size <= 1:
            return None
        return JsonlLog(role, fields, dp_rank)
    except Exception:
        logger.warning(
            "dp_balance_log: cannot open a log in %s", LOG_DIR, exc_info=True
        )
        return None


class EngineStepLog:
    """Engine-core side: one row per DP busy-loop step (a real step or a dummy)."""

    def __init__(self, log: JsonlLog):
        self.log = log
        self.fwd = None  # (forward RPC seconds, scheduled tokens, scheduled seqs), set by a real step
        self._gen = 0

    def step(self, t0, t1, real, sync_dt, scheduler) -> None:
        try:
            fwd = self.fwd if real else None
            self.fwd = None
            gen = int(getattr(scheduler, "total_generation_tokens", 0))
            bm = getattr(scheduler, "block_manager", None)
            kv = getattr(bm, "kv", None)
            self.log.row([
                round(t0, 6), round(t1 - t0, 6), int(real),
                None if sync_dt is None else round(sync_dt, 6),
                None if fwd is None else round(fwd[0], 6),
                0 if fwd is None else int(fwd[1]),
                0 if fwd is None else int(fwd[2]),
                len(scheduler.running), len(scheduler.waiting),
                int(getattr(scheduler, "_num_parked_remote_kv", 0)),
                None if kv is None else int(kv.num_used),
                gen - self._gen,
            ])  # fmt: skip
            self._gen = gen
        except Exception:
            self.log.fail("engine step")


class RunnerStepLog:
    """Model-runner side: one row per forward, written once its GPU time is known."""

    def __init__(self, log: JsonlLog, event_factory=None):
        self.log = log
        self.n = 0
        self.meta = None  # (seconds in ForwardMode.decide, its ForwardMode), set by prepare_model
        self._event = event_factory
        self._free = []
        self._pending = collections.deque()  # (row, start event, end event)
        self._start = None
        self.ep = None  # EpCounter, attached by the first sampled MoE layer
        self._ep_pending = None

    def _ev(self):
        return self._free.pop() if self._free else self._event()

    def launch(self) -> None:
        """Right before the model launch: the GPU time is measured from here."""
        if self._event is not None and self.log.ok:
            try:
                self._start = self._ev()
                self._start.record()
            except Exception:
                self.log.fail("launch event")

    def done(self, t0, t1, batch) -> None:
        try:
            self.n += 1
            meta, self.meta = self.meta, None
            # A DP dummy (`dummy_execution`) schedules no sequence.
            dummy = not batch.total_seqs_num
            ctx = 0 if dummy else int(np.asarray(batch.context_lens).sum())
            row = [round(t0, 6), self.n, round(t1 - t0, 6), None, int(dummy),
                   int(batch.has_prefill), int(batch.total_seqs_num), ctx,
                   None, None, None, None, None, None]  # fmt: skip
            if meta is not None:
                dt, fm = meta
                row[3] = round(dt, 6)
                row[8] = [int(x) for x in fm.sync.scheduled_tokens_per_dp_rank]
                row[9] = int(fm.sync.max_bs_across_dp)
                row[10] = int(fm.running_bs)
                row[11] = int(fm.running_tokens)
                row[12] = int(fm.use_cudagraph)
            start, self._start = self._start, None
            if start is not None and len(self._pending) < 512:
                end = self._ev()
                end.record()
                self._pending.append((row, start, end))
            else:
                if start is not None:
                    self._free.append(start)
                self.log.row(row)
            self._drain()
            if self.ep is not None and self.n % EP_EVERY == 0:
                self._ep_poll(t1)
        except Exception:
            self.log.fail("runner step")

    def _drain(self) -> None:
        while self._pending and self._pending[0][2].query():
            row, start, end = self._pending.popleft()
            row[13] = round(start.elapsed_time(end), 4)
            self._free += (start, end)
            self.log.row(row)

    def _ep_poll(self, now) -> None:
        if self._ep_pending is not None:
            t, ev, host = self._ep_pending
            if not ev.query():
                return
            self.log.row(["ep", round(t, 6), self.ep.layers, host.tolist()])
        self._ep_pending = self.ep.snapshot(now, self._ev())


class EpCounter:
    """Rows each rank's sampled MoE layers send to each EP rank, cumulative.

    One [EP_MAX_LAYERS, ep_size] int64 buffer on the device. Rows are assigned
    on eager forwards (warmup) and recorded into graphs at capture: a layer
    first met while capturing is never counted, as allocating there would
    put the buffer in the graph's pool.
    """

    MAX_LAYERS = 32

    def __init__(self, torch, ep_size: int, device):
        self.torch = torch
        self.ep_size = ep_size
        self.buf = torch.zeros(
            self.MAX_LAYERS, ep_size, dtype=torch.int64, device=device
        )
        self.host = torch.zeros_like(self.buf, device="cpu")
        if self.buf.is_cuda:
            self.host = self.host.pin_memory()
        self.slot = {}  # id(layer) -> row
        self.layers = []  # row -> layer prefix

    def record(self, name, key, topk_ids, per_rank, n_valid_gpu) -> None:
        torch = self.torch
        row = self.slot.get(key)
        if row is None:
            capturing = self.buf.is_cuda and torch.cuda.is_current_stream_capturing()
            if capturing or len(self.layers) >= self.MAX_LAYERS:
                return
            row = self.slot[key] = len(self.layers)
            self.layers.append(name)
        rows = topk_ids.shape[0]
        ids = topk_ids.to(torch.int64)
        # Routed ids only: a dispatch-space shared column (>= the routed range) stays local.
        w = ids < per_rank * self.ep_size
        if n_valid_gpu is not None:
            w &= (
                torch.arange(rows, device=ids.device) < n_valid_gpu.to(torch.int64)
            ).unsqueeze(1)
        owner = (ids // per_rank).clamp_(0, self.ep_size - 1)
        self.buf[row].index_add_(0, owner.reshape(-1), w.to(torch.int64).reshape(-1))

    def snapshot(self, now, event):
        self.host.copy_(self.buf, non_blocking=True)
        event.record()
        return (now, event, self.host[: len(self.layers)])


_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def ep_record(runner_log, layer, topk_ids, n_valid_gpu) -> None:
    """Count one MoE layer's routed rows per EP rank (target layers, every EP_STRIDE-th)."""
    if runner_log is None or EP_STRIDE <= 0 or not runner_log.log.ok:
        return
    try:
        m = _LAYER_RE.search(getattr(layer, "prefix", "") or "")
        if m is None or int(m.group(1)) % EP_STRIDE:
            return
        ep_size = int(layer.moe_parallel_config.ep_size)
        if ep_size <= 1:
            return
        if runner_log.ep is None:
            import torch

            runner_log.ep = EpCounter(torch, ep_size, topk_ids.device)
        per_rank = int(layer.expert_layout.routed_physical_per_rank)
        runner_log.ep.record(layer.prefix, id(layer), topk_ids, per_rank, n_valid_gpu)
    except Exception:
        runner_log.log.fail("ep record")


# The model runner's log, for the MoE layers (one runner per process).
RUNNER = None
