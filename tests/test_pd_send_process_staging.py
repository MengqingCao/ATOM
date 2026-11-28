# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""O1's send process (`ATOM_PD_SEND_PROCESS`) sends whole blocks, as a TP1 or
DEP decode takes them, from its staging slots like head cuts (#110)."""

import sys
import threading
from unittest.mock import patch

import pytest
import torch
from test_mha_landing import (  # noqa: F401 - autouse CPU stream and gather
    LAST,
    STAGES,
    Harness,
    byte_view,
    host_gather,
    host_streams,
    mc,
)
from test_page_unit_layout import LAYERS, MTP, stage_pages

from atom.kv_transfer.disaggregation.mooncake import send_process

SRC, DST = [3, 1, 2], [2, 0, 3]
# Built over the connector module the harness was imported with (under a stub
# AITER, then dropped from sys.modules).
with patch.dict(sys.modules, {mc.__name__: mc}):
    SendProcessConnector = send_process._connector_class()


def in_send_process(h, slots=1):
    """``h``'s producer as the send process's connector, staging ``slots``
    slots each a whole source PAGE wide."""
    c = h.c
    c.__class__ = SendProcessConnector
    c._transfer = threading.local()
    width = max(c._per_block_bytes_list)
    c._kv_staging = torch.empty((slots, width), dtype=torch.uint8, device=h.device)
    c._kv_staging_slots = mc.SlotPool(slots)
    return h


def whole_block_harness(stage, device="cpu", slots=1):
    """A producer stage sending to a TP1 decode, which holds every KV head."""
    h = in_send_process(Harness(stage, 0, offer=False, device=device), slots)
    h.theirs = stage_pages([*range(LAYERS), MTP], 4, draft=True)
    h.dst_views = [p.view.to(h.device).zero_() for p in h.theirs]
    h.req.update(
        consumer_tp_size=1,
        consumer_base_addrs=[v.data_ptr() for v in h.dst_views],
        consumer_region_roles=[p.region.semantic_role for p in h.theirs],
        consumer_block_bpb=[p.region.unit_bytes for p in h.theirs],
        consumer_unit_layout=mc._pack_unit_layout(h.theirs),
    )

    def write(target, srcs, dsts, sizes):
        h.writes.append((srcs, dsts, sizes))
        for src, dst, n in zip(srcs, dsts, sizes):
            source = byte_view([h.c._kv_staging, *h.src_views], src, n)
            byte_view(h.dst_views, dst, n).copy_(source)
        if h.device.type == "cuda":
            torch.cuda.synchronize()
        return 0

    h.c.transfer_engine.batch_transfer_sync_write = write
    return h


def check_whole_blocks(h, src, dst):
    by_spec = dict(zip((p.unit_spec for p in h.theirs), h.dst_views))
    for page, view in zip(h.ours, h.src_views):
        got = by_spec[page.unit_spec]
        for s, d in zip(src, dst):
            assert torch.equal(got[d], view[s]), page.unit_spec
    untouched = [i for i in range(h.dst_views[0].shape[0]) if i not in dst]
    assert all(not v[untouched].any() for v in h.dst_views)
    low = h.c._kv_staging.data_ptr()
    high = low + h.c._kv_staging.numel()
    sources = [a for srcs, _, _ in h.writes for a in srcs]
    assert sources and all(low <= a < high for a in sources)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU"),
        ),
    ],
)
@pytest.mark.parametrize("stage", STAGES)
def test_whole_block_sends_go_through_staging(stage, device):
    h = whole_block_harness(stage, device)
    assert h.send(SRC, DST)
    check_whole_blocks(h, SRC, DST)
    assert h.c._kv_staging_slots.free == [0]
    # A DCP decode's token relayout is not a whole-block send.
    h.req["consumer_dcp_size"] = 2
    assert h.c._block_plan(h.req, len(h.dst_views), 21).cuts is None


@pytest.mark.parametrize("stage", STAGES)
@pytest.mark.parametrize("rank", [0, 3])
def test_head_cuts_keep_their_plan_in_the_send_process(stage, rank):
    h = in_send_process(Harness(stage, rank, offer=False))
    n = len(h.dst_views)
    base = mc.MooncakeConnector._block_plan(h.c, h.req, n, 21)
    assert base.cuts is not None
    assert h.c._block_plan(h.req, n, 21) is base
    assert h.send(SRC, DST)
    h.check(SRC, DST)


def test_whole_blocks_wait_for_a_free_staging_slot():
    h = whole_block_harness(LAST)
    held = h.c._kv_staging_slots.acquire()
    result = []
    sender = threading.Thread(target=lambda: result.append(h.send(SRC, DST)))
    sender.start()
    sender.join(0.5)
    assert sender.is_alive() and not h.writes
    h.c._kv_staging_slots.release(held)
    sender.join(30)
    assert result == [True]
    check_whole_blocks(h, SRC, DST)
