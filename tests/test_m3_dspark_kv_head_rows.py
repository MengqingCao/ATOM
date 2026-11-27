# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""MengqingCao/ATOM#109: the MiniMax-M3.1 DSpark draft's block tables carry one
KV-head-encoded row per (block row, KV head) when a rank holds several KV heads
(attention TP1, as under DP attention), on the fused-metadata path
(`build_block_metadata`, ATOM_M3_DRAFT_FUSED_META) as on the generic one. The
fused path left one row per block row, so the draft attended the wrong pages and
real acceptance fell to ~1%.

`dspark.py` needs the AITER / Triton stack to import, so the two methods under
test are compiled from its source here, with its one kernel launch replaced by a
writer of known rows."""

import ast
from pathlib import Path
from types import SimpleNamespace

import torch

SOURCE = (
    Path(__file__).resolve().parents[1] / "atom" / "models" / "minimax_m3" / "dspark.py"
)
METHODS = ("_kv_head_rows", "build_block_metadata")
T = 3


def _kernel(*args, **kwargs):
    """`_block_meta_kernel`'s outputs: row r's pages 10 * r + column, length r + 1."""
    rows_bt, rows_len = args[12], args[13]
    rows = rows_bt.shape[0]
    rows_bt.copy_(
        10 * torch.arange(rows, dtype=torch.int32)[:, None]
        + torch.arange(rows_bt.shape[1], dtype=torch.int32)
    )
    rows_len.copy_(torch.arange(1, rows + 1, dtype=torch.int32))


class _Draft:
    """`MiniMaxM3DSpark` with only the two methods under test, from its source."""

    max_num_seqs = 4
    sliding_window = 4
    vocab_size = 10
    _pos_major = False

    def __init__(self, heads: int):
        self.kv_heads_per_rank = heads


def _bind_methods():
    tree = ast.parse(SOURCE.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "MiniMaxM3DSpark"
    )
    defs = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in METHODS]
    assert "build_block_metadata" in {d.name for d in defs}
    namespace = {
        "torch": torch,
        "triton": SimpleNamespace(
            cdiv=lambda a, b: -(-a // b),
            next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
        ),
        "AllocationCap": lambda *a, **k: None,
        "_block_meta_kernel": {(2,): _kernel},
    }
    module = ast.Module(body=defs, type_ignores=[])
    exec(compile(module, str(SOURCE), "exec"), namespace)
    for d in defs:
        setattr(_Draft, d.name, namespace[d.name])


_bind_methods()


def _fused_metadata(heads: int, running_bs: int = 2):
    md = SimpleNamespace()
    _Draft(heads).build_block_metadata(
        md,
        block_tables=torch.zeros(running_bs, 2, dtype=torch.int32),
        anchor_positions=torch.zeros(running_bs, dtype=torch.int64),
        block_positions=torch.zeros(running_bs, T, dtype=torch.int64),
        slots=torch.zeros(running_bs, T, dtype=torch.int64),
        ctx_lens=torch.zeros(running_bs, dtype=torch.int32),
        allocated_pages=torch.zeros(running_bs, dtype=torch.int32),
        scheduled_bs=running_bs,
        entry_tokens=128,
        pad_ctx_len=0,
    )
    return md.dspark_bt16_rows, md.dspark_ctx_rows


def test_fused_metadata_has_one_row_per_kv_head():
    H, rows = 4, 2 * T
    tables, lens = _fused_metadata(heads=H)
    assert tables.shape[0] == rows * H and lens.shape[0] == rows * H
    pw = tables.shape[1]
    base = 10 * torch.arange(rows)[:, None] + torch.arange(pw)
    for h in range(H):
        assert torch.equal(tables.view(rows, H, pw)[:, h], (base * H + h).int())
        assert torch.equal(lens.view(rows, H)[:, h], torch.arange(1, rows + 1).int())


def test_fused_metadata_one_kv_head_is_unchanged():
    rows = 2 * T
    tables, lens = _fused_metadata(heads=1)
    base = 10 * torch.arange(rows)[:, None] + torch.arange(tables.shape[1])
    assert torch.equal(tables, base.int())
    assert torch.equal(lens, torch.arange(1, rows + 1).int())
