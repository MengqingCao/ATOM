# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""MengqingCao/ATOM#109: under pure DP the MiniMax-M3.1 verify head takes its
logits through the LM head's DP-sharded collective, the one a peer with no
request enters through `compute_logits`; elsewhere (TP decodes) it keeps its
local logits. A rank that skipped the collective left its peers spinning."""

import importlib
import importlib.abc
import importlib.machinery
import sys
import types
from types import SimpleNamespace

import pytest
import torch

MODULE = "atom.models.minimax_m3.verify_head"
RQ = 8


class _Anything(types.ModuleType):
    """A stand-in for a GPU-stack module: every name is a callable no-op."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Obj()


class _Obj:
    def __call__(self, *args, **kwargs):
        # a decorator returns what it decorates
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return _Obj()

    def __getattr__(self, name):
        return _Obj()

    def __getitem__(self, key):
        return _Obj()

    def __mro_entries__(self, bases):
        return (object,)


class _GpuStackFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    roots = ("aiter", "triton", "amd_kernels")

    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in self.roots:
            return importlib.machinery.ModuleSpec(name, self, is_package=True)
        return None

    def create_module(self, spec):
        module = _Anything(spec.name)
        module.__path__ = []
        return module

    def exec_module(self, module):
        pass


@pytest.fixture
def verify_head_module(monkeypatch):
    """The real module; on a host without aiter / triton, imported over
    stand-ins for them (the code under test calls none of their kernels)."""
    try:
        return importlib.import_module(MODULE)
    except ImportError:
        pass
    before = set(sys.modules)
    monkeypatch.setattr(sys, "meta_path", [_GpuStackFinder(), *sys.meta_path])
    module = importlib.import_module(MODULE)
    yield module
    for name in set(sys.modules) - before:
        del sys.modules[name]


def _head(module, monkeypatch, dp_sharded: bool):
    """An M3VerifyHead whose LM head says whether this step runs DP-sharded,
    with `verify` returning the logits it was handed."""
    calls = []
    lm_head = SimpleNamespace(
        _can_use_dp_sharded_head=lambda ctx: dp_sharded,
        _dp_sharded_logits=lambda x, mode: calls.append("dp") or x + 1,
    )
    fake_dspark = types.ModuleType("atom.models.minimax_m3.dspark")
    fake_dspark.local_logits = lambda head, x: calls.append("local") or x
    monkeypatch.setitem(sys.modules, "atom.models.minimax_m3.dspark", fake_dspark)
    ctx = SimpleNamespace(is_prefill=False, is_draft=False)
    fwd = SimpleNamespace(context=ctx, attn_metadata=SimpleNamespace(max_seqlen_q=RQ))
    monkeypatch.setattr(module, "get_forward_context", lambda: fwd)

    head = object.__new__(module.M3VerifyHead)
    head.lm = SimpleNamespace(lm_head=lm_head, _step_input_ids=torch.zeros(RQ + 1))
    head.rq, head.max_bs = RQ, 4
    head.runner = SimpleNamespace(
        forward_vars={"temperatures": SimpleNamespace(gpu=torch.zeros(4))}
    )
    head._cond = ()
    head.verify = lambda logits, ids, temps, cond: logits
    return head, calls


def test_pure_dp_step_enters_the_dp_sharded_collective(verify_head_module, monkeypatch):
    head, calls = _head(verify_head_module, monkeypatch, dp_sharded=True)
    hidden = torch.zeros(RQ, 4)
    out = head._head(hidden)
    assert calls == ["dp"]
    assert torch.equal(out, hidden + 1)


def test_tp_step_keeps_local_logits(verify_head_module, monkeypatch):
    head, calls = _head(verify_head_module, monkeypatch, dp_sharded=False)
    hidden = torch.zeros(RQ, 4)
    assert torch.equal(head._head(hidden), hidden)
    assert calls == ["local"]
