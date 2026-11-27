# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
"""MengqingCao/ATOM#109: the M3.1 AgentX recipe's DEP4 + MegaMoE decode env turns
off the c17 DSpark decode features that hang a DP decode, and the TP4 reference
decode env keeps them at their defaults."""

import os
import subprocess
from pathlib import Path

import pytest

from atom.utils import envs

ENV_DIR = Path(__file__).resolve().parents[1] / "recipes" / "m31_agentx_1p2d" / "env"
C17_DECODE_FLAGS = (
    "ATOM_M3_VERIFY_HEAD",
    "ATOM_M3_DRAFT_CTX_IN_GRAPH",
    "ATOM_M3_DRAFT_FUSED_META",
    "ATOM_M3_DRAFT_FC_SHARDED",
)


def _decode_env(name: str) -> dict[str, str]:
    """What the recipe hands the decode role: `d_env_dump` of env file `name`."""
    out = subprocess.run(
        ["bash", "-c", f"source ./{name} && d_env_dump"],
        cwd=ENV_DIR,
        env={"PATH": os.environ["PATH"]},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return dict(line.split("=", 1) for line in out.splitlines() if "=" in line)


@pytest.mark.parametrize("flag", C17_DECODE_FLAGS)
def test_dep4_decode_turns_off_c17_feature(flag, monkeypatch):
    value = _decode_env("d_env_dep4_mega.sh").get(flag)
    assert value == "0", f"{flag} reaches the DEP4 decode as {value!r}"
    monkeypatch.setenv(flag, value)
    assert not getattr(envs, flag)


@pytest.mark.parametrize("flag", C17_DECODE_FLAGS)
def test_tp4_reference_decode_keeps_c17_default(flag, monkeypatch):
    assert flag not in _decode_env("d_env.sh")
    monkeypatch.delenv(flag, raising=False)
    assert getattr(envs, flag)
