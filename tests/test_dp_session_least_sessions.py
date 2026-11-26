# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Copyright (c) 2026 ATOM Project Contributors
# Least-session placement of new DP sessions (ATOM_DP_LEAST_SESSIONS), driven
# through CoreManager's real routing path on a bare manager.

from itertools import count

import pytest
from test_dp_load_balance import _FakeSeq, _make_mgr, _route

from atom.model_engine.dp_routing import policies

IDLE_S = 7200.0
_seq_ids = count()


@pytest.fixture
def clock(monkeypatch):
    now = [1000.0]
    monkeypatch.setattr(policies.time, "monotonic", lambda: now[0])
    return now


def _least_mgr(n_ranks=2):
    mgr = _make_mgr(n_ranks, strategy="least_requests", session_affinity=True)
    mgr._dp_session_idle_s = IDLE_S
    return mgr


def _turn(session, rank=None, tokens=100):
    return _FakeSeq(
        next(_seq_ids), tokens, data_parallel_rank=rank, dp_session_id=session
    )


def _sessions(mgr):
    return mgr.get_dp_router_statistics()["session_count_per_rank"]


def test_new_session_goes_to_rank_with_fewest_live_sessions(clock):
    mgr = _least_mgr()
    _route(mgr, [_turn(f"a{i}", rank=0) for i in range(3)] + [_turn("b", rank=1)])
    # Rank 1 is far busier, but only the session count decides.
    mgr._rank_tokens = [0, 1_000_000]
    assert _route(mgr, [_turn("new")]) == [1]
    assert _sessions(mgr) == [3, 2]


def test_full_tie_goes_to_rank_0(clock):
    mgr = _least_mgr()
    _route(mgr, [_turn(s, rank=r) for s, r in (("a", 0), ("b", 0), ("c", 1), ("d", 1))])
    # Rank 0 is the busier one; load-based placement would pick rank 1.
    mgr._rank_reqs = [10, 0]
    assert _route(mgr, [_turn("new")]) == [0]


def test_burst_spreads_across_ranks(clock):
    mgr = _least_mgr()
    assert _route(mgr, [_turn(f"s{i}") for i in range(4)]) == [0, 1, 0, 1]
    assert _sessions(mgr) == [2, 2]


def test_later_turn_stays_on_owner(clock):
    mgr = _least_mgr()
    first = _route(mgr, [_turn(f"s{i}") for i in range(4)])
    mgr._rank_reqs = [50, 0]
    assert _route(mgr, [_turn(f"s{i}") for i in range(4)]) == first
    counters = mgr.get_dp_router_statistics()
    assert counters["affinity_owner_hit_total"] == 4
    assert counters["affinity_spill_total"] == 0
    assert _sessions(mgr) == [2, 2]


def test_idle_session_stops_counting_and_next_turn_reaches_owner(clock):
    mgr = _least_mgr()
    assert _route(mgr, [_turn("keep"), _turn("idle")]) == [0, 1]
    clock[0] += IDLE_S
    _route(mgr, [_turn("keep")])
    clock[0] += 1
    # "idle" was last routed IDLE_S + 1 seconds ago; "keep" 1 second ago.
    assert _sessions(mgr) == [1, 0]
    assert _route(mgr, [_turn("new")]) == [1]
    assert _sessions(mgr) == [1, 1]
    assert _route(mgr, [_turn("idle")]) == [1]
    assert mgr._dp_session_owners["idle"] == 1
    assert _sessions(mgr) == [1, 2]


def test_session_at_window_edge_is_still_live(clock):
    mgr = _least_mgr()
    _route(mgr, [_turn("a")])
    clock[0] += IDLE_S
    assert _sessions(mgr) == [1, 0]


def test_migrate_margin_causes_no_spill(clock):
    mgr = _least_mgr()
    mgr._session_migrate_margin = 0.1
    assert _route(mgr, [_turn("s")]) == [0]
    # Load-based spill would move s off its crowded owner.
    mgr._rank_reqs = [40, 0]
    assert _route(mgr, [_turn("s")]) == [0]
    assert mgr._dp_route_counters["affinity_spill_total"] == 0


def test_reset_forgets_live_sessions(clock):
    mgr = _least_mgr()
    _route(mgr, [_turn("a"), _turn("b"), _turn("c")])
    mgr.reset_dp_router()
    assert _sessions(mgr) == [0, 0]
    assert mgr._get_dp_policy().seen == {}
