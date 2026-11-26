# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""DP placement policies and their request-lifecycle hooks."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from atom.model_engine.dp_routing.route import Route


class LoadPolicy:
    stateful = False
    router = None

    def __init__(self, lightest: Callable[[], int]) -> None:
        self.lightest = lightest

    def select(self, seq: Any, explicit_rank: int | None) -> Route:
        if explicit_rank is not None:
            return Route(explicit_rank, "explicit_total")
        return Route(self.lightest(), "load_balanced_total")

    def charge(self, seq: Any, credit: int) -> int:
        return credit

    def prefilling(self, seq_id: Any) -> bool:
        return False

    def prefill_complete(self, seq_id: Any) -> None:
        pass

    def release(self, seq_id: Any) -> dict:
        return {}

    def reset(self) -> None:
        pass


class SessionPolicy(LoadPolicy):
    """A session stays on the rank that holds its conversation.

    With `rank_reqs` -- given only when the ranks share their KV pools, so a
    session's history follows it -- a session whose rank is crowded moves to
    the emptiest one: its owner runs at least `margin` (a fraction of the mean)
    more requests than the mean while that one runs at least as many fewer.
    Under DP-attention every rank waits for the busiest one's step, and the
    move costs only the copy of its history from the old rank's pool.

    With `idle_s`, a new session goes instead to the rank holding the fewest
    live sessions -- routed to within the last `idle_s` seconds -- the lowest
    index on a tie.
    """

    stateful = True

    def __init__(
        self,
        lightest: Callable[[], int],
        select_session: Callable[[str], int],
        owners: dict,
        prompts: dict,
        rank_count: int,
        parent_ignored: Callable[[], None],
        rank_reqs: Callable[[], list[int]] | None = None,
        margin: float = 0.1,
        idle_s: float | None = None,
    ) -> None:
        super().__init__(lightest)
        self.select_session = select_session
        self.owners = owners
        self.prompts = prompts
        self.rank_count = rank_count
        self.parent_ignored = parent_ignored
        self.rank_reqs = rank_reqs
        self.margin = margin
        self.idle_s = idle_s
        # Session id -> time.monotonic() of its last routed request.
        self.seen: dict = {}

    def live_counts(self) -> list[int]:
        """Per rank, the sessions it owns that were routed within `idle_s`."""
        now = time.monotonic()
        live = [0] * self.rank_count
        for session, rank in self.owners.items():
            seen = self.seen.get(session)
            live_now = seen is not None and now - seen <= self.idle_s
            if live_now and 0 <= rank < self.rank_count:
                live[rank] += 1
        return live

    def _relieve(self, owner: int) -> int | None:
        """The rank a session on crowded `owner` moves to, or None."""
        reqs = self.rank_reqs()[: self.rank_count]
        mean = sum(reqs) / len(reqs)
        slack = max(1.0, self.margin * mean)
        if reqs[owner] < mean + slack:
            return None
        target = min(range(len(reqs)), key=lambda r: (reqs[r], r))
        if reqs[target] > mean - slack:
            return None
        return target

    def select(self, seq: Any, explicit_rank: int | None) -> Route:
        session = getattr(seq, "dp_session_id", None)
        if not session:
            return super().select(seq, explicit_rank)
        owner = self.owners.get(session)
        if explicit_rank is not None:
            if owner is not None and owner != explicit_rank:
                self.prompts.pop(session, None)
            owner = explicit_rank
            decision = "explicit_total"
        elif owner is None or not 0 <= owner < self.rank_count:
            if self.idle_s is None:
                owner = self.select_session(session)
            else:
                live = self.live_counts()
                owner = live.index(min(live))
            decision = "affinity_new_total"
            if getattr(seq, "dp_parent_session_id", None):
                self.parent_ignored()
        else:
            decision = "affinity_owner_hit_total"
            target = None if self.rank_reqs is None else self._relieve(owner)
            if target is not None:
                owner = target
                decision = "affinity_spill_total"
        self.owners[session] = owner
        self.seen[session] = time.monotonic()
        return Route(owner, decision, self.prompts.get(session, 0))

    def charge(self, seq: Any, credit: int) -> int:
        session = getattr(seq, "dp_session_id", None)
        if session:
            self.prompts[session] = int(getattr(seq, "num_prompt_tokens", 0) or 0)
        return credit

    def reset(self) -> None:
        self.owners.clear()
        self.prompts.clear()
        self.seen.clear()


class PrefixPolicy(LoadPolicy):
    stateful = True

    def __init__(self, router, rank_tokens, rank_reqs, rank_load, lightest) -> None:
        super().__init__(lightest)
        self.router = router
        self.rank_tokens = rank_tokens
        self.rank_reqs = rank_reqs
        self.rank_load = rank_load

    def select(self, seq: Any, explicit_rank: int | None) -> Route:
        rank_tokens = self.rank_tokens()
        rank_reqs = self.rank_reqs()
        if explicit_rank is not None:
            placed = self.router.place(seq, explicit_rank, len(rank_tokens))
            return Route(
                explicit_rank, "explicit_total", placed.credit, placed.moved_tokens
            )
        return self.router.select(
            seq, rank_tokens, rank_reqs, self.rank_load, self.lightest
        )

    def charge(self, seq: Any, credit: int) -> int:
        return self.router.charge(seq.id, credit)

    def prefilling(self, seq_id: Any) -> bool:
        return self.router.prefilling(seq_id)

    def prefill_complete(self, seq_id: Any) -> None:
        self.router.prefill_complete(seq_id)

    def release(self, seq_id: Any) -> dict:
        return self.router.release(seq_id)

    def reset(self) -> None:
        self.router.reset()


class ShareCachePolicy(PrefixPolicy):
    """Shared-tier placement; the router balances on `rank_tokens`, each rank's
    reported prefill backlog."""

    def charge(self, seq: Any, credit: int) -> int:
        seq.dp_est_cached_tokens = hit = super().charge(seq, credit)
        return hit
