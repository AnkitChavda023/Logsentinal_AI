"""
Persistent, Zipf-weighted synthetic user population.

Replaces the previous scheme (core/pipeline.py used to assign a fresh
`f"user_{random.randint(10000, 99999)}"` on *every single request*), which
had zero identity persistence — the same label was, for practical purposes,
never reused, so the generated data could never show a repeat visit or any
concentration of traffic in a subset of users. Real usage-analytics data
consistently shows the opposite: a small fraction of users generate most of
the traffic (Zipf's law / Pareto-style skew — the same shape as most
real-world popularity distributions).

Deliberately scoped to what this closes and what it doesn't: this gives
genuine persistent identity, repeat visits, and realistic popularity skew.
It does NOT implement per-user behavioral sequencing (e.g. a user's next
request being logically related to their last one, like browse -> cart ->
checkout) — that would require entry-point selection to be driven by a
per-user session/workflow state machine, which conflicts with how traffic
is currently generated (every service independently samples its own
Poisson-distributed root-level request load in GeneratorPipeline —
see core/traffic_model.py). Restructuring that is a bigger, separate change;
this is the well-bounded, low-risk increment.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np

from module1_log_generator.utils.random_utils import get_rng


class UserPool:
    """A fixed population of `num_users` synthetic users, selected with
    Zipf-style popularity skew rather than uniformly at random."""

    def __init__(self, num_users: int, activity_skew: float = 1.2) -> None:
        if num_users < 1:
            raise ValueError(f"num_users must be >= 1, got {num_users}")
        self._user_ids = [f"user_{i:06d}" for i in range(num_users)]

        # Zipf-style weights over a FINITE population: numpy's own zipf()
        # sampler has unbounded support (it can draw arbitrarily large
        # ranks), which doesn't fit a fixed-size pool without rejection
        # sampling or clamping. Instead, precompute weight ~ 1/rank^skew for
        # every user once, normalize, and take a cumulative sum so each draw
        # is a single uniform() + binary search (np.searchsorted) — O(log N)
        # per call, not O(N). That matters here: this runs once per
        # generated request, potentially tens of millions of times per run.
        ranks = np.arange(1, num_users + 1, dtype=np.float64)
        weights = 1.0 / np.power(ranks, activity_skew)
        probabilities = weights / weights.sum()
        self._cdf = np.cumsum(probabilities)
        self._cdf[-1] = 1.0  # guard against float rounding leaving the tail < 1.0

        self._last_active: dict[str, datetime] = {}

    def pick_user(self, sim_time: datetime) -> str:
        """Draw one user, weighted by their fixed, rank-based popularity,
        and record sim_time as their most recent activity."""
        u = float(get_rng().uniform())
        idx = int(np.searchsorted(self._cdf, u, side="right"))
        idx = min(idx, len(self._user_ids) - 1)
        user_id = self._user_ids[idx]
        self._last_active[user_id] = sim_time
        return user_id

    def last_active_before(self, user_id: str) -> Optional[datetime]:
        """The timestamp of user_id's most recent pick_user() call so far —
        call this BEFORE picking again for the same user to measure the gap
        since their previous appearance (pick_user() itself overwrites this
        the instant it's called, so reading it after would just return the
        current call's own timestamp, not the previous one)."""
        return self._last_active.get(user_id)

    def __len__(self) -> int:
        return len(self._user_ids)
