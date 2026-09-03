"""Live in-memory priority queue.

This is the long-lived operational state the driver queries ("what's next?").
It is deliberately separate from the LangGraph cognitive pipeline: the graph
turns each item into a ``TriageResult`` and pushes it here; this object can be
inspected at any moment, independent of the graph.

Ordering is a *pluggable strategy* (`key_fn`). v1 ships FIFO-within-level; the
urgency-aware strategy is wired and tested so the switch is one line.
"""

from __future__ import annotations

import heapq
import itertools
from typing import Callable

from .models import TriageResult

# An ordering strategy maps a TriageResult to a sort key (a tuple).
# Lower sorts first. The queue always appends a monotonic sequence number as the
# final tiebreaker, so two items are never compared by their payload.
KeyFn = Callable[[TriageResult], tuple]


def fifo_key(item: TriageResult) -> tuple:
    """v1 default: pure severity. Ties within a level fall through to arrival
    order (FIFO), supplied by the queue's sequence counter."""
    return (item.priority_level,)


def urgency_key(item: TriageResult) -> tuple:
    """v2 drop-in: severity, then intra-tier urgency (higher first). No other
    code changes needed -- the `urgency` field is already populated by triage."""
    return (item.priority_level, -(item.urgency or 0.0))


class TriageQueue:
    """A min-heap priority queue keyed by an injected ordering strategy."""

    def __init__(self, key_fn: KeyFn = fifo_key) -> None:
        self._key_fn = key_fn
        self._seq = itertools.count()
        self._heap: list[tuple[tuple, int, TriageResult]] = []

    def push(self, item: TriageResult) -> None:
        seq = next(self._seq)  # FIFO tiebreak + guarantees total order
        heapq.heappush(self._heap, (self._key_fn(item), seq, item))

    def pop(self) -> TriageResult | None:
        """Remove and return the highest-priority item, or None if empty."""
        if not self._heap:
            return None
        return heapq.heappop(self._heap)[2]

    def peek(self) -> TriageResult | None:
        """Return the highest-priority item without removing it ('what's next?')."""
        if not self._heap:
            return None
        return self._heap[0][2]

    def list_ordered(self) -> list[TriageResult]:
        """Snapshot of the current ordering, highest priority first."""
        return [entry[2] for entry in sorted(self._heap)]

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)
