"""Typed data contracts.

These models are the backbone of the safeguards: a parse/validation failure
becomes a *routable event* (dead-letter or heuristic fallback) rather than an
exception that takes the run down.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


class IncomingItem(BaseModel):
    """A validated raw item arriving at the SOC queue."""

    id: str
    source: str = "unknown"  # e.g. "EDR", "SIEM", "email-gateway", "user-ticket"
    text: str = Field(min_length=1)


class TriageResult(BaseModel):
    """The triage decision for one item.

    ``urgency`` is reserved for intra-tier ordering. It is unused by the v1
    FIFO queue strategy but populated now, so switching the queue's ``key_fn``
    to urgency-aware ordering needs no change to this contract (see queue.py).
    """

    item_id: str
    category: str  # short label, or "UNKNOWN"
    required_action: str
    priority_level: int  # 1..4 (lower = more urgent)
    confidence: float  # 0..1
    rationale: str = ""
    needs_human_review: bool = False
    urgency: float | None = None  # 0..1 intra-tier urgency; reserved for v2


@dataclass
class LabeledItem:
    """A generated item paired with its *intended* severity (weak ground truth
    for evaluation). The intended level is NEVER shown to triage; it exists only
    so eval can score triage's independent decision against it.
    """

    item: IncomingItem
    intended_level: int | None  # None for ambiguous / off-topic edge cases
    kind: str  # "normal" | "ambiguous" | "offtopic"
