from soc_triage.models import TriageResult
from soc_triage.queue import TriageQueue, fifo_key, urgency_key


def mk(level, item_id="i", conf=0.9, urgency=0.0, cat="x"):
    return TriageResult(
        item_id=item_id, category=cat, required_action="a",
        priority_level=level, confidence=conf, rationale="r", urgency=urgency,
    )


def test_emergency_jumps_the_queue():
    q = TriageQueue()
    for i in range(3):
        q.push(mk(3, f"s{i}"))
    assert q.peek().priority_level == 3          # routine work on top
    q.push(mk(1, "emergency"))
    assert q.peek().priority_level == 1          # emergency now first
    assert q.peek().item_id == "emergency"
    assert len(q) == 4


def test_fifo_within_level():
    q = TriageQueue()
    q.push(mk(2, "first"))
    q.push(mk(2, "second"))
    q.push(mk(2, "third"))
    assert [q.pop().item_id for _ in range(3)] == ["first", "second", "third"]


def test_full_ordering_across_levels_is_severity_then_fifo():
    q = TriageQueue()
    q.push(mk(4, "routine"))
    q.push(mk(2, "high_a"))
    q.push(mk(2, "high_b"))
    q.push(mk(1, "crit"))
    assert [r.item_id for r in q.list_ordered()] == ["crit", "high_a", "high_b", "routine"]


def test_default_strategy_is_fifo():
    q = TriageQueue()
    assert q._key_fn is fifo_key  # documents the v1 default


def test_urgency_strategy_reorders_within_tier_only():
    q = TriageQueue(key_fn=urgency_key)
    q.push(mk(2, "low_urg", urgency=0.2))
    q.push(mk(2, "high_urg", urgency=0.9))
    q.push(mk(1, "sev1", urgency=0.1))
    order = [r.item_id for r in q.list_ordered()]
    assert order[0] == "sev1"        # cross-tier ordering preserved
    assert order[1] == "high_urg"    # within Sev2, higher urgency first
    assert order[2] == "low_urg"


def test_pop_and_peek_on_empty():
    q = TriageQueue()
    assert q.peek() is None
    assert q.pop() is None
    assert not q
