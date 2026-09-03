from soc_triage.generate import generate_items
from soc_triage.graph import build_graph, process_item
from soc_triage.llm import make_chat_model
from soc_triage.models import IncomingItem
from soc_triage.queue import TriageQueue


def test_full_batch_ordering_and_safeguards():
    llm = make_chat_model(mock=True)
    g = build_graph(llm)
    q = TriageQueue()

    raws = [
        IncomingItem(id="a", source="user-ticket", text="user needs a password reset for a new phone"),  # Sev4
        IncomingItem(id="b", source="EDR", text="ransomware encrypting files on the domain controller; production down"),  # Sev1
        IncomingItem(id="c", source="SIEM", text="impossible-travel alert for a single user; account still active"),  # Sev3
        {"id": "d", "source": "SIEM"},  # malformed -> dead-letter
        IncomingItem(id="e", source="SIEM", text="AMBIGUOUS_EDGE could be recon or the new scanner"),  # UNKNOWN -> review Sev3
    ]
    finals = [process_item(g, r, q) for r in raws]

    outcomes = {f.get("result").item_id: f["outcome"] for f in finals if f.get("result")}
    assert outcomes["b"] == "normal"
    assert outcomes["e"] == "review"

    order = [r.item_id for r in q.list_ordered()]
    assert order[0] == "b"            # Sev1 first
    assert "d" not in order           # malformed never enqueued
    assert len(q) == 4                # a, b, c, e
    assert any(r.item_id == "e" and r.needs_human_review for r in q.list_ordered())


def test_generation_covers_all_levels_and_edge_cases():
    llm = make_chat_model(mock=True)
    items = generate_items(llm, n_per_level=2)
    normal = [li for li in items if li.kind == "normal"]
    assert {li.intended_level for li in normal} == {1, 2, 3, 4}  # full coverage
    assert any(li.kind == "ambiguous" for li in items)
    assert any(li.kind == "offtopic" for li in items)
    # no label leakage: text must not name an explicit severity label
    import re
    leak = re.compile(r"\bsev\s?[1-4]\b|\bseverity\b|\bpriority [1-4]\b", re.I)
    for li in items:
        assert not leak.search(li.item.text)
