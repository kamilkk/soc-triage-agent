from soc_triage.graph import build_graph, process_item
from soc_triage.llm import make_chat_model
from soc_triage.models import IncomingItem
from soc_triage.queue import TriageQueue


def setup():
    llm = make_chat_model(mock=True)
    return build_graph(llm), TriageQueue()


def test_clear_sev1_is_normal_and_enqueued():
    g, q = setup()
    raw = IncomingItem(id="a", source="EDR",
                       text="ransomware encrypting files on the domain controller; production down")
    final = process_item(g, raw, q)
    assert final["outcome"] == "normal"
    assert final["result"].priority_level == 1
    assert not final["result"].needs_human_review
    assert len(q) == 1


def test_unknown_routes_to_review_at_sev3():
    g, q = setup()
    raw = IncomingItem(id="amb", source="SIEM",
                       text="AMBIGUOUS_EDGE host scanning subnets, could be recon or the new scanner")
    final = process_item(g, raw, q)
    assert final["outcome"] == "review"
    assert final["result"].priority_level == 3
    assert final["result"].needs_human_review
    assert len(q) == 1  # still queued, just flagged


def test_low_confidence_routes_to_review():
    g, q = setup()
    raw = IncomingItem(id="lc", source="EDR",
                       text="LOW_CONF suspicious login for a single user on one endpoint")
    final = process_item(g, raw, q)
    assert final["outcome"] == "review"
    assert final["result"].needs_human_review
    assert final["result"].confidence < 0.6


def test_malformed_item_goes_to_dead_letter_without_crashing():
    g, q = setup()
    final = process_item(g, {"id": "bad", "source": "SIEM"}, q)  # missing text
    assert final["outcome"] == "dead_letter"
    assert "error" in final
    assert len(q) == 0  # never enters the priority queue


def test_llm_failure_falls_back_to_heuristic():
    g, q = setup()
    raw = IncomingItem(id="boom", source="EDR",
                       text="FORCE_LLM_ERROR ransomware encrypting the domain controller")
    final = process_item(g, raw, q)
    res = final["result"]
    assert res.needs_human_review            # heuristic always flags
    assert final["outcome"] == "review"
    assert res.priority_level == 1           # heuristic still catches ransomware as Sev1
    assert len(q) == 1
