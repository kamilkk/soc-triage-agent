"""Runnable demo / CLI.

    uv run python -m soc_triage.demo --mock      # offline, deterministic
    uv run python -m soc_triage.demo             # against a real GLM-5.2 vLLM endpoint

Shows: synthetic generation, per-item triage, the live priority queue answering
"what's next?", the emergency-jumps-the-queue dynamic, and the safeguards
(ambiguous -> review, malformed -> dead-letter, failed call -> heuristic).
"""

from __future__ import annotations

import argparse

from .generate import generate_items
from .graph import build_graph, process_item
from .llm import make_chat_model
from .models import IncomingItem
from .queue import TriageQueue, fifo_key, urgency_key
from .schema import name_for_level


def _line(char="-", n=78):
    print(char * n)


def _fmt(res) -> str:
    flag = "  [REVIEW]" if res.needs_human_review else ""
    return (f"{name_for_level(res.priority_level)} conf={res.confidence:.2f} "
            f"{res.category:<22} {res.item_id}{flag}")


def _show_queue(q: TriageQueue):
    print(f'\n  "what\'s next?" -> {_fmt(q.peek()) if q else "(queue empty)"}')
    print("  current ordering:")
    for i, res in enumerate(q.list_ordered(), 1):
        print(f"    {i:>2}. {_fmt(res)}")


def run(mock: bool, n_per_level: int, strategy: str):
    key_fn = urgency_key if strategy == "urgency" else fifo_key
    gen_llm = make_chat_model(mock=mock, temperature=0.6)
    triage_llm = make_chat_model(mock=mock, temperature=0.0)
    graph = build_graph(triage_llm)
    queue = TriageQueue(key_fn=key_fn)
    dead_letter: list = []

    _line("=")
    print(f"SOC TRIAGE AGENT  (mock={mock}, ordering={strategy})")
    _line("=")

    # 1) Generate a coverage-balanced batch.
    print("\n[1] Generating synthetic incoming items...")
    items = generate_items(gen_llm, n_per_level=n_per_level)
    for li in items:
        tag = "" if li.kind == "normal" else f"  <{li.kind}>"
        print(f"  - [{li.item.source}] {li.item.text[:90]}{tag}")

    # 2) Triage each through the LangGraph pipeline into the live queue.
    #    We also inject one malformed raw item to exercise the dead-letter path.
    print("\n[2] Triaging items into the live queue...")
    raws = [li.item for li in items] + [{"id": "malformed-1", "source": "SIEM"}]  # missing 'text'
    for raw in raws:
        final = process_item(graph, raw, queue)
        if final["outcome"] == "dead_letter":
            dead_letter.append(final)
        else:
            print(f"    {final['outcome']:<7} {_fmt(final['result'])}")

    _show_queue(queue)

    # 3) Emergency-jumps-the-queue: drain to a few routine items, then inject a Sev1.
    _line()
    print("[3] Emergency scenario: a Sev1 arrives while routine items are queued.")
    scenario_q = TriageQueue(key_fn=key_fn)
    routine = [
        IncomingItem(id="rt-1", source="user-ticket", text="user needs an MFA reset after a new phone"),
        IncomingItem(id="rt-2", source="user-ticket", text="please allow-list a vendor domain for a webinar"),
        IncomingItem(id="rt-3", source="SIEM", text="impossible-travel alert for a single user, still active"),
    ]
    for r in routine:
        process_item(graph, r, scenario_q)
    print("  three items queued; top of queue:")
    print(f'    "what\'s next?" -> {_fmt(scenario_q.peek())}')

    emergency = IncomingItem(id="EMERGENCY", source="EDR",
                             text="ransomware encrypting files across the production cluster; client portal is down")
    process_item(graph, emergency, scenario_q)
    print("  ...emergency arrives...")
    top = scenario_q.peek()
    print(f'    "what\'s next?" -> {_fmt(top)}')
    assert top is not None and top.item_id == "EMERGENCY", "emergency should jump to the front"
    print("  (the Sev1 jumped ahead of the queued items, as expected)")

    # 4) Failure path: a forced LLM error degrades to the heuristic, not a crash.
    _line()
    print("[4] Failure path: forced LLM error -> heuristic fallback (no crash).")
    boom = IncomingItem(id="boom", source="EDR",
                        text="FORCE_LLM_ERROR ransomware on the domain controller")
    final = process_item(graph, boom, queue)
    print(f"    -> {_fmt(final['result'])}  (rationale: {final['result'].rationale})")

    # 5) Dead-letter summary.
    _line()
    print(f"[5] Dead-letter (malformed/unprocessable): {len(dead_letter)} item(s)")
    for f in dead_letter:
        print(f"    {f.get('error')}")
    if not dead_letter:
        print("    (none in this batch)")
    _line("=")


def main():
    ap = argparse.ArgumentParser(description="SOC next-best-action triage agent demo")
    ap.add_argument("--mock", action="store_true", help="use the offline deterministic model")
    ap.add_argument("--n-per-level", type=int, default=2, help="synthetic items per severity level")
    ap.add_argument("--strategy", choices=["fifo", "urgency"], default="fifo",
                    help="queue ordering strategy (urgency demonstrates the pluggable key_fn)")
    args = ap.parse_args()
    run(mock=args.mock, n_per_level=args.n_per_level, strategy=args.strategy)


if __name__ == "__main__":
    main()
