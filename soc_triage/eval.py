"""Triage evaluation harness.

How do we know the triage is any good? We score triage's independent decision
against the *intended* severity baked into generation (the weak label that is
never shown to triage). Metrics emphasise the asymmetric cost in a SOC:
under-triaging a real Sev1 is far worse than over-triaging routine noise.

    uv run python -m soc_triage.eval --mock
"""

from __future__ import annotations

import argparse

from .generate import generate_items
from .graph import build_graph, process_item
from .llm import make_chat_model
from .queue import TriageQueue
from .schema import MAX_LEVEL, MIN_LEVEL, name_for_level


def run(mock: bool, n_per_level: int):
    gen_llm = make_chat_model(mock=mock, temperature=0.6)
    triage_llm = make_chat_model(mock=mock, temperature=0.0)
    graph = build_graph(triage_llm)

    labeled = [li for li in generate_items(gen_llm, n_per_level=n_per_level) if li.intended_level is not None]
    levels = list(range(MIN_LEVEL, MAX_LEVEL + 1))
    confusion = {a: {p: 0 for p in levels} for a in levels}

    exact = within1 = reviewed = 0
    sev1_total = sev1_caught = 0
    under = over = 0
    cost = 0.0  # asymmetric: under-triage penalised 3x over-triage

    for li in labeled:
        assert li.intended_level is not None  # guaranteed by the filter above
        actual = li.intended_level
        q = TriageQueue()
        final = process_item(graph, li.item, q)
        res = final["result"]
        pred = res.priority_level
        confusion[actual][pred] += 1
        if pred == actual:
            exact += 1
        if abs(pred - actual) <= 1:
            within1 += 1
        if res.needs_human_review:
            reviewed += 1
        if actual == 1:
            sev1_total += 1
            if pred == 1:
                sev1_caught += 1
        if pred > actual:       # higher number = lower urgency => under-triage
            under += 1
            cost += 3 * (pred - actual)
        elif pred < actual:
            over += 1
            cost += 1 * (actual - pred)

    n = len(labeled)
    print("=" * 60)
    print(f"TRIAGE EVALUATION  (mock={mock}, n={n})")
    print("=" * 60)
    print(f"Exact-level accuracy : {exact}/{n} = {exact / n:.0%}")
    print(f"Within +/-1 level    : {within1}/{n} = {within1 / n:.0%}")
    sev1_recall_str = f"{sev1_caught / sev1_total:.0%}" if sev1_total else "n/a"
    print(f"Sev1 recall (critical): {sev1_caught}/{sev1_total} = {sev1_recall_str}   <- the costly miss")
    print(f"Under-triaged        : {under}   Over-triaged: {over}")
    print(f"Cost-weighted error  : {cost:.1f}  (under-triage weighted 3x)")
    print(f"Routed to review     : {reviewed}/{n} = {reviewed / n:.0%}")
    print("\nConfusion matrix (rows = intended, cols = predicted):")
    header = "        " + "".join(f"{name_for_level(p):>7}" for p in levels)
    print(header)
    for a in levels:
        row = "".join(f"{confusion[a][p]:>7}" for p in levels)
        print(f"  {name_for_level(a):<5} {row}")
    print("=" * 60)
    print("Note: intended labels are a weak baseline from generation, not gold.")
    print("With more time: a hand-labeled set + an LLM-as-judge on action quality.")


def main():
    ap = argparse.ArgumentParser(description="Evaluate SOC triage quality")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--n-per-level", type=int, default=4)
    args = ap.parse_args()
    run(mock=args.mock, n_per_level=args.n_per_level)


if __name__ == "__main__":
    main()
