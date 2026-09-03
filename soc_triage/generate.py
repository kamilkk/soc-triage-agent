"""Synthetic item generation.

Reliable synthetic inputs matter here: the queue's behaviour and the entire
triage evaluation are only as trustworthy as the inputs. So generation is
*controlled*:
  - forced coverage: N items per severity level, so the queue and eval exercise
    the full range and the re-ordering behaviour;
  - deliberate edge injection: ambiguous + off-topic items, to exercise the
    safeguards rather than hope they trigger;
  - no label leakage: the generator knows the intended severity (a weak label
    for eval) but the prompt forbids the text from naming it, so triage can't
    cheat.
"""

from __future__ import annotations

import itertools
import json
import re

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .models import IncomingItem, LabeledItem
from .schema import PRIORITY_SCHEMA, schema_as_prompt_block

_id_counter = itertools.count(1)


def _next_id() -> str:
    return f"itm-{next(_id_counter):04d}"


_GEN_SYSTEM = (
    "You generate realistic, free-text security alerts/tickets that an L1 SOC "
    "analyst would actually receive (from EDR, SIEM, email gateway, or user "
    "tickets). Vary the wording and source.\n\n"
    "Priority schema (for your reference only):\n{schema}\n\n"
    "HARD RULE: never state or hint at the severity/level in the text itself "
    "(do not write 'Sev1', 'critical', 'low priority', etc.). Describe only what "
    "was observed, so a triager must infer urgency from the content."
)

_TARGET_GUIDANCE = {
    "AMBIGUOUS": "Make each item genuinely ambiguous between two severity levels.",
    "OFFTOPIC": "Make each item something that is NOT a security event at all.",
}


def _gen_messages(target: str, n: int) -> list:
    sys = _GEN_SYSTEM.format(schema=schema_as_prompt_block())
    guidance = _TARGET_GUIDANCE.get(target, f"Each item should be a realistic {target} situation.")
    human = (
        "TASK: GENERATE\n"
        f"TARGET: {target}\n"
        f"N: {n}\n"
        f"{guidance}\n"
        'Respond with ONLY a JSON array of objects, each {"source": string, "text": string}.'
    )
    return [SystemMessage(content=sys), HumanMessage(content=human)]


def _parse_array(content: str) -> list[dict]:
    cleaned = re.sub(r"^```(?:json)?|```$", "", content.strip(), flags=re.MULTILINE).strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", cleaned, flags=re.DOTALL)
        if not match:
            return []
        data = json.loads(match.group(0))
    return [d for d in data if isinstance(d, dict) and d.get("text")]


def _gen_for_target(llm: BaseChatModel, target: str, n: int) -> list[dict]:
    try:
        msg = llm.invoke(_gen_messages(target, n))
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        return _parse_array(content)[:n]
    except Exception:
        return []  # generation is best-effort; a failed batch just yields fewer items


def generate_items(
    llm: BaseChatModel,
    n_per_level: int = 2,
    include_edge_cases: bool = True,
) -> list[LabeledItem]:
    """Generate a coverage-balanced batch of labeled items."""
    out: list[LabeledItem] = []
    for row in PRIORITY_SCHEMA:
        level = row["level"]
        for rec in _gen_for_target(llm, f"L{level}", n_per_level):
            out.append(
                LabeledItem(
                    item=IncomingItem(id=_next_id(), source=rec.get("source", "SIEM"), text=rec["text"]),
                    intended_level=level,
                    kind="normal",
                )
            )
    if include_edge_cases:
        for target, kind in (("AMBIGUOUS", "ambiguous"), ("OFFTOPIC", "offtopic")):
            for rec in _gen_for_target(llm, target, 1):
                out.append(
                    LabeledItem(
                        item=IncomingItem(id=_next_id(), source=rec.get("source", "misc"), text=rec["text"]),
                        intended_level=None,
                        kind=kind,
                    )
                )
    return out
