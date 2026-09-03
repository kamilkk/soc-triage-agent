"""Triage one item into a TriageResult.

Layered for resilience (matters more with a local model):
  1. structured output via ``with_structured_output`` (tool calling);
  2. on failure, a plain call + manual JSON parse, wrapped in a retry;
  3. on parse failure, one stricter JSON reprompt;
  4. last resort, a deterministic keyword heuristic flagged for human review.

``triage_item`` therefore NEVER raises: a flaky model degrades to
"flagged-for-human", never to a crash or a silently wrong result.
"""

from __future__ import annotations

import json
import re
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from .models import IncomingItem, TriageResult
from .schema import MAX_LEVEL, MIN_LEVEL, schema_as_prompt_block

CONFIDENCE_THRESHOLD = 0.6  # below this -> route to human review

_SYSTEM = (
    "You are an L1 SOC analyst triaging incoming items. For each item decide: "
    "what kind of thing it is (category), the single next action required, the "
    "priority level from the schema, and your confidence.\n\n"
    "Priority schema:\n{schema}\n\n"
    "Rules:\n"
    "- If the item fits no category or is genuinely ambiguous, set "
    'category to "UNKNOWN" and a low confidence (< 0.6).\n'
    "- If it is ambiguous between two levels, pick the MORE severe one and "
    "lower your confidence.\n"
    "- confidence is 0..1. urgency is 0..1 (how urgent within its own level).\n"
)

_JSON_INSTRUCTION = (
    "Respond with ONLY a JSON object, no prose, no markdown fences, with keys: "
    'category (string), required_action (string), priority_level (integer 1-4), '
    "confidence (number 0-1), rationale (string), urgency (number 0-1)."
)


def _build_messages(item: IncomingItem, strict: bool = False) -> list:
    sys = _SYSTEM.format(schema=schema_as_prompt_block())
    human = (
        "TASK: TRIAGE\n"
        f"{_JSON_INSTRUCTION}\n"
        f"ITEM:\nsource={item.source}\n{item.text}"
    )
    if strict:
        human = "Return ONLY valid JSON. No explanation.\n" + human
    return [SystemMessage(content=sys), HumanMessage(content=human)]


def _clamp(res: TriageResult, item: IncomingItem) -> TriageResult:
    res.item_id = item.id
    res.priority_level = min(MAX_LEVEL, max(MIN_LEVEL, int(res.priority_level)))
    res.confidence = min(1.0, max(0.0, float(res.confidence)))
    if res.urgency is not None:
        res.urgency = min(1.0, max(0.0, float(res.urgency)))
    return res


def _extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?|```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return cast(dict[str, Any], json.loads(cleaned))
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        return cast(dict[str, Any], json.loads(match.group(0)))


def _structured_attempt(llm: BaseChatModel, item: IncomingItem) -> TriageResult | None:
    try:
        structured = llm.with_structured_output(TriageResult)
        res = structured.invoke(_build_messages(item))
        return _clamp(res, item) if isinstance(res, TriageResult) else None
    except Exception:
        return None


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.05, max=0.3), reraise=True)
def _invoke_with_retry(llm: BaseChatModel, messages: list):
    return llm.invoke(messages)


def _json_attempt(llm: BaseChatModel, item: IncomingItem, strict: bool = False) -> TriageResult | None:
    msg = _invoke_with_retry(llm, _build_messages(item, strict=strict))
    content = msg.content if isinstance(msg.content, str) else str(msg.content)
    data = _extract_json_object(content)
    data["item_id"] = item.id
    return _clamp(TriageResult.model_validate(data), item)


# Keyword heuristic -- the deterministic floor when the model is unavailable.
_HEURISTIC = [
    (1, ("ransom", "domain controller", "exfiltrat", "encrypting", "production down", "breach")),
    (2, ("phishing", "trojan", "malware", "lateral", "spreading", "degraded", "multiple hosts")),
    (4, ("password reset", "mfa reset", "access request", "allow-list", "whitelist", "routine")),
    (3, ("suspicious", "single user", "one endpoint", "impossible-travel", "unexpected")),
]


def heuristic_fallback(item: IncomingItem) -> TriageResult:
    low = item.text.lower()
    level = 3
    category = "UNKNOWN"
    for lvl, kws in _HEURISTIC:
        if any(k in low for k in kws):
            level, category = lvl, "heuristic-match"
            break
    return _clamp(
        TriageResult(
            item_id=item.id,
            category=category,
            required_action="Model unavailable -- triage by keyword heuristic; confirm manually.",
            priority_level=level,
            confidence=0.3,
            rationale="LLM triage failed; deterministic keyword fallback applied.",
            needs_human_review=True,
            urgency=0.5,
        ),
        item,
    )


def triage_item(llm: BaseChatModel, item: IncomingItem) -> TriageResult:
    """Triage one item, never raising."""
    res = _structured_attempt(llm, item)
    if res is not None:
        return res
    for strict in (False, True):
        try:
            res = _json_attempt(llm, item, strict=strict)
            if res is not None:
                return res
        except Exception:
            continue
    return heuristic_fallback(item)
