"""LLM boundary.

`make_chat_model()` returns a LangChain ``BaseChatModel``:
- real:  ``ChatOpenAI`` pointed at the local GLM-5.2 vLLM OpenAI-compatible
         endpoint (config via env vars). vLLM ignores the api key but the
         OpenAI client requires a non-empty string.
- mock:  a deterministic ``MockChatModel`` that needs no network. It returns
         canned, label-clean SOC scenarios for generation and keyword-based
         triage JSON for triage -- enough to run the demo/tests offline.

The mock intentionally does NOT support tool calling (`bind_tools` raises), so
`with_structured_output` fails and triage degrades to its JSON-parsing path --
exactly the behaviour we expect from a quantized local model, and a realistic
exercise of the safeguard.
"""

from __future__ import annotations

import json
import os
from typing import Any, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr


def make_chat_model(mock: bool = False, temperature: float = 0.0) -> BaseChatModel:
    if mock:
        return MockChatModel()
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=os.environ.get("LLM_MODEL", "zai-org/GLM-5.2-FP8"),
        base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
        api_key=os.environ.get("LLM_API_KEY", "local"),  # type: ignore[arg-type]
        temperature=temperature,
        timeout=60,
        max_retries=0,  # we own retries (tenacity) in triage.py
    )


# --------------------------------------------------------------------------- #
# Mock model
# --------------------------------------------------------------------------- #

# Canned, severity-clean SOC scenarios. The text never names its own severity
# (no "this is a Sev1") so triage can't cheat -- it must infer from content.
_GEN_BANK: dict[str, list[dict]] = {
    "L1": [
        {"source": "SIEM", "text": "Multiple servers in the production cluster for client Northstar are unreachable; monitoring shows mass file-rename activity and ransom notes appearing on shared drives."},
        {"source": "EDR", "text": "Credential-dumping detected on the primary domain controller with outbound traffic to a known exfiltration host; the client-facing portal is now down."},
    ],
    "L2": [
        {"source": "EDR", "text": "Endpoint protection quarantined a trojan on three finance workstations within ten minutes; users report the app is slow but still working."},
        {"source": "email-gateway", "text": "A targeted phishing wave is hitting the sales org; two users already entered credentials on a lookalike portal and mail flow is degraded."},
    ],
    "L3": [
        {"source": "EDR", "text": "One analyst reports an unexpected PowerShell window opened on their workstation this morning; no spread observed so far."},
        {"source": "SIEM", "text": "A single user's account triggered an impossible-travel alert from a new country; the account is still active."},
    ],
    "L4": [
        {"source": "user-ticket", "text": "A staff member opened a ticket asking to allow-list a vendor domain for a scheduled webinar next week."},
        {"source": "user-ticket", "text": "Routine request: a user needs an MFA reset after getting a new phone."},
    ],
    "AMBIGUOUS": [
        {"source": "SIEM", "text": "A tool flagged an internal host scanning several subnets at 2am -- could be an attacker doing recon, or just the new vulnerability scanner IT deployed without telling anyone."},
    ],
    "OFFTOPIC": [
        {"source": "facilities", "text": "Reminder: the coffee machine on floor 3 is out of order; facilities has been notified."},
    ],
}

# urgency baseline by level (intra-tier signal; only used if the queue runs the
# urgency strategy). Higher = more urgent within the tier.
_URGENCY_BY_LEVEL = {1: 0.9, 2: 0.65, 3: 0.4, 4: 0.1}


def _mock_triage(text: str) -> dict:
    """Deterministic keyword triage used by the mock model."""
    low = text.lower()

    def build(level, category, action, conf, rationale):
        return {
            "category": category,
            "required_action": action,
            "priority_level": level,
            "confidence": conf,
            "rationale": rationale,
            "urgency": _URGENCY_BY_LEVEL.get(level, 0.3),
        }

    force_low = "low_conf" in low

    # Ambiguous / off-topic -> UNKNOWN (route will send to review at Sev3).
    if "ambiguous_edge" in low or ("could be" in low and " or " in low):
        return build(3, "UNKNOWN", "Manually confirm whether this is benign or malicious before acting.", 0.45, "Reads as either recon or sanctioned scanning; cannot disambiguate from content alone.")
    if "coffee" in low or ("facilities" in low and "notified" in low):
        return build(3, "UNKNOWN", "Likely out of scope for the SOC; confirm and reassign.", 0.3, "Does not describe a security-relevant event.")

    # Severity by keyword.
    if any(k in low for k in ("ransom", "domain controller", "exfiltrat", "production cluster", "client-facing portal is now down", "encrypting")):
        r = build(1, "active-breach", "Page on-call IR; isolate affected hosts; begin containment immediately.", 0.92, "Indicators of an active, high-impact compromise / client production impact.")
    elif any(k in low for k in ("three finance", "phishing wave", "trojan", "quarantined", "lateral", "degraded", "multiple hosts", "lookalike portal")):
        r = build(2, "intrusion-in-progress", "Scope affected accounts/hosts; reset exposed credentials; contain spread.", 0.85, "Confirmed malicious activity affecting multiple users/hosts but not yet client-down.")
    elif any(k in low for k in ("one analyst", "single user", "one endpoint", "impossible-travel", "one workstation", "unexpected powershell", "suspicious login")):
        r = build(3, "suspicious-activity", "Investigate the single affected host/account during the shift; watch for spread.", 0.8, "Scoped, single-entity, suspicious-but-unconfirmed activity.")
    elif any(k in low for k in ("allow-list", "whitelist", "mfa reset", "password reset", "access request", "webinar", "routine request")):
        r = build(4, "routine-request", "Handle as a standard service request per runbook.", 0.88, "Routine administrative request, no security incident.")
    else:
        return build(3, "UNKNOWN", "No clear category; route to a human for manual triage.", 0.4, "Content does not match any known category.")

    if force_low:
        r["confidence"] = 0.4
    return r


class MockChatModel(BaseChatModel):
    """A deterministic, network-free chat model for demo + tests."""

    _counters: dict[str, int] = PrivateAttr(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "mock-soc"

    def bind_tools(self, *args: Any, **kwargs: Any):  # noqa: D401
        # Simulate a local model with no reliable tool-calling: force callers of
        # with_structured_output down the JSON-parsing fallback path.
        raise NotImplementedError("MockChatModel does not support tool calling")

    def _last_human(self, messages: list[BaseMessage]) -> str:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                return msg.content if isinstance(msg.content, str) else str(msg.content)
        return str(messages[-1].content) if messages else ""

    def _next(self, key: str, n: int) -> int:
        i = self._counters.get(key, 0)
        self._counters[key] = i + 1
        return i % max(n, 1)

    def _respond(self, prompt: str) -> str:
        if "FORCE_LLM_ERROR" in prompt:
            raise RuntimeError("simulated LLM/transport failure")

        if "TASK: GENERATE" in prompt:
            target = "L3"
            n = 1
            for line in prompt.splitlines():
                if line.startswith("TARGET:"):
                    target = line.split(":", 1)[1].strip()
                elif line.startswith("N:"):
                    try:
                        n = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        n = 1
            bank = _GEN_BANK.get(target, _GEN_BANK["L3"])
            items = []
            for _ in range(n):
                idx = self._next(target, len(bank))
                items.append(bank[idx])
            return json.dumps(items)

        # Default: triage. Item text follows the "ITEM:" marker if present.
        text = prompt
        if "ITEM:" in prompt:
            text = prompt.split("ITEM:", 1)[1].strip()
        return json.dumps(_mock_triage(text))

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: Optional[list[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        content = self._respond(self._last_human(messages))
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
