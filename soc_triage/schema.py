"""Priority schema for the SOC use case.

A schema level has a ``priority_name``, an integer ``level`` (lower = more
urgent), and a short ``description``. The schema is the single source of truth
for both generation (what kinds of items to invent) and triage (what levels an
item can map to).
"""

from __future__ import annotations

from typing import TypedDict


class PrioritySchemaRow(TypedDict):
    priority_name: str
    level: int
    description: str


PRIORITY_SCHEMA: list[PrioritySchemaRow] = [
    {
        "priority_name": "Sev1",
        "level": 1,
        "description": (
            "Critical: production down for a client, or an active, confirmed "
            "breach (ransomware encrypting hosts, domain-controller compromise, "
            "active data exfiltration). Immediate response, page on-call."
        ),
    },
    {
        "priority_name": "Sev2",
        "level": 2,
        "description": (
            "High: degraded or partial service, or a high-confidence intrusion "
            "in progress affecting multiple hosts/users (malware spreading, "
            "targeted phishing wave, lateral movement). Respond within minutes."
        ),
    },
    {
        "priority_name": "Sev3",
        "level": 3,
        "description": (
            "Moderate: a single-user or single-host, scoped issue, or "
            "suspicious-but-unconfirmed activity (one impossible-travel alert, "
            "one unexpected process). Investigate during the shift."
        ),
    },
    {
        "priority_name": "Sev4",
        "level": 4,
        "description": (
            "Low: routine request or informational/low-risk noise (access "
            "request, MFA reset, allow-list a vendor domain). Queue as routine."
        ),
    },
]

# Conservative default level for items that fit no category / are ambiguous /
# are low-confidence. Sev3 surfaces them above routine noise without crying
# wolf at Sev1 (alert fatigue) -- a human makes the final call.
DEFAULT_UNKNOWN_LEVEL: int = 3

VALID_LEVELS: set[int] = {row["level"] for row in PRIORITY_SCHEMA}
MIN_LEVEL: int = min(VALID_LEVELS)
MAX_LEVEL: int = max(VALID_LEVELS)

_NAME_BY_LEVEL: dict[int, str] = {row["level"]: row["priority_name"] for row in PRIORITY_SCHEMA}
_DESC_BY_LEVEL: dict[int, str] = {row["level"]: row["description"] for row in PRIORITY_SCHEMA}


def name_for_level(level: int) -> str:
    return _NAME_BY_LEVEL.get(level, f"L{level}")


def description_for_level(level: int) -> str:
    return _DESC_BY_LEVEL.get(level, "")


def schema_as_prompt_block() -> str:
    """Render the schema for inclusion in an LLM prompt."""
    lines = []
    for row in PRIORITY_SCHEMA:
        lines.append(f"- {row['priority_name']} (level {row['level']}): {row['description']}")
    return "\n".join(lines)
