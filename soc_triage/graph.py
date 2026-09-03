"""LangGraph cognitive pipeline (the 'brain').

A per-item subgraph turns one raw item into an enqueued ``TriageResult`` or
routes it aside:

    validate ──valid──► triage ──► route ──normal──► enqueue_normal ─► END
        │                               └─review──► enqueue_review ─► END
        └─malformed──► END (dead_letter)

The live ``TriageQueue`` is NOT graph state -- it is injected via config
(``configurable.queue``) as a long-lived singleton the orchestrator owns and
can inspect at any moment. This is the clean separation between the cognitive
pipeline and the operational state.
"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from .models import IncomingItem, TriageResult
from .queue import TriageQueue
from .schema import DEFAULT_UNKNOWN_LEVEL
from .triage import CONFIDENCE_THRESHOLD, triage_item


class AgentState(TypedDict, total=False):
    raw: Any
    item: IncomingItem
    result: TriageResult
    error: str
    outcome: str  # "dead_letter" | "normal" | "review"


def _get_queue(config: RunnableConfig) -> TriageQueue:
    try:
        return cast(TriageQueue, config["configurable"]["queue"])
    except (KeyError, TypeError) as exc:
        raise RuntimeError("TriageQueue must be injected via config['configurable']['queue']") from exc


def build_graph(llm: BaseChatModel):
    """Compile the per-item triage graph bound to a given LLM."""

    def validate(state: AgentState) -> dict:
        raw = state["raw"]
        try:
            item = raw if isinstance(raw, IncomingItem) else IncomingItem(**raw)
            return {"item": item}
        except Exception as exc:  # malformed -> dead-letter, never crash
            return {"error": f"{type(exc).__name__}: {exc}", "outcome": "dead_letter"}

    def triage(state: AgentState) -> dict:
        return {"result": triage_item(llm, state["item"])}

    def route(state: AgentState) -> dict:
        res = state["result"]
        outcome = "normal"
        if res.category.strip().upper() == "UNKNOWN":
            res.priority_level = DEFAULT_UNKNOWN_LEVEL  # conservative surface level
            res.needs_human_review = True
            outcome = "review"
        elif res.confidence < CONFIDENCE_THRESHOLD:
            res.needs_human_review = True
            outcome = "review"
        elif res.needs_human_review:  # set by heuristic fallback
            outcome = "review"
        return {"result": res, "outcome": outcome}

    def enqueue_normal(state: AgentState, config: RunnableConfig) -> dict:
        _get_queue(config).push(state["result"])
        return {"outcome": "normal"}

    def enqueue_review(state: AgentState, config: RunnableConfig) -> dict:
        res = state["result"]
        res.needs_human_review = True
        _get_queue(config).push(res)
        return {"result": res, "outcome": "review"}

    g = StateGraph(AgentState)
    g.add_node("validate", validate)
    g.add_node("triage", triage)
    g.add_node("route", route)
    g.add_node("enqueue_normal", enqueue_normal)
    g.add_node("enqueue_review", enqueue_review)

    g.add_edge(START, "validate")
    g.add_conditional_edges(
        "validate",
        lambda s: "dead_letter" if s.get("error") else "ok",
        {"dead_letter": END, "ok": "triage"},
    )
    g.add_edge("triage", "route")
    g.add_conditional_edges(
        "route",
        lambda s: s["outcome"],
        {"normal": "enqueue_normal", "review": "enqueue_review"},
    )
    g.add_edge("enqueue_normal", END)
    g.add_edge("enqueue_review", END)
    return g.compile()


def process_item(graph, raw: Any, queue: TriageQueue) -> AgentState:
    """Run one raw item through the graph against a live queue."""
    return cast(AgentState, graph.invoke({"raw": raw}, config={"configurable": {"queue": queue}}))
