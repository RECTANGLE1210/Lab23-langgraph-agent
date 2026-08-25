"""Metrics schema and helpers."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field


class ScenarioMetric(BaseModel):
    scenario_id: str
    success: bool
    expected_route: str
    actual_route: str | None = None
    nodes_visited: int = 0
    visited_nodes: list[str] = Field(default_factory=list)
    retry_count: int = 0
    interrupt_count: int = 0
    approval_count: int = 0
    approval_required: bool = False
    approval_observed: bool = False
    error_count: int = 0
    latency_ms: int = 0
    errors: list[str] = Field(default_factory=list)


class MetricsReport(BaseModel):
    total_scenarios: int
    success_rate: float
    avg_nodes_visited: float
    total_retries: int
    total_interrupts: int
    resume_success: bool = False
    scenario_metrics: list[ScenarioMetric]
    persistence_backend: str | None = None
    thread_ids: list[str] = Field(default_factory=list)
    checkpoint_count: int = 0


def metric_from_state(
    state: dict[str, Any], expected_route: str, approval_required: bool
) -> ScenarioMetric:
    """Extract execution metrics from one completed AgentState."""
    raw_events = state.get("events", []) or []
    events = [event for event in raw_events if isinstance(event, dict)]
    raw_errors = state.get("errors", []) or []
    errors = list(raw_errors) if isinstance(raw_errors, list) else []
    actual_route = state.get("route")
    visited_nodes = [str(event.get("node", "unknown")) for event in events]
    retry_count = sum(1 for node in visited_nodes if node == "retry")
    approval_events = sum(1 for node in visited_nodes if node == "approval")
    approval_state_present = state.get("approval") is not None
    approval_count = approval_events or int(approval_state_present)
    latency_ms = sum(_event_latency(event) for event in events)
    has_output = bool(state.get("final_answer") or state.get("pending_question"))
    has_finalize = "finalize" in visited_nodes
    approval_observed = approval_count > 0
    success = (
        actual_route == expected_route
        and has_output
        and has_finalize
        and (not approval_required or approval_observed)
    )
    return ScenarioMetric(
        scenario_id=str(state.get("scenario_id", "unknown")),
        success=success,
        expected_route=expected_route,
        actual_route=actual_route,
        nodes_visited=len(visited_nodes),
        visited_nodes=visited_nodes,
        retry_count=retry_count,
        interrupt_count=approval_count,
        approval_count=approval_count,
        approval_required=approval_required,
        approval_observed=approval_observed,
        error_count=len(errors),
        latency_ms=latency_ms,
        errors=errors,
    )


def summarize_metrics(
    items: list[ScenarioMetric],
    *,
    persistence_backend: str | None = None,
    thread_ids: list[str] | None = None,
    checkpoint_count: int = 0,
    resume_success: bool = False,
) -> MetricsReport:
    if not items:
        raise ValueError("No scenario metrics to summarize")
    return MetricsReport(
        total_scenarios=len(items),
        success_rate=sum(1 for item in items if item.success) / len(items),
        avg_nodes_visited=mean(item.nodes_visited for item in items),
        total_retries=sum(item.retry_count for item in items),
        total_interrupts=sum(item.interrupt_count for item in items),
        resume_success=resume_success,
        scenario_metrics=items,
        persistence_backend=persistence_backend,
        thread_ids=list(thread_ids or []),
        checkpoint_count=checkpoint_count,
    )


def write_metrics(report: MetricsReport, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.model_dump(), indent=2, ensure_ascii=False), encoding="utf-8")


def _event_latency(event: dict[str, Any]) -> int:
    value = event.get("latency_ms", 0)
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0
