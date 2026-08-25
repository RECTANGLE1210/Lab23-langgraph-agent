"""Markdown report generation using the official lab template structure."""

from __future__ import annotations

import os
from pathlib import Path

from .metrics import MetricsReport, ScenarioMetric

DEFAULT_MODEL = "google/gemini-3.7-flash"


def render_report(metrics: MetricsReport) -> str:
    """Render metrics into the official eight-section report structure."""
    model = os.getenv("LLM_MODEL") or DEFAULT_MODEL
    successful = sum(item.success for item in metrics.scenario_metrics)
    thread_ids = metrics.thread_ids or [
        f"thread-{item.scenario_id}" for item in metrics.scenario_metrics
    ]
    lines = [
        "# Day 08 Lab Report",
        "",
        "## 1. Team / student",
        "",
        "- Name:",
        "- Repo/commit:",
        "- Date:",
        "",
        "## 2. Architecture",
        "",
        (
            "The graph follows `START -> intake -> classify`, then uses conditional "
            "routing: simple -> answer; tool -> tool -> evaluate; missing_info -> "
            "clarify; risky -> risky_action -> approval -> tool -> evaluate; and "
            "error -> retry -> tool/dead_letter. Every path ends at `finalize -> END`."
        ),
        (
            f"`classify_node` uses OpenRouter with `{model}` and structured output. "
            "`answer_node` uses grounded LLM generation. The workflow has bounded "
            "retry, deterministic mock approval, and SQLite checkpoint persistence."
        ),
        "",
        "## 3. State schema",
        "",
        (
            "Important current-state fields overwrite their previous value; audit "
            "and execution histories append with `operator.add`."
        ),
        "",
        "| Field | Reducer | Why |",
        "|---|---|---|",
        *_state_schema_rows(),
        "",
        "## 4. Scenario results",
        "",
        (
            f"Observed execution summary: {metrics.total_scenarios} scenarios, "
            f"{successful} successful, success rate {metrics.success_rate:.1%}."
        ),
        "",
        "| Scenario | Expected route | Actual route | Success | Retries | Interrupts |",
        "|---|---|---|---:|---:|---:|",
        *_scenario_rows(metrics.scenario_metrics),
        "",
        "## 5. Failure analysis",
        "",
        *_failure_analysis(metrics.scenario_metrics),
        "",
        "## 6. Persistence / recovery evidence",
        "",
        *_persistence_lines(metrics, thread_ids),
        "",
        "## 7. Extension work",
        "",
        (
            "No optional extension has been finalized yet; SQLite persistence/"
            "recovery is implemented as the persistence requirement. A graph "
            "diagram or real HITL flow may be added in the final extension gate."
        ),
        "",
        "## 8. Improvement plan",
        "",
        "- Replace mock approval with real `interrupt()/resume` HITL.",
        "- Improve tool-result evaluation with LLM-as-judge or a stronger evaluator.",
        "- Add production observability for latency, provider, and checkpoint failures.",
        "",
    ]
    return "\n".join(lines)


def write_report(metrics: MetricsReport, output_path: str | Path) -> None:
    """Write the rendered report to a file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report(metrics), encoding="utf-8")


def _state_schema_rows() -> list[str]:
    overwrite_fields = (
        "thread_id",
        "scenario_id",
        "query",
        "route",
        "risk_level",
        "attempt",
        "max_attempts",
        "evaluation_result",
        "pending_question",
        "proposed_action",
        "approval",
        "final_answer",
    )
    append_fields = ("messages", "tool_results", "errors", "events")
    rows = [f"| {field} | overwrite | current workflow state |" for field in overwrite_fields]
    rows.extend(
        f"| {field} | append / operator.add | execution and audit history |"
        for field in append_fields
    )
    return rows


def _scenario_rows(items: list[ScenarioMetric]) -> list[str]:
    return [
        f"| {item.scenario_id} | {item.expected_route} | {item.actual_route or '-'} | "
        f"{item.success} | {item.retry_count} | {item.approval_count} |"
        for item in items
    ]


def _failure_analysis(items: list[ScenarioMetric]) -> list[str]:
    retry_items = [
        item
        for item in items
        if item.retry_count and "dead_letter" not in item.visited_nodes
    ]
    risky_items = [item for item in items if item.approval_count]
    dead_letter_items = [
        item for item in items if "dead_letter" in item.visited_nodes
    ]
    retry_ids = ", ".join(item.scenario_id for item in retry_items) or "none"
    risky_ids = ", ".join(item.scenario_id for item in risky_items) or "none"
    dead_letter_ids = ", ".join(item.scenario_id for item in dead_letter_items) or "none"
    retry_count = sum(item.retry_count for item in retry_items)
    recovery = (
        "recovered"
        if retry_items and all(item.success for item in retry_items)
        else "did not fully recover"
    )
    return [
        (
            f"1. Retry or tool failure: observed in `{retry_ids}`. The retry node "
            f"increments the attempt, `evaluate` requests another try when the "
            f"tool result contains an error, and the bounded workflow {recovery}. "
            f"Observed retry events: {retry_count}."
        ),
        (
            f"2. Risky action without approval: approval-path evidence was "
            f"observed in `{risky_ids}`. Risky classification reaches "
            "`risky_action` and then `approval`; only approval proceeds to the "
            f"tool, while rejection routes to clarification. Additional retry "
            f"exhaustion observed in `{dead_letter_ids}`."
        ),
    ]


def _persistence_lines(metrics: MetricsReport, thread_ids: list[str]) -> list[str]:
    backend = metrics.persistence_backend or "SQLite"
    checkpoint = (
        f"{metrics.checkpoint_count} checkpoint snapshots were observed."
        if metrics.checkpoint_count
        else "Checkpoint count is not embedded in this metrics object."
    )
    resume = "passed" if metrics.resume_success else "not recorded in this run"
    return [
        f"Backend: `langgraph.checkpoint.sqlite.SqliteSaver` (run backend: `{backend}`).",
        f"Thread IDs: {', '.join(f'`{thread_id}`' for thread_id in thread_ids)}.",
        (
            f"{checkpoint} State history is scoped by `thread_id`; "
            f"resume/state-history evidence: {resume}."
        ),
    ]
