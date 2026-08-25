"""CLI for the lab."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
import yaml

from .graph import build_graph
from .metrics import MetricsReport, metric_from_state, summarize_metrics, write_metrics
from .persistence import build_checkpointer
from .report import write_report
from .scenarios import load_scenarios
from .state import initial_state

app = typer.Typer(no_args_is_help=True)


@app.command("run-scenarios")
def run_scenarios(
    config: Annotated[Path, typer.Option("--config")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Run all grading scenarios and write metrics JSON."""
    cfg = yaml.safe_load(config.read_text(encoding="utf-8"))
    scenarios = load_scenarios(cfg["scenarios_path"])
    checkpointer = build_checkpointer(cfg.get("checkpointer", "memory"), cfg.get("database_url"))
    graph = build_graph(checkpointer=checkpointer)
    metrics = []
    thread_ids = []
    checkpoint_counts = []
    native_interrupts = 0
    for scenario in scenarios:
        state = initial_state(scenario)
        run_config = {"configurable": {"thread_id": state["thread_id"]}}
        invocation = graph.invoke(state, config=run_config)
        final_state = invocation
        if isinstance(invocation, dict) and invocation.get("__interrupt__"):
            from langgraph.types import Command

            native_interrupts += 1
            final_state = graph.invoke(Command(resume=True), config=run_config)
        metrics.append(
            metric_from_state(
                final_state,
                scenario.expected_route.value,
                scenario.requires_approval,
            )
        )
        thread_ids.append(state["thread_id"])
        if checkpointer is not None:
            checkpoint_counts.append(len(list(graph.get_state_history(run_config))))
    report = summarize_metrics(
        metrics,
        persistence_backend=cfg.get("checkpointer", "memory"),
        thread_ids=thread_ids,
        checkpoint_count=sum(checkpoint_counts),
        resume_success=bool(checkpoint_counts),
    )
    write_metrics(report, output)
    if cfg.get("report_path"):
        write_report(report, cfg["report_path"])
    if hasattr(checkpointer, "close"):
        checkpointer.close()
    typer.echo(f"Native interrupts observed: {native_interrupts}")
    typer.echo(f"Wrote metrics to {output}")


@app.command("validate-metrics")
def validate_metrics(metrics: Annotated[Path, typer.Option("--metrics")]) -> None:
    """Validate metrics JSON schema for grading."""
    payload = json.loads(metrics.read_text(encoding="utf-8"))
    report = MetricsReport.model_validate(payload)
    if report.total_scenarios < 6:
        raise typer.BadParameter("Expected at least 6 scenarios")
    typer.echo(f"Metrics valid. success_rate={report.success_rate:.2%}")


if __name__ == "__main__":
    app()
