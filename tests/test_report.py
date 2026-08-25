from pathlib import Path

from langgraph_agent_lab.metrics import MetricsReport, ScenarioMetric
from langgraph_agent_lab.report import render_report, write_report


def _metrics() -> MetricsReport:
    return MetricsReport(
        total_scenarios=2,
        success_rate=1.0,
        avg_nodes_visited=5.0,
        total_retries=1,
        total_interrupts=1,
        resume_success=True,
        persistence_backend="sqlite",
        thread_ids=["thread-alpha", "thread-beta"],
        checkpoint_count=8,
        scenario_metrics=[
            ScenarioMetric(
                scenario_id="alpha",
                success=True,
                expected_route="error",
                actual_route="error",
                visited_nodes=["retry", "tool", "evaluate", "answer", "finalize"],
                nodes_visited=5,
                retry_count=1,
                errors=["transient"],
                error_count=1,
            ),
            ScenarioMetric(
                scenario_id="beta",
                success=True,
                expected_route="risky",
                actual_route="risky",
                visited_nodes=["risky_action", "approval", "tool", "finalize"],
                nodes_visited=4,
                approval_count=1,
                interrupt_count=1,
                approval_required=True,
                approval_observed=True,
            ),
        ],
    )


def test_render_report_contains_required_sections() -> None:
    report = render_report(_metrics())

    for section in (
        "# Day 08 Lab Report",
        "## 1. Team / student",
        "## 2. Architecture",
        "## 3. State schema",
        "## 4. Scenario results",
        "## 5. Failure analysis",
        "## 6. Persistence / recovery evidence",
        "## 7. Extension work",
        "## 8. Improvement plan",
    ):
        assert section in report
    assert "alpha" in report
    assert "beta" in report
    assert "OpenRouter" in report
    assert "8 checkpoint snapshots" in report
    assert "Name: Nguyen" not in report


def test_write_report(tmp_path: Path) -> None:
    output = tmp_path / "lab_report.md"
    write_report(_metrics(), output)
    assert output.exists()
    assert "## 4. Scenario results" in output.read_text(encoding="utf-8")
