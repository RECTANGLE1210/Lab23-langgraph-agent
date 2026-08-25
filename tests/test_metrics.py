from langgraph_agent_lab.metrics import metric_from_state, summarize_metrics
from langgraph_agent_lab.state import make_event


def test_metric_from_state_success() -> None:
    state = {
        "scenario_id": "S",
        "route": "simple",
        "final_answer": "ok",
        "events": [
            make_event("intake", "completed", "ok"),
            make_event("answer", "completed", "ok"),
            make_event("finalize", "completed", "done"),
        ],
        "errors": [],
        "approval": None,
    }
    metric = metric_from_state(state, expected_route="simple", approval_required=False)
    assert metric.success is True
    assert metric.nodes_visited == 3
    assert metric.visited_nodes == ["intake", "answer", "finalize"]


def test_metric_from_state_route_mismatch() -> None:
    state = {
        "scenario_id": "S",
        "route": "tool",
        "final_answer": "ok",
        "events": [],
        "errors": [],
        "approval": None,
    }
    metric = metric_from_state(state, expected_route="simple", approval_required=False)
    assert metric.success is False


def test_metric_from_state_counts_execution_events() -> None:
    events = [
        make_event("intake", "completed", "ok"),
        make_event("retry", "completed", "retry"),
        make_event("tool", "completed", "error"),
        make_event("retry", "completed", "retry"),
        make_event("finalize", "completed", "done"),
    ]
    metric = metric_from_state(
        {
            "scenario_id": "retry-case",
            "route": "error",
            "final_answer": "manual follow-up",
            "events": events,
            "errors": ["first", "second"],
            "approval": None,
        },
        "error",
        False,
    )
    assert metric.success is True
    assert metric.retry_count == 2
    assert metric.error_count == 2
    assert metric.approval_count == 0


def test_metric_requires_finalize_and_risky_approval() -> None:
    state = {
        "scenario_id": "risky-case",
        "route": "risky",
        "final_answer": "approved",
        "events": [make_event("answer", "completed", "ok")],
        "errors": [],
        "approval": None,
    }
    metric = metric_from_state(state, "risky", True)
    assert metric.success is False


def test_summarize_metrics() -> None:
    m1 = metric_from_state(
        {
            "scenario_id": "1",
            "route": "simple",
            "final_answer": "ok",
            "events": [],
            "errors": [],
            "approval": None,
        },
        "simple",
        False,
    )
    m2 = metric_from_state(
        {
            "scenario_id": "2",
            "route": "tool",
            "final_answer": None,
            "events": [],
            "errors": [],
            "approval": None,
        },
        "tool",
        False,
    )
    report = summarize_metrics(
        [m1, m2],
        persistence_backend="sqlite",
        thread_ids=["thread-1", "thread-2"],
        checkpoint_count=4,
        resume_success=True,
    )
    assert report.total_scenarios == 2
    assert 0 <= report.success_rate <= 1
    assert report.persistence_backend == "sqlite"
    assert report.checkpoint_count == 4
