import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.persistence import build_checkpointer
from langgraph_agent_lab.state import (
    AgentState,
    ClassificationSchema,
    Route,
    Scenario,
    initial_state,
)


class FakeStructuredModel:
    def invoke(self, prompt: str) -> ClassificationSchema:
        return ClassificationSchema(route="simple")


class FakeLLM:
    def with_structured_output(self, schema: object, **kwargs: object) -> FakeStructuredModel:
        return FakeStructuredModel()

    def invoke(self, prompt: str) -> object:
        return SimpleNamespace(content="offline persisted answer")


def _state(query: str, thread_id: str) -> AgentState:
    scenario = Scenario(id=thread_id, query=query, expected_route=Route.SIMPLE)
    state = initial_state(scenario)
    state["thread_id"] = thread_id
    return state


def _invoke(graph: object, state: AgentState) -> AgentState:
    config = {"configurable": {"thread_id": state["thread_id"]}}
    return graph.invoke(state, config=config)


def _invoke_with_fake_llm(graph: object, state: AgentState) -> AgentState:
    with patch("langgraph_agent_lab.nodes.get_llm", return_value=FakeLLM()):
        return _invoke(graph, state)


def test_build_checkpointer(tmp_path: Path) -> None:
    database_path = tmp_path / "checkpoints.db"
    checkpointer = build_checkpointer(database_path)

    try:
        assert isinstance(checkpointer, SqliteSaver)
        assert database_path.exists()
        print(f"SQLite persistence backend: path={database_path}")
    finally:
        checkpointer.close()


def test_thread_checkpoint(tmp_path: Path) -> None:
    database_path = tmp_path / "thread-checkpoint.db"
    with build_checkpointer(database_path) as checkpointer:
        graph = build_graph(checkpointer)
        state = _state("first persisted query", "recovery-test")
        _invoke_with_fake_llm(graph, state)

        config = {"configurable": {"thread_id": "recovery-test"}}
        history = list(graph.get_state_history(config))
        assert history
        print(
            f"SQLite checkpoint: path={database_path}, "
            f"thread_id=recovery-test, history_entries={len(history)}"
        )


def test_thread_isolation(tmp_path: Path) -> None:
    database_path = tmp_path / "isolated.db"
    with build_checkpointer(database_path) as checkpointer:
        graph = build_graph(checkpointer)
        _invoke_with_fake_llm(graph, _state("query from A", "thread-A"))
        _invoke_with_fake_llm(graph, _state("query from B", "thread-B"))

        history_a = list(graph.get_state_history({"configurable": {"thread_id": "thread-A"}}))
        history_b = list(graph.get_state_history({"configurable": {"thread_id": "thread-B"}}))
        serialized_a = json.dumps([snapshot.values for snapshot in history_a], default=str)
        serialized_b = json.dumps([snapshot.values for snapshot in history_b], default=str)

        assert "query from A" in serialized_a
        assert "query from B" not in serialized_a
        assert "query from B" in serialized_b
        assert "query from A" not in serialized_b
        print(
            f"SQLite isolation: thread-A entries={len(history_a)}, "
            f"thread-B entries={len(history_b)}, status=passed"
        )


def test_resume_same_thread(tmp_path: Path) -> None:
    database_path = tmp_path / "resume.db"
    with build_checkpointer(database_path) as checkpointer:
        graph = build_graph(checkpointer)
        first_state = _state("before resume", "resume-thread")

        _invoke_with_fake_llm(graph, first_state)
        config = {"configurable": {"thread_id": "resume-thread"}}
        before_count = len(list(graph.get_state_history(config)))

    with build_checkpointer(database_path) as reopened_checkpointer:
        reopened_graph = build_graph(reopened_checkpointer)
        second_state = _state("after resume", "resume-thread")
        _invoke_with_fake_llm(reopened_graph, second_state)
        history = list(reopened_graph.get_state_history(config))
        serialized = json.dumps([snapshot.values for snapshot in history], default=str)

        assert len(history) > before_count
        assert "before resume" in serialized
        assert "after resume" in serialized
        print(
            f"SQLite resume: thread_id=resume-thread, "
            f"history_entries={len(history)}, status=passed"
        )
