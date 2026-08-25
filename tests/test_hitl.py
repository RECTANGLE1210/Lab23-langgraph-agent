from types import SimpleNamespace
from unittest.mock import patch

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.state import ClassificationSchema, Route, Scenario, initial_state


class FakeStructuredModel:
    def invoke(self, prompt: str) -> ClassificationSchema:
        return ClassificationSchema(route="risky")


class FakeLLM:
    def with_structured_output(self, schema: object, **kwargs: object) -> FakeStructuredModel:
        return FakeStructuredModel()

    def invoke(self, prompt: str) -> object:
        return SimpleNamespace(content="offline approved answer")


def _scenario(thread_id: str) -> tuple[dict, dict]:
    scenario = Scenario(
        id=thread_id,
        query="Refund this customer",
        expected_route=Route.RISKY,
        requires_approval=True,
    )
    state = initial_state(scenario)
    state["thread_id"] = thread_id
    return state, {"configurable": {"thread_id": thread_id}}


def _pause(graph: object, state: dict, config: dict) -> dict:
    with patch("langgraph_agent_lab.nodes.get_llm", return_value=FakeLLM()):
        return graph.invoke(state, config=config)


def test_native_hitl_approval_resumes_and_runs_tool() -> None:
    graph = build_graph(checkpointer=MemorySaver())
    state, config = _scenario("hitl-approved")

    paused = _pause(graph, state, config)
    assert paused.get("__interrupt__")
    payload = paused["__interrupt__"][0].value
    assert payload["type"] == "approval_required"
    assert payload["proposed_action"]
    assert not any(event["node"] == "tool" for event in paused.get("events", []))

    with patch("langgraph_agent_lab.nodes.get_llm", return_value=FakeLLM()):
        resumed = graph.invoke(Command(resume=True), config=config)

    assert resumed["approval"]["approved"] is True
    assert resumed["approval"]["reviewer"] == "native-hitl"
    visited = [event["node"] for event in resumed["events"]]
    assert "approval" in visited
    assert "tool" in visited
    assert "finalize" in visited


def test_native_hitl_rejection_routes_to_clarify_without_tool() -> None:
    graph = build_graph(checkpointer=MemorySaver())
    state, config = _scenario("hitl-rejected")
    paused = _pause(graph, state, config)
    assert paused.get("__interrupt__")

    with patch("langgraph_agent_lab.nodes.get_llm", return_value=FakeLLM()):
        resumed = graph.invoke(Command(resume=False), config=config)

    assert resumed["approval"]["approved"] is False
    visited = [event["node"] for event in resumed["events"]]
    assert "clarify" in visited
    assert "finalize" in visited
    assert "tool" not in visited


def test_native_hitl_checkpoint_history_continues_on_same_thread() -> None:
    graph = build_graph(checkpointer=MemorySaver())
    state, config = _scenario("hitl-history")
    paused = _pause(graph, state, config)
    before = list(graph.get_state_history(config))
    assert paused.get("__interrupt__")
    assert before

    with patch("langgraph_agent_lab.nodes.get_llm", return_value=FakeLLM()):
        graph.invoke(Command(resume=True), config=config)
    after = list(graph.get_state_history(config))

    assert len(after) > len(before)
    print(
        f"Native HITL checkpoint: thread_id=hitl-history, "
        f"before={len(before)}, after={len(after)}, resume=passed"
    )
