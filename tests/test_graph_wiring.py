from types import SimpleNamespace
from unittest.mock import patch

from langgraph.checkpoint.memory import MemorySaver

from langgraph_agent_lab.graph import build_graph
from langgraph_agent_lab.state import ClassificationSchema, Route, Scenario, initial_state

NODE_NAMES = {
    "intake",
    "classify",
    "tool",
    "evaluate",
    "answer",
    "clarify",
    "risky_action",
    "approval",
    "retry",
    "dead_letter",
    "finalize",
}


class FakeStructuredModel:
    def __init__(self, route: str) -> None:
        self.route = route

    def invoke(self, prompt: str) -> ClassificationSchema:
        return ClassificationSchema(route=self.route)


class FakeLLM:
    def __init__(self, route: str) -> None:
        self.structured = FakeStructuredModel(route)

    def with_structured_output(self, schema: object, **kwargs: object) -> FakeStructuredModel:
        return self.structured

    def invoke(self, prompt: str) -> object:
        return SimpleNamespace(content="offline generated answer")


def test_graph_registers_nodes_and_edges() -> None:
    graph = build_graph()
    view = graph.get_graph()
    nodes = set(view.nodes) - {"__start__", "__end__"}
    edges = {(edge.source, edge.target) for edge in view.edges}

    assert nodes == NODE_NAMES
    assert {
        ("__start__", "intake"),
        ("intake", "classify"),
        ("tool", "evaluate"),
        ("risky_action", "approval"),
        ("answer", "finalize"),
        ("clarify", "finalize"),
        ("dead_letter", "finalize"),
        ("finalize", "__end__"),
        ("classify", "answer"),
        ("classify", "tool"),
        ("classify", "clarify"),
        ("classify", "risky_action"),
        ("classify", "retry"),
        ("evaluate", "answer"),
        ("evaluate", "retry"),
        ("retry", "tool"),
        ("retry", "dead_letter"),
        ("approval", "tool"),
        ("approval", "clarify"),
    } <= edges


def test_graph_compiles_with_memory_saver() -> None:
    assert build_graph(checkpointer=MemorySaver()) is not None


def test_graph_routes_all_paths_and_bounds_retries() -> None:
    cases = [
        ("simple", Route.SIMPLE, 3, ["answer", "finalize"]),
        ("tool", Route.TOOL, 3, ["tool", "evaluate", "answer", "finalize"]),
        ("missing_info", Route.MISSING_INFO, 3, ["clarify", "finalize"]),
        ("risky", Route.RISKY, 3, ["risky_action", "approval", "tool"]),
        ("error", Route.ERROR, 3, ["retry", "tool", "evaluate", "answer"]),
        ("error", Route.ERROR, 1, ["retry", "dead_letter", "finalize"]),
    ]

    for index, (fake_route, expected_route, max_attempts, required_nodes) in enumerate(cases):
        scenario = Scenario(
            id=f"offline-{index}",
            query="offline test request",
            expected_route=expected_route,
            max_attempts=max_attempts,
        )
        state = initial_state(scenario)
        graph = build_graph(checkpointer=MemorySaver())
        fake_llm = FakeLLM(fake_route)

        with patch("langgraph_agent_lab.nodes.get_llm", return_value=fake_llm):
            result = graph.invoke(
                state,
                config={"configurable": {"thread_id": state["thread_id"]}},
            )

        visited = [event["node"] for event in result["events"]]
        assert result["route"] == expected_route.value
        assert "finalize" in visited
        for node in required_nodes:
            assert node in visited
        if max_attempts == 1:
            assert "tool" not in visited
