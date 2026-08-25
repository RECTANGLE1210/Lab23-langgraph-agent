from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from langgraph_agent_lab.nodes import (
    answer_node,
    approval_node,
    ask_clarification_node,
    classify_node,
    dead_letter_node,
    evaluate_node,
    finalize_node,
    retry_or_fallback_node,
    risky_action_node,
    tool_node,
)
from langgraph_agent_lab.state import ClassificationSchema


class FakeStructuredModel:
    def __init__(self, route: str) -> None:
        self.route = route
        self.prompts: list[str] = []

    def invoke(self, prompt: str) -> ClassificationSchema:
        self.prompts.append(prompt)
        return ClassificationSchema(route=self.route)


class FakeLLM:
    def __init__(self, route: str = "simple", answer: str = "generated answer") -> None:
        self.structured = FakeStructuredModel(route)
        self.answer = answer
        self.structured_schema = None
        self.structured_kwargs = None
        self.prompts: list[str] = []

    def with_structured_output(
        self, schema: object, **kwargs: object
    ) -> FakeStructuredModel:
        self.structured_schema = schema
        self.structured_kwargs = kwargs
        return self.structured

    def invoke(self, prompt: str) -> object:
        self.prompts.append(prompt)
        return SimpleNamespace(content=self.answer)


def test_classify_uses_structured_llm_and_does_not_mutate_state() -> None:
    state = {"query": "Please delete this account", "route": "", "events": []}
    original = deepcopy(state)
    fake_llm = FakeLLM(route="risky")

    with patch("langgraph_agent_lab.nodes.get_llm", return_value=fake_llm):
        result = classify_node(state)

    assert state == original
    assert result["route"] == "risky"
    assert result["risk_level"] == "high"
    assert fake_llm.structured_schema is ClassificationSchema
    assert fake_llm.structured_kwargs == {"method": "json_schema"}
    assert fake_llm.structured.prompts
    assert result["events"][0]["node"] == "classify"


def test_tool_errors_until_attempt_two_and_does_not_mutate_results() -> None:
    for attempt in (0, 1):
        state = {"route": "error", "attempt": attempt, "tool_results": []}
        result = tool_node(state)
        assert "ERROR" in result["tool_results"][0]
        assert state["tool_results"] == []

    assert "ERROR" not in tool_node({"route": "error", "attempt": 2})["tool_results"][0]
    assert "ERROR" not in tool_node({"route": "tool", "attempt": 0})["tool_results"][0]


@pytest.mark.parametrize(
    ("results", "expected"),
    [([], "needs_retry"), (["ERROR: failure"], "needs_retry"), (["SUCCESS: done"], "success")],
)
def test_evaluate_is_safe_for_empty_and_latest_results(
    results: list[str], expected: str
) -> None:
    result = evaluate_node({"tool_results": results})
    assert result["evaluation_result"] == expected
    assert result["events"][0]["node"] == "evaluate"


def test_answer_uses_generated_content_and_grounding_context() -> None:
    fake_llm = FakeLLM(answer="The generated grounded response")
    state = {
        "query": "Where is order 123?",
        "tool_results": ["SUCCESS: order 123 is in transit"],
        "approval": {"approved": True},
        "proposed_action": None,
    }

    with patch("langgraph_agent_lab.nodes.get_llm", return_value=fake_llm):
        result = answer_node(state)

    assert result["final_answer"] == "The generated grounded response"
    assert "Where is order 123?" in fake_llm.prompts[0]
    assert "order 123 is in transit" in fake_llm.prompts[0]
    assert '"approved": true' in fake_llm.prompts[0]
    assert result["events"][0]["node"] == "answer"


def test_remaining_nodes_return_expected_contracts() -> None:
    clarify = ask_clarification_node({"query": "Can you fix it?"})
    assert clarify["pending_question"] and clarify["final_answer"]
    assert clarify["events"][0]["node"] == "clarify"

    risky = risky_action_node({"query": "Refund the payment"})
    assert "approval" in risky["proposed_action"].lower()
    assert risky["events"][0]["node"] == "risky_action"

    with patch("langgraph.types.interrupt", return_value=True):
        approval = approval_node({"proposed_action": "Delete the account"})
    assert approval["approval"]["approved"] is True
    assert approval["approval"]["reviewer"] == "native-hitl"
    assert approval["events"][0]["node"] == "approval"

    retry = retry_or_fallback_node({"attempt": 1})
    assert retry["attempt"] == 2
    assert retry["errors"]
    assert retry["events"][0]["node"] == "retry"

    dead_letter = dead_letter_node({"attempt": 1, "max_attempts": 1})
    assert dead_letter["final_answer"]
    assert "successfully completed" not in dead_letter["final_answer"].lower()
    assert dead_letter["events"][0]["node"] == "dead_letter"

    assert finalize_node({})["events"][0]["node"] == "finalize"
