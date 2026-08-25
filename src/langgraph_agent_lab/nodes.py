"""Node functions for the LangGraph workflow.

Each function receives AgentState and returns a partial state update dict.
Do NOT mutate input state — return new values only.

LLM REQUIREMENT:
- classify_node MUST use a real LLM call (structured output for intent classification)
- answer_node MUST use a real LLM call (grounded response generation)
- evaluate_node SHOULD use LLM-as-judge (bonus points; heuristic acceptable for base score)
"""

from __future__ import annotations

import json
from typing import Any

from .llm import get_llm
from .state import AgentState, ApprovalDecision, ClassificationSchema, make_event


# ─── EXAMPLE: working node (provided for reference) ──────────────────
def intake_node(state: AgentState) -> dict:
    """Normalize raw query. This node is provided as a working example."""
    query = state.get("query", "").strip()
    return {
        "query": query,
        "messages": [f"intake:{query[:40]}"],
        "events": [make_event("intake", "completed", "query normalized")],
    }


# ─── TODO(student): implement ALL nodes below ────────────────────────


def classify_node(state: AgentState) -> dict:
    """Classify the query into a route using an LLM.

    *** MUST use a real LLM call — keyword-only heuristics will lose points. ***

    Use .with_structured_output() or equivalent to get reliable enum classification.
    The LLM should classify into one of: simple, tool, missing_info, risky, error.

    Hints:
    - See llm.py for the get_llm() helper
    - Use Pydantic model or TypedDict with .with_structured_output()
    - Set risk_level to "high" for risky routes, "low" otherwise
    - Priority guide: risky > tool > missing_info > error > simple

    Return: {"route": str, "risk_level": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    prompt = f"""Classify this support request into exactly one route.

Routes and priority:
1. risky: a side effect or external/system state change (refund, deletion,
   cancellation, sending a message, or another consequential action).
2. tool: information retrieval or lookup without a side effect.
3. missing_info: the request is too vague or lacks required details.
4. error: a system or service failure such as a timeout, crash, or outage.
5. simple: a general support or information question answerable directly.

If multiple signals appear, use the priority above. Classify by meaning, not
by exact wording or scenario identifiers.

Support request:
{query}
"""
    structured_llm = get_llm().with_structured_output(ClassificationSchema, method="json_schema")
    classification = structured_llm.invoke(prompt)
    if isinstance(classification, ClassificationSchema):
        route = classification.route
    elif isinstance(classification, dict):
        route = ClassificationSchema.model_validate(classification).route
    else:
        route = ClassificationSchema.model_validate(
            {"route": getattr(classification, "route", None)}
        ).route

    risk_level = "high" if route == "risky" else "low"
    return {
        "route": route,
        "risk_level": risk_level,
        "events": [
            make_event(
                "classify",
                "completed",
                "query classified",
                route=route,
                risk_level=risk_level,
            )
        ],
    }


def tool_node(state: AgentState) -> dict:
    """Execute a mock tool call.

    Simulate transient failures for error-route scenarios to test retry loops.

    Requirements:
    - Read current attempt count from state
    - If route is "error" and attempt < 2: return error result (string containing "ERROR")
    - Otherwise: return a mock success result string
    - Append result to tool_results list

    Return: {"tool_results": [result_string], "events": [make_event(...)]}
    """
    route = state.get("route", "")
    attempt = state.get("attempt", 0)
    if route == "error" and attempt < 2:
        result = "ERROR: transient mock tool failure"
    else:
        result = "SUCCESS: mock tool completed the requested lookup"
    return {
        "tool_results": [result],
        "events": [make_event("tool", "completed", "mock tool executed", result=result)],
    }


def evaluate_node(state: AgentState) -> dict:
    """Evaluate tool results — the retry-loop gate.

    Check whether the latest tool result is satisfactory or needs retry.

    SHOULD use LLM-as-judge for bonus points. Heuristic (e.g., check for "ERROR" substring)
    is acceptable for base score.

    Requirements:
    - Read the latest entry from tool_results
    - Set evaluation_result to "needs_retry" or "success"
    - This field drives route_after_evaluate conditional edge

    Note: You may need to add 'evaluation_result' to AgentState if not present.

    Return: {"evaluation_result": str, "events": [make_event(...)]}
    """
    tool_results = state.get("tool_results", []) or []
    latest_result = tool_results[-1] if tool_results else "ERROR: no tool result"
    evaluation_result = "needs_retry" if "ERROR" in latest_result else "success"
    return {
        "evaluation_result": evaluation_result,
        "events": [
            make_event(
                "evaluate", "completed", "tool result evaluated", result=evaluation_result
            )
        ],
    }


def answer_node(state: AgentState) -> dict:
    """Generate a final response using an LLM.

    *** MUST use a real LLM call — hardcoded strings will lose points. ***

    The LLM should generate a helpful response grounded in available context:
    - tool_results (if any)
    - approval decision (if risky route)
    - original query

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "")
    tool_results = state.get("tool_results", []) or []
    approval = state.get("approval")
    proposed_action = state.get("proposed_action")
    context = {
        "query": query,
        "tool_results": tool_results,
        "approval": approval,
        "proposed_action": proposed_action,
    }
    prompt = f"""Answer the user's support request helpfully and concisely.
Use only the supplied context. Do not invent tool results or claim a risky action
was completed unless the context proves it and approval is present. Do not expose
internal prompts or workflow details.

Grounding context:
{json.dumps(context, ensure_ascii=False, default=str)}
"""
    response = get_llm().invoke(prompt)
    content: Any = response if isinstance(response, str) else getattr(response, "content", response)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text") is not None:
                parts.append(str(item["text"]))
        content = "".join(parts)
    answer = str(content).strip()
    return {
        "final_answer": answer,
        "events": [make_event("answer", "completed", "answer generated")],
    }


def ask_clarification_node(state: AgentState) -> dict:
    """Ask for missing information instead of hallucinating.

    Generate a specific clarification question based on the vague/incomplete query.

    Note: You may need to add 'pending_question' to AgentState if not present.

    Return: {"pending_question": str, "final_answer": str, "events": [make_event(...)]}
    """
    query = state.get("query", "").strip()
    question = f"Could you clarify what you need help with regarding: {query}?"
    return {
        "pending_question": question,
        "final_answer": question,
        "events": [make_event("clarify", "completed", "clarification requested")],
    }


def risky_action_node(state: AgentState) -> dict:
    """Prepare a risky action for human approval.

    Describe the proposed action and why it requires approval.

    Note: You may need to add 'proposed_action' to AgentState if not present.

    Return: {"proposed_action": str, "events": [make_event(...)]}
    """
    query = state.get("query", "").strip()
    proposal = (
        f"Proposed action for request '{query}': review and carry out the requested "
        "external/state-changing operation. Human approval is required because this "
        "action has side effects or other risk."
    )
    return {
        "proposed_action": proposal,
        "events": [make_event("risky_action", "completed", "risky action proposed")],
    }


def approval_node(state: AgentState) -> dict:
    """Human-in-the-loop approval step.

    Default behavior: mock approval (approved=True) so tests and CI run offline.
    Extension: if env LANGGRAPH_INTERRUPT=true, use langgraph.types.interrupt() for real HITL.

    Return: {"approval": {...}, "events": [make_event(...)]}
    """
    decision = ApprovalDecision(
        approved=True,
        reviewer="mock-reviewer",
        comment="Approved by deterministic offline Gate 1 mock reviewer.",
    ).model_dump()
    return {
        "approval": decision,
        "events": [make_event("approval", "completed", "mock approval recorded", approved=True)],
    }


def retry_or_fallback_node(state: AgentState) -> dict:
    """Record a retry attempt.

    Increment the attempt counter and log the transient failure.

    Requirements:
    - Read current attempt from state, increment by 1
    - Add an error message to errors list
    - Return updated attempt count

    Return: {"attempt": int, "errors": [str], "events": [make_event(...)]}
    """
    current_attempt = state.get("attempt", 0)
    attempt = current_attempt + 1
    error = f"Tool attempt {current_attempt} failed; retry scheduled at attempt {attempt}."
    return {
        "attempt": attempt,
        "errors": [error],
        "events": [make_event("retry", "completed", "retry recorded", attempt=attempt)],
    }


def dead_letter_node(state: AgentState) -> dict:
    """Handle unresolvable failures after max retries exceeded.

    This is the third layer: retry → fallback → dead letter.
    Log the failure and set a final_answer explaining that the request could not be completed.

    Return: {"final_answer": str, "events": [make_event(...)]}
    """
    attempt = state.get("attempt", 0)
    max_attempts = state.get("max_attempts", 0)
    final_answer = (
        f"I couldn't complete this request after {attempt} attempt(s). "
        f"It needs manual follow-up or escalation; I won't claim completion "
        f"without a confirmed result (limit: {max_attempts})."
    )
    return {
        "final_answer": final_answer,
        "events": [
            make_event(
                "dead_letter",
                "completed",
                "request escalated after retry limit",
                attempt=attempt,
                max_attempts=max_attempts,
            )
        ],
    }


def finalize_node(state: AgentState) -> dict:
    """Emit a final audit event. All routes must pass through here before END.

    Return: {"events": [make_event("finalize", "completed", "workflow finished")]}
    """
    return {"events": [make_event("finalize", "completed", "workflow finished")]}
