# Day 08 Lab Report

## 1. Team / student

- Name: Nguyen Cong Hung
- Repo/commit: https://github.com/RECTANGLE1210/Lab23-langgraph-agent
- Date: 25/08/2026

## 2. Architecture

The graph follows `START -> intake -> classify`, then uses conditional routing: simple -> answer; tool -> tool -> evaluate; missing_info -> clarify; risky -> risky_action -> approval -> tool -> evaluate; and error -> retry -> tool/dead_letter. Every path ends at `finalize -> END`.

`classify_node` uses OpenRouter with `google/gemini-3.7-flash` and structured output. `answer_node` uses grounded LLM generation. The workflow has bounded retry, native LangGraph HITL approval with `interrupt()`/`Command(resume=...)`, and SQLite checkpoint persistence. Batch evaluation auto-approves the native interrupt for deterministic evaluation.

## 3. State schema

Important current-state fields overwrite their previous value; audit and execution histories append with `operator.add`.

| Field | Reducer | Why |
|---|---|---|
| thread_id | overwrite | current workflow state |
| scenario_id | overwrite | current workflow state |
| query | overwrite | current workflow state |
| route | overwrite | current workflow state |
| risk_level | overwrite | current workflow state |
| attempt | overwrite | current workflow state |
| max_attempts | overwrite | current workflow state |
| evaluation_result | overwrite | current workflow state |
| pending_question | overwrite | current workflow state |
| proposed_action | overwrite | current workflow state |
| approval | overwrite | current workflow state |
| final_answer | overwrite | current workflow state |
| messages | append / operator.add | execution and audit history |
| tool_results | append / operator.add | execution and audit history |
| errors | append / operator.add | execution and audit history |
| events | append / operator.add | execution and audit history |

## 4. Scenario results

Observed execution summary: 7 scenarios, 7 successful, success rate 100.0%.

| Scenario | Expected route | Actual route | Success | Retries | Interrupts |
|---|---|---|---:|---:|---:|
| S01_simple | simple | simple | True | 0 | 0 |
| S02_tool | tool | tool | True | 0 | 0 |
| S03_missing | missing_info | missing_info | True | 0 | 0 |
| S04_risky | risky | risky | True | 0 | 1 |
| S05_error | error | error | True | 2 | 0 |
| S06_delete | risky | risky | True | 0 | 1 |
| S07_dead_letter | error | error | True | 1 | 0 |

`Interrupts` counts the approval events produced after native LangGraph interrupts are resumed. Risky scenarios pause before the tool, then the batch runner resumes the same `thread_id` with approval.

## 5. Failure analysis

1. Retry or tool failure: observed in `S05_error`. The retry node increments the attempt, `evaluate` requests another try when the tool result contains an error, and the bounded workflow recovered. Observed retry events: 2.
2. Risky action without approval: native HITL evidence was observed in `S04_risky, S06_delete`. The first invocation pauses at `approval` before `tool`; resume with approval continues to `tool`, while a separate rejection test resumes `False` and routes to `clarify` without executing `tool`. Additional retry exhaustion was observed in `S07_dead_letter`.

## 6. Persistence / recovery evidence

Backend: `langgraph.checkpoint.sqlite.SqliteSaver` (run backend: `sqlite`).

Thread-isolation evidence:
- `gate3-report-thread-A`: query `How do I reset my password?`, final route `simple`, history count `6`.
- `gate3-report-thread-B`: query `Please lookup order status for order 12345`, final route `tool`, history count `8`.
- `ISOLATION_CHECK: True`.

Recovery evidence:
- After closing and reopening the SQLite database, `resume-thread` continued from the persisted state and its history increased from `6` to `12` checkpoints.
- The seven-scenario evaluation observed `59` checkpoint snapshots.

## 7. Extension work

Native LangGraph HITL approval was implemented with `interrupt()` and `Command(resume=...)` using a thread-scoped checkpointer. Both approved and rejected paths were tested. The batch runner auto-approves only for deterministic evaluation; a separate rejection test resumes `False` and verifies that the risky tool is not executed. The graph topology was also exported as `outputs/graph.mmd`.

## 8. Improvement plan

- Connect native HITL to an authenticated operator or UI workflow.
- Improve tool-result evaluation with LLM-as-judge or a stronger evaluator.
- Add production observability for latency, provider, and checkpoint failures.
