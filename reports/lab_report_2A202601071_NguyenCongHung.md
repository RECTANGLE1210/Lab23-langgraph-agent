# Day 08 Lab Report

## 1. Team / student

- Name: Nguyen Cong Hung
- Repo/commit: https://github.com/RECTANGLE1210/Lab23-langgraph-agent
- Date: 25/08/2026

## 2. Architecture

The graph follows `START -> intake -> classify`, then uses conditional routing: simple -> answer; tool -> tool -> evaluate; missing_info -> clarify; risky -> risky_action -> approval -> tool -> evaluate; and error -> retry -> tool/dead_letter. Every path ends at `finalize -> END`.

`classify_node` uses OpenRouter with `google/gemini-3.7-flash` and structured output. `answer_node` uses grounded LLM generation. The workflow has bounded retry, deterministic mock approval, and SQLite checkpoint persistence.

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

`Interrupts` currently counts observed approval checkpoints/events in the lab's deterministic mock approval path; native LangGraph `interrupt()/resume` HITL is not yet implemented.

## 5. Failure analysis

1. Retry or tool failure: observed in `S05_error`. The retry node increments the attempt, `evaluate` requests another try when the tool result contains an error, and the bounded workflow recovered. Observed retry events: 2.
2. Risky action without approval: approval-path evidence was observed in `S04_risky, S06_delete`. Risky classification reaches `risky_action` and then `approval`; only approval proceeds to the tool, while rejection routes to clarification. Additional retry exhaustion observed in `S07_dead_letter`.

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

No optional extension has been finalized yet; SQLite persistence/recovery is implemented as the persistence requirement. A graph diagram or real HITL flow may be added in the final extension gate.

## 8. Improvement plan

- Replace mock approval with real `interrupt()/resume` HITL.
- Improve tool-result evaluation with LLM-as-judge or a stronger evaluator.
- Add production observability for latency, provider, and checkpoint failures.
