from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, NotRequired, TypedDict

from django.conf import settings

from sessions.models import (
    CoachAgentExecution,
    CoachAgentExecutionStatus,
    CoachAgentKind,
    CoachOrchestrationRun,
    CoachOrchestrationRunStatus,
    LedgerEntryKind,
)

from .ledger import (
    append_ledger_entry,
    create_agent_execution,
    mark_agent_completed,
    mark_agent_failed,
    mark_agent_processing,
    mark_run_completed,
    mark_run_failed,
    mark_run_processing,
    touch_agent_heartbeat,
)
from .live_ledger import (
    append_live_ledger_entry,
    get_live_ledger_latest_sequence,
    read_live_ledger_slice,
)
from .orchestrator import run_primary_structured_reasoning
from .subagent_workflow import finalize_subagent_run

FLAGSHIP_FINAL_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "overall_impression": {"type": "string"},
        "strengths": {"type": "array", "items": {"type": "string"}},
        "improvements": {"type": "array", "items": {"type": "string"}},
        "priority_actions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "overall_impression",
        "strengths",
        "improvements",
        "priority_actions",
    ],
    "additionalProperties": False,
}

DEFAULT_FLAGSHIP_FINAL_SYSTEM_PROMPT = (
    "You are the final speech coaching synthesis model. Read the full ledger and "
    "produce a concise overall impression, key strengths, key improvements, and "
    "priority coaching actions."
)


class FlagshipFinalWorkflowError(RuntimeError):
    """Raised when the flagship-final reconciliation workflow cannot proceed."""


class FlagshipFinalState(TypedDict):
    run_id: str
    system_prompt: str
    run_index: NotRequired[int]
    session_id: NotRequired[str]
    used_live_ledger: NotRequired[bool]
    input_seq_to: NotRequired[int]
    ledger_entries: NotRequired[list[dict[str, Any]]]
    user_prompt: NotRequired[str]
    request_payload: NotRequired[dict[str, Any]]
    model_name: NotRequired[str]
    usage: NotRequired[dict[str, int]]
    response_metadata: NotRequired[dict[str, Any]]
    structured_output: NotRequired[dict[str, Any]]
    final_summary_content: NotRequired[str]
    final_summary_payload: NotRequired[dict[str, Any]]
    final_agent_execution_id: NotRequired[str]
    output_seq_to: NotRequired[int]
    finalized_result: NotRequired[dict[str, Any]]


def _resolve_system_prompt(system_prompt: str | None) -> str:
    """Resolve and validate the configured flagship-final system prompt."""
    resolved = (
        str(system_prompt)
        if system_prompt is not None
        else str(
            getattr(
                settings,
                "GEMINI_FLAGSHIP_FINAL_SYSTEM_PROMPT",
                DEFAULT_FLAGSHIP_FINAL_SYSTEM_PROMPT,
            )
        )
    ).strip()
    if not resolved:
        raise FlagshipFinalWorkflowError(
            "Missing flagship-final system prompt configuration."
        )
    return resolved


def _normalize_string_list(value: Any) -> list[str]:
    """Normalize candidate list payloads into non-empty strings."""
    if not isinstance(value, list):
        return []
    normalized: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_final_summary(
    structured_output: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    """Normalize structured flagship output into final ledger content and payload."""
    overall_impression = str(structured_output.get("overall_impression", "")).strip()
    if not overall_impression:
        overall_impression = "Final coaching reconciliation is ready."
    strengths = _normalize_string_list(structured_output.get("strengths"))
    improvements = _normalize_string_list(structured_output.get("improvements"))
    priority_actions = _normalize_string_list(structured_output.get("priority_actions"))
    payload = {
        "title": "Final reconciliation",
        "overall_impression": overall_impression,
        "strengths": strengths,
        "improvements": improvements,
        "priority_actions": priority_actions,
    }
    return overall_impression, payload


def _serialize_db_entry(entry) -> dict[str, Any]:
    """Convert a persisted ledger entry into the reconciliation input shape."""
    return {
        "sequence": int(entry.sequence),
        "entry_kind": str(entry.entry_kind),
        "agent_kind": str(entry.agent_kind) if entry.agent_kind else None,
        "agent_name": str(entry.agent_name or ""),
        "window_start_ms": entry.window_start_ms,
        "window_end_ms": entry.window_end_ms,
        "content": str(entry.content),
        "payload": entry.payload if isinstance(entry.payload, dict) else {},
        "created_at": entry.created_at.isoformat(),
    }


def _load_reconciliation_input(state: FlagshipFinalState) -> FlagshipFinalState:
    """Load run context and full ledger history for final reconciliation."""
    run = CoachOrchestrationRun.objects.select_related("session").get(id=state["run_id"])
    active_execution_exists = run.agent_executions.filter(
        status__in=[
            CoachAgentExecutionStatus.QUEUED,
            CoachAgentExecutionStatus.PROCESSING,
        ]
    ).exists()
    if active_execution_exists:
        raise FlagshipFinalWorkflowError(
            f"Run {run.id} still has queued/processing executions."
        )

    live_entries = read_live_ledger_slice(run_id=str(run.id), sequence_gt=0)
    live_entries.sort(key=lambda item: int(item.get("sequence", 0)))
    used_live_ledger = bool(live_entries)

    if used_live_ledger:
        ledger_entries = [
            {
                "sequence": int(entry.get("sequence", 0)),
                "entry_kind": str(entry.get("entry_kind", "")),
                "agent_kind": (
                    str(entry.get("agent_kind"))
                    if entry.get("agent_kind") is not None
                    else None
                ),
                "agent_name": str(entry.get("agent_name", "")),
                "window_start_ms": entry.get("window_start_ms"),
                "window_end_ms": entry.get("window_end_ms"),
                "content": str(entry.get("content", "")),
                "payload": (
                    entry.get("payload")
                    if isinstance(entry.get("payload"), dict)
                    else {}
                ),
                "created_at": str(entry.get("created_at", "")),
            }
            for entry in live_entries
        ]
        input_seq_to = get_live_ledger_latest_sequence(run_id=str(run.id))
    else:
        db_entries = list(run.ledger_entries.order_by("sequence", "created_at"))
        ledger_entries = [_serialize_db_entry(item) for item in db_entries]
        input_seq_to = run.latest_ledger_sequence

    return {
        **state,
        "run_index": int(run.run_index),
        "session_id": str(run.session_id),
        "used_live_ledger": used_live_ledger,
        "input_seq_to": int(input_seq_to),
        "ledger_entries": ledger_entries,
    }


def _load_collective_metrics_stub(
    *,
    run_id: str,
    session_id: str,
) -> dict[str, Any] | None:
    """Return overall speech metrics for flagship-final input once table is available."""
    # TODO(ev): Replace this stub with a DB read from the upcoming
    # "overall speech metrics" table. Keep return shape stable so the
    # flagship-final prompt always receives request_payload.collective_metrics.
    _ = run_id
    _ = session_id
    return None


def _build_reconciliation_prompt(state: FlagshipFinalState) -> FlagshipFinalState:
    """Build the fixed user prompt from the loaded full-ledger context."""
    collective_metrics = _load_collective_metrics_stub(
        run_id=state["run_id"],
        session_id=str(state.get("session_id", "")),
    )
    request_payload = {
        "run_id": state["run_id"],
        "session_id": state.get("session_id", ""),
        "run_index": state.get("run_index"),
        "ledger_entries": list(state.get("ledger_entries", [])),
        "collective_metrics": collective_metrics,
    }
    user_prompt = (
        "Reconcile this full speech-coaching ledger into a final coaching summary.\n"
        "Return JSON output only according to the configured schema.\n"
        f"LEDGER_INPUT_JSON:\n{json.dumps(request_payload, sort_keys=True)}"
    )
    return {
        **state,
        "request_payload": request_payload,
        "user_prompt": user_prompt,
    }


def _invoke_reconciliation_model(state: FlagshipFinalState) -> FlagshipFinalState:
    """Run the primary model to compute structured final reconciliation output."""
    metadata = {
        "run_id": state["run_id"],
        "session_id": state.get("session_id", ""),
        "run_index": state.get("run_index"),
        "input_seq_to": state.get("input_seq_to"),
    }
    result = run_primary_structured_reasoning(
        system_prompt=state["system_prompt"],
        user_prompt=state.get("user_prompt", ""),
        structured_schema=FLAGSHIP_FINAL_OUTPUT_SCHEMA,
        metadata=metadata,
        request_payload=dict(state.get("request_payload", {})),
    )
    return {
        **state,
        "model_name": result.model_name,
        "usage": dict(result.usage),
        "response_metadata": dict(result.response_metadata),
        "structured_output": dict(result.structured_output),
    }


def _persist_reconciliation_entry(state: FlagshipFinalState) -> FlagshipFinalState:
    """Persist one flagship-final ledger entry in live ledger or DB."""
    run = CoachOrchestrationRun.objects.get(id=state["run_id"])
    if run.status == CoachOrchestrationRunStatus.QUEUED:
        run = mark_run_processing(run=run)

    input_seq_to = int(state.get("input_seq_to", 0))
    agent_name = f"flagship-final-run-{run.run_index}"
    execution = create_agent_execution(
        run=run,
        agent_kind=CoachAgentKind.FLAGSHIP_FINAL,
        agent_name=agent_name,
        input_seq_from=1 if input_seq_to > 0 else None,
        input_seq_to=input_seq_to if input_seq_to > 0 else None,
    )
    execution = mark_agent_processing(execution=execution)
    touch_agent_heartbeat(execution=execution)

    try:
        final_summary_content, final_summary_payload = _normalize_final_summary(
            dict(state.get("structured_output", {}))
        )
        final_summary_payload.update(
            {
                "model_name": state.get("model_name", ""),
                "usage": dict(state.get("usage", {})),
            }
        )

        if bool(state.get("used_live_ledger", False)):
            live_entry = append_live_ledger_entry(
                run_id=str(run.id),
                entry_kind=LedgerEntryKind.FLAGSHIP_FINAL,
                content=final_summary_content,
                agent_execution_id=str(execution.id),
                agent_kind=CoachAgentKind.FLAGSHIP_FINAL,
                agent_name=execution.agent_name,
                payload=final_summary_payload,
            )
            output_seq_to = int(live_entry["sequence"])
        else:
            db_entry = append_ledger_entry(
                run=run,
                entry_kind=LedgerEntryKind.FLAGSHIP_FINAL,
                content=final_summary_content,
                agent_execution=execution,
                agent_kind=CoachAgentKind.FLAGSHIP_FINAL,
                agent_name=execution.agent_name,
                payload=final_summary_payload,
            )
            output_seq_to = int(db_entry.sequence)

        execution = mark_agent_completed(execution=execution, output_seq_to=output_seq_to)
        touch_agent_heartbeat(execution=execution)
        return {
            **state,
            "final_summary_content": final_summary_content,
            "final_summary_payload": final_summary_payload,
            "final_agent_execution_id": str(execution.id),
            "output_seq_to": output_seq_to,
        }
    except Exception as exc:
        mark_agent_failed(execution=execution, error_message=str(exc))
        raise


def _finalize_reconciliation(state: FlagshipFinalState) -> FlagshipFinalState:
    """Finalize run completion after flagship-final entry persistence."""
    run = CoachOrchestrationRun.objects.get(id=state["run_id"])
    if bool(state.get("used_live_ledger", False)):
        finalized_result = finalize_subagent_run(run_id=str(run.id))
    else:
        run = mark_run_completed(run=run)
        finalized_result = {
            "run_id": str(run.id),
            "status": str(run.status),
            "flushed_entries": 0,
            "latest_ledger_sequence": run.latest_ledger_sequence,
        }
    return {**state, "finalized_result": finalized_result}


def build_flagship_final_graph():
    """Build and compile the LangGraph for flagship-final reconciliation."""
    from langgraph.graph import END, StateGraph

    graph_builder = StateGraph(FlagshipFinalState)
    graph_builder.add_node("load_reconciliation_input", _load_reconciliation_input)
    graph_builder.add_node("build_reconciliation_prompt", _build_reconciliation_prompt)
    graph_builder.add_node("invoke_reconciliation_model", _invoke_reconciliation_model)
    graph_builder.add_node("persist_reconciliation_entry", _persist_reconciliation_entry)
    graph_builder.add_node("finalize_reconciliation", _finalize_reconciliation)
    graph_builder.set_entry_point("load_reconciliation_input")
    graph_builder.add_edge("load_reconciliation_input", "build_reconciliation_prompt")
    graph_builder.add_edge("build_reconciliation_prompt", "invoke_reconciliation_model")
    graph_builder.add_edge("invoke_reconciliation_model", "persist_reconciliation_entry")
    graph_builder.add_edge("persist_reconciliation_entry", "finalize_reconciliation")
    graph_builder.add_edge("finalize_reconciliation", END)
    return graph_builder.compile()


@lru_cache(maxsize=1)
def _compiled_flagship_final_graph():
    """Build and cache a compiled flagship-final graph instance for reuse."""
    return build_flagship_final_graph()


def clear_flagship_final_graph_cache() -> None:
    """Clear the cached flagship-final graph to pick up code/config updates."""
    _compiled_flagship_final_graph.cache_clear()


def run_flagship_final_reconciliation(
    *,
    run_id: str,
    system_prompt: str | None = None,
    graph: Any | None = None,
) -> dict[str, Any]:
    """Execute flagship-final reconciliation end-to-end for a run."""
    resolved_prompt = _resolve_system_prompt(system_prompt)
    target_graph = graph or _compiled_flagship_final_graph()
    initial_state: FlagshipFinalState = {
        "run_id": str(run_id),
        "system_prompt": resolved_prompt,
    }
    try:
        final_state: FlagshipFinalState = target_graph.invoke(initial_state)
        return {
            "run_id": str(run_id),
            "status": "completed",
            "used_live_ledger": bool(final_state.get("used_live_ledger", False)),
            "final_agent_execution_id": str(
                final_state.get("final_agent_execution_id", "")
            ),
            "output_seq_to": int(final_state.get("output_seq_to", 0)),
            "finalized_result": dict(final_state.get("finalized_result", {})),
        }
    except Exception as exc:
        try:
            run = CoachOrchestrationRun.objects.get(id=run_id)
            mark_run_failed(run=run, error_message=str(exc))
        except CoachOrchestrationRun.DoesNotExist:
            pass
        raise
