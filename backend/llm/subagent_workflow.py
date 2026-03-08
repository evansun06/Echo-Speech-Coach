from __future__ import annotations

import json
import re
from typing import Any
from uuid import UUID

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
    clear_live_ledger,
    get_live_ledger_latest_sequence,
    read_live_ledger_slice,
)
from .orchestrator import run_subagent_structured_reasoning

SUBAGENT_OUTPUT_SCHEMA: dict[str, Any] = {
    "title": "subagent_window_output",
    "description": "Structured note payload for one subagent window.",
    "type": "object",
    "properties": {
        "notes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "event_id": {"type": "string"},
                    "note": {"type": "string"},
                },
                "required": ["event_id", "note"],
                "additionalProperties": False,
            },
        },
        "impression": {"type": "string"},
    },
    "required": ["notes", "impression"],
    "additionalProperties": False,
}

DEFAULT_WINDOW_IMPRESSION = "No clear trend was detected in this window."
SUBAGENT_SYSTEM_PROMPT = (
    "You are a Speech Coach subagent operating on exactly one 30-second window.\n\n"
    "Your role:\n"
    "- Analyze only the provided WINDOW_INPUT_JSON for this window.\n"
    "- Produce localized observations tied to events in this window.\n"
    "- Do not make global/session-wide judgments.\n\n"
    "Output contract:\n"
    "- Return valid JSON only, matching the provided schema exactly.\n"
    '- Required keys: "notes" (array), "impression" (string).\n'
    '- Each note object must contain: "event_id", "note".\n'
    "- Use only event_id values that exist in the input events list.\n"
    "- At most one note per event_id.\n"
    "- Do not add extra keys.\n\n"
    "Reasoning policy:\n"
    "- Prefer salient-only notes: include an event note only when there is a clear, meaningful signal.\n"
    "- Keep notes objective, concise, and specific to what happened in this window.\n"
    "- Avoid prescriptions, motivational language, or long explanations.\n"
    "- If evidence is weak/noisy, skip the event instead of forcing a note.\n"
    "- Never invent events, metrics, timestamps, or transcript content.\n\n"
    "Impression policy:\n"
    "- Write exactly one concise sentence summarizing the main local trend in this window.\n"
    "- Keep it neutral and evidence-based.\n"
    '- If no clear trend is present, output: "No clear trend was detected in this window."'
)


class SubagentWorkflowError(RuntimeError):
    """Raised for subagent workflow state or orchestration failures."""


class SubagentInputValidationError(SubagentWorkflowError):
    """Raised when subagent input payload shape is invalid."""


def _resolve_subagent_system_prompt() -> str:
    """Return the deterministic configured subagent system prompt constant."""
    prompt = str(SUBAGENT_SYSTEM_PROMPT).strip()
    if not prompt:
        raise SubagentWorkflowError("SUBAGENT_SYSTEM_PROMPT cannot be blank.")
    return prompt


def _normalize_window_bounds(*, window_start_ms: int, window_end_ms: int) -> tuple[int, int]:
    """Validate and normalize required window start/end bounds."""
    start = int(window_start_ms)
    end = int(window_end_ms)
    if start < 0:
        raise SubagentInputValidationError("window_start_ms must be >= 0.")
    if end < start:
        raise SubagentInputValidationError("window_end_ms must be >= window_start_ms.")
    return start, end


def _normalize_word_map(*, word_map: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalize word-level timing payload for the 30s window."""
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(word_map):
        if not isinstance(item, dict):
            raise SubagentInputValidationError(
                f"word_map[{index}] must be an object."
            )
        word = str(item.get("word", "")).strip()
        start_ms = item.get("start_ms")
        end_ms = item.get("end_ms")
        if not word:
            raise SubagentInputValidationError(
                f"word_map[{index}].word is required."
            )
        if not isinstance(start_ms, int) or not isinstance(end_ms, int):
            raise SubagentInputValidationError(
                f"word_map[{index}] requires integer start_ms/end_ms."
            )
        normalized.append(
            {
                **item,
                "word": word,
                "start_ms": start_ms,
                "end_ms": end_ms,
            }
        )
    return normalized


def _normalize_events(*, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate and normalize event payloads expected by subagent window reasoning."""
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(events):
        if not isinstance(item, dict):
            raise SubagentInputValidationError(f"events[{index}] must be an object.")
        event_id_raw = item.get("event_id")
        event_type_raw = item.get("event_type")
        start_ms = item.get("start_ms")
        end_ms = item.get("end_ms")
        metadata = item.get("metadata")

        event_id = str(event_id_raw).strip()
        event_type = str(event_type_raw).strip()
        if not event_id:
            raise SubagentInputValidationError(
                f"events[{index}].event_id is required."
            )
        if not event_type:
            raise SubagentInputValidationError(
                f"events[{index}].event_type is required."
            )
        if not isinstance(start_ms, int) or not isinstance(end_ms, int):
            raise SubagentInputValidationError(
                f"events[{index}] requires integer start_ms/end_ms."
            )
        if not isinstance(metadata, dict):
            raise SubagentInputValidationError(
                f"events[{index}].metadata must be an object."
            )

        normalized.append(
            {
                **item,
                "event_id": event_id,
                "event_type": event_type,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "metadata": dict(metadata),
            }
        )
    return normalized


def _to_one_sentence(value: str) -> str:
    """Normalize text into a concise one-sentence impression."""
    normalized = " ".join(value.split()).strip()
    if not normalized:
        return ""
    sentence_match = re.search(r"[.!?]", normalized)
    if sentence_match is None:
        return normalized
    return normalized[: sentence_match.end()].strip()


def _parse_subagent_output(
    *,
    structured_output: dict[str, Any],
    events_by_id: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    """Parse and validate model structured output against known event IDs."""
    notes_raw = structured_output.get("notes")
    parsed_notes: list[dict[str, Any]] = []
    seen_event_ids: set[str] = set()
    if isinstance(notes_raw, list):
        for note_item in notes_raw:
            if not isinstance(note_item, dict):
                continue
            event_id = str(note_item.get("event_id", "")).strip()
            note_body = str(note_item.get("note", "")).strip()
            if not event_id or not note_body:
                continue
            if event_id not in events_by_id:
                continue
            if event_id in seen_event_ids:
                continue
            seen_event_ids.add(event_id)
            event_payload = events_by_id[event_id]
            parsed_notes.append(
                {
                    "event_id": event_id,
                    "event_type": event_payload.get("event_type", ""),
                    "note": note_body,
                }
            )

    impression_raw = str(structured_output.get("impression", "")).strip()
    impression = _to_one_sentence(impression_raw)
    if not impression:
        impression = DEFAULT_WINDOW_IMPRESSION
    return parsed_notes, impression


def _build_subagent_user_prompt(*, request_payload: dict[str, Any]) -> str:
    """Build the fixed user prompt format for the pre-defined subagent workflow."""
    return (
        "Analyze the following 30-second speech window JSON.\n"
        "Return JSON output only according to the configured schema.\n"
        f"WINDOW_INPUT_JSON:\n{json.dumps(request_payload, sort_keys=True)}"
    )


def create_subagent_execution_for_window(
    *,
    run: CoachOrchestrationRun,
    window_start_ms: int,
    window_end_ms: int,
    input_seq_from: int | None = None,
    input_seq_to: int | None = None,
) -> CoachAgentExecution:
    """Create a queued subagent execution row for one 30-second window."""
    normalized_start, normalized_end = _normalize_window_bounds(
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
    )
    agent_name = f"subagent-window-{normalized_start}-{normalized_end}"
    return create_agent_execution(
        run=run,
        agent_kind=CoachAgentKind.SUBAGENT,
        agent_name=agent_name,
        window_start_ms=normalized_start,
        window_end_ms=normalized_end,
        input_seq_from=input_seq_from,
        input_seq_to=input_seq_to,
    )


def run_subagent_execution(
    *,
    execution_id: str,
    session_id: str,
    events: list[dict[str, Any]],
    word_map: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    graph: Any | None = None,
) -> dict[str, Any]:
    """Run one subagent execution using deterministic prompt and append ledger updates."""
    execution = CoachAgentExecution.objects.select_related("run__session").get(id=execution_id)
    run = execution.run
    if str(run.session_id) != str(session_id):
        raise SubagentWorkflowError(
            "execution session_id does not match the provided session_id."
        )
    if execution.status == CoachAgentExecutionStatus.COMPLETED:
        return {
            "execution_id": str(execution.id),
            "run_id": str(run.id),
            "status": "already_completed",
            "output_seq_to": execution.output_seq_to,
        }
    if execution.status == CoachAgentExecutionStatus.PROCESSING:
        return {
            "execution_id": str(execution.id),
            "run_id": str(run.id),
            "status": "already_processing",
        }

    if run.status == CoachOrchestrationRunStatus.QUEUED:
        run = mark_run_processing(run=run)
    execution = mark_agent_processing(execution=execution)
    touch_agent_heartbeat(execution=execution)

    try:
        normalized_start, normalized_end = _normalize_window_bounds(
            window_start_ms=execution.window_start_ms or 0,
            window_end_ms=execution.window_end_ms or 0,
        )
        normalized_events = _normalize_events(events=events)
        normalized_word_map = _normalize_word_map(word_map=word_map)
        events_by_id = {item["event_id"]: item for item in normalized_events}
        request_payload = {
            "session_id": str(session_id),
            "run_id": str(run.id),
            "window_start_ms": normalized_start,
            "window_end_ms": normalized_end,
            "events": normalized_events,
            "word_map": normalized_word_map,
        }
        request_metadata = {
            "session_id": str(session_id),
            "run_id": str(run.id),
            "execution_id": str(execution.id),
            "window_start_ms": normalized_start,
            "window_end_ms": normalized_end,
            **dict(metadata or {}),
        }
        user_prompt = _build_subagent_user_prompt(request_payload=request_payload)
        resolved_system_prompt = _resolve_subagent_system_prompt()

        reasoning_result = run_subagent_structured_reasoning(
            system_prompt=resolved_system_prompt,
            user_prompt=user_prompt,
            structured_schema=SUBAGENT_OUTPUT_SCHEMA,
            metadata=request_metadata,
            request_payload=request_payload,
            graph=graph,
        )
        notes, impression = _parse_subagent_output(
            structured_output=reasoning_result.structured_output,
            events_by_id=events_by_id,
        )

        last_sequence = get_live_ledger_latest_sequence(run_id=str(run.id))
        for note in notes:
            entry = append_live_ledger_entry(
                run_id=str(run.id),
                entry_kind=LedgerEntryKind.SUBAGENT_NOTE,
                content=note["note"],
                agent_execution_id=str(execution.id),
                agent_kind=CoachAgentKind.SUBAGENT,
                agent_name=execution.agent_name,
                window_start_ms=execution.window_start_ms,
                window_end_ms=execution.window_end_ms,
                payload={
                    "title": f"{note['event_type']} ({note['event_id']})",
                    "note_type": "event_note",
                    "event_id": note["event_id"],
                    "event_type": note["event_type"],
                    "model_name": reasoning_result.model_name,
                },
            )
            last_sequence = entry["sequence"]

        impression_entry = append_live_ledger_entry(
            run_id=str(run.id),
            entry_kind=LedgerEntryKind.SUBAGENT_NOTE,
            content=impression,
            agent_execution_id=str(execution.id),
            agent_kind=CoachAgentKind.SUBAGENT,
            agent_name=execution.agent_name,
            window_start_ms=execution.window_start_ms,
            window_end_ms=execution.window_end_ms,
            payload={
                "title": "Window impression",
                "note_type": "window_impression",
                "model_name": reasoning_result.model_name,
            },
        )
        last_sequence = impression_entry["sequence"]

        execution = mark_agent_completed(
            execution=execution,
            output_seq_to=last_sequence,
        )
        touch_agent_heartbeat(execution=execution)
        return {
            "execution_id": str(execution.id),
            "run_id": str(run.id),
            "status": "completed",
            "model_name": reasoning_result.model_name,
            "output_seq_to": execution.output_seq_to,
            "notes_count": len(notes),
            "impression": impression,
            "usage": dict(reasoning_result.usage),
        }
    except Exception as exc:
        mark_agent_failed(execution=execution, error_message=str(exc))
        raise


def finalize_subagent_run(*, run_id: str) -> dict[str, Any]:
    """Flush run-scoped live-ledger entries into DB and mark run as completed."""
    run = CoachOrchestrationRun.objects.select_related("session").get(id=run_id)
    active_execution_exists = run.agent_executions.filter(
        status__in=[
            CoachAgentExecutionStatus.QUEUED,
            CoachAgentExecutionStatus.PROCESSING,
        ]
    ).exists()
    if active_execution_exists:
        raise SubagentWorkflowError(
            f"Run {run.id} still has queued/processing agent executions."
        )

    try:
        live_entries = read_live_ledger_slice(run_id=str(run.id), sequence_gt=0)
        live_entries.sort(key=lambda item: item["sequence"])

        execution_ids = [
            UUID(entry["agent_execution_id"])
            for entry in live_entries
            if isinstance(entry.get("agent_execution_id"), str)
            and entry["agent_execution_id"].strip()
        ]
        executions_by_id = CoachAgentExecution.objects.in_bulk(execution_ids)

        run.refresh_from_db()
        next_expected_sequence = run.latest_ledger_sequence + 1
        flushed_entries = 0

        for entry in live_entries:
            sequence = entry["sequence"]
            if sequence < next_expected_sequence:
                continue
            if sequence != next_expected_sequence:
                raise SubagentWorkflowError(
                    "Live ledger sequence is non-contiguous; cannot flush safely."
                )

            execution: CoachAgentExecution | None = None
            raw_execution_id = entry.get("agent_execution_id")
            if isinstance(raw_execution_id, str) and raw_execution_id.strip():
                execution = executions_by_id.get(UUID(raw_execution_id))

            db_entry = append_ledger_entry(
                run=run,
                entry_kind=entry["entry_kind"],
                content=entry["content"],
                agent_execution=execution,
                agent_kind=entry.get("agent_kind"),
                agent_name=entry.get("agent_name", ""),
                window_start_ms=entry.get("window_start_ms"),
                window_end_ms=entry.get("window_end_ms"),
                payload=dict(entry.get("payload") or {}),
            )
            flushed_entries += 1
            next_expected_sequence = db_entry.sequence + 1

        run = mark_run_completed(run=run)
        clear_live_ledger(run_id=str(run.id))
        return {
            "run_id": str(run.id),
            "status": str(run.status),
            "flushed_entries": flushed_entries,
            "latest_ledger_sequence": run.latest_ledger_sequence,
        }
    except Exception as exc:
        mark_run_failed(run=run, error_message=str(exc))
        raise
