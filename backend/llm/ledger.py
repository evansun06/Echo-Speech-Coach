from __future__ import annotations

from typing import Any

from django.db import transaction
from django.utils import timezone

from sessions.models import (
    CoachAgentExecution,
    CoachAgentExecutionStatus,
    CoachAgentKind,
    CoachLedgerEntry,
    CoachOrchestrationRun,
    CoachOrchestrationRunStatus,
    CoachingSession,
    LedgerEntryKind,
)


class RunStateError(RuntimeError):
    pass


class LedgerValidationError(RuntimeError):
    pass


ACTIVE_RUN_STATUSES = (
    CoachOrchestrationRunStatus.QUEUED,
    CoachOrchestrationRunStatus.PROCESSING,
)


def _lock_session(session_id) -> CoachingSession:
    """Lock and return a session row for transactional orchestration updates."""
    return CoachingSession.objects.select_for_update().get(id=session_id)


def _lock_run(run_id) -> CoachOrchestrationRun:
    """Lock and return an orchestration run row inside a transaction."""
    return CoachOrchestrationRun.objects.select_for_update().get(id=run_id)


def _lock_execution(execution_id) -> CoachAgentExecution:
    """Lock and return an agent execution row inside a transaction."""
    return CoachAgentExecution.objects.select_for_update().get(id=execution_id)


def create_orchestration_run(*, session: CoachingSession) -> CoachOrchestrationRun:
    """Create the next queued orchestration run for a session if none is active."""
    with transaction.atomic():
        locked_session = _lock_session(session.id)
        has_active_run = CoachOrchestrationRun.objects.filter(
            session=locked_session,
            status__in=ACTIVE_RUN_STATUSES,
        ).exists()
        if has_active_run:
            raise RunStateError(
                f"Session {locked_session.id} already has an active coach run."
            )

        last_run = (
            CoachOrchestrationRun.objects.filter(session=locked_session)
            .order_by("-run_index")
            .first()
        )
        run_index = 1 if last_run is None else last_run.run_index + 1
        return CoachOrchestrationRun.objects.create(
            session=locked_session,
            run_index=run_index,
            status=CoachOrchestrationRunStatus.QUEUED,
        )


def mark_run_processing(*, run: CoachOrchestrationRun) -> CoachOrchestrationRun:
    """Mark a run as processing and ensure its start timestamp is set."""
    with transaction.atomic():
        locked_run = _lock_run(run.id)
        locked_run.status = CoachOrchestrationRunStatus.PROCESSING
        if locked_run.started_at is None:
            locked_run.started_at = timezone.now()
        locked_run.save(update_fields=["status", "started_at", "updated_at"])
        return locked_run


def mark_run_completed(*, run: CoachOrchestrationRun) -> CoachOrchestrationRun:
    """Mark a run as completed and clear failure metadata."""
    with transaction.atomic():
        locked_run = _lock_run(run.id)
        now = timezone.now()
        locked_run.status = CoachOrchestrationRunStatus.COMPLETED
        if locked_run.started_at is None:
            locked_run.started_at = now
        locked_run.completed_at = now
        locked_run.failed_at = None
        locked_run.error_message = ""
        locked_run.save(
            update_fields=[
                "status",
                "started_at",
                "completed_at",
                "failed_at",
                "error_message",
                "updated_at",
            ]
        )
        return locked_run


def mark_run_failed(
    *, run: CoachOrchestrationRun, error_message: str
) -> CoachOrchestrationRun:
    """Mark a run as failed and persist a normalized error message."""
    with transaction.atomic():
        locked_run = _lock_run(run.id)
        now = timezone.now()
        locked_run.status = CoachOrchestrationRunStatus.FAILED
        if locked_run.started_at is None:
            locked_run.started_at = now
        locked_run.completed_at = None
        locked_run.failed_at = now
        locked_run.error_message = error_message.strip()
        locked_run.save(
            update_fields=[
                "status",
                "started_at",
                "completed_at",
                "failed_at",
                "error_message",
                "updated_at",
            ]
        )
        return locked_run


def create_agent_execution(
    *,
    run: CoachOrchestrationRun,
    agent_kind: str,
    agent_name: str,
    window_start_ms: int | None = None,
    window_end_ms: int | None = None,
    input_seq_from: int | None = None,
    input_seq_to: int | None = None,
) -> CoachAgentExecution:
    """Create the next queued execution record for a run and agent window."""
    with transaction.atomic():
        locked_run = _lock_run(run.id)
        last_execution = (
            CoachAgentExecution.objects.filter(run=locked_run)
            .order_by("-execution_index")
            .first()
        )
        execution_index = 1 if last_execution is None else last_execution.execution_index + 1
        return CoachAgentExecution.objects.create(
            run=locked_run,
            execution_index=execution_index,
            agent_kind=agent_kind,
            agent_name=agent_name,
            status=CoachAgentExecutionStatus.QUEUED,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            input_seq_from=input_seq_from,
            input_seq_to=input_seq_to,
        )


def mark_agent_processing(*, execution: CoachAgentExecution) -> CoachAgentExecution:
    """Mark an agent execution as processing and set first-start timestamp."""
    with transaction.atomic():
        locked_execution = _lock_execution(execution.id)
        locked_execution.status = CoachAgentExecutionStatus.PROCESSING
        if locked_execution.started_at is None:
            locked_execution.started_at = timezone.now()
        locked_execution.save(update_fields=["status", "started_at", "updated_at"])
        return locked_execution


def touch_agent_heartbeat(
    *, execution: CoachAgentExecution, beat_at=None
) -> CoachAgentExecution:
    """Update the latest heartbeat timestamp for a running agent execution."""
    with transaction.atomic():
        locked_execution = _lock_execution(execution.id)
        locked_execution.last_heartbeat_at = beat_at or timezone.now()
        locked_execution.save(update_fields=["last_heartbeat_at", "updated_at"])
        return locked_execution


def mark_agent_completed(
    *, execution: CoachAgentExecution, output_seq_to: int | None = None
) -> CoachAgentExecution:
    """Mark an execution as completed and optionally persist output sequence."""
    with transaction.atomic():
        locked_execution = _lock_execution(execution.id)
        now = timezone.now()
        locked_execution.status = CoachAgentExecutionStatus.COMPLETED
        locked_execution.finished_at = now
        locked_execution.failed_at = None
        locked_execution.error_message = ""
        if output_seq_to is not None:
            locked_execution.output_seq_to = output_seq_to
        locked_execution.save(
            update_fields=[
                "status",
                "finished_at",
                "failed_at",
                "error_message",
                "output_seq_to",
                "updated_at",
            ]
        )
        return locked_execution


def mark_agent_failed(
    *, execution: CoachAgentExecution, error_message: str
) -> CoachAgentExecution:
    """Mark an execution as failed and persist a normalized error message."""
    with transaction.atomic():
        locked_execution = _lock_execution(execution.id)
        now = timezone.now()
        locked_execution.status = CoachAgentExecutionStatus.FAILED
        locked_execution.finished_at = None
        locked_execution.failed_at = now
        locked_execution.error_message = error_message.strip()
        locked_execution.save(
            update_fields=[
                "status",
                "finished_at",
                "failed_at",
                "error_message",
                "updated_at",
            ]
        )
        return locked_execution


def append_ledger_entry(
    *,
    run: CoachOrchestrationRun,
    entry_kind: str,
    content: str,
    agent_execution: CoachAgentExecution | None = None,
    agent_kind: str | None = None,
    agent_name: str = "",
    window_start_ms: int | None = None,
    window_end_ms: int | None = None,
    payload: dict[str, Any] | None = None,
) -> CoachLedgerEntry:
    """Append a new ledger entry and atomically advance run sequence tracking."""
    if agent_execution is not None and agent_execution.run_id != run.id:
        raise LedgerValidationError(
            "agent_execution does not belong to the provided run."
        )

    with transaction.atomic():
        locked_run = _lock_run(run.id)
        sequence = locked_run.latest_ledger_sequence + 1

        resolved_agent_kind = (
            agent_kind
            if agent_kind is not None
            else (agent_execution.agent_kind if agent_execution is not None else None)
        )
        resolved_agent_name = (
            agent_name
            if agent_name
            else (agent_execution.agent_name if agent_execution is not None else "")
        )
        resolved_window_start_ms = (
            window_start_ms
            if window_start_ms is not None
            else (
                agent_execution.window_start_ms
                if agent_execution is not None
                else None
            )
        )
        resolved_window_end_ms = (
            window_end_ms
            if window_end_ms is not None
            else (
                agent_execution.window_end_ms
                if agent_execution is not None
                else None
            )
        )

        entry = CoachLedgerEntry.objects.create(
            run=locked_run,
            agent_execution=agent_execution,
            sequence=sequence,
            entry_kind=entry_kind,
            agent_kind=resolved_agent_kind,
            agent_name=resolved_agent_name,
            window_start_ms=resolved_window_start_ms,
            window_end_ms=resolved_window_end_ms,
            content=content,
            payload=dict(payload or {}),
        )
        locked_run.latest_ledger_sequence = sequence
        locked_run.save(update_fields=["latest_ledger_sequence", "updated_at"])
        return entry


def read_ledger_slice(
    *,
    run: CoachOrchestrationRun,
    sequence_gt: int = 0,
    sequence_lte: int | None = None,
    entry_kind: str | None = None,
    limit: int | None = None,
) -> list[CoachLedgerEntry]:
    """Read ordered ledger entries for a run with sequence and kind filters."""
    queryset = CoachLedgerEntry.objects.filter(run=run).order_by("sequence", "created_at")
    if sequence_gt is not None:
        queryset = queryset.filter(sequence__gt=sequence_gt)
    if sequence_lte is not None:
        queryset = queryset.filter(sequence__lte=sequence_lte)
    if entry_kind:
        queryset = queryset.filter(entry_kind=entry_kind)
    if limit is not None:
        queryset = queryset[:limit]
    return list(queryset)


__all__ = [
    "ACTIVE_RUN_STATUSES",
    "LedgerValidationError",
    "RunStateError",
    "append_ledger_entry",
    "create_agent_execution",
    "create_orchestration_run",
    "mark_agent_completed",
    "mark_agent_failed",
    "mark_agent_processing",
    "mark_run_completed",
    "mark_run_failed",
    "mark_run_processing",
    "read_ledger_slice",
    "touch_agent_heartbeat",
    "CoachAgentKind",
    "LedgerEntryKind",
]
