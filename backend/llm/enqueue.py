from __future__ import annotations

from typing import Any

from celery import chain, chord
from celery.result import AsyncResult

from sessions.models import (
    CoachOrchestrationRun,
    CoachOrchestrationRunStatus,
    CoachingSession,
    SessionStatus,
)

from .ledger import create_orchestration_run, mark_run_failed, mark_run_processing
from .subagent_workflow import create_subagent_execution_for_window
from .tasks import (
    finalize_subagent_run_task,
    run_flagship_final_reconcile_task,
    run_subagent_window_task,
)


def _normalize_subagent_windows(
    *,
    windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate and sort subagent windows for deterministic orchestration dispatch."""
    normalized_windows: list[dict[str, Any]] = []
    for index, item in enumerate(windows):
        if not isinstance(item, dict):
            raise ValueError(f"windows[{index}] must be an object.")
        if "window_start_ms" not in item or "window_end_ms" not in item:
            raise ValueError(
                f"windows[{index}] must include window_start_ms and window_end_ms."
            )
        window_start_ms = int(item["window_start_ms"])
        window_end_ms = int(item["window_end_ms"])
        if window_start_ms < 0:
            raise ValueError(f"windows[{index}].window_start_ms must be >= 0.")
        if window_end_ms < window_start_ms:
            raise ValueError(
                f"windows[{index}].window_end_ms must be >= window_start_ms."
            )

        events = item.get("events", [])
        word_map = item.get("word_map", [])
        metadata = item.get("metadata", {})
        if not isinstance(events, list):
            raise ValueError(f"windows[{index}].events must be an array.")
        if not isinstance(word_map, list):
            raise ValueError(f"windows[{index}].word_map must be an array.")
        if not isinstance(metadata, dict):
            raise ValueError(f"windows[{index}].metadata must be an object.")

        normalized_windows.append(
            {
                "window_start_ms": window_start_ms,
                "window_end_ms": window_end_ms,
                "events": list(events),
                "word_map": list(word_map),
                "metadata": dict(metadata),
            }
        )

    return sorted(
        normalized_windows,
        key=lambda item: (item["window_start_ms"], item["window_end_ms"]),
    )


def enqueue_full_coach_workflow_job(
    *,
    session_id: str,
    windows: list[dict[str, Any]],
    subagent_metadata: dict[str, Any] | None = None,
    flagship_final_system_prompt: str | None = None,
) -> dict[str, Any]:
    """Create a run and enqueue full coach workflow: subagents -> finalize -> flagship final."""
    normalized_windows = _normalize_subagent_windows(windows=windows)
    session = CoachingSession.objects.get(id=session_id)
    if session.status not in {
        SessionStatus.ML_READY,
        SessionStatus.PROCESSING_COACH,
    }:
        raise ValueError(
            "Session must be in ml_ready or processing_coach before coach orchestration."
        )

    run = create_orchestration_run(session=session)
    if session.status != SessionStatus.PROCESSING_COACH:
        session.status = SessionStatus.PROCESSING_COACH
        session.save(update_fields=["status", "updated_at"])

    try:
        header_signatures = []
        execution_ids: list[str] = []
        for window in normalized_windows:
            execution = create_subagent_execution_for_window(
                run=run,
                window_start_ms=int(window["window_start_ms"]),
                window_end_ms=int(window["window_end_ms"]),
            )
            execution_ids.append(str(execution.id))
            merged_metadata = {
                **dict(subagent_metadata or {}),
                **dict(window.get("metadata", {})),
            }
            header_signatures.append(
                run_subagent_window_task.si(
                    execution_id=str(execution.id),
                    session_id=str(session.id),
                    events=list(window.get("events", [])),
                    word_map=list(window.get("word_map", [])),
                    metadata=merged_metadata,
                )
            )

        finalize_signature = finalize_subagent_run_task.si(run_id=str(run.id))
        if flagship_final_system_prompt is None:
            flagship_final_signature = run_flagship_final_reconcile_task.si(
                run_id=str(run.id),
            )
        else:
            flagship_final_signature = run_flagship_final_reconcile_task.si(
                run_id=str(run.id),
                system_prompt=str(flagship_final_system_prompt),
            )

        completion_chain = chain(finalize_signature, flagship_final_signature)
        if header_signatures:
            workflow_result = chord(header_signatures)(completion_chain)
        else:
            workflow_result = completion_chain.apply_async()
        return {
            "session_id": str(session.id),
            "run_id": str(run.id),
            "workflow_task_id": str(workflow_result.id),
            "subagent_task_count": len(header_signatures),
            "subagent_execution_ids": execution_ids,
        }
    except Exception as exc:
        mark_run_failed(
            run=run,
            error_message=f"Failed to enqueue full coach workflow: {exc}",
        )
        raise


def enqueue_subagent_window_job(
    *,
    run: CoachOrchestrationRun,
    session_id: str,
    window_start_ms: int,
    window_end_ms: int,
    events: list[dict[str, Any]],
    word_map: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> tuple[AsyncResult, str]:
    """Create one queued window execution and enqueue its subagent task."""
    if run.status == CoachOrchestrationRunStatus.QUEUED:
        run = mark_run_processing(run=run)

    execution = create_subagent_execution_for_window(
        run=run,
        window_start_ms=window_start_ms,
        window_end_ms=window_end_ms,
    )
    async_result = run_subagent_window_task.apply_async(
        kwargs={
            "execution_id": str(execution.id),
            "session_id": str(session_id),
            "events": events,
            "word_map": word_map,
            "metadata": dict(metadata or {}),
        }
    )
    return async_result, str(execution.id)


def enqueue_subagent_window_jobs(
    *,
    run: CoachOrchestrationRun,
    session_id: str,
    windows: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Enqueue many subagent window jobs in deterministic chronological order."""
    sorted_windows = sorted(
        windows,
        key=lambda item: (
            int(item.get("window_start_ms", 0)),
            int(item.get("window_end_ms", 0)),
        ),
    )
    jobs: list[dict[str, str]] = []
    for window in sorted_windows:
        async_result, execution_id = enqueue_subagent_window_job(
            run=run,
            session_id=str(session_id),
            window_start_ms=int(window["window_start_ms"]),
            window_end_ms=int(window["window_end_ms"]),
            events=list(window.get("events", [])),
            word_map=list(window.get("word_map", [])),
            metadata={**dict(metadata or {}), **dict(window.get("metadata", {}))},
        )
        jobs.append({"task_id": async_result.id, "execution_id": execution_id})
    return jobs


def enqueue_subagent_finalize_job(*, run_id: str) -> AsyncResult:
    """Enqueue the explicit finalize job that flushes Redis live-ledger to DB."""
    return finalize_subagent_run_task.apply_async(kwargs={"run_id": str(run_id)})


def enqueue_flagship_final_reconciliation_job(
    *,
    run_id: str,
    system_prompt: str | None = None,
) -> AsyncResult:
    """Enqueue flagship-final reconciliation graph execution for a run."""
    kwargs: dict[str, Any] = {"run_id": str(run_id)}
    if system_prompt is not None:
        kwargs["system_prompt"] = str(system_prompt)
    return run_flagship_final_reconcile_task.apply_async(kwargs=kwargs)
