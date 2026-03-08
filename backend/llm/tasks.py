from __future__ import annotations

import json
import logging
from typing import Any
from uuid import UUID

from celery import shared_task
from django.db import transaction

from sessions.models import CoachOrchestrationRun, SessionStatus

from .flagship_final_workflow import run_flagship_final_reconciliation
from .subagent_workflow import finalize_subagent_run, run_subagent_execution

logger = logging.getLogger(__name__)


def _set_session_status_for_run(*, run_id: str, status: str) -> bool:
    try:
        UUID(str(run_id))
    except (TypeError, ValueError):
        return False

    try:
        with transaction.atomic():
            run = (
                CoachOrchestrationRun.objects.select_related("session")
                .select_for_update()
                .get(id=run_id)
            )
            run.session.status = status
            run.session.save(update_fields=["status", "updated_at"])
            return True
    except CoachOrchestrationRun.DoesNotExist:
        return False
    except Exception:
        logger.warning(
            "Unable to update session status for run %s to %s.",
            run_id,
            status,
            exc_info=True,
        )
        return False


def _log_finalized_ledger(*, run_id: str) -> None:
    try:
        UUID(str(run_id))
    except (TypeError, ValueError):
        return

    try:
        run = CoachOrchestrationRun.objects.get(id=run_id)
    except CoachOrchestrationRun.DoesNotExist:
        logger.info("Cannot print final ledger; run %s was not found.", run_id)
        return
    except Exception:
        logger.warning(
            "Unable to read finalized ledger for run %s.",
            run_id,
            exc_info=True,
        )
        return

    entries = list(run.ledger_entries.order_by("sequence", "created_at"))
    logger.info(
        "Finalized coach ledger for run %s (session=%s, entries=%s).",
        run_id,
        run.session_id,
        len(entries),
    )
    if not entries:
        logger.info("Finalized coach ledger is empty for run %s.", run_id)
        return

    for entry in entries:
        payload_text = ""
        if isinstance(entry.payload, dict) and entry.payload:
            payload_text = f" | payload={json.dumps(entry.payload, sort_keys=True)}"
        logger.info(
            "LEDGER[%s] kind=%s agent=%s content=%s%s",
            entry.sequence,
            entry.entry_kind,
            entry.agent_name or "-",
            entry.content,
            payload_text,
        )


@shared_task(name="llm.subagent.run_window")
def run_subagent_window_task(
    *,
    execution_id: str,
    session_id: str,
    events: list[dict[str, Any]],
    word_map: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one queued subagent window job and append live-ledger updates."""
    return run_subagent_execution(
        execution_id=execution_id,
        session_id=session_id,
        events=events,
        word_map=word_map,
        metadata=metadata,
    )


@shared_task(name="llm.subagent.finalize_run")
def finalize_subagent_run_task(*, run_id: str) -> dict[str, Any]:
    """Flush Redis live-ledger entries to DB and complete the run."""
    return finalize_subagent_run(run_id=run_id)


@shared_task(name="llm.flagship.final_reconcile")
def run_flagship_final_reconcile_task(
    *,
    run_id: str,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Run the flagship-final reconciliation graph for one completed run."""
    try:
        result = run_flagship_final_reconciliation(
            run_id=run_id,
            system_prompt=system_prompt,
        )
    except Exception:
        _set_session_status_for_run(
            run_id=run_id,
            status=SessionStatus.COACH_FAILED,
        )
        raise

    _set_session_status_for_run(
        run_id=run_id,
        status=SessionStatus.READY,
    )
    _log_finalized_ledger(run_id=run_id)
    return result
