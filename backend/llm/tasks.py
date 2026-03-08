from __future__ import annotations

from typing import Any

from celery import shared_task

from .flagship_final_workflow import run_flagship_final_reconciliation
from .subagent_workflow import finalize_subagent_run, run_subagent_execution


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
    return run_flagship_final_reconciliation(
        run_id=run_id,
        system_prompt=system_prompt,
    )
