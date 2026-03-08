from __future__ import annotations

from typing import Any
from uuid import UUID

from django.db import transaction

from .models import (
    CoachingSession,
    SessionAlignedWord,
    SessionEvent,
    SessionOverallMetrics,
)


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    return int(value)


def _coerce_float(value: Any, default: float) -> float:
    if value is None:
        return default
    return float(value)


def persist_canonical_payload(
    *,
    canonical_payload: dict[str, Any],
    skip_missing_session: bool = True,
) -> dict[str, Any]:
    meta = canonical_payload.get("meta")
    if not isinstance(meta, dict):
        raise ValueError("canonical_payload.meta must be an object")

    session_uuid = meta.get("session_uuid")
    if not isinstance(session_uuid, str):
        raise ValueError("canonical_payload.meta.session_uuid is required")
    UUID(session_uuid)

    session = CoachingSession.objects.filter(id=session_uuid).only("id").first()
    if session is None:
        if skip_missing_session:
            return {
                "session_uuid": session_uuid,
                "written": False,
                "reason": "missing_session",
                "overall_rows": 0,
                "aligned_rows": 0,
                "event_rows": 0,
            }
        raise CoachingSession.DoesNotExist(f"CoachingSession {session_uuid} does not exist")

    overall_metrics = canonical_payload.get("overall_metrics", {})
    aligned_rows = canonical_payload.get("aligned_table", [])
    event_rows = canonical_payload.get("events", [])

    if not isinstance(aligned_rows, list):
        raise ValueError("canonical_payload.aligned_table must be a list")
    if not isinstance(event_rows, list):
        raise ValueError("canonical_payload.events must be a list")

    with transaction.atomic():
        SessionOverallMetrics.objects.update_or_create(
            session=session,
            defaults={"metrics": overall_metrics},
        )

        SessionAlignedWord.objects.filter(session=session).delete()
        SessionEvent.objects.filter(session=session).delete()

        aligned_payload = [
            SessionAlignedWord(
                session=session,
                word_id=_coerce_int(row.get("word_id"), i),
                word_json=row,
                start_time=_coerce_float(row.get("start_sec"), 0.0),
                end_time=_coerce_float(row.get("end_sec"), 0.0),
            )
            for i, row in enumerate(aligned_rows)
            if isinstance(row, dict)
        ]
        if aligned_payload:
            SessionAlignedWord.objects.bulk_create(aligned_payload, batch_size=1000)

        event_payload = [
            SessionEvent(
                session=session,
                event_id=_coerce_int(row.get("event_id"), i),
                event_json=row,
                start_time=_coerce_float(row.get("start_sec"), 0.0),
                end_time=_coerce_float(row.get("end_sec"), 0.0),
            )
            for i, row in enumerate(event_rows)
            if isinstance(row, dict)
        ]
        if event_payload:
            SessionEvent.objects.bulk_create(event_payload, batch_size=1000)

    return {
        "session_uuid": session_uuid,
        "written": True,
        "reason": "",
        "overall_rows": 1,
        "aligned_rows": len(aligned_payload),
        "event_rows": len(event_payload),
    }
