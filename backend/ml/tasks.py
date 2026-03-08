from __future__ import annotations

import logging
import random
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from django.conf import settings
from django.db import transaction

from llm.enqueue import enqueue_full_coach_workflow_job
from sessions.models import CoachingSession, SessionStatus
from sessions.services import persist_canonical_payload

from celery import shared_task

from .pipeline_facade import run_pipeline

WINDOW_SIZE_MS = 30_000
logger = logging.getLogger(__name__)


def _update_session_fields(
    *,
    session_id: str,
    **fields: Any,
) -> CoachingSession:
    with transaction.atomic():
        session = CoachingSession.objects.select_for_update().get(id=session_id)
        for field_name, value in fields.items():
            setattr(session, field_name, value)
        update_fields = list(fields.keys())
        if "updated_at" not in update_fields:
            update_fields.append("updated_at")
        session.save(update_fields=update_fields)
        return session


def _default_sample_media_paths() -> tuple[Path, Path]:
    ml_root = Path(settings.BASE_DIR) / "ml"
    return ml_root / "evan_test.wav", ml_root / "evan_test.mp4"


def _extract_audio_from_video(
    *,
    session_id: str,
    video_path: Path,
) -> Path | None:
    extract_root = Path(settings.BASE_DIR) / "ml" / "run_outputs" / str(session_id) / "audio_extracts"
    extract_root.mkdir(parents=True, exist_ok=True)
    output_path = extract_root / f"{uuid.uuid4().hex}.wav"

    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        str(output_path),
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
        if completed.stderr:
            logger.info(
                "ffmpeg audio extraction stderr for session %s: %s",
                session_id,
                completed.stderr.strip(),
            )
    except FileNotFoundError:
        logger.warning(
            "ffmpeg is not installed; falling back to bundled sample media for session %s.",
            session_id,
        )
        return None
    except subprocess.CalledProcessError as exc:
        logger.warning(
            "ffmpeg audio extraction failed for session %s (exit=%s): %s",
            session_id,
            exc.returncode,
            (exc.stderr or "").strip(),
        )
        return None
    except OSError as exc:
        logger.warning(
            "ffmpeg audio extraction OS error for session %s: %s",
            session_id,
            str(exc),
        )
        return None

    if not output_path.exists():
        logger.warning(
            "ffmpeg reported success but output audio file was not created for session %s.",
            session_id,
        )
        return None
    return output_path


def _resolve_ml_media_inputs(*, session: CoachingSession) -> dict[str, str]:
    sample_audio_path, sample_video_path = _default_sample_media_paths()
    if not sample_audio_path.exists() or not sample_video_path.exists():
        raise RuntimeError("Bundled fallback media files are missing in backend/ml.")

    fallback_reason = "missing_session_video"
    session_video_path: Path | None = None
    if session.video_file:
        try:
            maybe_video_path = Path(session.video_file.path)
        except (ValueError, NotImplementedError):
            fallback_reason = "unavailable_session_video_path"
        else:
            if maybe_video_path.exists():
                session_video_path = maybe_video_path
            else:
                fallback_reason = "session_video_file_missing_on_disk"

    if session_video_path is not None:
        extracted_audio_path = _extract_audio_from_video(
            session_id=str(session.id),
            video_path=session_video_path,
        )
        if extracted_audio_path is not None:
            return {
                "audio_path": str(extracted_audio_path),
                "video_path": str(session_video_path),
                "source": "session_media",
                "fallback_reason": "",
            }
        fallback_reason = "failed_audio_extract_from_session_video"

    return {
        "audio_path": str(sample_audio_path),
        "video_path": str(sample_video_path),
        "source": "sample_fallback",
        "fallback_reason": fallback_reason,
    }


def _to_milliseconds(value: Any) -> int | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None
    if seconds < 0:
        seconds = 0.0
    return int(round(seconds * 1000))


def _build_llm_windows_from_canonical_payload(
    *,
    canonical_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    aligned_rows = canonical_payload.get("aligned_table", [])
    event_rows = canonical_payload.get("events", [])
    if not isinstance(aligned_rows, list):
        aligned_rows = []
    if not isinstance(event_rows, list):
        event_rows = []

    events: list[dict[str, Any]] = []
    for index, row in enumerate(event_rows):
        if not isinstance(row, dict):
            continue
        start_ms = _to_milliseconds(row.get("start_sec"))
        end_ms = _to_milliseconds(row.get("end_sec"))
        event_type = str(row.get("event_type", "")).strip()
        if start_ms is None or end_ms is None or not event_type:
            continue
        normalized_end = max(start_ms, end_ms)
        raw_event_id = row.get("event_id")
        event_id = str(raw_event_id).strip() if raw_event_id is not None else ""
        if not event_id:
            event_id = f"event-{index}"
        events.append(
            {
                "event_id": event_id,
                "event_type": event_type,
                "start_ms": start_ms,
                "end_ms": normalized_end,
                "metadata": dict(row),
            }
        )

    word_map: list[dict[str, Any]] = []
    for row in aligned_rows:
        if not isinstance(row, dict):
            continue
        word = str(row.get("word", "")).strip()
        start_ms = _to_milliseconds(row.get("start_sec"))
        end_ms = _to_milliseconds(row.get("end_sec"))
        if not word or start_ms is None or end_ms is None:
            continue
        word_map.append(
            {
                "word": word,
                "start_ms": start_ms,
                "end_ms": max(start_ms, end_ms),
            }
        )

    events.sort(key=lambda item: (item["start_ms"], item["end_ms"], item["event_id"]))
    word_map.sort(key=lambda item: (item["start_ms"], item["end_ms"], item["word"]))

    max_end_ms = 0
    if events:
        max_end_ms = max(max_end_ms, max(item["end_ms"] for item in events))
    if word_map:
        max_end_ms = max(max_end_ms, max(item["end_ms"] for item in word_map))
    if max_end_ms <= 0:
        return []

    windows: list[dict[str, Any]] = []
    window_start_ms = 0
    window_index = 0
    while window_start_ms <= max_end_ms:
        window_end_ms = window_start_ms + WINDOW_SIZE_MS
        window_events = [
            item
            for item in events
            if item["end_ms"] >= window_start_ms and item["start_ms"] < window_end_ms
        ]
        window_words = [
            item
            for item in word_map
            if item["end_ms"] >= window_start_ms and item["start_ms"] < window_end_ms
        ]
        if window_events or window_words:
            windows.append(
                {
                    "window_start_ms": window_start_ms,
                    "window_end_ms": window_end_ms,
                    "events": window_events,
                    "word_map": window_words,
                    "metadata": {
                        "window_index": window_index,
                        "event_count": len(window_events),
                        "word_count": len(window_words),
                    },
                }
            )
        window_start_ms = window_end_ms
        window_index += 1
    return windows


@shared_task(name="ml.demo.random_sleep")
def random_sleep_demo_task(
    *,
    min_seconds: int = 1,
    max_seconds: int = 8,
    label: str = "demo",
) -> dict[str, int | str]:
    if min_seconds < 0:
        raise ValueError("min_seconds must be non-negative")
    if max_seconds < min_seconds:
        raise ValueError("max_seconds must be greater than or equal to min_seconds")

    sleep_seconds = random.randint(min_seconds, max_seconds)
    time.sleep(sleep_seconds)
    return {
        "label": label,
        "sleep_seconds": sleep_seconds,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


@shared_task(name="ml.session.run_workflow")
def run_session_ml_workflow_task(*, session_id: str) -> dict[str, Any]:
    workflow_t0 = time.time()
    logger.info("ML workflow task started | session_id=%s", session_id)
    session = CoachingSession.objects.get(id=session_id)
    logger.info(
        "ML workflow loaded session | session_id=%s current_status=%s",
        session_id,
        session.status,
    )
    if session.status not in {SessionStatus.QUEUED_ML, SessionStatus.PROCESSING_ML}:
        raise ValueError(
            "Session must be queued_ml or processing_ml before ML workflow starts."
        )

    session = _update_session_fields(
        session_id=str(session.id),
        status=SessionStatus.PROCESSING_ML,
    )
    logger.info(
        "ML workflow status updated | session_id=%s new_status=%s",
        session.id,
        SessionStatus.PROCESSING_ML,
    )
    ml_ready_reached = False

    try:
        media_inputs = _resolve_ml_media_inputs(session=session)
        logger.info(
            "ML media resolved | session_id=%s source=%s fallback_reason=%s audio_path=%s video_path=%s",
            session.id,
            media_inputs["source"],
            media_inputs["fallback_reason"] or "none",
            media_inputs["audio_path"],
            media_inputs["video_path"],
        )
        logger.info("ML pipeline invoke | session_id=%s", session.id)
        pipeline_result = run_pipeline(
            audio_path=media_inputs["audio_path"],
            video_path=media_inputs["video_path"],
            session_uuid=str(session.id),
        )
        logger.info(
            "ML pipeline finished | session_id=%s run_dir=%s run_report_path=%s",
            session.id,
            str(pipeline_result.get("run_dir", "")),
            str(pipeline_result.get("run_report_path", "")),
        )
        canonical_payload = dict(pipeline_result.get("canonical_payload") or {})
        logger.info(
            "ML payload built | session_id=%s aligned_rows=%s event_rows=%s",
            session.id,
            len(canonical_payload.get("aligned_table") or []),
            len(canonical_payload.get("events") or []),
        )
        persist_result = persist_canonical_payload(
            canonical_payload=canonical_payload,
            skip_missing_session=False,
        )
        logger.info(
            "ML payload persisted | session_id=%s written=%s overall_rows=%s aligned_rows=%s event_rows=%s",
            session.id,
            persist_result.get("written"),
            persist_result.get("overall_rows"),
            persist_result.get("aligned_rows"),
            persist_result.get("event_rows"),
        )

        _update_session_fields(
            session_id=str(session.id),
            status=SessionStatus.ML_READY,
        )
        ml_ready_reached = True
        logger.info(
            "ML workflow status updated | session_id=%s new_status=%s",
            session.id,
            SessionStatus.ML_READY,
        )

        windows = _build_llm_windows_from_canonical_payload(
            canonical_payload=canonical_payload,
        )
        logger.info(
            "ML windows built | session_id=%s windows_count=%s",
            session.id,
            len(windows),
        )
        coach_result = enqueue_full_coach_workflow_job(
            session_id=str(session.id),
            windows=windows,
            subagent_metadata={
                "ml_media_source": media_inputs["source"],
            },
        )

        workflow_task_id = str(coach_result.get("workflow_task_id", "")).strip()
        run_id = str(coach_result.get("run_id", "")).strip()
        logger.info(
            "Coach workflow enqueued | session_id=%s workflow_task_id=%s run_id=%s",
            session.id,
            workflow_task_id or "none",
            run_id or "none",
        )
        _update_session_fields(
            session_id=str(session.id),
            status=SessionStatus.PROCESSING_COACH,
            coach_task_id=workflow_task_id or None,
        )
        logger.info(
            "ML workflow handoff complete | session_id=%s new_status=%s duration_sec=%.3f",
            session.id,
            SessionStatus.PROCESSING_COACH,
            round(time.time() - workflow_t0, 3),
        )

        return {
            "status": "ok",
            "session_id": str(session.id),
            "ml_media_source": media_inputs["source"],
            "ml_media_fallback_reason": media_inputs["fallback_reason"],
            "windows_count": len(windows),
            "workflow_task_id": workflow_task_id,
            "run_id": run_id,
            "persist_result": persist_result,
        }
    except Exception:
        failure_status = (
            SessionStatus.COACH_FAILED
            if ml_ready_reached
            else SessionStatus.FAILED
        )
        try:
            _update_session_fields(
                session_id=str(session.id),
                status=failure_status,
            )
        except CoachingSession.DoesNotExist:
            pass
        logger.exception(
            "Session ML workflow failed | session_id=%s fallback_status=%s duration_sec=%.3f",
            session.id,
            failure_status,
            round(time.time() - workflow_t0, 3),
        )
        raise
