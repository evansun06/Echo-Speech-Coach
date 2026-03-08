from __future__ import annotations

from collections import defaultdict
from typing import Any

from rest_framework import serializers

from llm.live_ledger import get_live_ledger_latest_sequence, read_live_ledger_slice

from .models import (
    CoachAgentExecutionStatus,
    CoachOrchestrationRun,
    CoachOrchestrationRunStatus,
    CoachingSession,
    SessionStatus,
)


def _absolute_file_url(request, file_field) -> str | None:
    if not file_field:
        return None

    try:
        url = file_field.url
    except ValueError:
        return None

    if request is None:
        return url
    return request.build_absolute_uri(url)


def _iso_or_none(value) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def _coach_progress_status_from_session(session_status: str) -> str:
    if session_status == SessionStatus.READY:
        return "completed"
    if session_status in {SessionStatus.COACH_FAILED, SessionStatus.FAILED}:
        return "failed"
    if session_status in {
        SessionStatus.QUEUED_ML,
        SessionStatus.PROCESSING_ML,
        SessionStatus.ML_READY,
        SessionStatus.PROCESSING_COACH,
    }:
        return "processing_coach"
    return "pending"


def _coach_progress_status_from_run(run_status: str) -> str:
    if run_status in {
        CoachOrchestrationRunStatus.QUEUED,
        CoachOrchestrationRunStatus.PROCESSING,
    }:
        return "processing_coach"
    if run_status == CoachOrchestrationRunStatus.COMPLETED:
        return "completed"
    return "failed"


def _agent_ui_status(status: str) -> str:
    if status == CoachAgentExecutionStatus.QUEUED:
        return "pending"
    if status == CoachAgentExecutionStatus.PROCESSING:
        return "processing"
    if status == CoachAgentExecutionStatus.COMPLETED:
        return "completed"
    return "failed"


def _extract_evidence_refs(payload: dict[str, Any]) -> list[str]:
    maybe_refs = payload.get("evidence_refs")
    if not isinstance(maybe_refs, list):
        return []
    return [item for item in maybe_refs if isinstance(item, str)]


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _serialize_live_ledger_entry(entry: dict[str, Any]) -> dict[str, Any]:
    payload = entry.get("payload")
    return {
        "sequence": _safe_int(entry.get("sequence", 0)),
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
        "payload": payload if isinstance(payload, dict) else {},
        "created_at": str(entry.get("created_at", "")),
    }


def _serialize_db_ledger_entry(entry) -> dict[str, Any]:
    payload = entry.payload if isinstance(entry.payload, dict) else {}
    return {
        "sequence": int(entry.sequence),
        "entry_kind": str(entry.entry_kind),
        "agent_kind": str(entry.agent_kind) if entry.agent_kind else None,
        "agent_name": str(entry.agent_name or ""),
        "window_start_ms": entry.window_start_ms,
        "window_end_ms": entry.window_end_ms,
        "content": str(entry.content),
        "payload": payload,
        "created_at": entry.created_at.isoformat(),
    }


def _select_progress_run(session: CoachingSession) -> CoachOrchestrationRun | None:
    active_run = (
        session.coach_runs.filter(
            status__in=[
                CoachOrchestrationRunStatus.QUEUED,
                CoachOrchestrationRunStatus.PROCESSING,
            ]
        )
        .order_by("-run_index", "-created_at")
        .first()
    )
    if active_run is not None:
        return active_run
    return session.coach_runs.order_by("-run_index", "-created_at").first()


def _can_use_live_ledger(run: CoachOrchestrationRun) -> bool:
    """Return whether this run can expose live Redis-backed ledger updates."""
    return run.status in {
        CoachOrchestrationRunStatus.QUEUED,
        CoachOrchestrationRunStatus.PROCESSING,
    }


def _read_live_ledger_entries(
    run: CoachOrchestrationRun,
) -> tuple[list[dict[str, Any]], int]:
    """Read live-ledger entries safely, returning empty results on transient errors."""
    if not _can_use_live_ledger(run):
        return [], 0
    try:
        entries = read_live_ledger_slice(run_id=str(run.id), sequence_gt=0)
        latest_sequence = get_live_ledger_latest_sequence(run_id=str(run.id))
    except Exception:
        return [], 0
    if entries:
        latest_sequence = max(
            latest_sequence,
            max(
                int(item.get("sequence", 0))
                for item in entries
                if isinstance(item, dict)
            ),
        )
    return entries, latest_sequence


class CreateSessionSerializer(serializers.Serializer):
    title = serializers.CharField(required=False, max_length=255)

    def validate_title(self, value: str) -> str:
        title = value.strip()
        if not title:
            raise serializers.ValidationError("Title cannot be blank.")
        return title


class UploadSessionVideoSerializer(serializers.Serializer):
    video_file = serializers.FileField(required=True)


class UploadSessionAssetsSerializer(serializers.Serializer):
    supplementary_pdf_1 = serializers.FileField(required=False)
    supplementary_pdf_2 = serializers.FileField(required=False)
    supplementary_pdf_3 = serializers.FileField(required=False)
    speaker_context = serializers.CharField(required=False, allow_blank=True)

    def validate(self, attrs):
        update_fields = {
            "supplementary_pdf_1",
            "supplementary_pdf_2",
            "supplementary_pdf_3",
            "speaker_context",
        }
        if not any(field in self.initial_data for field in update_fields):
            raise serializers.ValidationError(
                "At least one of supplementary_pdf_1, supplementary_pdf_2, "
                "supplementary_pdf_3, or speaker_context is required."
            )
        return attrs


class SessionListItemSerializer(serializers.ModelSerializer):
    class Meta:
        model = CoachingSession
        fields = (
            "id",
            "title",
            "status",
            "created_at",
            "updated_at",
        )
        read_only_fields = fields


class SessionDetailSerializer(serializers.ModelSerializer):
    video_file_url = serializers.SerializerMethodField()
    supplementary_pdf_1_url = serializers.SerializerMethodField()
    supplementary_pdf_2_url = serializers.SerializerMethodField()
    supplementary_pdf_3_url = serializers.SerializerMethodField()
    coach_progress = serializers.SerializerMethodField()

    class Meta:
        model = CoachingSession
        fields = (
            "id",
            "title",
            "status",
            "created_at",
            "updated_at",
            "video_file_url",
            "supplementary_pdf_1_url",
            "supplementary_pdf_2_url",
            "supplementary_pdf_3_url",
            "speaker_context",
            "coach_progress",
        )
        read_only_fields = fields

    def get_video_file_url(self, obj: CoachingSession) -> str | None:
        request = self.context.get("request")
        return _absolute_file_url(request, obj.video_file)

    def get_supplementary_pdf_1_url(self, obj: CoachingSession) -> str | None:
        request = self.context.get("request")
        return _absolute_file_url(request, obj.supplementary_pdf_1)

    def get_supplementary_pdf_2_url(self, obj: CoachingSession) -> str | None:
        request = self.context.get("request")
        return _absolute_file_url(request, obj.supplementary_pdf_2)

    def get_supplementary_pdf_3_url(self, obj: CoachingSession) -> str | None:
        request = self.context.get("request")
        return _absolute_file_url(request, obj.supplementary_pdf_3)

    def get_coach_progress(self, obj: CoachingSession) -> dict[str, Any]:
        run = _select_progress_run(obj)
        if run is None:
            return {
                "status": _coach_progress_status_from_session(obj.status),
                "active_run_id": None,
                "run_index": None,
                "latest_ledger_sequence": 0,
                "updated_at": obj.updated_at.isoformat(),
                "current_stage": "",
                "agent_progress": [],
                "stages": [],
                "ledger_entries": [],
            }

        executions = list(
            run.agent_executions.order_by("execution_index", "created_at")
        )
        ledger_entries = list(run.ledger_entries.order_by("sequence", "created_at"))
        live_entries, live_latest_sequence = _read_live_ledger_entries(run)
        use_live_entries = bool(live_entries)
        if use_live_entries:
            live_entries = sorted(
                live_entries,
                key=lambda item: _safe_int(item.get("sequence", 0)),
            )
            executions = sorted(
                executions,
                key=lambda item: (
                    item.window_start_ms is None,
                    item.window_start_ms or 0,
                    item.execution_index,
                    item.created_at,
                ),
            )

        if use_live_entries:
            serialized_ledger_entries = [
                _serialize_live_ledger_entry(entry)
                for entry in live_entries
                if isinstance(entry, dict)
            ]
        else:
            serialized_ledger_entries = [
                _serialize_db_ledger_entry(entry)
                for entry in ledger_entries
            ]

        notes_by_execution_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
        if use_live_entries:
            for entry in live_entries:
                payload = entry.get("payload")
                normalized_payload = payload if isinstance(payload, dict) else {}
                title = normalized_payload.get("title")
                note_title = (
                    title
                    if isinstance(title, str) and title
                    else str(entry.get("entry_kind", "Note"))
                )
                note_payload = {
                    "note_id": f"live-{entry.get('sequence', '')}",
                    "title": note_title,
                    "body": str(entry.get("content", "")),
                    "evidence_refs": _extract_evidence_refs(normalized_payload),
                    "default_collapsed": True,
                }
                execution_id = entry.get("agent_execution_id")
                if not isinstance(execution_id, str) or not execution_id:
                    continue
                notes_by_execution_id[execution_id].append(note_payload)
        else:
            for entry in ledger_entries:
                payload = entry.payload if isinstance(entry.payload, dict) else {}
                title = payload.get("title")
                note_title = (
                    title if isinstance(title, str) and title else entry.get_entry_kind_display()
                )
                note_payload = {
                    "note_id": str(entry.id),
                    "title": note_title,
                    "body": entry.content,
                    "evidence_refs": _extract_evidence_refs(payload),
                    "default_collapsed": True,
                }
                if entry.agent_execution_id is None:
                    continue
                notes_by_execution_id[str(entry.agent_execution_id)].append(note_payload)

        agent_progress = []
        stages = []
        current_stage = ""
        for execution in executions:
            ui_status = _agent_ui_status(execution.status)
            completed_at = execution.finished_at or execution.failed_at
            stage_notes = notes_by_execution_id.get(str(execution.id), [])
            label = execution.agent_name or execution.get_agent_kind_display()
            stage_key = f"agent-{execution.execution_index}"

            if ui_status == "processing":
                current_stage = stage_key

            agent_progress.append(
                {
                    "agent_execution_id": str(execution.id),
                    "execution_index": execution.execution_index,
                    "agent_kind": execution.agent_kind,
                    "agent_name": execution.agent_name,
                    "status": ui_status,
                    "window_start_ms": execution.window_start_ms,
                    "window_end_ms": execution.window_end_ms,
                    "input_seq_from": execution.input_seq_from,
                    "input_seq_to": execution.input_seq_to,
                    "output_seq_to": execution.output_seq_to,
                    "started_at": _iso_or_none(execution.started_at),
                    "completed_at": _iso_or_none(completed_at),
                    "last_heartbeat_at": _iso_or_none(execution.last_heartbeat_at),
                }
            )
            stages.append(
                {
                    "stage_key": stage_key,
                    "label": label,
                    "status": ui_status,
                    "notes": stage_notes,
                }
            )

        if not current_stage and stages:
            current_stage = stages[-1]["stage_key"]

        latest_ledger_sequence = (
            max(run.latest_ledger_sequence, live_latest_sequence)
            if use_live_entries
            else run.latest_ledger_sequence
        )

        return {
            "status": _coach_progress_status_from_run(run.status),
            "active_run_id": str(run.id),
            "run_index": run.run_index,
            "latest_ledger_sequence": latest_ledger_sequence,
            "updated_at": run.updated_at.isoformat(),
            "current_stage": current_stage,
            "agent_progress": agent_progress,
            "stages": stages,
            "ledger_entries": serialized_ledger_entries,
        }
