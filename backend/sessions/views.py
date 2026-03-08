from __future__ import annotations

import mimetypes
import uuid

from django.core.exceptions import ValidationError as DjangoValidationError
from django.db import transaction
from django.http import FileResponse, Http404
from django.shortcuts import get_object_or_404
from drf_spectacular.utils import extend_schema, extend_schema_view
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.request import Request
from rest_framework.response import Response

from ml.enqueue import enqueue_session_ml_workflow_job

from .models import CoachingSession, SessionStatus
from .serializers import (
    CreateSessionSerializer,
    SessionDetailSerializer,
    SessionListItemSerializer,
    UploadSessionAssetsSerializer,
    UploadSessionVideoSerializer,
)


def _get_owned_session(
    *,
    user,
    session_id: str,
    for_update: bool = False,
) -> CoachingSession:
    try:
        parsed_id = uuid.UUID(session_id)
    except ValueError as exc:
        raise Http404 from exc

    queryset = CoachingSession.objects.filter(user=user)
    if for_update:
        queryset = queryset.select_for_update()
    return get_object_or_404(queryset, id=parsed_id)


def _validation_error_response(exc: DjangoValidationError) -> Response:
    if hasattr(exc, "message_dict"):
        data = exc.message_dict
    else:
        data = {"detail": exc.messages}
    return Response(data, status=status.HTTP_400_BAD_REQUEST)


def _status_conflict_response(
    *,
    session: CoachingSession,
    expected_status: str,
    operation: str,
) -> Response:
    return Response(
        {
            "detail": (
                f"Cannot {operation} while session status is '{session.status}'. "
                f"Expected '{expected_status}'."
            )
        },
        status=status.HTTP_409_CONFLICT,
    )


@extend_schema(tags=["sessions"])
def create_session(request: Request) -> Response:
    """Create a new coaching session."""
    serializer = CreateSessionSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)

    create_kwargs = {}
    if "title" in serializer.validated_data:
        create_kwargs["title"] = serializer.validated_data["title"]

    session = CoachingSession.objects.create(
        user=request.user,
        **create_kwargs,
    )
    output = SessionDetailSerializer(session, context={"request": request})
    return Response(output.data, status=status.HTTP_201_CREATED)


@extend_schema(tags=["sessions"])
def list_sessions(request: Request) -> Response:
    """List coaching sessions with status metadata."""
    sessions = CoachingSession.objects.filter(user=request.user).order_by("-created_at")
    serializer = SessionListItemSerializer(sessions, many=True)
    return Response(serializer.data, status=status.HTTP_200_OK)


@extend_schema_view(
    get=extend_schema(tags=["sessions"]),
    post=extend_schema(tags=["sessions"]),
)
@api_view(["GET", "POST"])
def sessions_collection(request: Request) -> Response:
    """Dispatch GET/POST requests for the sessions collection route."""
    if request.method == "POST":
        return create_session(request)
    return list_sessions(request)


@extend_schema(tags=["sessions"])
@api_view(["POST"])
def upload_session_video(request: Request, id: str) -> Response:
    """Attach or upload a video file for the given session."""
    serializer = UploadSessionVideoSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)

    with transaction.atomic():
        session = _get_owned_session(
            user=request.user,
            session_id=id,
            for_update=True,
        )
        if session.status != SessionStatus.DRAFT:
            return _status_conflict_response(
                session=session,
                expected_status=SessionStatus.DRAFT,
                operation="upload video",
            )

        session.video_file = serializer.validated_data["video_file"]
        session.status = SessionStatus.MEDIA_ATTACHED

        try:
            session.full_clean()
        except DjangoValidationError as exc:
            return _validation_error_response(exc)

        session.save()

    output = SessionDetailSerializer(session, context={"request": request})
    return Response(output.data, status=status.HTTP_200_OK)


@extend_schema(tags=["sessions"])
@api_view(["POST"])
def upload_session_assets(request: Request, id: str) -> Response:
    """Attach optional assets such as script or slides to a session."""
    serializer = UploadSessionAssetsSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)

    with transaction.atomic():
        session = _get_owned_session(
            user=request.user,
            session_id=id,
            for_update=True,
        )
        if session.status != SessionStatus.MEDIA_ATTACHED:
            return _status_conflict_response(
                session=session,
                expected_status=SessionStatus.MEDIA_ATTACHED,
                operation="upload assets",
            )

        for field_name in (
            "supplementary_pdf_1",
            "supplementary_pdf_2",
            "supplementary_pdf_3",
            "speaker_context",
        ):
            if field_name in serializer.validated_data:
                setattr(session, field_name, serializer.validated_data[field_name])

        try:
            session.full_clean()
        except DjangoValidationError as exc:
            return _validation_error_response(exc)

        session.save()

    output = SessionDetailSerializer(session, context={"request": request})
    return Response(output.data, status=status.HTTP_200_OK)


@extend_schema(tags=["sessions"])
@api_view(["POST"])
def start_session_analysis(request: Request, id: str) -> Response:
    """Start asynchronous analysis for the specified session."""
    allowed_statuses = {SessionStatus.MEDIA_ATTACHED, SessionStatus.COACH_FAILED}

    with transaction.atomic():
        session = _get_owned_session(
            user=request.user,
            session_id=id,
            for_update=True,
        )
        if session.status not in allowed_statuses:
            return Response(
                {
                    "detail": (
                        f"Cannot start analysis while session status is '{session.status}'. "
                        f"Expected one of: {', '.join(sorted(allowed_statuses))}."
                    )
                },
                status=status.HTTP_409_CONFLICT,
            )

        session.status = SessionStatus.QUEUED_ML
        session.coach_task_id = None
        session.ml_task_id = None
        session.save(update_fields=["status", "coach_task_id", "ml_task_id", "updated_at"])

        async_result = enqueue_session_ml_workflow_job(session_id=str(session.id))
        session.ml_task_id = async_result.id
        session.save(update_fields=["ml_task_id", "updated_at"])

    return Response(
        {
            "session_id": str(session.id),
            "status": session.status,
            "ml_task_id": session.ml_task_id,
        },
        status=status.HTTP_202_ACCEPTED,
    )


@extend_schema(tags=["sessions"])
@api_view(["GET"])
def get_session(request: Request, id: str) -> Response:
    """Fetch a single session, including coach progress details."""
    session = _get_owned_session(user=request.user, session_id=id)
    serializer = SessionDetailSerializer(session, context={"request": request})
    return Response(serializer.data, status=status.HTTP_200_OK)


@extend_schema(tags=["sessions"])
@api_view(["GET"])
def get_session_timeline(request: Request, id: str) -> Response:
    """Return timeline events generated for a session."""
    session = _get_owned_session(user=request.user, session_id=id)
    events = session.events.order_by("start_time", "event_id")

    timeline_items: list[dict[str, object]] = []
    for event in events:
        event_json = event.event_json if isinstance(event.event_json, dict) else {}
        event_type = str(event_json.get("event_type", "")).strip() or "event"
        source = event_json.get("source")
        source_value = source if source in {"audio", "video"} else "audio"

        confidence_raw = event_json.get("confidence")
        try:
            confidence = float(confidence_raw)
        except (TypeError, ValueError):
            confidence = 0.65
        confidence = max(0.0, min(1.0, confidence))

        severity_raw = event_json.get("severity")
        severity = severity_raw if severity_raw in {"low", "medium", "high"} else "medium"

        summary = (
            str(event_json.get("summary", "")).strip()
            or str(event_json.get("text_span", "")).strip()
            or event_type.replace("_", " ").title()
        )
        timeline_items.append(
            {
                "id": f"{session.id}-{event.event_id}",
                "event_type": event_type,
                "source": source_value,
                "start_ms": int(round(event.start_time * 1000)),
                "end_ms": int(round(event.end_time * 1000)),
                "severity": severity,
                "confidence": confidence,
                "summary": summary,
                "metadata": event_json,
            }
        )

    return Response(timeline_items, status=status.HTTP_200_OK)


@extend_schema(tags=["sessions"])
@api_view(["GET"])
def get_session_chat_context(request: Request, id: str) -> Response:
    """Return prepared chat context for a session."""
    return Response({}, status=status.HTTP_200_OK)


@extend_schema(tags=["sessions"])
@api_view(["GET"])
def get_session_video_stream(request: Request, id: str) -> Response:
    """Stream the session video resource for playback."""
    session = _get_owned_session(user=request.user, session_id=id)

    if not session.video_file:
        return Response(
            {"detail": "Video not available"},
            status=status.HTTP_404_NOT_FOUND,
        )

    try:
        file_path = session.video_file.path
    except (ValueError, NotImplementedError):
        return Response(
            {"detail": "Video not available"},
            status=status.HTTP_404_NOT_FOUND,
        )

    if not session.video_file.storage.exists(session.video_file.name):
        return Response(
            {"detail": "Video not available"},
            status=status.HTTP_404_NOT_FOUND,
        )

    content_type, _ = mimetypes.guess_type(file_path)
    content_type = content_type or "video/mp4"

    response = FileResponse(
        open(file_path, "rb"),
        content_type=content_type,
        as_attachment=False,
    )
    response["Accept-Ranges"] = "bytes"
    return response
