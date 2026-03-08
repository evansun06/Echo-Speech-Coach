from __future__ import annotations

import uuid

from django.core.exceptions import ValidationError as DjangoValidationError
from django.db import transaction
from django.http import Http404
from django.shortcuts import get_object_or_404
from drf_spectacular.utils import extend_schema, extend_schema_view
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.request import Request
from rest_framework.response import Response

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
    return Response({}, status=status.HTTP_200_OK)


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
    return Response({}, status=status.HTTP_200_OK)


@extend_schema(tags=["sessions"])
@api_view(["GET"])
def get_session_chat_context(request: Request, id: str) -> Response:
    """Return prepared chat context for a session."""
    return Response({}, status=status.HTTP_200_OK)


@extend_schema(tags=["sessions"])
@api_view(["GET"])
def get_session_video_stream(request: Request, id: str) -> Response:
    """Stream the session video resource for playback."""
    return Response({}, status=status.HTTP_200_OK)
