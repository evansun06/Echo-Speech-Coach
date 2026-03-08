from __future__ import annotations

import json
import re
import uuid
from typing import Any

from django.db import transaction
from django.http import Http404, StreamingHttpResponse
from django.shortcuts import get_object_or_404
from django.utils import timezone
from drf_spectacular.utils import extend_schema
from rest_framework import status
from rest_framework.decorators import api_view, renderer_classes
from rest_framework.renderers import BaseRenderer
from rest_framework.request import Request
from rest_framework.response import Response
from rest_framework.serializers import CharField, Serializer, ValidationError

from llm.orchestrator import stream_chat_response_tokens
from sessions.models import CoachOrchestrationRunStatus, CoachingSession, SessionStatus

from .models import ChatMessage, ChatMessageRole, ChatResponse, ChatResponseStatus

CHAT_SYSTEM_PROMPT = (
    "You are a concise speech coach assistant for one recorded presentation session. "
    "Treat the latest finalized coaching ledger as the source of truth for this speech. "
    "Ground every claim in that ledger and the provided session context. "
    "Give actionable, specific guidance, and clearly state when evidence is missing."
)
RECENT_CHAT_MESSAGE_LIMIT = 20


class ServerSentEventRenderer(BaseRenderer):
    media_type = "text/event-stream"
    format = "sse"
    charset = "utf-8"
    render_style = "text"

    def render(self, data, accepted_media_type=None, renderer_context=None):
        if data is None:
            return b""
        if isinstance(data, bytes):
            return data
        if isinstance(data, str):
            return data.encode(self.charset)
        return json.dumps(data).encode(self.charset)


class CreateChatMessageSerializer(Serializer):
    content = CharField(required=True, allow_blank=False, trim_whitespace=True)

    def validate_content(self, value: str) -> str:
        content = value.strip()
        if not content:
            raise ValidationError("content is required.")
        return content


def _get_owned_session(*, user, session_id: str) -> CoachingSession:
    try:
        parsed_id = uuid.UUID(session_id)
    except ValueError as exc:
        raise Http404 from exc
    return get_object_or_404(CoachingSession.objects.filter(user=user), id=parsed_id)


def _get_owned_response(*, session: CoachingSession, response_id: str) -> ChatResponse:
    try:
        parsed_id = uuid.UUID(response_id)
    except ValueError as exc:
        raise Http404 from exc
    return get_object_or_404(
        ChatResponse.objects.select_related("session", "assistant_message"),
        session=session,
        response_id=parsed_id,
    )


def _status_conflict_response(*, session: CoachingSession, operation: str) -> Response:
    return Response(
        {
            "detail": (
                f"Cannot {operation} while session status is '{session.status}'. "
                f"Expected '{SessionStatus.READY}'."
            )
        },
        status=status.HTTP_409_CONFLICT,
    )


def _serialize_message(message: ChatMessage) -> dict[str, Any]:
    return {
        "id": str(message.id),
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at.isoformat(),
    }


def _sse_event(event_name: str, payload: dict[str, Any] | str) -> str:
    if isinstance(payload, str):
        data = payload
    else:
        data = json.dumps(payload)
    return f"event: {event_name}\ndata: {data}\n\n"


def _iter_answer_tokens(answer_text: str):
    for token in re.findall(r"\S+\s*", answer_text):
        yield token
    if answer_text and not re.search(r"\S", answer_text):
        yield answer_text


def _latest_finalized_ledger_block(session: CoachingSession) -> str:
    latest_completed_run = (
        session.coach_runs.filter(status=CoachOrchestrationRunStatus.COMPLETED)
        .order_by("-run_index", "-created_at")
        .first()
    )
    if latest_completed_run is None:
        return "No finalized ledger entries were found for this session."

    entries = latest_completed_run.ledger_entries.order_by("sequence", "created_at")
    lines: list[str] = [
        f"Latest finalized run: {latest_completed_run.id} (run_index={latest_completed_run.run_index})"
    ]
    for entry in entries:
        line = f"[{entry.sequence}] {entry.entry_kind}: {entry.content}"
        if isinstance(entry.payload, dict) and entry.payload:
            line += f" | payload={json.dumps(entry.payload, sort_keys=True)}"
        lines.append(line)
    if len(lines) == 1:
        lines.append("No ledger rows were recorded in that finalized run.")
    return "\n".join(lines)


def _recent_chat_history_block(session: CoachingSession) -> str:
    recent_messages = list(
        ChatMessage.objects.filter(session=session)
        .order_by("-created_at")[:RECENT_CHAT_MESSAGE_LIMIT]
    )
    if not recent_messages:
        return "No prior chat messages."
    recent_messages.reverse()
    return "\n".join(
        f"{message.role.upper()}: {message.content}" for message in recent_messages
    )


def _build_user_prompt(*, session: CoachingSession, user_message: ChatMessage) -> str:
    session_context = {
        "session_id": str(session.id),
        "title": session.title,
        "speaker_context": session.speaker_context,
    }
    return (
        "SESSION_CONTEXT_JSON:\n"
        f"{json.dumps(session_context, sort_keys=True)}\n\n"
        "LATEST_FINALIZED_LEDGER:\n"
        f"{_latest_finalized_ledger_block(session)}\n\n"
        "RECENT_CHAT_HISTORY:\n"
        f"{_recent_chat_history_block(session)}\n\n"
        "CURRENT_USER_MESSAGE:\n"
        f"{user_message.content}"
    )


def _mark_response_streaming(response: ChatResponse) -> ChatResponse:
    with transaction.atomic():
        locked = ChatResponse.objects.select_for_update().get(id=response.id)
        if locked.status != ChatResponseStatus.QUEUED:
            return locked
        now = timezone.now()
        locked.status = ChatResponseStatus.STREAMING
        locked.started_at = now
        locked.failed_at = None
        locked.error_message = ""
        locked.save(
            update_fields=[
                "status",
                "started_at",
                "failed_at",
                "error_message",
                "updated_at",
            ]
        )
        return locked


def _mark_response_failed(response: ChatResponse, *, message: str) -> ChatResponse:
    with transaction.atomic():
        locked = ChatResponse.objects.select_for_update().get(id=response.id)
        now = timezone.now()
        locked.status = ChatResponseStatus.FAILED
        locked.error_message = message
        locked.failed_at = now
        if locked.started_at is None:
            locked.started_at = now
        locked.save(
            update_fields=[
                "status",
                "error_message",
                "failed_at",
                "started_at",
                "updated_at",
            ]
        )
        return locked


def _mark_response_completed(response: ChatResponse, *, answer_text: str) -> tuple[ChatResponse, ChatMessage]:
    with transaction.atomic():
        locked = ChatResponse.objects.select_for_update().get(id=response.id)
        now = timezone.now()
        assistant_message = locked.assistant_message
        if assistant_message is None:
            assistant_message = ChatMessage.objects.create(
                session=locked.session,
                role=ChatMessageRole.ASSISTANT,
                content=answer_text,
                response=locked,
            )
        else:
            assistant_message.content = answer_text
            assistant_message.save(update_fields=["content"])
        if locked.started_at is None:
            locked.started_at = now
        locked.status = ChatResponseStatus.COMPLETED
        locked.assistant_message = assistant_message
        locked.answer_text = answer_text
        locked.error_message = ""
        locked.completed_at = now
        locked.failed_at = None
        locked.save(
            update_fields=[
                "status",
                "assistant_message",
                "answer_text",
                "error_message",
                "started_at",
                "completed_at",
                "failed_at",
                "updated_at",
            ]
        )
        return locked, assistant_message


def _stream_tokens(*, response: ChatResponse, answer_text: str, message_id: str | None):
    yield _sse_event("start", {"response_id": str(response.response_id)})
    yield _sse_event("heartbeat", {})
    for token in _iter_answer_tokens(answer_text):
        yield _sse_event("token", {"phase": "answer", "token": token})
    yield _sse_event(
        "complete",
        {
            "response_id": str(response.response_id),
            "message_id": message_id,
        },
    )


def _stream_error(*, response: ChatResponse, message: str):
    yield _sse_event("start", {"response_id": str(response.response_id)})
    yield _sse_event("error", message)


@extend_schema(tags=["chat"])
@api_view(["POST"])
def create_chat_message(request: Request, id: str) -> Response:
    """Create a new chat message for the specified session."""
    session = _get_owned_session(user=request.user, session_id=id)
    if session.status != SessionStatus.READY:
        return _status_conflict_response(session=session, operation="send chat message")

    serializer = CreateChatMessageSerializer(data=request.data)
    serializer.is_valid(raise_exception=True)

    with transaction.atomic():
        user_message = ChatMessage.objects.create(
            session=session,
            role=ChatMessageRole.USER,
            content=serializer.validated_data["content"],
        )
        response = ChatResponse.objects.create(
            session=session,
            user_message=user_message,
            status=ChatResponseStatus.QUEUED,
        )
        user_message.response = response
        user_message.save(update_fields=["response"])

    return Response(
        {"response_id": str(response.response_id)},
        status=status.HTTP_201_CREATED,
    )


@extend_schema(tags=["chat"])
@api_view(["GET"])
@renderer_classes([ServerSentEventRenderer])
def stream_chat_response(request: Request, id: str, response_id: str) -> Response:
    """Stream chat response events for a previously created response."""
    session = _get_owned_session(user=request.user, session_id=id)
    if session.status != SessionStatus.READY:
        return _status_conflict_response(session=session, operation="stream chat response")

    response = _get_owned_response(session=session, response_id=response_id)

    if response.status == ChatResponseStatus.STREAMING:
        return Response(
            {
                "detail": (
                    "This response is already streaming. Wait for completion and retry."
                )
            },
            status=status.HTTP_409_CONFLICT,
        )

    if response.status == ChatResponseStatus.COMPLETED:
        assistant_message_id = (
            str(response.assistant_message_id) if response.assistant_message_id else None
        )
        stream = _stream_tokens(
            response=response,
            answer_text=response.answer_text,
            message_id=assistant_message_id,
        )
        return StreamingHttpResponse(
            streaming_content=stream,
            content_type="text/event-stream",
        )

    if response.status == ChatResponseStatus.FAILED:
        stream = _stream_error(
            response=response,
            message=response.error_message or "Chat response generation failed.",
        )
        return StreamingHttpResponse(
            streaming_content=stream,
            content_type="text/event-stream",
        )

    locked_response = _mark_response_streaming(response)
    if locked_response.status != ChatResponseStatus.STREAMING:
        if locked_response.status == ChatResponseStatus.COMPLETED:
            assistant_message_id = (
                str(locked_response.assistant_message_id)
                if locked_response.assistant_message_id
                else None
            )
            stream = _stream_tokens(
                response=locked_response,
                answer_text=locked_response.answer_text,
                message_id=assistant_message_id,
            )
            return StreamingHttpResponse(
                streaming_content=stream,
                content_type="text/event-stream",
            )
        if locked_response.status == ChatResponseStatus.FAILED:
            stream = _stream_error(
                response=locked_response,
                message=locked_response.error_message
                or "Chat response generation failed.",
            )
            return StreamingHttpResponse(
                streaming_content=stream,
                content_type="text/event-stream",
            )
        return Response(
            {
                "detail": (
                    f"Cannot stream response while status is '{locked_response.status}'."
                )
            },
            status=status.HTTP_409_CONFLICT,
        )

    def generate():
        stream_started = False
        try:
            user_message = locked_response.user_message
            if user_message is None:
                raise RuntimeError("Missing user message for chat response.")
            prompt = _build_user_prompt(session=session, user_message=user_message)
            answer_chunks: list[str] = []

            yield _sse_event(
                "start",
                {"response_id": str(locked_response.response_id)},
            )
            stream_started = True
            yield _sse_event("heartbeat", {})

            for token in stream_chat_response_tokens(
                system_prompt=CHAT_SYSTEM_PROMPT,
                user_prompt=prompt,
                metadata={
                    "session_id": str(session.id),
                    "response_id": str(locked_response.response_id),
                    "chat_mode": "session_chat",
                },
            ):
                answer_chunks.append(token)
                yield _sse_event(
                    "token",
                    {"phase": "answer", "token": token},
                )

            answer_text = "".join(answer_chunks).strip()
            if not answer_text:
                answer_text = "I do not have enough evidence to coach this yet."
            completed_response, assistant_message = _mark_response_completed(
                locked_response,
                answer_text=answer_text,
            )
            yield _sse_event(
                "complete",
                {
                    "response_id": str(completed_response.response_id),
                    "message_id": str(assistant_message.id),
                },
            )
        except Exception as exc:
            failed_response = _mark_response_failed(
                locked_response,
                message=str(exc) or "Chat response generation failed.",
            )
            if not stream_started:
                yield _sse_event("start", {"response_id": str(failed_response.response_id)})
            yield _sse_event("error", failed_response.error_message)

    return StreamingHttpResponse(
        streaming_content=generate(),
        content_type="text/event-stream",
    )


@extend_schema(tags=["chat"])
@api_view(["GET"])
def get_chat_history(request: Request, id: str) -> Response:
    """Return prior chat messages for the specified session."""
    session = _get_owned_session(user=request.user, session_id=id)
    messages = ChatMessage.objects.filter(session=session).order_by("created_at")
    payload = [_serialize_message(message) for message in messages]
    return Response(payload, status=status.HTTP_200_OK)
