from django.contrib.auth import get_user_model
from django.urls import reverse
from django.test import TestCase
from rest_framework import status
from rest_framework.test import APIClient
from unittest.mock import patch

from sessions.models import (
    CoachAgentKind,
    CoachLedgerEntry,
    CoachOrchestrationRun,
    CoachOrchestrationRunStatus,
    CoachingSession,
    LedgerEntryKind,
    SessionStatus,
)

from .models import ChatMessage, ChatMessageRole, ChatResponse, ChatResponseStatus


class ChatPersistenceModelTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(
            username="chat-owner@example.com",
            email="chat-owner@example.com",
            password="password123",
        )
        self.session = CoachingSession.objects.create(
            user=self.user,
            status=SessionStatus.READY,
            video_file="sessions/videos/2026/03/08/demo.mp4",
        )

    def test_chat_message_defaults(self):
        message = ChatMessage.objects.create(
            session=self.session,
            role=ChatMessageRole.USER,
            content="What should I improve?",
        )

        self.assertEqual(message.role, ChatMessageRole.USER)
        self.assertEqual(message.content, "What should I improve?")
        self.assertEqual(message.metadata, {})
        self.assertIsNone(message.response)

    def test_chat_response_can_link_user_and_assistant_messages(self):
        user_message = ChatMessage.objects.create(
            session=self.session,
            role=ChatMessageRole.USER,
            content="How was my pacing?",
        )
        response = ChatResponse.objects.create(
            session=self.session,
            user_message=user_message,
            status=ChatResponseStatus.STREAMING,
        )

        assistant_message = ChatMessage.objects.create(
            session=self.session,
            role=ChatMessageRole.ASSISTANT,
            content="Your pace improved after the first minute.",
            response=response,
        )
        response.assistant_message = assistant_message
        response.status = ChatResponseStatus.COMPLETED
        response.answer_text = assistant_message.content
        response.save()

        self.assertEqual(response.user_message_id, user_message.id)
        self.assertEqual(response.assistant_message_id, assistant_message.id)
        self.assertEqual(response.answer_text, assistant_message.content)
        self.assertEqual(response.status, ChatResponseStatus.COMPLETED)


class ChatApiEndpointTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(
            username="chat-api-owner@example.com",
            email="chat-api-owner@example.com",
            password="password123",
        )
        self.other_user = User.objects.create_user(
            username="chat-api-other@example.com",
            email="chat-api-other@example.com",
            password="password123",
        )
        self.ready_session = CoachingSession.objects.create(
            user=self.user,
            status=SessionStatus.READY,
            video_file="sessions/videos/2026/03/08/demo.mp4",
            title="Demo Session",
            speaker_context="Pitch to investors.",
        )
        self.non_ready_session = CoachingSession.objects.create(
            user=self.user,
            status=SessionStatus.MEDIA_ATTACHED,
            video_file="sessions/videos/2026/03/08/non-ready.mp4",
        )
        self.other_user_session = CoachingSession.objects.create(
            user=self.other_user,
            status=SessionStatus.READY,
            video_file="sessions/videos/2026/03/08/other.mp4",
        )

        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        self.unauthenticated_client = APIClient()

    def _messages_url(self, session: CoachingSession) -> str:
        return reverse("api:chat-messages", kwargs={"id": str(session.id)})

    def _history_url(self, session: CoachingSession) -> str:
        return reverse("api:chat-history", kwargs={"id": str(session.id)})

    def _stream_url(self, session: CoachingSession, response_id: str) -> str:
        return reverse(
            "api:chat-stream",
            kwargs={"id": str(session.id), "response_id": response_id},
        )

    def _stream_body(self, response) -> str:
        return b"".join(response.streaming_content).decode("utf-8")

    def test_chat_endpoints_require_authentication(self):
        create_response = self.unauthenticated_client.post(
            self._messages_url(self.ready_session),
            {"content": "Hello"},
            format="json",
        )
        history_response = self.unauthenticated_client.get(
            self._history_url(self.ready_session)
        )
        queued = ChatResponse.objects.create(session=self.ready_session)
        stream_response = self.unauthenticated_client.get(
            self._stream_url(self.ready_session, str(queued.response_id))
        )

        self.assertIn(
            create_response.status_code,
            {status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN},
        )
        self.assertIn(
            history_response.status_code,
            {status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN},
        )
        self.assertIn(
            stream_response.status_code,
            {status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN},
        )

    def test_create_chat_message_creates_user_message_and_queued_response(self):
        response = self.client.post(
            self._messages_url(self.ready_session),
            {"content": "  How can I improve my closing?  "},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        created_response = ChatResponse.objects.get(response_id=response.data["response_id"])
        self.assertEqual(created_response.status, ChatResponseStatus.QUEUED)
        self.assertIsNotNone(created_response.user_message)
        self.assertEqual(created_response.user_message.role, ChatMessageRole.USER)
        self.assertEqual(
            created_response.user_message.content,
            "How can I improve my closing?",
        )
        self.assertEqual(created_response.user_message.response_id, created_response.id)

    def test_create_chat_message_requires_non_empty_content(self):
        response = self.client.post(
            self._messages_url(self.ready_session),
            {"content": "   "},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("content", response.data)

    def test_create_chat_message_requires_ready_session(self):
        response = self.client.post(
            self._messages_url(self.non_ready_session),
            {"content": "Can I chat now?"},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_409_CONFLICT)
        self.assertIn("detail", response.data)

    def test_create_chat_message_enforces_session_ownership(self):
        response = self.client.post(
            self._messages_url(self.other_user_session),
            {"content": "Hello"},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_get_chat_history_returns_ordered_messages(self):
        first = ChatMessage.objects.create(
            session=self.ready_session,
            role=ChatMessageRole.USER,
            content="First message",
        )
        second = ChatMessage.objects.create(
            session=self.ready_session,
            role=ChatMessageRole.ASSISTANT,
            content="Second message",
        )

        response = self.client.get(self._history_url(self.ready_session))

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(len(response.data), 2)
        self.assertEqual(response.data[0]["id"], str(first.id))
        self.assertEqual(response.data[1]["id"], str(second.id))
        self.assertEqual(
            set(response.data[0].keys()),
            {"id", "role", "content", "created_at"},
        )

    def test_get_chat_history_enforces_session_ownership(self):
        response = self.client.get(self._history_url(self.other_user_session))
        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    @patch("chatbot.views.stream_chat_response_tokens")
    def test_stream_chat_response_generates_and_completes_from_queued(self, stream_tokens_mock):
        completed_run = CoachOrchestrationRun.objects.create(
            session=self.ready_session,
            run_index=1,
            status=CoachOrchestrationRunStatus.COMPLETED,
        )
        CoachLedgerEntry.objects.create(
            run=completed_run,
            sequence=1,
            entry_kind=LedgerEntryKind.FLAGSHIP_FINAL,
            agent_kind=CoachAgentKind.FLAGSHIP_FINAL,
            agent_name="flagship-final",
            content="Final summary: improve transitions and pacing.",
            payload={"title": "Final Summary"},
        )
        user_message = ChatMessage.objects.create(
            session=self.ready_session,
            role=ChatMessageRole.USER,
            content="What are the top two improvements?",
        )
        queued_response = ChatResponse.objects.create(
            session=self.ready_session,
            user_message=user_message,
            status=ChatResponseStatus.QUEUED,
        )
        user_message.response = queued_response
        user_message.save(update_fields=["response"])

        stream_tokens_mock.return_value = [
            "Focus on ",
            "slower transitions ",
            "and cleaner openings.",
        ]

        response = self.client.get(
            self._stream_url(self.ready_session, str(queued_response.response_id))
        )
        body = self._stream_body(response)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response["Content-Type"], "text/event-stream")
        self.assertIn("event: start", body)
        self.assertIn("event: token", body)
        self.assertIn("event: complete", body)
        stream_tokens_mock.assert_called_once()
        prompt = stream_tokens_mock.call_args.kwargs["user_prompt"]
        self.assertIn("LATEST_FINALIZED_LEDGER", prompt)
        self.assertIn("Final summary: improve transitions and pacing.", prompt)

        queued_response.refresh_from_db()
        self.assertEqual(queued_response.status, ChatResponseStatus.COMPLETED)
        self.assertTrue(queued_response.answer_text)
        self.assertIsNotNone(queued_response.assistant_message_id)

    @patch("chatbot.views.stream_chat_response_tokens")
    def test_stream_chat_response_replays_completed_response(self, stream_tokens_mock):
        assistant_message = ChatMessage.objects.create(
            session=self.ready_session,
            role=ChatMessageRole.ASSISTANT,
            content="Replay answer.",
        )
        completed_response = ChatResponse.objects.create(
            session=self.ready_session,
            assistant_message=assistant_message,
            status=ChatResponseStatus.COMPLETED,
            answer_text="Replay answer.",
        )

        response = self.client.get(
            self._stream_url(self.ready_session, str(completed_response.response_id))
        )
        body = self._stream_body(response)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("event: token", body)
        self.assertIn("Replay", body)
        self.assertIn("event: complete", body)
        stream_tokens_mock.assert_not_called()

    def test_stream_chat_response_accepts_text_event_stream_header(self):
        assistant_message = ChatMessage.objects.create(
            session=self.ready_session,
            role=ChatMessageRole.ASSISTANT,
            content="Replay answer.",
        )
        completed_response = ChatResponse.objects.create(
            session=self.ready_session,
            assistant_message=assistant_message,
            status=ChatResponseStatus.COMPLETED,
            answer_text="Replay answer.",
        )

        response = self.client.get(
            self._stream_url(self.ready_session, str(completed_response.response_id)),
            HTTP_ACCEPT="text/event-stream",
        )
        body = self._stream_body(response)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response["Content-Type"], "text/event-stream")
        self.assertIn("event: token", body)

    def test_stream_chat_response_emits_failed_error_event(self):
        failed_response = ChatResponse.objects.create(
            session=self.ready_session,
            status=ChatResponseStatus.FAILED,
            error_message="Model request failed.",
        )

        response = self.client.get(
            self._stream_url(self.ready_session, str(failed_response.response_id))
        )
        body = self._stream_body(response)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("event: error", body)
        self.assertIn("Model request failed.", body)

    def test_stream_chat_response_rejects_streaming_in_progress(self):
        streaming_response = ChatResponse.objects.create(
            session=self.ready_session,
            status=ChatResponseStatus.STREAMING,
        )

        response = self.client.get(
            self._stream_url(self.ready_session, str(streaming_response.response_id))
        )

        self.assertEqual(response.status_code, status.HTTP_409_CONFLICT)
        self.assertIn("detail", response.data)

    @patch("chatbot.views.stream_chat_response_tokens")
    def test_stream_chat_response_marks_failed_when_generation_raises(self, stream_tokens_mock):
        user_message = ChatMessage.objects.create(
            session=self.ready_session,
            role=ChatMessageRole.USER,
            content="Will this fail?",
        )
        queued_response = ChatResponse.objects.create(
            session=self.ready_session,
            user_message=user_message,
            status=ChatResponseStatus.QUEUED,
        )
        user_message.response = queued_response
        user_message.save(update_fields=["response"])
        stream_tokens_mock.side_effect = RuntimeError("Gemini error")

        response = self.client.get(
            self._stream_url(self.ready_session, str(queued_response.response_id))
        )
        body = self._stream_body(response)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn("event: error", body)
        self.assertIn("Gemini error", body)

        queued_response.refresh_from_db()
        self.assertEqual(queued_response.status, ChatResponseStatus.FAILED)
        self.assertEqual(queued_response.error_message, "Gemini error")
        self.assertIsNotNone(queued_response.failed_at)

    def test_stream_chat_response_requires_ready_session(self):
        queued_response = ChatResponse.objects.create(
            session=self.non_ready_session,
            status=ChatResponseStatus.QUEUED,
        )

        response = self.client.get(
            self._stream_url(self.non_ready_session, str(queued_response.response_id))
        )

        self.assertEqual(response.status_code, status.HTTP_409_CONFLICT)
        self.assertIn("detail", response.data)

    def test_stream_chat_response_enforces_session_ownership(self):
        queued_response = ChatResponse.objects.create(
            session=self.other_user_session,
            status=ChatResponseStatus.QUEUED,
        )

        response = self.client.get(
            self._stream_url(self.other_user_session, str(queued_response.response_id))
        )

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)
