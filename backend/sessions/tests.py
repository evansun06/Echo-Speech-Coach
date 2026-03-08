import importlib.util
import sys
import tempfile
import types
import unittest
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.core.exceptions import ValidationError
from django.core.management import CommandError, call_command
from django.db import IntegrityError
from django.test import SimpleTestCase, TestCase, override_settings
from django.urls import reverse
from django.utils import timezone
from rest_framework import status
from rest_framework.test import APIClient

from llm.coach_graph import build_reasoning_graph, run_reasoning_graph
from llm.ledger import (
    LedgerValidationError,
    RunStateError,
    append_ledger_entry,
    create_agent_execution,
    create_orchestration_run,
    mark_agent_completed,
    mark_agent_failed,
    mark_agent_processing,
    mark_run_completed,
    mark_run_failed,
    mark_run_processing,
    read_ledger_slice,
    touch_agent_heartbeat,
)
from llm.provider import ModelConfigurationError, ReasoningModels, build_reasoning_models
from llm.schemas import ReasoningInput
from ml.enqueue import enqueue_random_sleep_demo_job, enqueue_random_sleep_demo_jobs
from sessions.models import (
    CoachAgentExecution,
    CoachAgentExecutionStatus,
    CoachAgentKind,
    CoachLedgerEntry,
    CoachOrchestrationRun,
    CoachOrchestrationRunStatus,
    LedgerEntryKind,
    MAX_SUPPLEMENTARY_PDF_FILE_SIZE_BYTES,
    MAX_VIDEO_FILE_SIZE_BYTES,
    CoachingSession,
    MaxFileSizeValidator,
    SessionStatus,
)

HAS_LANGGRAPH_STACK = (
    importlib.util.find_spec("langgraph") is not None
    and importlib.util.find_spec("langchain_core") is not None
)


class DemoEnqueueWrapperTests(SimpleTestCase):
    @patch("ml.enqueue.random_sleep_demo_task.apply_async")
    def test_enqueue_single_job_dispatches_expected_kwargs(self, apply_async_mock):
        apply_async_mock.return_value = SimpleNamespace(id="task-1")

        result = enqueue_random_sleep_demo_job(
            min_seconds=2,
            max_seconds=4,
            label="demo-1",
        )

        apply_async_mock.assert_called_once_with(
            kwargs={"min_seconds": 2, "max_seconds": 4, "label": "demo-1"}
        )
        self.assertEqual(result.id, "task-1")

    @patch("ml.enqueue.enqueue_random_sleep_demo_job")
    def test_enqueue_many_jobs_returns_task_ids(self, enqueue_mock):
        enqueue_mock.side_effect = [
            SimpleNamespace(id="task-1"),
            SimpleNamespace(id="task-2"),
            SimpleNamespace(id="task-3"),
        ]

        task_ids = enqueue_random_sleep_demo_jobs(
            count=3,
            min_seconds=1,
            max_seconds=2,
            label_prefix="load",
        )

        self.assertEqual(task_ids, ["task-1", "task-2", "task-3"])
        self.assertEqual(enqueue_mock.call_count, 3)


class EnqueueDemoJobsCommandTests(SimpleTestCase):
    @patch("sessions.management.commands.enqueue_demo_jobs.enqueue_random_sleep_demo_jobs")
    def test_command_enqueues_expected_count(self, enqueue_many_mock):
        enqueue_many_mock.return_value = ["task-1", "task-2"]
        stdout = StringIO()

        call_command(
            "enqueue_demo_jobs",
            "--count",
            "2",
            "--min-seconds",
            "1",
            "--max-seconds",
            "3",
            "--prefix",
            "batch",
            stdout=stdout,
        )

        enqueue_many_mock.assert_called_once_with(
            count=2,
            min_seconds=1,
            max_seconds=3,
            label_prefix="batch",
        )
        output = stdout.getvalue()
        self.assertIn("Enqueued 2 demo jobs", output)
        self.assertIn("task-1", output)
        self.assertIn("task-2", output)

    def test_command_rejects_invalid_range(self):
        with self.assertRaises(CommandError):
            call_command(
                "enqueue_demo_jobs",
                "--count",
                "2",
                "--min-seconds",
                "5",
                "--max-seconds",
                "1",
            )


class GeminiReasoningProviderTests(SimpleTestCase):
    def test_build_reasoning_models_requires_api_key(self):
        with self.assertRaises(ModelConfigurationError):
            build_reasoning_models(api_key="")

    def test_build_reasoning_models_uses_configured_model_ids(self):
        captured_configs: list[dict[str, object]] = []

        class FakeChatGoogleGenerativeAI:
            def __init__(self, **kwargs):
                captured_configs.append(kwargs)
                self.model = kwargs.get("model")

        fake_module = types.ModuleType("langchain_google_genai")
        fake_module.ChatGoogleGenerativeAI = FakeChatGoogleGenerativeAI

        with patch.dict(sys.modules, {"langchain_google_genai": fake_module}):
            with override_settings(
                GEMINI_API_KEY="test-api-key",
                GEMINI_SUBAGENT_MODEL="gemini-2.0-flash",
                GEMINI_PRIMARY_MODEL="gemini-3.0-pro",
                GEMINI_SUBAGENT_TEMPERATURE=0.2,
                GEMINI_PRIMARY_TEMPERATURE=0.1,
            ):
                models = build_reasoning_models()

        self.assertEqual(models.subagent_model_name, "gemini-2.0-flash")
        self.assertEqual(models.primary_model_name, "gemini-3.0-pro")
        self.assertEqual(len(captured_configs), 2)
        self.assertEqual(captured_configs[0]["model"], "gemini-2.0-flash")
        self.assertEqual(captured_configs[1]["model"], "gemini-3.0-pro")
        self.assertEqual(captured_configs[0]["google_api_key"], "test-api-key")
        self.assertEqual(captured_configs[1]["google_api_key"], "test-api-key")
        self.assertEqual(captured_configs[0]["temperature"], 0.2)
        self.assertEqual(captured_configs[1]["temperature"], 0.1)


@unittest.skipUnless(
    HAS_LANGGRAPH_STACK,
    "langgraph/langchain-core dependencies are not installed",
)
class LangGraphReasoningTests(SimpleTestCase):
    def test_graph_routes_subagent_and_primary_to_expected_models(self):
        class FakeModel:
            def __init__(self, response_text: str):
                self.response_text = response_text
                self.calls = []

            def invoke(self, messages):
                self.calls.append(messages)
                return SimpleNamespace(
                    content=self.response_text,
                    usage_metadata={
                        "input_tokens": 11,
                        "output_tokens": 7,
                        "total_tokens": 18,
                    },
                    response_metadata={"finish_reason": "stop"},
                )

        subagent = FakeModel("subagent note")
        primary = FakeModel("primary impression")
        models = ReasoningModels(
            subagent=subagent,
            primary=primary,
            subagent_model_name="gemini-2.0-flash",
            primary_model_name="gemini-3.0-pro",
        )
        graph = build_reasoning_graph(models=models)

        subagent_result = run_reasoning_graph(
            graph=graph,
            reasoning_input=ReasoningInput(
                role="subagent",
                system_prompt="system",
                user_prompt="window events",
                metadata={"window_start_ms": 0},
            ),
        )
        primary_result = run_reasoning_graph(
            graph=graph,
            reasoning_input=ReasoningInput(
                role="primary",
                system_prompt="system",
                user_prompt="ledger updates",
                metadata={"input_seq_from": 1},
            ),
        )

        self.assertEqual(len(subagent.calls), 1)
        self.assertEqual(len(primary.calls), 1)
        self.assertEqual(subagent_result.output_text, "subagent note")
        self.assertEqual(primary_result.output_text, "primary impression")
        self.assertEqual(subagent_result.model_name, "gemini-2.0-flash")
        self.assertEqual(primary_result.model_name, "gemini-3.0-pro")
        self.assertEqual(subagent_result.usage["total_tokens"], 18)
        self.assertEqual(primary_result.response_metadata["finish_reason"], "stop")


class CoachingSessionModelTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(
            username="coach@example.com",
            email="coach@example.com",
            password="password123",
        )

    def test_session_defaults(self):
        session = CoachingSession.objects.create(user=self.user)

        self.assertEqual(session.title, "Untitled Session")
        self.assertEqual(session.status, SessionStatus.DRAFT)
        self.assertEqual(session.speaker_context, "")
        self.assertFalse(session.video_file)
        self.assertIsNone(session.ml_task_id)
        self.assertIsNone(session.coach_task_id)

    def test_session_can_store_custom_title(self):
        session = CoachingSession.objects.create(
            user=self.user,
            title="Demo Day Runthrough",
        )

        self.assertEqual(session.title, "Demo Day Runthrough")

    def test_user_can_have_multiple_sessions(self):
        first = CoachingSession.objects.create(user=self.user)
        second = CoachingSession.objects.create(user=self.user)

        self.assertNotEqual(first.id, second.id)
        self.assertEqual(
            CoachingSession.objects.filter(user=self.user).count(),
            2,
        )

    def test_status_choices_are_validated(self):
        session = CoachingSession(user=self.user, status="invalid_status")

        with self.assertRaises(ValidationError):
            session.full_clean()

    def test_video_file_rejects_non_supported_extension(self):
        session = CoachingSession(
            user=self.user,
            video_file=SimpleUploadedFile("recording.mov", b"fake-video"),
        )

        with self.assertRaises(ValidationError) as error:
            session.full_clean()

        self.assertIn("video_file", error.exception.message_dict)

    def test_video_file_accepts_webm_extension(self):
        session = CoachingSession(
            user=self.user,
            video_file=SimpleUploadedFile("recording.webm", b"fake-video"),
        )

        session.full_clean()

    def test_supplementary_pdf_rejects_non_pdf_extension(self):
        session = CoachingSession(
            user=self.user,
            supplementary_pdf_1=SimpleUploadedFile("script.txt", b"fake-text"),
        )

        with self.assertRaises(ValidationError) as error:
            session.full_clean()

        self.assertIn("supplementary_pdf_1", error.exception.message_dict)

    def test_max_file_size_validator_rejects_oversized_video(self):
        validator = MaxFileSizeValidator(
            max_bytes=MAX_VIDEO_FILE_SIZE_BYTES,
            label="Video file",
        )

        with self.assertRaises(ValidationError):
            validator(
                SimpleNamespace(
                    name="recording.mp4",
                    size=MAX_VIDEO_FILE_SIZE_BYTES + 1,
                )
            )

    def test_max_file_size_validator_rejects_oversized_pdf(self):
        validator = MaxFileSizeValidator(
            max_bytes=MAX_SUPPLEMENTARY_PDF_FILE_SIZE_BYTES,
            label="Supplementary PDF",
        )

        with self.assertRaises(ValidationError):
            validator(
                SimpleNamespace(
                    name="slides.pdf",
                    size=MAX_SUPPLEMENTARY_PDF_FILE_SIZE_BYTES + 1,
                )
            )

    def test_non_draft_status_requires_video(self):
        with self.assertRaises(IntegrityError):
            CoachingSession.objects.create(
                user=self.user,
                status=SessionStatus.MEDIA_ATTACHED,
            )

    def test_non_draft_status_allows_video(self):
        session = CoachingSession.objects.create(
            user=self.user,
            status=SessionStatus.MEDIA_ATTACHED,
            video_file="sessions/videos/2026/03/08/demo.mp4",
        )

        self.assertEqual(session.status, SessionStatus.MEDIA_ATTACHED)
        self.assertEqual(
            session.video_file.name,
            "sessions/videos/2026/03/08/demo.mp4",
        )


class CoachOrchestrationModelsTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(
            username="orchestrator@example.com",
            email="orchestrator@example.com",
            password="password123",
        )
        self.session = CoachingSession.objects.create(
            user=self.user,
            status=SessionStatus.PROCESSING_COACH,
            video_file="sessions/videos/2026/03/08/demo.mp4",
        )

    def test_run_index_must_be_unique_per_session(self):
        CoachOrchestrationRun.objects.create(
            session=self.session,
            run_index=1,
            status=CoachOrchestrationRunStatus.QUEUED,
        )

        with self.assertRaises(IntegrityError):
            CoachOrchestrationRun.objects.create(
                session=self.session,
                run_index=1,
                status=CoachOrchestrationRunStatus.FAILED,
            )

    def test_only_one_active_run_allowed_per_session(self):
        CoachOrchestrationRun.objects.create(
            session=self.session,
            run_index=1,
            status=CoachOrchestrationRunStatus.QUEUED,
        )

        with self.assertRaises(IntegrityError):
            CoachOrchestrationRun.objects.create(
                session=self.session,
                run_index=2,
                status=CoachOrchestrationRunStatus.PROCESSING,
            )

    def test_agent_window_bounds_are_validated(self):
        run = CoachOrchestrationRun.objects.create(
            session=self.session,
            run_index=1,
            status=CoachOrchestrationRunStatus.PROCESSING,
        )

        with self.assertRaises(IntegrityError):
            CoachAgentExecution.objects.create(
                run=run,
                execution_index=1,
                agent_kind=CoachAgentKind.SUBAGENT,
                agent_name="subagent-1",
                status=CoachAgentExecutionStatus.PROCESSING,
                window_start_ms=40_000,
                window_end_ms=10_000,
            )

    def test_ledger_sequence_must_be_unique_per_run(self):
        run = CoachOrchestrationRun.objects.create(
            session=self.session,
            run_index=1,
            status=CoachOrchestrationRunStatus.PROCESSING,
        )
        CoachLedgerEntry.objects.create(
            run=run,
            sequence=1,
            entry_kind=LedgerEntryKind.SUBAGENT_NOTE,
            agent_kind=CoachAgentKind.SUBAGENT,
            agent_name="subagent-1",
            content="First note",
        )

        with self.assertRaises(IntegrityError):
            CoachLedgerEntry.objects.create(
                run=run,
                sequence=1,
                entry_kind=LedgerEntryKind.FLAGSHIP_IMPRESSION,
                agent_kind=CoachAgentKind.FLAGSHIP_PERIODIC,
                agent_name="flagship",
                content="Duplicate sequence",
            )


class CoachLedgerServiceTests(TestCase):
    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(
            username="ledger-service@example.com",
            email="ledger-service@example.com",
            password="password123",
        )
        self.session = CoachingSession.objects.create(
            user=self.user,
            status=SessionStatus.PROCESSING_COACH,
            video_file="sessions/videos/2026/03/08/demo.mp4",
        )

    def test_create_orchestration_run_increments_run_index(self):
        CoachOrchestrationRun.objects.create(
            session=self.session,
            run_index=1,
            status=CoachOrchestrationRunStatus.COMPLETED,
        )

        run = create_orchestration_run(session=self.session)

        self.assertEqual(run.run_index, 2)
        self.assertEqual(run.status, CoachOrchestrationRunStatus.QUEUED)

    def test_create_orchestration_run_rejects_session_with_active_run(self):
        CoachOrchestrationRun.objects.create(
            session=self.session,
            run_index=1,
            status=CoachOrchestrationRunStatus.QUEUED,
        )

        with self.assertRaises(RunStateError):
            create_orchestration_run(session=self.session)

    def test_create_agent_execution_increments_execution_index(self):
        run = CoachOrchestrationRun.objects.create(
            session=self.session,
            run_index=1,
            status=CoachOrchestrationRunStatus.PROCESSING,
        )

        first = create_agent_execution(
            run=run,
            agent_kind=CoachAgentKind.SUBAGENT,
            agent_name="subagent-window-1",
            window_start_ms=0,
            window_end_ms=30_000,
        )
        second = create_agent_execution(
            run=run,
            agent_kind=CoachAgentKind.FLAGSHIP_PERIODIC,
            agent_name="flagship-pass-1",
        )

        self.assertEqual(first.execution_index, 1)
        self.assertEqual(second.execution_index, 2)

    def test_append_ledger_entry_allocates_sequence_and_updates_run(self):
        run = CoachOrchestrationRun.objects.create(
            session=self.session,
            run_index=1,
            status=CoachOrchestrationRunStatus.PROCESSING,
        )
        execution = create_agent_execution(
            run=run,
            agent_kind=CoachAgentKind.SUBAGENT,
            agent_name="subagent-window-1",
            window_start_ms=0,
            window_end_ms=30_000,
        )

        first = append_ledger_entry(
            run=run,
            entry_kind=LedgerEntryKind.SUBAGENT_NOTE,
            content="Local pacing improved in this window.",
            agent_execution=execution,
            payload={"title": "Pacing"},
        )
        second = append_ledger_entry(
            run=run,
            entry_kind=LedgerEntryKind.FLAGSHIP_IMPRESSION,
            content="Overall pacing trend is stable.",
            agent_kind=CoachAgentKind.FLAGSHIP_PERIODIC,
            agent_name="flagship-pass-1",
        )

        run.refresh_from_db()
        self.assertEqual(first.sequence, 1)
        self.assertEqual(second.sequence, 2)
        self.assertEqual(run.latest_ledger_sequence, 2)
        self.assertEqual(first.agent_name, "subagent-window-1")
        self.assertEqual(second.agent_name, "flagship-pass-1")

    def test_append_ledger_entry_validates_execution_belongs_to_run(self):
        run = CoachOrchestrationRun.objects.create(
            session=self.session,
            run_index=1,
            status=CoachOrchestrationRunStatus.PROCESSING,
        )
        second_run = CoachOrchestrationRun.objects.create(
            session=self.session,
            run_index=2,
            status=CoachOrchestrationRunStatus.FAILED,
        )
        execution = create_agent_execution(
            run=run,
            agent_kind=CoachAgentKind.SUBAGENT,
            agent_name="subagent-window-1",
        )

        with self.assertRaises(LedgerValidationError):
            append_ledger_entry(
                run=second_run,
                entry_kind=LedgerEntryKind.SUBAGENT_NOTE,
                content="Invalid linkage",
                agent_execution=execution,
            )

    def test_read_ledger_slice_filters_by_sequence_and_kind(self):
        run = CoachOrchestrationRun.objects.create(
            session=self.session,
            run_index=1,
            status=CoachOrchestrationRunStatus.PROCESSING,
        )
        append_ledger_entry(
            run=run,
            entry_kind=LedgerEntryKind.SUBAGENT_NOTE,
            content="Entry 1",
            agent_kind=CoachAgentKind.SUBAGENT,
            agent_name="subagent-1",
        )
        append_ledger_entry(
            run=run,
            entry_kind=LedgerEntryKind.FLAGSHIP_IMPRESSION,
            content="Entry 2",
            agent_kind=CoachAgentKind.FLAGSHIP_PERIODIC,
            agent_name="flagship",
        )
        append_ledger_entry(
            run=run,
            entry_kind=LedgerEntryKind.SUBAGENT_NOTE,
            content="Entry 3",
            agent_kind=CoachAgentKind.SUBAGENT,
            agent_name="subagent-2",
        )

        entries = read_ledger_slice(
            run=run,
            sequence_gt=1,
            sequence_lte=3,
            entry_kind=LedgerEntryKind.SUBAGENT_NOTE,
        )
        self.assertEqual([entry.sequence for entry in entries], [3])

    def test_run_and_agent_lifecycle_helpers_update_status_and_timestamps(self):
        run = create_orchestration_run(session=self.session)
        run = mark_run_processing(run=run)
        self.assertEqual(run.status, CoachOrchestrationRunStatus.PROCESSING)
        self.assertIsNotNone(run.started_at)

        execution = create_agent_execution(
            run=run,
            agent_kind=CoachAgentKind.SUBAGENT,
            agent_name="subagent-window-1",
        )
        execution = mark_agent_processing(execution=execution)
        execution = touch_agent_heartbeat(execution=execution)
        execution = mark_agent_completed(execution=execution, output_seq_to=4)
        self.assertEqual(execution.status, CoachAgentExecutionStatus.COMPLETED)
        self.assertIsNotNone(execution.last_heartbeat_at)
        self.assertEqual(execution.output_seq_to, 4)

        run = mark_run_completed(run=run)
        self.assertEqual(run.status, CoachOrchestrationRunStatus.COMPLETED)
        self.assertIsNotNone(run.completed_at)

    def test_failed_lifecycle_helpers_capture_error_message(self):
        run = create_orchestration_run(session=self.session)
        run = mark_run_failed(run=run, error_message="failure on run")
        self.assertEqual(run.status, CoachOrchestrationRunStatus.FAILED)
        self.assertEqual(run.error_message, "failure on run")

        execution = create_agent_execution(
            run=run,
            agent_kind=CoachAgentKind.FLAGSHIP_FINAL,
            agent_name="flagship-final",
        )
        execution = mark_agent_failed(
            execution=execution,
            error_message="failure on final pass",
        )
        self.assertEqual(execution.status, CoachAgentExecutionStatus.FAILED)
        self.assertEqual(execution.error_message, "failure on final pass")


class CoachingSessionApiTests(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls._media_dir = tempfile.TemporaryDirectory()
        cls._media_override = override_settings(
            MEDIA_ROOT=cls._media_dir.name,
            MEDIA_URL="/media/",
        )
        cls._media_override.enable()

    @classmethod
    def tearDownClass(cls):
        cls._media_override.disable()
        cls._media_dir.cleanup()
        super().tearDownClass()

    def setUp(self):
        User = get_user_model()
        self.user = User.objects.create_user(
            username="owner@example.com",
            email="owner@example.com",
            password="password123",
        )
        self.other_user = User.objects.create_user(
            username="other@example.com",
            email="other@example.com",
            password="password123",
        )
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)
        self.unauthenticated_client = APIClient()
        self.collection_url = reverse("api:sessions-collection")

    def _video_file(
        self, *, name: str = "recording.mp4", content_type: str = "video/mp4"
    ) -> SimpleUploadedFile:
        return SimpleUploadedFile(name, b"fake-video", content_type=content_type)

    def _pdf_file(self, *, name: str = "slides.pdf") -> SimpleUploadedFile:
        return SimpleUploadedFile(name, b"fake-pdf", content_type="application/pdf")

    def _assert_auth_required(self, response):
        self.assertIn(
            response.status_code,
            {status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN},
        )

    def test_endpoints_require_authentication(self):
        session = CoachingSession.objects.create(user=self.user)
        detail_url = reverse("api:session-detail", kwargs={"id": session.id})
        video_url = reverse("api:session-video", kwargs={"id": session.id})
        assets_url = reverse("api:session-assets", kwargs={"id": session.id})

        responses = [
            self.unauthenticated_client.get(self.collection_url),
            self.unauthenticated_client.post(
                self.collection_url,
                {"title": "Draft"},
                format="json",
            ),
            self.unauthenticated_client.get(detail_url),
            self.unauthenticated_client.post(
                video_url,
                {"video_file": self._video_file()},
                format="multipart",
            ),
            self.unauthenticated_client.post(
                assets_url,
                {"speaker_context": "Brief context"},
                format="multipart",
            ),
        ]

        for response in responses:
            self._assert_auth_required(response)

    def test_create_session_with_default_title(self):
        response = self.client.post(self.collection_url, {}, format="json")

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        created = CoachingSession.objects.get(id=response.data["id"])
        self.assertEqual(created.user, self.user)
        self.assertEqual(created.title, "Untitled Session")
        self.assertEqual(created.status, SessionStatus.DRAFT)

    def test_create_session_with_custom_title(self):
        response = self.client.post(
            self.collection_url,
            {"title": "Boardroom Rehearsal"},
            format="json",
        )

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        created = CoachingSession.objects.get(id=response.data["id"])
        self.assertEqual(created.title, "Boardroom Rehearsal")
        self.assertEqual(response.data["status"], SessionStatus.DRAFT)

    def test_list_sessions_returns_only_authenticated_user_sessions(self):
        older = CoachingSession.objects.create(user=self.user, title="Older")
        newer = CoachingSession.objects.create(user=self.user, title="Newer")
        CoachingSession.objects.create(user=self.other_user, title="Other User Session")

        response = self.client.get(self.collection_url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(
            [item["id"] for item in response.data],
            [str(newer.id), str(older.id)],
        )
        self.assertEqual(
            set(response.data[0].keys()),
            {"id", "title", "status", "created_at", "updated_at"},
        )

    def test_get_session_returns_detail_for_owner(self):
        session = CoachingSession.objects.create(user=self.user, title="Session Detail")
        detail_url = reverse("api:session-detail", kwargs={"id": session.id})

        response = self.client.get(detail_url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["id"], str(session.id))
        self.assertEqual(response.data["title"], "Session Detail")
        self.assertEqual(response.data["status"], SessionStatus.DRAFT)
        self.assertIsNone(response.data["video_file_url"])
        self.assertIsNone(response.data["supplementary_pdf_1_url"])
        self.assertIsNone(response.data["supplementary_pdf_2_url"])
        self.assertIsNone(response.data["supplementary_pdf_3_url"])
        self.assertEqual(response.data["speaker_context"], "")
        self.assertEqual(response.data["coach_progress"]["status"], "pending")
        self.assertEqual(response.data["coach_progress"]["current_stage"], "")
        self.assertEqual(response.data["coach_progress"]["agent_progress"], [])
        self.assertEqual(response.data["coach_progress"]["stages"], [])
        self.assertEqual(response.data["coach_progress"]["latest_ledger_sequence"], 0)

    def test_get_session_returns_404_for_non_owner(self):
        session = CoachingSession.objects.create(user=self.other_user)
        detail_url = reverse("api:session-detail", kwargs={"id": session.id})

        response = self.client.get(detail_url)

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_get_session_can_return_queued_status_if_set_externally(self):
        session = CoachingSession.objects.create(
            user=self.user,
            status=SessionStatus.QUEUED_ML,
            video_file="sessions/videos/2026/03/08/demo.mp4",
        )
        detail_url = reverse("api:session-detail", kwargs={"id": session.id})

        response = self.client.get(detail_url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertEqual(response.data["status"], SessionStatus.QUEUED_ML)
        self.assertEqual(response.data["coach_progress"]["status"], "processing_coach")

    def test_get_session_returns_agent_progress_from_active_orchestration_run(self):
        session = CoachingSession.objects.create(
            user=self.user,
            title="Session Detail",
            status=SessionStatus.PROCESSING_COACH,
            video_file="sessions/videos/2026/03/08/demo.mp4",
        )
        old_run = CoachOrchestrationRun.objects.create(
            session=session,
            run_index=1,
            status=CoachOrchestrationRunStatus.COMPLETED,
            latest_ledger_sequence=2,
            completed_at=timezone.now(),
        )
        new_run = CoachOrchestrationRun.objects.create(
            session=session,
            run_index=2,
            status=CoachOrchestrationRunStatus.PROCESSING,
            latest_ledger_sequence=5,
            started_at=timezone.now(),
        )
        CoachAgentExecution.objects.create(
            run=old_run,
            execution_index=1,
            agent_kind=CoachAgentKind.SUBAGENT,
            agent_name="old-subagent",
            status=CoachAgentExecutionStatus.COMPLETED,
            started_at=timezone.now(),
            finished_at=timezone.now(),
        )
        first_execution = CoachAgentExecution.objects.create(
            run=new_run,
            execution_index=1,
            agent_kind=CoachAgentKind.SUBAGENT,
            agent_name="subagent-window-1",
            status=CoachAgentExecutionStatus.QUEUED,
            window_start_ms=0,
            window_end_ms=30_000,
        )
        second_execution = CoachAgentExecution.objects.create(
            run=new_run,
            execution_index=2,
            agent_kind=CoachAgentKind.FLAGSHIP_PERIODIC,
            agent_name="flagship-pass-1",
            status=CoachAgentExecutionStatus.COMPLETED,
            input_seq_from=1,
            input_seq_to=4,
            output_seq_to=5,
            started_at=timezone.now(),
            finished_at=timezone.now(),
        )
        CoachLedgerEntry.objects.create(
            run=new_run,
            agent_execution=second_execution,
            sequence=5,
            entry_kind=LedgerEntryKind.FLAGSHIP_IMPRESSION,
            agent_kind=CoachAgentKind.FLAGSHIP_PERIODIC,
            agent_name="flagship-pass-1",
            content="Confidence improved over the final minute.",
            payload={
                "title": "Global trend",
                "evidence_refs": ["01:00-01:40"],
            },
        )

        detail_url = reverse("api:session-detail", kwargs={"id": session.id})
        response = self.client.get(detail_url)

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        coach_progress = response.data["coach_progress"]
        self.assertEqual(coach_progress["status"], "processing_coach")
        self.assertEqual(coach_progress["active_run_id"], str(new_run.id))
        self.assertEqual(coach_progress["run_index"], 2)
        self.assertEqual(coach_progress["latest_ledger_sequence"], 5)
        self.assertEqual(coach_progress["current_stage"], "agent-2")
        self.assertEqual(len(coach_progress["agent_progress"]), 2)
        self.assertEqual(
            coach_progress["agent_progress"][0]["agent_execution_id"],
            str(first_execution.id),
        )
        self.assertEqual(coach_progress["agent_progress"][0]["status"], "pending")
        self.assertEqual(
            coach_progress["agent_progress"][1]["agent_execution_id"],
            str(second_execution.id),
        )
        self.assertEqual(coach_progress["agent_progress"][1]["status"], "completed")
        self.assertEqual(len(coach_progress["stages"]), 2)
        self.assertEqual(coach_progress["stages"][0]["stage_key"], "agent-1")
        self.assertEqual(coach_progress["stages"][1]["stage_key"], "agent-2")
        self.assertEqual(len(coach_progress["stages"][1]["notes"]), 1)
        self.assertEqual(
            coach_progress["stages"][1]["notes"][0]["title"],
            "Global trend",
        )
        self.assertEqual(
            coach_progress["stages"][1]["notes"][0]["evidence_refs"],
            ["01:00-01:40"],
        )

    def test_upload_video_moves_session_to_media_attached(self):
        session = CoachingSession.objects.create(user=self.user, status=SessionStatus.DRAFT)
        video_url = reverse("api:session-video", kwargs={"id": session.id})

        response = self.client.post(
            video_url,
            {"video_file": self._video_file()},
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        session.refresh_from_db()
        self.assertEqual(session.status, SessionStatus.MEDIA_ATTACHED)
        self.assertTrue(session.video_file.name.endswith(".mp4"))
        self.assertEqual(response.data["status"], SessionStatus.MEDIA_ATTACHED)

    def test_upload_video_accepts_webm_extension(self):
        session = CoachingSession.objects.create(user=self.user, status=SessionStatus.DRAFT)
        video_url = reverse("api:session-video", kwargs={"id": session.id})

        response = self.client.post(
            video_url,
            {
                "video_file": self._video_file(
                    name="recording.webm",
                    content_type="video/webm",
                )
            },
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        session.refresh_from_db()
        self.assertEqual(session.status, SessionStatus.MEDIA_ATTACHED)
        self.assertTrue(session.video_file.name.endswith(".webm"))
        self.assertEqual(response.data["status"], SessionStatus.MEDIA_ATTACHED)

    def test_upload_video_requires_draft_status(self):
        session = CoachingSession.objects.create(
            user=self.user,
            status=SessionStatus.MEDIA_ATTACHED,
            video_file="sessions/videos/2026/03/08/original.mp4",
        )
        video_url = reverse("api:session-video", kwargs={"id": session.id})

        response = self.client.post(
            video_url,
            {"video_file": self._video_file(name="replacement.mp4")},
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_409_CONFLICT)
        session.refresh_from_db()
        self.assertEqual(session.video_file.name, "sessions/videos/2026/03/08/original.mp4")

    def test_upload_video_rejects_invalid_extension(self):
        session = CoachingSession.objects.create(user=self.user, status=SessionStatus.DRAFT)
        video_url = reverse("api:session-video", kwargs={"id": session.id})

        response = self.client.post(
            video_url,
            {
                "video_file": SimpleUploadedFile(
                    "recording.mov",
                    b"fake-video",
                    content_type="video/quicktime",
                )
            },
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("video_file", response.data)

    def test_upload_video_returns_404_for_non_owner(self):
        session = CoachingSession.objects.create(user=self.other_user, status=SessionStatus.DRAFT)
        video_url = reverse("api:session-video", kwargs={"id": session.id})

        response = self.client.post(
            video_url,
            {"video_file": self._video_file()},
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    def test_upload_assets_updates_partial_fields_without_changing_status(self):
        session = CoachingSession.objects.create(
            user=self.user,
            status=SessionStatus.MEDIA_ATTACHED,
            video_file="sessions/videos/2026/03/08/demo.mp4",
            supplementary_pdf_2="sessions/assets/2026/03/08/existing.pdf",
        )
        assets_url = reverse("api:session-assets", kwargs={"id": session.id})

        response = self.client.post(
            assets_url,
            {
                "supplementary_pdf_1": self._pdf_file(),
                "speaker_context": "Pitch for enterprise audience",
            },
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        session.refresh_from_db()
        self.assertEqual(session.status, SessionStatus.MEDIA_ATTACHED)
        self.assertTrue(session.supplementary_pdf_1.name.endswith(".pdf"))
        self.assertEqual(session.supplementary_pdf_2.name, "sessions/assets/2026/03/08/existing.pdf")
        self.assertEqual(session.speaker_context, "Pitch for enterprise audience")

    def test_upload_assets_requires_media_attached_status(self):
        session = CoachingSession.objects.create(user=self.user, status=SessionStatus.DRAFT)
        assets_url = reverse("api:session-assets", kwargs={"id": session.id})

        response = self.client.post(
            assets_url,
            {"supplementary_pdf_1": self._pdf_file()},
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_409_CONFLICT)

    def test_upload_assets_requires_at_least_one_field(self):
        session = CoachingSession.objects.create(
            user=self.user,
            status=SessionStatus.MEDIA_ATTACHED,
            video_file="sessions/videos/2026/03/08/demo.mp4",
        )
        assets_url = reverse("api:session-assets", kwargs={"id": session.id})

        response = self.client.post(assets_url, {}, format="multipart")

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("non_field_errors", response.data)

    def test_upload_assets_rejects_invalid_pdf_extension(self):
        session = CoachingSession.objects.create(
            user=self.user,
            status=SessionStatus.MEDIA_ATTACHED,
            video_file="sessions/videos/2026/03/08/demo.mp4",
        )
        assets_url = reverse("api:session-assets", kwargs={"id": session.id})

        response = self.client.post(
            assets_url,
            {
                "supplementary_pdf_1": SimpleUploadedFile(
                    "notes.txt",
                    b"not-a-pdf",
                    content_type="text/plain",
                )
            },
            format="multipart",
        )

        self.assertEqual(response.status_code, status.HTTP_400_BAD_REQUEST)
        self.assertIn("supplementary_pdf_1", response.data)
