from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.exceptions import ValidationError
from django.core.management import CommandError, call_command
from django.db import IntegrityError
from django.test import SimpleTestCase, TestCase
from django.core.files.uploadedfile import SimpleUploadedFile

from ml.enqueue import enqueue_random_sleep_demo_job, enqueue_random_sleep_demo_jobs
from sessions.models import (
    MAX_SUPPLEMENTARY_PDF_FILE_SIZE_BYTES,
    MAX_VIDEO_FILE_SIZE_BYTES,
    CoachingSession,
    MaxFileSizeValidator,
    SessionStatus,
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

        self.assertEqual(session.status, SessionStatus.DRAFT)
        self.assertEqual(session.speaker_context, "")
        self.assertFalse(session.video_file)
        self.assertIsNone(session.ml_task_id)
        self.assertIsNone(session.coach_task_id)

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

    def test_video_file_rejects_non_mp4_extension(self):
        session = CoachingSession(
            user=self.user,
            video_file=SimpleUploadedFile("recording.mov", b"fake-video"),
        )

        with self.assertRaises(ValidationError) as error:
            session.full_clean()

        self.assertIn("video_file", error.exception.message_dict)

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
