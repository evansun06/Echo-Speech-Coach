import tempfile
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
from rest_framework import status
from rest_framework.test import APIClient

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

    def _video_file(self, *, name: str = "recording.mp4") -> SimpleUploadedFile:
        return SimpleUploadedFile(name, b"fake-video", content_type="video/mp4")

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
