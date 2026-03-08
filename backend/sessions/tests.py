from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from django.core.management import CommandError, call_command
from django.test import SimpleTestCase

from ml.enqueue import enqueue_random_sleep_demo_job, enqueue_random_sleep_demo_jobs


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
