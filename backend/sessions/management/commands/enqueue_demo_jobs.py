from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from ml.enqueue import enqueue_random_sleep_demo_jobs


class Command(BaseCommand):
    help = "Enqueue random-sleep Celery demo jobs for queue/Flower validation."

    def add_arguments(self, parser) -> None:
        parser.add_argument("--count", type=int, default=10)
        parser.add_argument("--min-seconds", type=int, default=1)
        parser.add_argument("--max-seconds", type=int, default=8)
        parser.add_argument("--prefix", type=str, default="demo")

    def handle(self, *args, **options) -> None:
        count = options["count"]
        min_seconds = options["min_seconds"]
        max_seconds = options["max_seconds"]
        prefix = options["prefix"]

        if count <= 0:
            raise CommandError("--count must be greater than 0")
        if min_seconds < 0:
            raise CommandError("--min-seconds must be non-negative")
        if max_seconds < min_seconds:
            raise CommandError("--max-seconds must be >= --min-seconds")

        task_ids = enqueue_random_sleep_demo_jobs(
            count=count,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            label_prefix=prefix,
        )

        self.stdout.write(
            self.style.SUCCESS(f"Enqueued {len(task_ids)} demo jobs:")
        )
        for task_id in task_ids:
            self.stdout.write(task_id)
