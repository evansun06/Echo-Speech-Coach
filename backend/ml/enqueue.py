from __future__ import annotations

from celery.result import AsyncResult

from .tasks import random_sleep_demo_task


def enqueue_random_sleep_demo_job(
    *,
    min_seconds: int = 1,
    max_seconds: int = 8,
    label: str = "demo",
) -> AsyncResult:
    return random_sleep_demo_task.apply_async(
        kwargs={
            "min_seconds": min_seconds,
            "max_seconds": max_seconds,
            "label": label,
        }
    )


def enqueue_random_sleep_demo_jobs(
    *,
    count: int,
    min_seconds: int = 1,
    max_seconds: int = 8,
    label_prefix: str = "demo",
) -> list[str]:
    task_ids: list[str] = []
    for index in range(1, count + 1):
        async_result = enqueue_random_sleep_demo_job(
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            label=f"{label_prefix}-{index}",
        )
        task_ids.append(async_result.id)
    return task_ids
