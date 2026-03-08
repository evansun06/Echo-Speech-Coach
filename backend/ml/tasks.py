from __future__ import annotations

import random
import time
from datetime import datetime, timezone

from celery import shared_task


@shared_task(name="ml.demo.random_sleep")
def random_sleep_demo_task(
    *,
    min_seconds: int = 1,
    max_seconds: int = 8,
    label: str = "demo",
) -> dict[str, int | str]:
    if min_seconds < 0:
        raise ValueError("min_seconds must be non-negative")
    if max_seconds < min_seconds:
        raise ValueError("max_seconds must be greater than or equal to min_seconds")

    sleep_seconds = random.randint(min_seconds, max_seconds)
    time.sleep(sleep_seconds)
    return {
        "label": label,
        "sleep_seconds": sleep_seconds,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
