from __future__ import annotations

import json
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any, TypedDict

from django.conf import settings
from redis import Redis


class LiveLedgerEntry(TypedDict):
    sequence: int
    run_id: str
    entry_kind: str
    content: str
    payload: dict[str, Any]
    agent_execution_id: str | None
    agent_kind: str | None
    agent_name: str
    window_start_ms: int | None
    window_end_ms: int | None
    created_at: str


class LiveLedgerError(RuntimeError):
    """Raised when live-ledger input or state is invalid."""


def _run_key(*, run_id: str, suffix: str) -> str:
    """Build a run-scoped Redis key for live ledger storage."""
    return f"speechcoach:llm:live_ledger:run:{run_id}:{suffix}"


def _sequence_key(*, run_id: str) -> str:
    """Return the Redis key tracking the latest run sequence counter."""
    return _run_key(run_id=run_id, suffix="sequence")


def _entry_index_key(*, run_id: str) -> str:
    """Return the sorted-set key indexing sequences for ordered reads."""
    return _run_key(run_id=run_id, suffix="entry_index")


def _entry_payload_key(*, run_id: str) -> str:
    """Return the hash key storing serialized ledger entries by sequence."""
    return _run_key(run_id=run_id, suffix="entry_payload")


@lru_cache(maxsize=1)
def get_live_ledger_redis_client() -> Redis:
    """Build and cache the Redis client used for live-ledger reads/writes."""
    redis_url = getattr(settings, "LLM_LEDGER_REDIS_URL", settings.CELERY_BROKER_URL)
    return Redis.from_url(redis_url, decode_responses=True)


def clear_live_ledger_redis_client_cache() -> None:
    """Clear the cached Redis client so new configuration takes effect."""
    get_live_ledger_redis_client.cache_clear()


def _ledger_ttl_seconds() -> int:
    """Return the configured TTL for run-scoped live-ledger keys."""
    configured = getattr(settings, "LLM_LEDGER_REDIS_TTL_SECONDS", 86_400)
    return max(int(configured), 0)


def _entry_from_json(*, raw_entry: str) -> LiveLedgerEntry:
    """Parse one JSON-serialized live-ledger entry with basic shape validation."""
    decoded = json.loads(raw_entry)
    if not isinstance(decoded, dict):
        raise LiveLedgerError("Live ledger entry must decode to an object.")
    if not isinstance(decoded.get("sequence"), int):
        raise LiveLedgerError("Live ledger entry is missing a valid sequence.")
    return decoded  # type: ignore[return-value]


def append_live_ledger_entry(
    *,
    run_id: str,
    entry_kind: str,
    content: str,
    payload: dict[str, Any] | None = None,
    agent_execution_id: str | None = None,
    agent_kind: str | None = None,
    agent_name: str = "",
    window_start_ms: int | None = None,
    window_end_ms: int | None = None,
    now_iso: str | None = None,
) -> LiveLedgerEntry:
    """Append one run-scoped live-ledger entry and allocate its sequence atomically."""
    normalized_run_id = str(run_id).strip()
    if not normalized_run_id:
        raise LiveLedgerError("run_id is required.")
    normalized_entry_kind = str(entry_kind).strip()
    if not normalized_entry_kind:
        raise LiveLedgerError("entry_kind is required.")
    normalized_content = str(content).strip()
    if not normalized_content:
        raise LiveLedgerError("content is required.")

    entry: LiveLedgerEntry = {
        "sequence": 0,
        "run_id": normalized_run_id,
        "entry_kind": normalized_entry_kind,
        "content": normalized_content,
        "payload": dict(payload or {}),
        "agent_execution_id": (
            str(agent_execution_id).strip() if agent_execution_id is not None else None
        ),
        "agent_kind": str(agent_kind).strip() if agent_kind is not None else None,
        "agent_name": str(agent_name).strip(),
        "window_start_ms": window_start_ms,
        "window_end_ms": window_end_ms,
        "created_at": now_iso or datetime.now(UTC).isoformat(),
    }

    sequence_key = _sequence_key(run_id=normalized_run_id)
    entry_index_key = _entry_index_key(run_id=normalized_run_id)
    entry_payload_key = _entry_payload_key(run_id=normalized_run_id)
    ttl_seconds = _ledger_ttl_seconds()

    client = get_live_ledger_redis_client()
    sequence = int(client.incr(sequence_key))
    entry["sequence"] = sequence
    raw_entry = json.dumps(entry, separators=(",", ":"), sort_keys=True)

    pipeline = client.pipeline(transaction=True)
    pipeline.hset(entry_payload_key, str(sequence), raw_entry)
    pipeline.zadd(entry_index_key, {str(sequence): sequence})
    if ttl_seconds > 0:
        pipeline.expire(sequence_key, ttl_seconds)
        pipeline.expire(entry_index_key, ttl_seconds)
        pipeline.expire(entry_payload_key, ttl_seconds)
    pipeline.execute()
    return entry


def read_live_ledger_slice(
    *,
    run_id: str,
    sequence_gt: int = 0,
    sequence_lte: int | None = None,
    limit: int | None = None,
) -> list[LiveLedgerEntry]:
    """Read ordered live-ledger entries for a run using sequence bounds."""
    normalized_run_id = str(run_id).strip()
    if not normalized_run_id:
        return []

    minimum = sequence_gt + 1
    maximum: int | str = "+inf" if sequence_lte is None else int(sequence_lte)

    client = get_live_ledger_redis_client()
    entry_index_key = _entry_index_key(run_id=normalized_run_id)
    entry_payload_key = _entry_payload_key(run_id=normalized_run_id)

    if limit is None:
        sequence_ids = client.zrangebyscore(entry_index_key, minimum, maximum)
    else:
        sequence_ids = client.zrangebyscore(
            entry_index_key,
            minimum,
            maximum,
            start=0,
            num=max(int(limit), 0),
        )
    if not sequence_ids:
        return []

    raw_entries = client.hmget(entry_payload_key, sequence_ids)
    entries: list[LiveLedgerEntry] = []
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, str):
            continue
        parsed = _entry_from_json(raw_entry=raw_entry)
        entries.append(parsed)
    entries.sort(key=lambda item: item["sequence"])
    return entries


def get_live_ledger_latest_sequence(*, run_id: str) -> int:
    """Return the latest allocated live-ledger sequence for a run."""
    normalized_run_id = str(run_id).strip()
    if not normalized_run_id:
        return 0
    raw_value = get_live_ledger_redis_client().get(_sequence_key(run_id=normalized_run_id))
    if raw_value is None:
        return 0
    return int(raw_value)


def clear_live_ledger(*, run_id: str) -> None:
    """Delete all run-scoped keys used by the live-ledger store."""
    normalized_run_id = str(run_id).strip()
    if not normalized_run_id:
        return
    get_live_ledger_redis_client().delete(
        _sequence_key(run_id=normalized_run_id),
        _entry_index_key(run_id=normalized_run_id),
        _entry_payload_key(run_id=normalized_run_id),
    )

