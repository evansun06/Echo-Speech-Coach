from __future__ import annotations

import argparse
import json
import logging
import time
import traceback
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

try:
    from .events import compute_events
    from .fuse import align_word_features
    from .mp_features import aggregate_windows, compute_overall_features, extract_frame_features
    from .os_features import SpeechFeatureExtractor
    from .stt_features import compute_overall_transcript_metrics, transcribe_words_google
except ImportError:
    # Fallback for direct script execution: `python ml/pipeline_facade.py`
    from .events import compute_events
    from .fuse import align_word_features
    from .mp_features import aggregate_windows, compute_overall_features, extract_frame_features
    from .os_features import SpeechFeatureExtractor
    from .stt_features import compute_overall_transcript_metrics, transcribe_words_google

logger = logging.getLogger(__name__)


@dataclass
class StageResult:
    name: str
    status: str
    duration_sec: float
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    traceback_snippet: Optional[str] = None


def _now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _sanitize(value: Any) -> Any:
    """Convert numpy/pandas values into JSON-safe Python values."""
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        if pd.isna(value):
            return None
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value


def _df_to_records(df: pd.DataFrame) -> list[dict]:
    if df is None or df.empty:
        return []
    cleaned = df.where(pd.notna(df), None)
    return _sanitize(cleaned.to_dict(orient="records"))


def _save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize(obj), f, indent=2, ensure_ascii=False)


def _save_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _run_stage(
    stage_name: str,
    fn,
    stage_results: list[StageResult],
    *,
    session_uuid: str,
) -> Any:
    t0 = time.time()
    try:
        logger.info(
            "ML stage started | session_id=%s stage=%s",
            session_uuid,
            stage_name,
        )
        out = fn()
        duration = round(time.time() - t0, 3)
        stage_results.append(
            StageResult(name=stage_name, status="ok", duration_sec=duration)
        )
        logger.info(
            "ML stage completed | session_id=%s stage=%s duration_sec=%.3f",
            session_uuid,
            stage_name,
            duration,
        )
        return out
    except Exception as exc:
        duration = round(time.time() - t0, 3)
        stage_results.append(
            StageResult(
                name=stage_name,
                status="failed",
                duration_sec=duration,
                error_type=type(exc).__name__,
                error_message=str(exc),
                traceback_snippet=traceback.format_exc(limit=6),
            )
        )
        logger.exception(
            "ML stage failed | session_id=%s stage=%s duration_sec=%.3f error_type=%s",
            session_uuid,
            stage_name,
            duration,
            type(exc).__name__,
        )
        raise


def _canonicalize_payload(
    session_uuid: str,
    generated_at: str,
    audio_path: str,
    video_path: str,
    run_dir: Path,
    stt_overall: dict,
    os_overall: dict,
    mp_overall: dict,
    aligned_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> dict:
    """
    Canonical payload for DB+debug JSON.
    Contains only: meta, overall_metrics, aligned_table, events.
    """
    aligned_records = _df_to_records(aligned_df)
    event_records = _df_to_records(events_df)

    # Force compact integer IDs for DB-friendly child rows.
    for i, row in enumerate(aligned_records):
        row["word_id"] = i
    for i, row in enumerate(event_records):
        row["event_id"] = i

    return {
        "meta": {
            "schema_version": "v2",
            "generated_at": generated_at,
            "session_uuid": session_uuid,
            "audio_path": str(Path(audio_path).resolve()),
            "video_path": str(Path(video_path).resolve()),
            "run_dir": str(run_dir.resolve()),
        },
        "overall_metrics": {
            "stt": stt_overall,
            "opensmile": os_overall,
            "mediapipe": mp_overall,
        },
        "aligned_table": aligned_records,
        "events": event_records,
    }


def write_to_postgres(canonical_payload: dict, db_url: str) -> dict:
    """
    Write canonical payload into 3 Postgres tables keyed by session_uuid.

    Tables:
    - session_overall_metrics(session_uuid, metrics)
    - session_aligned_words(session_uuid, word_id, word_json, start_time, end_time)
    - session_events(session_uuid, event_id, event_json, start_time, end_time)
    """
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError(
            "psycopg is not installed. Install it before using --write-db."
        ) from exc

    session_uuid = canonical_payload["meta"]["session_uuid"]
    # Validate early so DB errors are clearer.
    uuid.UUID(session_uuid)

    overall_metrics = canonical_payload["overall_metrics"]
    aligned_rows = canonical_payload.get("aligned_table", [])
    event_rows = canonical_payload.get("events", [])

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            # DDL
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS session_overall_metrics (
                  session_uuid UUID PRIMARY KEY,
                  metrics JSONB NOT NULL
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS session_aligned_words (
                  session_uuid UUID NOT NULL,
                  word_id INTEGER NOT NULL,
                  word_json JSONB NOT NULL,
                  start_time DOUBLE PRECISION NOT NULL,
                  end_time DOUBLE PRECISION NOT NULL,
                  PRIMARY KEY (session_uuid, word_id),
                  FOREIGN KEY (session_uuid)
                    REFERENCES session_overall_metrics(session_uuid)
                    ON DELETE CASCADE
                )
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS session_events (
                  session_uuid UUID NOT NULL,
                  event_id INTEGER NOT NULL,
                  event_json JSONB NOT NULL,
                  start_time DOUBLE PRECISION NOT NULL,
                  end_time DOUBLE PRECISION NOT NULL,
                  PRIMARY KEY (session_uuid, event_id),
                  FOREIGN KEY (session_uuid)
                    REFERENCES session_overall_metrics(session_uuid)
                    ON DELETE CASCADE
                )
                """
            )

            # Upsert parent row
            cur.execute(
                """
                INSERT INTO session_overall_metrics (session_uuid, metrics)
                VALUES (%s, %s::jsonb)
                ON CONFLICT (session_uuid)
                DO UPDATE SET metrics = EXCLUDED.metrics
                """,
                (session_uuid, json.dumps(_sanitize(overall_metrics))),
            )

            # Replace child rows for idempotent reruns per session_uuid.
            cur.execute("DELETE FROM session_aligned_words WHERE session_uuid = %s", (session_uuid,))
            cur.execute("DELETE FROM session_events WHERE session_uuid = %s", (session_uuid,))

            if aligned_rows:
                aligned_payload = [
                    (
                        session_uuid,
                        int(row.get("word_id", i)),
                        json.dumps(_sanitize(row)),
                        float(row.get("start_sec", 0.0)),
                        float(row.get("end_sec", 0.0)),
                    )
                    for i, row in enumerate(aligned_rows)
                ]
                cur.executemany(
                    """
                    INSERT INTO session_aligned_words
                      (session_uuid, word_id, word_json, start_time, end_time)
                    VALUES (%s, %s, %s::jsonb, %s, %s)
                    """,
                    aligned_payload,
                )

            if event_rows:
                events_payload = [
                    (
                        session_uuid,
                        int(row.get("event_id", i)),
                        json.dumps(_sanitize(row)),
                        float(row.get("start_sec", 0.0)),
                        float(row.get("end_sec", 0.0)),
                    )
                    for i, row in enumerate(event_rows)
                ]
                cur.executemany(
                    """
                    INSERT INTO session_events
                      (session_uuid, event_id, event_json, start_time, end_time)
                    VALUES (%s, %s, %s::jsonb, %s, %s)
                    """,
                    events_payload,
                )

        conn.commit()

    return {
        "session_uuid": session_uuid,
        "overall_rows": 1,
        "aligned_rows": len(aligned_rows),
        "event_rows": len(event_rows),
    }


def run_pipeline(
    audio_path: str,
    video_path: str,
    session_uuid: str,
    *,
    language_code: str = "en-US",
    sample_rate_hz: int = 16000,
    os_interval_sec: float = 1.0,
    mp_window_sec: float = 1.0,
    output_root: str = "ml/run_outputs",
) -> dict:
    """Live-only pipeline; returns canonical payload + run metadata."""
    # Validate UUID at the API boundary.
    uuid.UUID(session_uuid)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_root) / session_uuid / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    stage_results: list[StageResult] = []
    generated_at = _now_iso()
    pipeline_t0 = time.time()
    logger.info(
        "ML pipeline started | session_id=%s audio_path=%s video_path=%s run_dir=%s",
        session_uuid,
        str(Path(audio_path).resolve()),
        str(Path(video_path).resolve()),
        str(run_dir.resolve()),
    )

    try:
        # 1) STT
        word_df, _transcript_text = _run_stage(
            "stt_transcription",
            lambda: transcribe_words_google(
                audio_path=audio_path,
                language_code=language_code,
                sample_rate_hz=sample_rate_hz,
            ),
            stage_results,
            session_uuid=session_uuid,
        )
        stt_overall = _run_stage(
            "stt_overall",
            lambda: compute_overall_transcript_metrics(word_df),
            stage_results,
            session_uuid=session_uuid,
        )
        logger.info(
            "ML STT summary | session_id=%s words=%s",
            session_uuid,
            len(word_df),
        )

        # 2) OpenSMILE
        extractor = SpeechFeatureExtractor(interval_seconds=os_interval_sec)
        os_interval_df = _run_stage(
            "opensmile_intervals",
            lambda: extractor.extract_interval_features(audio_path),
            stage_results,
            session_uuid=session_uuid,
        )
        os_overall = _run_stage(
            "opensmile_overall",
            lambda: extractor.extract_overall_features(audio_path),
            stage_results,
            session_uuid=session_uuid,
        )
        logger.info(
            "ML OpenSMILE summary | session_id=%s intervals=%s",
            session_uuid,
            len(os_interval_df),
        )

        # 3) MediaPipe
        frame_df, effective_fps = _run_stage(
            "mediapipe_frame_features",
            lambda: extract_frame_features(video_path),
            stage_results,
            session_uuid=session_uuid,
        )
        mp_window_df = _run_stage(
            "mediapipe_windows",
            lambda: aggregate_windows(frame_df, window_sec=mp_window_sec),
            stage_results,
            session_uuid=session_uuid,
        )
        mp_overall = _run_stage(
            "mediapipe_overall",
            lambda: compute_overall_features(frame_df, mp_window_df),
            stage_results,
            session_uuid=session_uuid,
        )
        mp_overall["effective_fps"] = float(effective_fps)
        logger.info(
            "ML MediaPipe summary | session_id=%s windows=%s effective_fps=%.2f",
            session_uuid,
            len(mp_window_df),
            effective_fps,
        )

        # 4) Alignment
        aligned_df = _run_stage(
            "fusion_alignment",
            lambda: align_word_features(
                words_df=word_df,
                opensmile_df=os_interval_df,
                mediapipe_df=mp_window_df,
            ),
            stage_results,
            session_uuid=session_uuid,
        )
        logger.info(
            "ML fusion summary | session_id=%s aligned_rows=%s",
            session_uuid,
            len(aligned_df),
        )

        # 5) Events
        _, events_df = _run_stage(
            "event_detection",
            lambda: compute_events(aligned_df),
            stage_results,
            session_uuid=session_uuid,
        )
        logger.info(
            "ML events summary | session_id=%s events=%s",
            session_uuid,
            len(events_df),
        )

        # Save minimal debugging artifacts
        _save_table(aligned_df, run_dir / "aligned.csv")
        _save_table(events_df, run_dir / "events.csv")

        canonical_payload = _canonicalize_payload(
            session_uuid=session_uuid,
            generated_at=generated_at,
            audio_path=audio_path,
            video_path=video_path,
            run_dir=run_dir,
            stt_overall=stt_overall,
            os_overall=os_overall,
            mp_overall=mp_overall,
            aligned_df=aligned_df,
            events_df=events_df,
        )

        _save_json(canonical_payload, run_dir / "canonical_payload.json")
        logger.info(
            "ML artifacts saved | session_id=%s run_dir=%s",
            session_uuid,
            str(run_dir.resolve()),
        )

        run_report = {
            "meta": {
                "session_uuid": session_uuid,
                "generated_at": generated_at,
                "run_dir": str(run_dir.resolve()),
                "status": "ok",
            },
            "stages": [_sanitize(s.__dict__) for s in stage_results],
            "artifact_paths": {
                "aligned": str((run_dir / "aligned.csv").resolve()),
                "events": str((run_dir / "events.csv").resolve()),
                "canonical_payload": str((run_dir / "canonical_payload.json").resolve()),
            },
        }
        _save_json(run_report, run_dir / "run_report.json")
        logger.info(
            "ML pipeline completed | session_id=%s duration_sec=%.3f run_report=%s",
            session_uuid,
            round(time.time() - pipeline_t0, 3),
            str((run_dir / "run_report.json").resolve()),
        )

        return {
            "status": "ok",
            "run_dir": str(run_dir.resolve()),
            "canonical_payload": canonical_payload,
            "canonical_payload_path": str((run_dir / "canonical_payload.json").resolve()),
            "run_report_path": str((run_dir / "run_report.json").resolve()),
        }

    except Exception as exc:
        failed_stage = stage_results[-1].name if stage_results else "unknown"
        run_report = {
            "meta": {
                "session_uuid": session_uuid,
                "generated_at": generated_at,
                "run_dir": str(run_dir.resolve()),
                "status": "failed",
                "failed_stage": failed_stage,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
            "stages": [_sanitize(s.__dict__) for s in stage_results],
        }
        _save_json(run_report, run_dir / "run_report.json")
        logger.exception(
            "ML pipeline failed | session_id=%s failed_stage=%s duration_sec=%.3f run_report=%s error_type=%s",
            session_uuid,
            failed_stage,
            round(time.time() - pipeline_t0, 3),
            str((run_dir / "run_report.json").resolve()),
            type(exc).__name__,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Facade entrypoint for 3-table session payload")
    parser.add_argument("--audio", required=True, help="Path to input WAV audio")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--session-uuid", required=True, help="Session UUID")
    parser.add_argument("--output-root", default="ml/run_outputs", help="Root folder for run artifacts")
    parser.add_argument("--language-code", default="en-US", help="Google STT language code")
    parser.add_argument("--sample-rate-hz", type=int, default=16000, help="Audio sample rate for STT")
    parser.add_argument("--os-interval-sec", type=float, default=1.0, help="OpenSMILE interval size")
    parser.add_argument("--mp-window-sec", type=float, default=1.0, help="MediaPipe window size")
    parser.add_argument("--write-db", action="store_true", help="Write canonical payload into Postgres")
    parser.add_argument("--db-url", default=None, help="Postgres connection URL")

    args = parser.parse_args()

    result = run_pipeline(
        audio_path=args.audio,
        video_path=args.video,
        session_uuid=args.session_uuid,
        language_code=args.language_code,
        sample_rate_hz=args.sample_rate_hz,
        os_interval_sec=args.os_interval_sec,
        mp_window_sec=args.mp_window_sec,
        output_root=args.output_root,
    )

    # CLI default: show JSON payload preview for easy inspection.
    print("\n=== Canonical Payload (JSON Preview) ===")
    print(json.dumps(_sanitize(result["canonical_payload"]), indent=2, ensure_ascii=False))

    if args.write_db:
        if not args.db_url:
            raise ValueError("--write-db requires --db-url")
        db_result = write_to_postgres(result["canonical_payload"], args.db_url)
        print("\n=== DB Write Result ===")
        print(json.dumps(_sanitize(db_result), indent=2))

    print("\nPipeline completed.")
    print(f"Run dir: {result['run_dir']}")
    print(f"Canonical payload: {result['canonical_payload_path']}")
    print(f"Run report: {result['run_report_path']}")


if __name__ == "__main__":
    main()
