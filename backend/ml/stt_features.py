from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from google.cloud import speech
import numpy as np

logger = logging.getLogger(__name__)


FILLER_WORDS = {
    "um", "uh", "erm", "ah", "like", "you know", "i mean", "sort of", "kind of"
}
SYNC_RECOGNIZE_API_PAYLOAD_LIMIT_BYTES = 10_485_760
DEFAULT_SYNC_RECOGNIZE_MAX_PAYLOAD_BYTES = 9_500_000


@dataclass
class WordItem:
    word: str
    start_sec: float
    end_sec: float
    confidence: Optional[float]


@dataclass
class SentenceChunk:
    chunk_id: int
    text: str
    start_sec: float
    end_sec: float
    duration_sec: float
    word_count: int
    wpm: float
    pause_before_sec: float
    filler_count: int
    mean_confidence: Optional[float]


def _duration_to_sec(duration) -> float:
    if hasattr(duration, "total_seconds"):
        return duration.total_seconds()
    return duration.seconds + duration.nanos / 1e9

def _normalize_word(word: str) -> str:
    return re.sub(r"[^\w']", "", word.lower()).strip()

def _is_sentence_end(word: str) -> bool:
    return word.endswith((".", "!", "?"))


def _env_positive_int(name: str, default: int) -> int:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return default
    try:
        parsed = int(raw_value)
    except ValueError:
        logger.warning(
            "Invalid %s=%r; falling back to %s.",
            name,
            raw_value,
            default,
        )
        return default
    return max(parsed, 1)


def _resolve_max_sync_payload_bytes() -> int:
    requested_payload_limit = _env_positive_int(
        "ML_STT_SYNC_MAX_PAYLOAD_BYTES",
        DEFAULT_SYNC_RECOGNIZE_MAX_PAYLOAD_BYTES,
    )
    payload_limit = min(
        requested_payload_limit,
        SYNC_RECOGNIZE_API_PAYLOAD_LIMIT_BYTES,
    )
    if payload_limit != requested_payload_limit:
        logger.warning(
            "ML_STT_SYNC_MAX_PAYLOAD_BYTES=%s exceeds API limit; clamping to %s.",
            requested_payload_limit,
            payload_limit,
        )
    return payload_limit

def transcribe_words_google(
    audio_path: str,
    language_code: str = "en-US",
    sample_rate_hz: int = 16000,
) -> tuple[pd.DataFrame, str]:
    """
    Transcribe a local WAV file with Google STT and return:
    1) word-level dataframe
    2) transcript text
    """
    audio_file = Path(audio_path)
    timeout_seconds = float(os.environ.get("ML_STT_RECOGNIZE_TIMEOUT_SECONDS", "30"))
    requested_transport = os.environ.get("ML_STT_TRANSPORT", "rest").strip().lower()
    transport = requested_transport if requested_transport in {"grpc", "rest"} else "rest"
    file_size_bytes = audio_file.stat().st_size
    # LINEAR16 mono -> ~2 bytes per sample
    estimated_duration_sec = file_size_bytes / max(sample_rate_hz * 2.0, 1.0)

    logger.info(
        (
            "STT request start | audio_path=%s file_size_mb=%.2f "
            "estimated_duration_sec=%.2f language=%s sample_rate_hz=%s timeout_sec=%.1f"
        ),
        str(audio_file.resolve()),
        file_size_bytes / (1024 * 1024),
        estimated_duration_sec,
        language_code,
        sample_rate_hz,
        timeout_seconds,
    )

    logger.info(
        "STT client init | transport=%s pid=%s",
        transport,
        os.getpid(),
    )
    client = speech.SpeechClient(transport=transport)
    max_payload_bytes = _resolve_max_sync_payload_bytes()
    if file_size_bytes > max_payload_bytes:
        raise ValueError(
            "Audio payload exceeds sync STT limit. "
            f"file_size_bytes={file_size_bytes} max_allowed_bytes={max_payload_bytes}. "
            "Upload a shorter/smaller video."
        )
    with open(audio_file, "rb") as f:
        content = f.read()

    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate_hz,
        language_code=language_code,
        enable_word_time_offsets=True,
        enable_word_confidence=True,
        enable_automatic_punctuation=True,
    )
    request_t0 = time.time()
    words: List[WordItem] = []
    transcript_parts: List[str] = []
    try:
        response = client.recognize(
            config=config,
            audio=audio,
            timeout=timeout_seconds,
        )
    except Exception as exc:
        logger.exception(
            "STT request failed | error_type=%s timeout_sec=%.1f elapsed_sec=%.3f",
            type(exc).__name__,
            timeout_seconds,
            time.time() - request_t0,
        )
        raise

    for result in response.results:
        if not result.alternatives:
            continue
        alt = result.alternatives[0]
        transcript = str(alt.transcript).strip()
        if transcript:
            transcript_parts.append(transcript)
        for token in alt.words:
            words.append(
                WordItem(
                    word=token.word,
                    start_sec=_duration_to_sec(token.start_time),
                    end_sec=_duration_to_sec(token.end_time),
                    confidence=(
                        float(token.confidence)
                        if token.confidence is not None
                        else None
                    ),
                )
            )

    word_df = pd.DataFrame(
        [
            {
                "word": w.word,
                "start_sec": w.start_sec,
                "end_sec": w.end_sec,
                "duration_sec": w.end_sec - w.start_sec,
                "confidence": w.confidence,
            }
            for w in words
        ]
    )

    transcript_text = " ".join(transcript_parts).strip()
    logger.info(
        "STT request completed | results=%s words=%s elapsed_sec=%.3f",
        len(response.results),
        len(word_df),
        time.time() - request_t0,
    )
    return word_df, transcript_text


def compute_overall_transcript_metrics(word_df: pd.DataFrame) -> Dict[str, float]:
    if word_df.empty:
        return {
            "overall_wpm": np.nan,
            "pause_ratio": np.nan,
            "mean_pause_sec": np.nan,
            "long_pause_ratio": np.nan,
            "mean_word_confidence": np.nan,
            "filler_rate_per_100_words": np.nan,
            "articulation_rate_wpm": np.nan,
        }

    total_duration = float(word_df["end_sec"].max() - word_df["start_sec"].min())
    total_duration = max(total_duration, 1e-6)

    word_count = int(len(word_df))
    overall_wpm = word_count / (total_duration / 60.0)

    pauses = []
    for i in range(len(word_df) - 1):
        gap = float(word_df.iloc[i + 1]["start_sec"] - word_df.iloc[i]["end_sec"])
        pauses.append(max(0.0, gap))

    pauses = np.array(pauses, dtype=float) if pauses else np.array([], dtype=float)
    total_pause_time = float(pauses.sum()) if len(pauses) else 0.0
    pause_ratio = total_pause_time / total_duration

    mean_pause_sec = float(pauses.mean()) if len(pauses) else 0.0
    long_pause_ratio = (
        float((pauses >= 0.7).mean()) if len(pauses) else 0.0
    )

    confidences = word_df["confidence"].dropna()
    mean_word_confidence = float(confidences.mean()) if len(confidences) else np.nan

    norm_words = [_normalize_word(w) for w in word_df["word"].tolist()]
    filler_count = sum(1 for w in norm_words if w in FILLER_WORDS)
    filler_rate_per_100_words = 100.0 * filler_count / max(word_count, 1)

    speaking_time = total_duration - total_pause_time
    speaking_time = max(speaking_time, 1e-6)
    articulation_rate_wpm = word_count / (speaking_time / 60.0)

    return {
        "overall_wpm": overall_wpm,
        "pause_ratio": pause_ratio,
        "mean_pause_sec": mean_pause_sec,
        "long_pause_ratio": long_pause_ratio,
        "mean_word_confidence": mean_word_confidence,
        "filler_rate_per_100_words": filler_rate_per_100_words,
        "articulation_rate_wpm": articulation_rate_wpm,
    }

def build_sentence_chunks(word_df: pd.DataFrame, pause_threshold: float = 0.6) -> pd.DataFrame:
    if word_df.empty:
        return pd.DataFrame()

    rows = word_df.to_dict("records")
    chunks = []
    current_chunk = []

    for i, row in enumerate(rows):
        current_chunk.append(row)

        should_split = False

        # split on punctuation
        if _is_sentence_end(row["word"]):
            should_split = True

        # split on long pause before next word
        if i < len(rows) - 1:
            next_row = rows[i + 1]
            gap = max(0.0, next_row["start_sec"] - row["end_sec"])
            if gap >= pause_threshold:
                should_split = True

        if should_split:
            chunks.append(current_chunk)
            current_chunk = []

    if current_chunk:
        chunks.append(current_chunk)

    chunk_rows = []

    for chunk_id, chunk_words in enumerate(chunks):
        text = " ".join(w["word"] for w in chunk_words).strip()
        start_sec = chunk_words[0]["start_sec"]
        end_sec = chunk_words[-1]["end_sec"]
        duration_sec = end_sec - start_sec
        word_count = len(chunk_words)

        confidences = [
            w["confidence"] for w in chunk_words
            if w["confidence"] is not None
        ]

        filler_count = sum(
            1 for w in chunk_words
            if _normalize_word(w["word"]) in FILLER_WORDS
        )

        # speaking time = sum of word durations only
        speaking_time = sum(w["duration_sec"] for w in chunk_words)
        speaking_time = max(speaking_time, 1e-6)

        wpm = word_count / (speaking_time / 60.0)

        chunk_rows.append({
            "chunk_id": chunk_id,
            "text": text,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": duration_sec,
            "word_count": word_count,
            "wpm": wpm,
            "filler_count": filler_count,
            "filler_ratio": filler_count / max(word_count, 1),
            "mean_confidence": float(np.mean(confidences)) if confidences else np.nan,
        })

    chunk_df = pd.DataFrame(chunk_rows)

    # add pause_before_sec and pause_after_sec
    # pause_before = []
    # pause_after = []

    # for i in range(len(chunk_df)):
    #     if i == 0:
    #         pause_before.append(0.0)
    #     else:
    #         gap = chunk_df.iloc[i]["start_sec"] - chunk_df.iloc[i - 1]["end_sec"]
    #         pause_before.append(max(0.0, float(gap)))

    #     if i == len(chunk_df) - 1:
    #         pause_after.append(0.0)
    #     else:
    #         gap = chunk_df.iloc[i + 1]["start_sec"] - chunk_df.iloc[i]["end_sec"]
    #         pause_after.append(max(0.0, float(gap)))

    # chunk_df["pause_before_sec"] = pause_before
    # chunk_df["pause_after_sec"] = pause_after

    return chunk_df


if __name__ == "__main__":
    audio_path = "evan_test.wav"
    
    # cache_base = Path(audio_path).stem

    # df.to_csv(f"{cache_base}_words.csv", index=False)

    # with open(f"{cache_base}_transcript.txt", "w") as f:
    #     f.write(transcript)
    
    word_df = pd.read_csv("evan_test_words.csv")
    
    with open("evan_test_transcript.txt", "r") as f:
        transcript = f.read()
    
    overall_stats = compute_overall_transcript_metrics(word_df)
    sentence_df = build_sentence_chunks(word_df)
    
    print("====== Words ======\n")
    print(word_df.head())
    print("===================\n")
    
    print("====== Transcript ======\n")
    print(transcript)
    print("========================\n")
    
    print("====== Overall Metrics ======\n")
    print(overall_stats)
    print("=============================\n")
    
    print("====== Sentences ======\n")
    print(sentence_df[["chunk_id", 
                       "start_sec", 
                       "end_sec", 
                       "duration_sec", 
                       "word_count", 
                       "wpm"]].head())
    print("=======================\n")
    
    
