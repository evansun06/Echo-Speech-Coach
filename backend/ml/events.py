from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# Small filler list for a simple v1 lexical hesitation signal.
FILLER_WORDS = {"um", "uh", "erm", "ah", "like"}


def _normalize_word(text: str) -> str:
    """Lowercase and remove punctuation so lexical rules are consistent."""
    if pd.isna(text):
        return ""
    return re.sub(r"[^\w']", "", str(text).lower()).strip()


def _compute_local_wpm(df: pd.DataFrame, window_sec: float = 1.5) -> pd.Series:
    """
    Compute local speaking rate around each word.
    Window is centered on the word midpoint: [mid - window_sec, mid + window_sec].
    """
    mids = (df["start_sec"].to_numpy(dtype=float) + df["end_sec"].to_numpy(dtype=float)) * 0.5
    starts = df["start_sec"].to_numpy(dtype=float)
    ends = df["end_sec"].to_numpy(dtype=float)
    durations = df["effective_duration"].to_numpy(dtype=float)

    local_wpm = np.zeros(len(df), dtype=float)
    for i, mid in enumerate(mids):
        left = mid - window_sec
        right = mid + window_sec

        # Keep words that overlap the local time window.
        mask = (ends > left) & (starts < right)
        count = int(mask.sum())
        speaking_time_min = float(durations[mask].sum()) / 60.0
        speaking_time_min = max(speaking_time_min, 1e-6)

        local_wpm[i] = count / speaking_time_min

    return pd.Series(local_wpm, index=df.index)


def _build_word_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Create boolean per-word flags used to build segment events."""
    out = df.copy()

    # Minimal baselines: only what we need for simple adaptive thresholds.
    median_effective_duration = float(out["effective_duration"].median())
    median_local_wpm = float(out["local_wpm"].median())
    median_confidence = float(out["confidence"].median())
    median_pitch_std = float(out["os_pitch_std"].median())
    median_loudness = float(out["os_loudness_mean"].median())
    median_attention = float(out["mp_forward_attention_ratio"].median())
    median_posture = float(out["mp_posture_deviation"].median())
    median_fidget = float(out["mp_fidget_index"].median())
    median_expressiveness = float(out["mp_expressiveness_score"].median())

    # Word-level lexical/event primitives.
    out["flag_filler_word"] = out["word_norm"].isin(FILLER_WORDS)
    out["flag_hesitation_pause"] = out["gap_before_sec"] >= 0.45
    out["flag_stretched_word"] = out["effective_duration"] >= (1.8 * median_effective_duration)

    # Repeat rule: same normalized word appears back-to-back within <= 1.0 sec.
    prev_norm = out["word_norm"].shift(1).fillna("")
    prev_start = out["start_sec"].shift(1)
    out["flag_stuttering_repeat"] = (
        (out["word_norm"] != "")
        & (out["word_norm"] == prev_norm)
        & ((out["start_sec"] - prev_start).fillna(np.inf) <= 1.0)
    )

    # Simple local rate events.
    out["flag_rushing"] = out["local_wpm"] >= (1.25 * median_local_wpm)
    out["flag_slow_delivery"] = out["local_wpm"] <= (0.75 * median_local_wpm)

    # Combined hesitation flag makes later merging easier.
    out["flag_hesitation"] = (
        out["flag_filler_word"] | out["flag_hesitation_pause"] | out["flag_stretched_word"]
    )

    # -------------------------
    # Positive event primitives
    # -------------------------
    # Confident delivery: high STT confidence + healthy loudness + smooth flow + moderate pitch variation.
    out["flag_confident_delivery"] = (
        out["confidence"].fillna(-np.inf) >= median_confidence
    ) & (
        out["os_loudness_mean"].fillna(-np.inf) >= median_loudness
    ) & (
        out["gap_before_sec"].fillna(np.inf) <= 0.20
    ) & (
        out["os_pitch_std"].fillna(np.inf).between(0.6 * median_pitch_std, 1.6 * median_pitch_std)
    )

    # Engaged presence: strong attention with stable body/fidget signals.
    out["flag_engaged_presence"] = (
        out["mp_forward_attention_ratio"].fillna(-np.inf) >= median_attention
    ) & (
        out["mp_posture_deviation"].fillna(np.inf) <= median_posture
    ) & (
        out["mp_fidget_index"].fillna(np.inf) <= median_fidget
    )

    # Expressive moment: above-normal visual expressiveness with non-flat pitch movement.
    out["flag_expressive_moment"] = (
        out["mp_expressiveness_score"].fillna(-np.inf) >= median_expressiveness
    ) & (
        out["os_pitch_std"].fillna(-np.inf) >= median_pitch_std
    ) & (
        out["os_pitch_std"].fillna(np.inf) <= 2.0 * median_pitch_std
    )

    # Steady pacing: local speed stays near personal median, low pause spikes, and no immediate hesitation/stutter.
    out["flag_steady_pacing"] = (
        (out["local_wpm"] - median_local_wpm).abs() <= (0.15 * max(median_local_wpm, 1e-6))
    ) & (
        out["gap_before_sec"].fillna(np.inf) < 0.25
    ) & (
        ~out["flag_hesitation"]
    ) & (
        ~out["flag_stuttering_repeat"]
    )

    return out


def _merge_flags_to_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge consecutive flagged words into event segments.
    Output is one row per event segment.
    """
    event_map: Dict[str, str] = {
        "hesitation": "flag_hesitation",
        "stuttering": "flag_stuttering_repeat",
        "rushing": "flag_rushing",
        "slow_delivery": "flag_slow_delivery",
        "confident_delivery": "flag_confident_delivery",
        "engaged_presence": "flag_engaged_presence",
        "expressive_moment": "flag_expressive_moment",
        "steady_pacing": "flag_steady_pacing",
    }

    event_rows: List[Dict] = []
    event_id = 1

    for event_type, flag_col in event_map.items():
        flagged_idx = df.index[df[flag_col].fillna(False)].tolist()
        if not flagged_idx:
            continue

        # Group consecutive row indices into contiguous segments.
        groups: List[List[int]] = []
        curr: List[int] = [flagged_idx[0]]
        for idx in flagged_idx[1:]:
            if idx == curr[-1] + 1:
                curr.append(idx)
            else:
                groups.append(curr)
                curr = [idx]
        groups.append(curr)

        for g in groups:
            seg = df.loc[g]

            # Keep tempo/body-style events only when at least 3 words are consecutive.
            if event_type in {"rushing", "slow_delivery", "engaged_presence", "steady_pacing"} and len(seg) < 3:
                continue
            # Keep other positive events at minimum 2 words to reduce one-word noise.
            if event_type in {"confident_delivery", "expressive_moment"} and len(seg) < 2:
                continue

            start_sec = float(seg["start_sec"].min())
            end_sec = float(seg["end_sec"].max())
            word_start_idx = int(seg["word_idx"].min())
            word_end_idx = int(seg["word_idx"].max())
            word_count = int(len(seg))
            text_span = " ".join(seg["word"].astype(str).tolist())

            # Very simple deterministic confidence score.
            confidence = 0.65
            if word_count >= 3:
                confidence += 0.10
            midpoint_ratio = float((seg["alignment_method"] == "midpoint_fallback").mean())
            if midpoint_ratio > 0.5:
                confidence -= 0.10
            # Small bonus for positive events sustained over longer segments.
            if event_type in {"confident_delivery", "engaged_presence", "expressive_moment", "steady_pacing"} and word_count >= 4:
                confidence += 0.05
            confidence = float(max(0.0, min(1.0, confidence)))

            event_rows.append(
                {
                    "event_id": event_id,
                    "event_type": event_type,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "word_start_idx": word_start_idx,
                    "word_end_idx": word_end_idx,
                    "word_count": word_count,
                    "text_span": text_span,
                    "confidence": confidence,
                }
            )
            event_id += 1

    events_df = pd.DataFrame(event_rows)
    if not events_df.empty:
        events_df = events_df.sort_values(["start_sec", "event_type"]).reset_index(drop=True)

    return events_df


def compute_events(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Main DataFrame-based API.
    Returns:
    - word_flags_df: aligned words + derived columns + flag columns
    - events_df: merged segment-level events
    """
    work = df.copy()

    # Basic ordering and ids so segment boundaries are deterministic.
    work = work.sort_values(["start_sec", "end_sec"]).reset_index(drop=True)
    work["word_idx"] = np.arange(len(work), dtype=int)

    # Lightweight preprocessing: normalize token and safe duration for math.
    work["word_norm"] = work["word"].apply(_normalize_word)
    work["effective_duration"] = work["duration_sec"].fillna(0.0).clip(lower=0.05)

    # Local speaking-rate feature used by rushing/slow rules.
    work["local_wpm"] = _compute_local_wpm(work, window_sec=1.5)

    # Build per-word flags then merge to event segments.
    word_flags_df = _build_word_flags(work)
    events_df = _merge_flags_to_events(word_flags_df)

    return word_flags_df, events_df


def main() -> None:
    """Read aligned.csv, compute events, and write simple outputs."""
    aligned_df = pd.read_csv("aligned.csv")
    word_flags_df, events_df = compute_events(aligned_df)

    word_flags_df.to_csv("word_flags.csv", index=False)
    events_df.to_csv("events.csv", index=False)

    print(f"Aligned rows: {len(aligned_df)}")
    print(f"Word-flag rows: {len(word_flags_df)}")
    print(f"Events: {len(events_df)}")
    if not events_df.empty:
        print(events_df["event_type"].value_counts())
        print(events_df.head())
    else:
        print("No events detected.")


if __name__ == "__main__":
    main()
