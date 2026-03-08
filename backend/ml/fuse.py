from __future__ import annotations

import numpy as np
import pandas as pd


def align_word_features(
    words_df: pd.DataFrame,
    opensmile_df: pd.DataFrame,
    mediapipe_df: pd.DataFrame,
    min_word_duration: float = 0.05,
) -> pd.DataFrame:
    """
    Align OpenSMILE + MediaPipe interval/window features to each STT word row.

    Inputs:
    - words_df: must contain at least ["word", "start_sec", "end_sec"]
    - opensmile_df: must contain at least ["interval_start", "interval_end", ...features]
    - mediapipe_df: must contain at least ["start_sec", "end_sec", ...features]

    Output:
    - DataFrame with original word rows + aligned audio/visual features.
      Added columns:
      - gap_before_sec, gap_after_sec
      - alignment_method: "overlap" or "midpoint_fallback"
      - alignment_coverage: overlap / effective_word_duration in [0, 1]
    """
    words_df = words_df.copy()
    os_df = opensmile_df.copy()
    mp_df = mediapipe_df.copy()

    # Sort words by time so gap computation is deterministic.
    words_df = words_df.sort_values(["start_sec", "end_sec"]).reset_index(drop=True)

    # Pause/gap helpers between neighboring words (useful for hesitation logic later).
    words_df["gap_before_sec"] = 0.0
    words_df["gap_after_sec"] = 0.0
    if len(words_df) > 1:
        prev_end = words_df["end_sec"].shift(1)
        next_start = words_df["start_sec"].shift(-1)
        words_df["gap_before_sec"] = (words_df["start_sec"] - prev_end).clip(lower=0).fillna(0.0)
        words_df["gap_after_sec"] = (next_start - words_df["end_sec"]).clip(lower=0).fillna(0.0)

    # Standardize interval column names so both sources are treated the same.
    os_df = os_df.rename(columns={"interval_start": "start_sec", "interval_end": "end_sec"})

    # Only align numeric feature columns (skip time bounds).
    os_feature_cols = [
        c for c in os_df.columns
        if c not in {"start_sec", "end_sec"} and pd.api.types.is_numeric_dtype(os_df[c])
    ]
    mp_feature_cols = [
        c for c in mp_df.columns
        if c not in {"window_id", "start_sec", "end_sec"} and pd.api.types.is_numeric_dtype(mp_df[c])
    ]

    # Prefix feature names so source is explicit in the merged table.
    os_prefixed = {c: f"os_{c}" for c in os_feature_cols}
    mp_prefixed = {c: f"mp_{c}" for c in mp_feature_cols}
    os_df = os_df.rename(columns=os_prefixed)
    mp_df = mp_df.rename(columns=mp_prefixed)

    # Fast numpy views for repeated interval overlap calculations.
    os_start = os_df["start_sec"].to_numpy(dtype=float)
    os_end = os_df["end_sec"].to_numpy(dtype=float)
    mp_start = mp_df["start_sec"].to_numpy(dtype=float)
    mp_end = mp_df["end_sec"].to_numpy(dtype=float)

    os_values = {c: os_df[c].to_numpy(dtype=float) for c in os_prefixed.values()}
    mp_values = {c: mp_df[c].to_numpy(dtype=float) for c in mp_prefixed.values()}

    aligned_rows = []

    for row in words_df.itertuples(index=False):
        ws = float(row.start_sec)
        we = float(row.end_sec)
        wd = max(we - ws, min_word_duration)

        # Overlap amount between this word span and each source interval.
        os_overlap = np.maximum(0.0, np.minimum(we, os_end) - np.maximum(ws, os_start))
        mp_overlap = np.maximum(0.0, np.minimum(we, mp_end) - np.maximum(ws, mp_start))

        os_weight_sum = float(os_overlap.sum())
        mp_weight_sum = float(mp_overlap.sum())

        out = row._asdict()
        out["alignment_method"] = "overlap"
        out["alignment_coverage"] = min(1.0, (os_weight_sum + mp_weight_sum) / (2.0 * wd))

        # If overlap exists, compute weighted mean for each feature.
        if os_weight_sum > 0.0:
            for c, arr in os_values.items():
                out[c] = float(np.nansum(arr * os_overlap) / os_weight_sum)
        else:
            for c in os_values:
                out[c] = np.nan

        if mp_weight_sum > 0.0:
            for c, arr in mp_values.items():
                out[c] = float(np.nansum(arr * mp_overlap) / mp_weight_sum)
        else:
            for c in mp_values:
                out[c] = np.nan

        # Fallback: if either source had no overlap (very short or edge-timestamp words),
        # sample the closest interval by word midpoint.
        if os_weight_sum <= 0.0 or mp_weight_sum <= 0.0:
            mid = 0.5 * (ws + we)
            out["alignment_method"] = "midpoint_fallback"

            if os_weight_sum <= 0.0 and len(os_df) > 0:
                idx = int(np.argmin(np.abs(((os_start + os_end) * 0.5) - mid)))
                for c, arr in os_values.items():
                    out[c] = float(arr[idx]) if not np.isnan(arr[idx]) else np.nan

            if mp_weight_sum <= 0.0 and len(mp_df) > 0:
                idx = int(np.argmin(np.abs(((mp_start + mp_end) * 0.5) - mid)))
                for c, arr in mp_values.items():
                    out[c] = float(arr[idx]) if not np.isnan(arr[idx]) else np.nan

        aligned_rows.append(out)

    return pd.DataFrame(aligned_rows)


if __name__ == "__main__":
    try:
        from .stt_features import compute_overall_transcript_metrics, build_sentence_chunks
        from .os_features import SpeechFeatureExtractor
        from .mp_features import extract_frame_features, aggregate_windows
    except ImportError:
        from stt_features import compute_overall_transcript_metrics, build_sentence_chunks
        from os_features import SpeechFeatureExtractor
        from mp_features import extract_frame_features, aggregate_windows

    # # STT component (using existing word table artifact)
    # words_df = pd.read_csv("evan_test_words.csv")
    # stt_overall = compute_overall_transcript_metrics(words_df)
    # stt_sentence_df = build_sentence_chunks(words_df)

    # # OpenSMILE component
    # os_extractor = SpeechFeatureExtractor(interval_seconds=1.0)
    # os_interval_df = os_extractor.extract_interval_features("evan_test.wav")

    # # MediaPipe component
    # frame_df, _effective_fps = extract_frame_features("evan_test.mp4")
    # mp_window_df = aggregate_windows(frame_df, window_sec=2.0)
    
    words_df = pd.read_csv("evan_test_words.csv")
    opensmile_df = pd.read_csv("interval_features.csv")
    mediapipe_df = pd.read_csv("visual_outputs/window_features.csv")
        
    # Fusion alignment
    aligned_df = align_word_features(words_df, opensmile_df, mediapipe_df)
    aligned_df.to_csv("aligned.csv", index=False)
    
    # print("STT overall metrics:", stt_overall)
    # print("STT sentence rows:", len(stt_sentence_df))
    # print("OpenSMILE interval rows:", len(os_interval_df))
    # print("MediaPipe window rows:", len(mp_window_df))
    print("Aligned rows:", len(aligned_df))
    print(aligned_df.head())
