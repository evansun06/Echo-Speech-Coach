import opensmile
import pandas as pd
import numpy as np


class SpeechFeatureExtractor:
    def __init__(self, interval_seconds=2.0):
        self.interval_seconds = interval_seconds

        # openSMILE extractor
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.LowLevelDescriptors,
        )

    def extract_frame_features(self, wav_path):
        """
        Extract low-level frame-based features from the audio file.
        Returns a dataframe indexed by time.
        """
        df = self.smile.process_file(wav_path)

        # Keep only the columns we care about
        needed = [
            "F0semitoneFrom27.5Hz_sma3nz",    # pitch
            "Loudness_sma3",               # loudness
            "spectralFlux_sma3",           # spectral flux
        ]

        missing = [col for col in needed if col not in df.columns]
        if missing:
            raise ValueError(
                f"These expected openSMILE columns were not found: {missing}\n"
                f"Available columns include: {list(df.columns)[:20]} ..."
            )

        return df[needed].copy()

    def _get_time_seconds(self, df):
        """
        Convert the MultiIndex start times to seconds.
        openSMILE usually returns index like (file, start, end).
        """
        if isinstance(df.index, pd.MultiIndex):
            start_times = df.index.get_level_values(1)
        else:
            raise ValueError("Expected openSMILE dataframe to have a MultiIndex.")

        return np.array([t.total_seconds() for t in start_times])

    def extract_interval_features(self, wav_path):
        """
        Compute features over fixed intervals (e.g. every 2 seconds).
        Returns one row per interval.
        """
        df = self.extract_frame_features(wav_path)
        times = self._get_time_seconds(df)

        max_time = times.max()
        interval_rows = []

        start = 0.0
        while start < max_time:
            end = start + self.interval_seconds
            mask = (times >= start) & (times < end)
            chunk = df.loc[mask]

            if len(chunk) == 0:
                start = end
                continue

            # voicingFinalUnclipped_sma is commonly used as a voicing indicator/proxy
            voiced_vals = chunk["F0semitoneFrom27.5Hz_sma3nz"].replace(0, np.nan).notna().mean()

            # pitch: only use frames where pitch exists and is voiced
            pitch_vals = chunk["F0semitoneFrom27.5Hz_sma3nz"].replace(0, np.nan).dropna()

            loudness_vals = chunk["Loudness_sma3"].dropna()
            spectral_flux_vals = chunk["spectralFlux_sma3"].dropna()

            row = {
                "interval_start": start,
                "interval_end": end,
                "voiced_ratio": float(voiced_vals) if not np.isnan(voiced_vals) else np.nan,
                "pitch_mean": float(pitch_vals.mean()) if len(pitch_vals) > 0 else np.nan,
                "pitch_std": float(pitch_vals.std()) if len(pitch_vals) > 1 else np.nan,
                "loudness_mean": float(loudness_vals.mean()) if len(loudness_vals) > 0 else np.nan,
                "spectral_flux_mean": float(spectral_flux_vals.mean()) if len(spectral_flux_vals) > 0 else np.nan,
            }

            interval_rows.append(row)
            start = end

        return pd.DataFrame(interval_rows)

    def extract_overall_features(self, wav_path):
        """
        Compute overall speech-level features for the whole file.
        """
        df = self.extract_frame_features(wav_path)

        pitch_series = df["F0semitoneFrom27.5Hz_sma3nz"].replace(0, np.nan)
        voiced_ratio = pitch_series.notna().mean() if len(df) > 0 else np.nan
        
        pitch_vals = df["F0semitoneFrom27.5Hz_sma3nz"].replace(0, np.nan).dropna()
        loudness_vals = df["Loudness_sma3"].dropna()

        overall = {
            "pitch_std": float(pitch_vals.std()) if len(pitch_vals) > 1 else np.nan,
            "voiced_ratio": voiced_ratio,
            "loudness_mean": float(loudness_vals.mean()) if len(loudness_vals) > 0 else np.nan,
        }

        return overall


if __name__ == "__main__":
    wav_file = "evan_test.wav"   # change this

    extractor = SpeechFeatureExtractor(interval_seconds=1.0)

    interval_df = extractor.extract_interval_features(wav_file)
    overall_features = extractor.extract_overall_features(wav_file)

    print("\n=== Interval Features ===")
    print(interval_df)

    print("\n=== Overall Features ===")
    print(overall_features)

    # Optional: save interval features
    interval_df.to_csv("interval_features.csv", index=False)