"""End-to-end SmogNet pipeline entry point."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.alert_generation import generate_alerts_dataframe
from src.anomaly_detection import (
    compute_rolling_baselines,
    detect_spikes,
    learn_anomaly_thresholds,
    score_test_data,
)
from src.config import (
    DEFAULT_ROLLING_WINDOW,
    OUTPUT_DATA_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from src.data_loader import inspect_data_files, load_air_quality_data
from src.preprocessing import detect_pollutant_columns, preprocess_air_quality_data
from src.source_classification import (
    attach_dust_ratio_thresholds,
    classify_all_spikes,
    fit_dust_ratio_reference,
)


def _ensure_dirs() -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)
    print(f"[save] {path.relative_to(path.parents[2]) if len(path.parents) >= 3 else path}: {len(df)} row(s)")


def main() -> None:
    """Run the complete raw-to-alert SmogNet workflow."""
    _ensure_dirs()
    print("[pipeline] Starting SmogNet pipeline")
    inspect_data_files(RAW_DATA_DIR)

    print("[pipeline] Loading raw data")
    try:
        raw_train, raw_test, split_metadata = load_air_quality_data(RAW_DATA_DIR)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"[pipeline] {exc}") from exc
    print(f"[pipeline] Split strategy: {split_metadata['split_strategy']}")
    ignored_generic = split_metadata.get("generic_train_files_ignored", [])
    if ignored_generic:
        print(
            "[pipeline] Using city-specific training files for city-aware baselines; "
            f"ignored generic training file(s): {ignored_generic}"
        )

    print("[pipeline] Preprocessing train/test data")
    train_df = preprocess_air_quality_data(raw_train, is_train=True)
    test_df = preprocess_air_quality_data(raw_test, is_train=False)

    pollutant_cols = [
        pollutant
        for pollutant in detect_pollutant_columns(train_df)
        if pollutant in test_df.columns
    ]
    if not pollutant_cols:
        raise ValueError(
            "No common pollutant columns remain after preprocessing train and test data. "
            "Check the raw files for consistent pollutant fields."
        )
    print(f"[pipeline] Pollutants used: {pollutant_cols}")

    _save_csv(train_df, PROCESSED_DATA_DIR / "train_processed.csv")
    _save_csv(test_df, PROCESSED_DATA_DIR / "test_processed.csv")

    print("[pipeline] Learning adaptive rolling baselines from training data only")
    train_scored, baseline_reference = compute_rolling_baselines(
        train_df, pollutant_cols, window=DEFAULT_ROLLING_WINDOW
    )
    thresholds = learn_anomaly_thresholds(train_scored, pollutant_cols)

    print("[pipeline] Scoring test data against learned baselines")
    test_scored = score_test_data(test_df, baseline_reference, pollutant_cols)
    _save_csv(train_scored, PROCESSED_DATA_DIR / "train_scored.csv")
    _save_csv(test_scored, PROCESSED_DATA_DIR / "test_scored.csv")

    print("[pipeline] Detecting spikes using learned score thresholds")
    detected_spikes = detect_spikes(test_scored, thresholds, pollutant_cols)
    if detected_spikes.empty:
        print(
            "[pipeline] Warning: no spikes detected at the learned 95th-percentile thresholds. "
            "Thresholds were not relaxed automatically to avoid inventing events."
        )
    _save_csv(detected_spikes, OUTPUT_DATA_DIR / "detected_spikes.csv")

    print("[pipeline] Classifying probable sources")
    dust_reference = fit_dust_ratio_reference(train_df)
    spikes_with_ratios = attach_dust_ratio_thresholds(detected_spikes, dust_reference)
    classified_spikes = classify_all_spikes(spikes_with_ratios)
    _save_csv(classified_spikes, OUTPUT_DATA_DIR / "classified_spikes.csv")

    print("[pipeline] Generating public alerts")
    public_alerts = generate_alerts_dataframe(classified_spikes)
    if public_alerts.empty:
        print("[pipeline] Warning: no public alerts generated because no classified spikes were available.")
    _save_csv(public_alerts, OUTPUT_DATA_DIR / "public_alerts.csv")
    print("[pipeline] Complete")


if __name__ == "__main__":
    main()
