"""Interpretable probable-source classification for detected pollution spikes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.config import EPSILON, MIN_GROUP_SIZE


SOURCE_LABELS = {
    "crop_score": "Crop Burning",
    "vehicular_score": "Vehicular Emissions",
    "industrial_score": "Industrial Emissions",
    "dust_score": "Dust Storm",
}


def _normalised_signal(row: pd.Series, pollutant: str) -> float:
    score = row.get(f"{pollutant}_score", np.nan)
    threshold = row.get(f"{pollutant}_moderate_threshold", np.nan)
    if pd.isna(score) or pd.isna(threshold) or threshold <= 0:
        return 0.0
    return float(max(score, 0.0) / max(threshold, EPSILON))


def fit_dust_ratio_reference(train_df: pd.DataFrame) -> dict[str, Any]:
    """Learn PM10/PM2.5 ratio references from training data only."""
    if not {"pm10", "pm2_5"}.issubset(train_df.columns):
        return {"lookups": {}, "min_group_size": 0}

    ratio_df = train_df.loc[train_df["pm2_5"] > EPSILON, ["city", "season", "pm10", "pm2_5"]].copy()
    ratio_df["pm10_pm2_5_ratio"] = ratio_df["pm10"] / ratio_df["pm2_5"].clip(lower=EPSILON)
    if ratio_df.empty:
        return {"lookups": {}, "min_group_size": 0}

    def build_table(group_cols: list[str], level: str) -> pd.DataFrame:
        records: list[dict[str, Any]] = []
        grouped = ratio_df.groupby(group_cols, sort=False, dropna=False) if group_cols else [((), ratio_df)]
        for keys, group in grouped:
            if not isinstance(keys, tuple):
                keys = (keys,)
            payload = dict(zip(group_cols, keys, strict=False))
            ratios = group["pm10_pm2_5_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
            if ratios.empty:
                continue
            records.append(
                {
                    **payload,
                    "ratio_threshold": float(ratios.quantile(0.95)),
                    "n_obs": int(ratios.shape[0]),
                    "level": level,
                }
            )
        return pd.DataFrame.from_records(records)

    tables = {
        "city_season": build_table(["city", "season"], "city_season"),
        "city": build_table(["city"], "city"),
        "global_season": build_table(["season"], "global_season"),
        "global": build_table([], "global"),
    }

    def lookup(table: pd.DataFrame, key_cols: list[str]) -> dict[tuple[Any, ...], dict[str, Any]]:
        return {
            tuple(row[col] for col in key_cols): row
            for row in table.to_dict("records")
        }

    return {
        "tables": tables,
        "lookups": {
            "city_season": lookup(tables["city_season"], ["city", "season"]),
            "city": lookup(tables["city"], ["city"]),
            "global_season": lookup(tables["global_season"], ["season"]),
            "global": lookup(tables["global"], []),
        },
        "min_group_size": min(MIN_GROUP_SIZE, max(3, len(ratio_df))),
    }


def _lookup_ratio_threshold(city: Any, season: Any, reference: dict[str, Any]) -> dict[str, Any] | None:
    if not reference.get("lookups"):
        return None
    min_group_size = reference["min_group_size"]
    candidates = (
        ("city_season", (city, season)),
        ("city", (city,)),
        ("global_season", (season,)),
        ("global", ()),
    )
    for level, key in candidates:
        row = reference["lookups"][level].get(key)
        if row is None:
            continue
        if level == "global" or row["n_obs"] >= min_group_size:
            return row
    return None


def attach_dust_ratio_thresholds(spikes_df: pd.DataFrame, reference: dict[str, Any]) -> pd.DataFrame:
    """Attach training-derived dust-ratio references to spike rows."""
    result = spikes_df.copy()
    if result.empty:
        result["pm10_pm2_5_ratio"] = pd.Series(dtype=float)
        result["dust_ratio_threshold"] = pd.Series(dtype=float)
        result["dust_ratio_threshold_level"] = pd.Series(dtype="object")
        return result

    if not {"pm10", "pm2_5"}.issubset(result.columns):
        result["pm10_pm2_5_ratio"] = np.nan
        result["dust_ratio_threshold"] = np.nan
        result["dust_ratio_threshold_level"] = pd.NA
        return result

    result["pm10_pm2_5_ratio"] = result["pm10"] / result["pm2_5"].clip(lower=EPSILON)
    lookups = [
        _lookup_ratio_threshold(city, season, reference)
        for city, season in zip(result["city"], result["season"], strict=False)
    ]
    result["dust_ratio_threshold"] = [
        row["ratio_threshold"] if row is not None else np.nan for row in lookups
    ]
    result["dust_ratio_threshold_level"] = [
        row["level"] if row is not None else pd.NA for row in lookups
    ]
    return result


def compute_source_scores(row: pd.Series) -> dict[str, float]:
    """Calculate interpretable source scores from normalised pollutant evidence."""
    nh3 = _normalised_signal(row, "nh3")
    co = _normalised_signal(row, "co")
    no = _normalised_signal(row, "no")
    no2 = _normalised_signal(row, "no2")
    so2 = _normalised_signal(row, "so2")
    pm10 = _normalised_signal(row, "pm10")
    pm2_5 = _normalised_signal(row, "pm2_5")

    crop_score = min(nh3, co) if nh3 and co else 0.0
    vehicular_score = min(no, no2) if no and no2 else 0.0
    industrial_score = so2

    dust_score = 0.0
    ratio = row.get("pm10_pm2_5_ratio", np.nan)
    ratio_threshold = row.get("dust_ratio_threshold", np.nan)
    if (
        pd.notna(ratio)
        and pd.notna(ratio_threshold)
        and ratio_threshold > EPSILON
        and ratio > ratio_threshold
    ):
        ratio_excess = ratio / ratio_threshold - 1.0
        dust_score = max(pm10 - pm2_5, 0.0) + float(ratio_excess)

    primary_scores = [crop_score, vehicular_score, industrial_score, dust_score]
    scores_desc = sorted(primary_scores, reverse=True)
    second_best = scores_desc[1] if len(scores_desc) > 1 else 0.0
    mixed_score = second_best if sum(score >= 1.0 for score in primary_scores) >= 2 else 0.0

    return {
        "crop_score": float(crop_score),
        "vehicular_score": float(vehicular_score),
        "industrial_score": float(industrial_score),
        "dust_score": float(dust_score),
        "mixed_score": float(mixed_score),
    }


def classify_source(row: pd.Series) -> tuple[str, str]:
    """Return a probable source label and confidence."""
    scores = {name: float(row[name]) for name in SOURCE_LABELS}
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_name, top_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    high_sources = [name for name, score in ranked if score >= 1.0]

    if top_score <= 0 or len(high_sources) >= 2 or second_score >= 0.8 * top_score:
        return "Mixed Sources", "Low"

    if top_score >= 1.5 and second_score <= 0.5 * top_score:
        confidence = "High"
    elif top_score >= 1.0:
        confidence = "Medium"
    else:
        confidence = "Low"
    return SOURCE_LABELS[top_name], confidence


def build_source_reason(row: pd.Series) -> str:
    """Explain the probable source classification in plain language."""
    source = row["probable_source"]
    if source == "Crop Burning":
        return "Probable crop-burning fingerprint: NH3 and CO both rose unusually together."
    if source == "Vehicular Emissions":
        return "Probable traffic fingerprint: NO and NO2 both rose unusually together."
    if source == "Industrial Emissions":
        return "Probable industrial fingerprint: SO2 showed the strongest unusual increase."
    if source == "Dust Storm":
        return (
            "Probable dust fingerprint: PM10 was unusually elevated and much higher than PM2.5 "
            "relative to the learned local ratio."
        )
    return "Overlapping pollutant fingerprints were present, so no single probable source dominated."


def classify_all_spikes(spikes_df: pd.DataFrame) -> pd.DataFrame:
    """Classify every detected spike with probable-source evidence."""
    output_cols = [
        "timestamp",
        "city",
        "severity",
        "dominant_pollutant",
        "combined_anomaly_score",
        "probable_source",
        "source_confidence",
        "source_reason",
        "crop_score",
        "vehicular_score",
        "industrial_score",
        "dust_score",
        "mixed_score",
    ]
    if spikes_df.empty:
        return pd.DataFrame(columns=output_cols)

    classified = spikes_df.copy()
    score_frame = classified.apply(lambda row: pd.Series(compute_source_scores(row)), axis=1)
    classified = pd.concat([classified, score_frame], axis=1)
    source_labels = classified.apply(classify_source, axis=1)
    classified["probable_source"] = [item[0] for item in source_labels]
    classified["source_confidence"] = [item[1] for item in source_labels]
    classified["source_reason"] = classified.apply(build_source_reason, axis=1)

    leading = output_cols
    trailing = [col for col in classified.columns if col not in leading]
    return classified[leading + trailing].sort_values("timestamp").reset_index(drop=True)

