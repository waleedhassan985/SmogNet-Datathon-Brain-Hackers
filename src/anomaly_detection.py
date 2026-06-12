"""Adaptive, training-derived anomaly detection for pollution spikes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from src.config import EPSILON, MIN_GROUP_SIZE


def _rolling_min_periods(series_length: int, window: int) -> int:
    candidate = max(6, min(window, series_length) // 3)
    return max(3, min(series_length, candidate))


def _rolling_stats(series: pd.Series, window: int) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    min_periods = _rolling_min_periods(len(series), window)
    rolling = series.rolling(window=window, min_periods=min_periods)
    median = rolling.median()
    q1 = rolling.quantile(0.25)
    q3 = rolling.quantile(0.75)
    iqr = q3 - q1
    return median, q1, q3, iqr


def _group_rolling_stat(
    df: pd.DataFrame,
    pollutant: str,
    group_cols: list[str],
    window: int,
    stat: str,
) -> pd.Series:
    def transform(series: pd.Series) -> pd.Series:
        median, q1, q3, iqr = _rolling_stats(series.astype(float), window)
        return {"median": median, "q1": q1, "q3": q3, "iqr": iqr}[stat]

    if not group_cols:
        return transform(df[pollutant])
    return df.groupby(group_cols, sort=False, dropna=False)[pollutant].transform(transform)


def _valid_baseline(median: pd.Series, iqr: pd.Series) -> pd.Series:
    return median.notna() & iqr.notna() & np.isfinite(iqr) & (iqr > 0)


def _build_reference_table(
    train_df: pd.DataFrame,
    pollutant_cols: list[str],
    group_cols: list[str],
    window: int,
    level_name: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []

    grouped: Iterable[tuple[Any, pd.DataFrame]]
    if group_cols:
        grouped = train_df.groupby(group_cols, sort=False, dropna=False)
    else:
        grouped = [((), train_df)]

    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_payload = dict(zip(group_cols, keys, strict=False))

        for pollutant in pollutant_cols:
            values = group[pollutant].dropna().astype(float)
            if values.empty:
                continue
            median, _, _, iqr = _rolling_stats(values.reset_index(drop=True), window)
            baseline_median = median.dropna().median()
            if pd.isna(baseline_median):
                baseline_median = values.median()

            positive_iqr = iqr[(iqr > 0) & iqr.notna()]
            baseline_iqr = positive_iqr.median()
            if pd.isna(baseline_iqr) or baseline_iqr <= 0:
                raw_iqr = values.quantile(0.75) - values.quantile(0.25)
                baseline_iqr = raw_iqr if pd.notna(raw_iqr) and raw_iqr > 0 else EPSILON

            records.append(
                {
                    **key_payload,
                    "pollutant": pollutant,
                    "baseline_median": float(baseline_median),
                    "baseline_iqr": float(max(baseline_iqr, EPSILON)),
                    "n_obs": int(values.shape[0]),
                    "level": level_name,
                }
            )

    return pd.DataFrame.from_records(records)


def _table_to_lookup(table: pd.DataFrame, key_cols: list[str]) -> dict[tuple[Any, ...], dict[str, Any]]:
    if table.empty:
        return {}
    lookup: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in table.to_dict("records"):
        key = tuple(row[col] for col in key_cols) + (row["pollutant"],)
        lookup[key] = row
    return lookup


def _lookup_baseline(
    city: Any,
    season: Any,
    pollutant: str,
    baseline_reference: dict[str, Any],
) -> dict[str, Any]:
    min_group_size = baseline_reference["min_group_size"]
    candidates = (
        ("city_season", (city, season, pollutant)),
        ("city", (city, pollutant)),
        ("global_season", (season, pollutant)),
        ("global", (pollutant,)),
    )
    for level, key in candidates:
        row = baseline_reference["lookups"][level].get(key)
        if row is None:
            continue
        if level == "global" or row["n_obs"] >= min_group_size:
            return row
    raise ValueError(f"No usable training baseline found for pollutant {pollutant!r}.")


def compute_rolling_baselines(
    train_df: pd.DataFrame,
    pollutant_cols: list[str],
    window: int = 72,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score training rows using leakage-safe rolling baselines and build references."""
    if train_df.empty:
        raise ValueError("Training data is empty; cannot learn rolling baselines.")
    if not pollutant_cols:
        raise ValueError("At least one pollutant column is required for anomaly detection.")

    scored = train_df.sort_values(["city", "timestamp"]).reset_index(drop=True).copy()
    min_group_size = min(MIN_GROUP_SIZE, max(3, len(scored)))

    city_season_sizes = scored.groupby(["city", "season"], dropna=False)["city"].transform("size")
    city_sizes = scored.groupby(["city"], dropna=False)["city"].transform("size")
    season_sizes = scored.groupby(["season"], dropna=False)["season"].transform("size")

    for pollutant in pollutant_cols:
        cs_median = _group_rolling_stat(scored, pollutant, ["city", "season"], window, "median")
        cs_iqr = _group_rolling_stat(scored, pollutant, ["city", "season"], window, "iqr")
        city_median = _group_rolling_stat(scored, pollutant, ["city"], window, "median")
        city_iqr = _group_rolling_stat(scored, pollutant, ["city"], window, "iqr")
        season_median = _group_rolling_stat(scored, pollutant, ["season"], window, "median")
        season_iqr = _group_rolling_stat(scored, pollutant, ["season"], window, "iqr")
        global_median = _group_rolling_stat(scored, pollutant, [], window, "median")
        global_iqr = _group_rolling_stat(scored, pollutant, [], window, "iqr")

        chosen_median = pd.Series(np.nan, index=scored.index, dtype=float)
        chosen_iqr = pd.Series(np.nan, index=scored.index, dtype=float)
        chosen_level = pd.Series(pd.NA, index=scored.index, dtype="object")

        selectors = (
            ("city_season", city_season_sizes >= min_group_size, cs_median, cs_iqr),
            ("city", city_sizes >= min_group_size, city_median, city_iqr),
            ("global_season", season_sizes >= min_group_size, season_median, season_iqr),
            ("global", pd.Series(True, index=scored.index), global_median, global_iqr),
        )
        for level, enough_data, median, iqr in selectors:
            mask = chosen_median.isna() & enough_data & _valid_baseline(median, iqr)
            chosen_median.loc[mask] = median.loc[mask]
            chosen_iqr.loc[mask] = iqr.loc[mask]
            chosen_level.loc[mask] = level

        if chosen_median.isna().any():
            chosen_median = chosen_median.fillna(scored[pollutant].median())
        if chosen_iqr.isna().any():
            fallback_iqr = scored[pollutant].quantile(0.75) - scored[pollutant].quantile(0.25)
            chosen_iqr = chosen_iqr.fillna(fallback_iqr if fallback_iqr > 0 else EPSILON)
            chosen_level = chosen_level.fillna("global")

        scored[f"{pollutant}_baseline_median"] = chosen_median
        scored[f"{pollutant}_baseline_iqr"] = chosen_iqr.clip(lower=EPSILON)
        scored[f"{pollutant}_baseline_level"] = chosen_level
        scored[f"{pollutant}_score"] = (
            scored[pollutant] - scored[f"{pollutant}_baseline_median"]
        ) / scored[f"{pollutant}_baseline_iqr"].clip(lower=EPSILON)

    baseline_tables = {
        "city_season": _build_reference_table(
            scored, pollutant_cols, ["city", "season"], window, "city_season"
        ),
        "city": _build_reference_table(scored, pollutant_cols, ["city"], window, "city"),
        "global_season": _build_reference_table(
            scored, pollutant_cols, ["season"], window, "global_season"
        ),
        "global": _build_reference_table(scored, pollutant_cols, [], window, "global"),
    }
    baseline_reference = {
        "tables": baseline_tables,
        "lookups": {
            "city_season": _table_to_lookup(baseline_tables["city_season"], ["city", "season"]),
            "city": _table_to_lookup(baseline_tables["city"], ["city"]),
            "global_season": _table_to_lookup(baseline_tables["global_season"], ["season"]),
            "global": _table_to_lookup(baseline_tables["global"], []),
        },
        "min_group_size": min_group_size,
        "window": window,
    }
    return scored, baseline_reference


def _threshold_table(
    scored_df: pd.DataFrame,
    pollutant_cols: list[str],
    group_cols: list[str],
    level_name: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    grouped = scored_df.groupby(group_cols, sort=False, dropna=False) if group_cols else [((), scored_df)]

    for keys, group in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_payload = dict(zip(group_cols, keys, strict=False))
        for pollutant in pollutant_cols:
            scores = group[f"{pollutant}_score"].dropna().astype(float)
            if scores.empty:
                continue
            moderate, high, severe = scores.quantile([0.95, 0.975, 0.99]).tolist()
            records.append(
                {
                    **key_payload,
                    "pollutant": pollutant,
                    "moderate_threshold": float(moderate),
                    "high_threshold": float(max(high, moderate)),
                    "severe_threshold": float(max(severe, high, moderate)),
                    "n_obs": int(scores.shape[0]),
                    "level": level_name,
                }
            )
    return pd.DataFrame.from_records(records)


def learn_anomaly_thresholds(train_scored_df: pd.DataFrame, pollutant_cols: list[str]) -> dict[str, Any]:
    """Learn quantile thresholds from training-score distributions only."""
    tables = {
        "city_season": _threshold_table(
            train_scored_df, pollutant_cols, ["city", "season"], "city_season"
        ),
        "city": _threshold_table(train_scored_df, pollutant_cols, ["city"], "city"),
        "global_season": _threshold_table(
            train_scored_df, pollutant_cols, ["season"], "global_season"
        ),
        "global": _threshold_table(train_scored_df, pollutant_cols, [], "global"),
    }
    min_group_size = min(MIN_GROUP_SIZE, max(3, len(train_scored_df)))
    return {
        "tables": tables,
        "lookups": {
            "city_season": _table_to_lookup(tables["city_season"], ["city", "season"]),
            "city": _table_to_lookup(tables["city"], ["city"]),
            "global_season": _table_to_lookup(tables["global_season"], ["season"]),
            "global": _table_to_lookup(tables["global"], []),
        },
        "min_group_size": min_group_size,
    }


def _lookup_thresholds(
    city: Any,
    season: Any,
    pollutant: str,
    thresholds: dict[str, Any],
) -> dict[str, Any]:
    min_group_size = thresholds["min_group_size"]
    candidates = (
        ("city_season", (city, season, pollutant)),
        ("city", (city, pollutant)),
        ("global_season", (season, pollutant)),
        ("global", (pollutant,)),
    )
    for level, key in candidates:
        row = thresholds["lookups"][level].get(key)
        if row is None:
            continue
        if level == "global" or row["n_obs"] >= min_group_size:
            return row
    raise ValueError(f"No usable learned thresholds found for pollutant {pollutant!r}.")


def score_test_data(
    test_df: pd.DataFrame,
    baseline_reference: dict[str, Any],
    pollutant_cols: list[str],
) -> pd.DataFrame:
    """Score test rows against training-derived baseline references only."""
    if test_df.empty:
        return test_df.copy()

    scored = test_df.sort_values(["city", "timestamp"]).reset_index(drop=True).copy()
    for pollutant in pollutant_cols:
        references = [
            _lookup_baseline(city, season, pollutant, baseline_reference)
            for city, season in zip(scored["city"], scored["season"], strict=False)
        ]
        scored[f"{pollutant}_baseline_median"] = [row["baseline_median"] for row in references]
        scored[f"{pollutant}_baseline_iqr"] = [
            max(float(row["baseline_iqr"]), EPSILON) for row in references
        ]
        scored[f"{pollutant}_baseline_level"] = [row["level"] for row in references]
        scored[f"{pollutant}_score"] = (
            scored[pollutant] - scored[f"{pollutant}_baseline_median"]
        ) / scored[f"{pollutant}_baseline_iqr"].clip(lower=EPSILON)
    return scored


def assign_severity(score: float, thresholds: dict[str, float]) -> str:
    """Assign severity from the dominant pollutant's learned thresholds."""
    if score >= thresholds["severe_threshold"]:
        return "severe"
    if score >= thresholds["high_threshold"]:
        return "high"
    if score >= thresholds["moderate_threshold"]:
        return "moderate"
    return "not_anomaly"


def build_anomaly_explanation(row: pd.Series) -> str:
    """Build a concise, interpretable explanation for a detected spike."""
    pollutant = row["dominant_pollutant"]
    score = row["combined_anomaly_score"]
    baseline_level = row.get(f"{pollutant}_baseline_level", "training-derived")
    return (
        f"{pollutant.upper()} is {score:.2f} robust-IQR units above its "
        f"{baseline_level.replace('_', ' ')} baseline for {row['city']} in {row['season']}."
    )


def detect_spikes(
    test_scored_df: pd.DataFrame,
    thresholds: dict[str, Any],
    pollutant_cols: list[str],
) -> pd.DataFrame:
    """Flag rows where at least one pollutant exceeds its learned threshold."""
    if test_scored_df.empty:
        return pd.DataFrame()

    scored = test_scored_df.copy()
    for pollutant in pollutant_cols:
        threshold_rows = [
            _lookup_thresholds(city, season, pollutant, thresholds)
            for city, season in zip(scored["city"], scored["season"], strict=False)
        ]
        for threshold_name in ("moderate_threshold", "high_threshold", "severe_threshold"):
            scored[f"{pollutant}_{threshold_name}"] = [
                row[threshold_name] for row in threshold_rows
            ]
        scored[f"{pollutant}_threshold_level"] = [row["level"] for row in threshold_rows]
        scored[f"{pollutant}_is_anomalous"] = (
            scored[f"{pollutant}_score"] >= scored[f"{pollutant}_moderate_threshold"]
        )

    anomaly_flag_cols = [f"{pollutant}_is_anomalous" for pollutant in pollutant_cols]
    scored["is_anomaly"] = scored[anomaly_flag_cols].any(axis=1)
    spikes = scored.loc[scored["is_anomaly"]].copy()
    if spikes.empty:
        return spikes

    score_cols = [f"{pollutant}_score" for pollutant in pollutant_cols]
    anomalous_score_frame = spikes[score_cols].copy()
    for pollutant in pollutant_cols:
        anomalous_score_frame.loc[
            ~spikes[f"{pollutant}_is_anomalous"], f"{pollutant}_score"
        ] = np.nan

    spikes["combined_anomaly_score"] = anomalous_score_frame.max(axis=1)
    spikes["dominant_pollutant"] = (
        anomalous_score_frame
        .idxmax(axis=1)
        .str.replace("_score", "", regex=False)
    )

    def row_severity(row: pd.Series) -> str:
        severity_rank = {"not_anomaly": 0, "moderate": 1, "high": 2, "severe": 3}
        row_severities: list[str] = []
        for pollutant in pollutant_cols:
            row_severities.append(
                assign_severity(
                    float(row[f"{pollutant}_score"]),
                    {
                        "moderate_threshold": float(row[f"{pollutant}_moderate_threshold"]),
                        "high_threshold": float(row[f"{pollutant}_high_threshold"]),
                        "severe_threshold": float(row[f"{pollutant}_severe_threshold"]),
                    },
                )
            )
        return max(row_severities, key=lambda severity: severity_rank[severity])

    spikes["severity"] = spikes.apply(row_severity, axis=1)
    spikes["anomaly_explanation"] = spikes.apply(build_anomaly_explanation, axis=1)

    leading_cols = [
        "timestamp",
        "city",
        "season",
        *pollutant_cols,
        *score_cols,
        "combined_anomaly_score",
        "dominant_pollutant",
        "severity",
        "anomaly_explanation",
    ]
    remaining_cols = [col for col in spikes.columns if col not in leading_cols]
    return spikes[leading_cols + remaining_cols].sort_values("timestamp").reset_index(drop=True)
