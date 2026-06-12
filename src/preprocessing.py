"""Preprocessing utilities for heterogeneous air quality files."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from src.config import CANONICAL_POLLUTANTS
from src.data_loader import detect_column_roles, parse_datetime_series


def _snake_case(name: str) -> str:
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", cleaned).strip("_")


def _canonicalise_pollutant_name(name: str) -> str:
    aliases = {
        "pm25": "pm2_5",
        "pm_25": "pm2_5",
        "pm2_5": "pm2_5",
        "pm_2_5": "pm2_5",
        "pm2p5": "pm2_5",
        "pm10": "pm10",
        "pm_10": "pm10",
        "no": "no",
        "no2": "no2",
        "no_2": "no2",
        "so2": "so2",
        "so_2": "so2",
        "nh3": "nh3",
        "nh_3": "nh3",
        "co": "co",
        "o3": "o3",
        "o_3": "o3",
        "ozone": "o3",
        "components_pm25": "pm2_5",
        "components_pm_25": "pm2_5",
        "components_pm2_5": "pm2_5",
        "components_pm_2_5": "pm2_5",
        "components_pm2p5": "pm2_5",
        "components_pm10": "pm10",
        "components_pm_10": "pm10",
        "components_no": "no",
        "components_no2": "no2",
        "components_no_2": "no2",
        "components_so2": "so2",
        "components_so_2": "so2",
        "components_nh3": "nh3",
        "components_nh_3": "nh3",
        "components_co": "co",
        "components_o3": "o3",
        "components_o_3": "o3",
        "components_ozone": "o3",
    }
    return aliases.get(name, name)


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with safe lowercase snake_case column names."""
    renamed = [_canonicalise_pollutant_name(_snake_case(col)) for col in df.columns]
    seen: dict[str, int] = {}
    unique_names: list[str] = []
    for name in renamed:
        seen[name] = seen.get(name, 0) + 1
        unique_names.append(name if seen[name] == 1 else f"{name}_{seen[name]}")

    result = df.copy()
    result.columns = unique_names
    return result


def parse_timestamp_column(df: pd.DataFrame) -> pd.DataFrame:
    """Detect and parse a timestamp column into canonical ``timestamp``."""
    result = df.copy()
    roles = detect_column_roles(result)
    timestamp_col = roles["timestamp_column"]

    if timestamp_col is None and {"date", "time"}.issubset(result.columns):
        result["timestamp"] = pd.to_datetime(
            result["date"].astype(str).str.strip() + " " + result["time"].astype(str).str.strip(),
            errors="coerce",
        )
        return result

    if timestamp_col is None:
        raise ValueError(
            "Missing timestamp column. Add a parseable date/time field such as "
            "'timestamp', 'datetime', or separate 'date' and 'time' columns."
        )

    result["timestamp"] = parse_datetime_series(result[timestamp_col])
    return result


def detect_city_column(df: pd.DataFrame) -> str:
    """Return the detected city column name, or raise a clear error."""
    roles = detect_column_roles(df)
    city_col = roles["city_column"]
    if city_col is None:
        raise ValueError(
            "Missing city column. Expected a field such as 'city', 'city_name', "
            "'location', or 'station_city'."
        )
    return city_col


def detect_pollutant_columns(df: pd.DataFrame) -> list[str]:
    """Return canonical pollutant columns present in ``df``."""
    pollutant_cols: list[str] = []
    for col in df.columns:
        canonical = _canonicalise_pollutant_name(_snake_case(col))
        if canonical in CANONICAL_POLLUTANTS and canonical == col:
            pollutant_cols.append(col)
    return pollutant_cols


def assign_season(month: int | float | None) -> str | float:
    """Map calendar month to the competition-friendly seasonal buckets."""
    if month in (12, 1, 2):
        return "Winter"
    if month in (3, 4, 5):
        return "Spring"
    if month in (6, 7, 8, 9):
        return "Monsoon/Summer"
    if month in (10, 11):
        return "Autumn"
    return np.nan


def _coerce_numeric(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


def preprocess_air_quality_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """Standardise, enrich, and impute a raw air-quality dataframe."""
    label = "train" if is_train else "test"
    processed = standardize_column_names(df)
    processed = parse_timestamp_column(processed)

    invalid_dates = int(processed["timestamp"].isna().sum())
    if invalid_dates:
        print(f"[preprocess] Warning: dropping {invalid_dates} {label} row(s) with invalid timestamps.")
        processed = processed.loc[processed["timestamp"].notna()].copy()
    if processed.empty:
        raise ValueError(f"No valid timestamped rows remain in the {label} dataset after parsing.")

    city_col = detect_city_column(processed)
    if city_col != "city":
        processed = processed.rename(columns={city_col: "city"})
    processed["city"] = (
        processed["city"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .replace("", "Unknown")
    )

    pollutant_cols = detect_pollutant_columns(processed)
    if not pollutant_cols:
        raise ValueError(
            "No pollutant columns detected. Expected one or more of: "
            "PM2.5, PM10, NO, NO2, SO2, NH3, CO, or O3/ozone."
        )

    for col in pollutant_cols:
        processed[col] = _coerce_numeric(processed[col])

    all_missing_pollutants = [col for col in pollutant_cols if processed[col].notna().sum() == 0]
    if all_missing_pollutants:
        print(
            "[preprocess] Warning: dropping pollutant column(s) with no valid numeric values: "
            f"{all_missing_pollutants}"
        )
        processed = processed.drop(columns=all_missing_pollutants)
        pollutant_cols = [col for col in pollutant_cols if col not in all_missing_pollutants]
    if not pollutant_cols:
        raise ValueError(
            "Pollutant columns were detected by name, but none contained usable numeric values."
        )

    processed["hour"] = processed["timestamp"].dt.hour
    processed["day"] = processed["timestamp"].dt.day
    processed["month"] = processed["timestamp"].dt.month
    processed["year"] = processed["timestamp"].dt.year
    processed["date"] = processed["timestamp"].dt.date
    processed["season"] = processed["month"].map(assign_season)

    processed = processed.sort_values(["city", "timestamp"]).reset_index(drop=True)

    for col in pollutant_cols:
        processed[col] = processed.groupby("city", dropna=False)[col].transform(
            lambda series: series.interpolate(method="linear", limit_direction="both")
        )
        processed[col] = processed[col].fillna(
            processed.groupby("city", dropna=False)[col].transform("median")
        )
        processed[col] = processed[col].fillna(processed[col].median())

    return processed
