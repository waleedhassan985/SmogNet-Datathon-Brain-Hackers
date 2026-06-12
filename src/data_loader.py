"""Utilities for discovering, auditing, and loading raw air quality files."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import DEFAULT_TEST_FRACTION


SUPPORTED_SUFFIXES = {".csv", ".xlsx"}


def _normalise_name(name: str) -> str:
    normalised = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", normalised).strip("_")


def _read_tabular_file(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path, engine="openpyxl")
    raise ValueError(f"Unsupported file type for {path.name!r}. Expected CSV or XLSX.")


def _infer_dayfirst(series: pd.Series) -> bool:
    dayfirst_votes = 0
    monthfirst_votes = 0
    pattern = re.compile(r"^\s*(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})")
    for value in series.dropna().astype(str).head(500):
        match = pattern.match(value)
        if not match:
            continue
        first, second = int(match.group(1)), int(match.group(2))
        if first > 12 and second <= 12:
            dayfirst_votes += 1
        elif second > 12 and first <= 12:
            monthfirst_votes += 1
    return dayfirst_votes > monthfirst_votes


def parse_datetime_series(series: pd.Series) -> pd.Series:
    """Parse datetimes while inferring day-first strings when the data reveal it."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce")
    return pd.to_datetime(
        series,
        errors="coerce",
        dayfirst=_infer_dayfirst(series),
        format="mixed",
    )


def _infer_city_from_path(path: Path) -> str | None:
    known_cities = ("islamabad", "karachi", "lahore", "peshawar", "quetta")
    stem = path.stem.lower()
    for city in known_cities:
        if stem.startswith(city):
            return city.title()
    return None


def _read_with_inferred_city(path: Path) -> pd.DataFrame:
    df = _read_tabular_file(path)
    roles = detect_column_roles(df)
    inferred_city = _infer_city_from_path(path)
    if roles["city_column"] is None and inferred_city is not None:
        df = df.copy()
        df["city"] = inferred_city
    return df


def list_data_files(raw_dir: str | Path = "data/raw") -> list[Path]:
    """Return CSV/XLSX files found anywhere under ``raw_dir``."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return []
    return sorted(
        [
            path
            for path in raw_path.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        ]
    )


def identify_train_test_files(files: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    """Split filenames into likely train, test, and unclassified groups."""
    train_patterns = ("train", "training")
    test_patterns = ("test", "testing", "valid", "validation", "eval", "evaluation")

    train_files: list[Path] = []
    test_files: list[Path] = []
    other_files: list[Path] = []

    for path in files:
        searchable_text = " ".join(part.lower() for part in path.parts)
        if any(pattern in searchable_text for pattern in train_patterns):
            train_files.append(path)
        elif any(pattern in searchable_text for pattern in test_patterns):
            test_files.append(path)
        else:
            other_files.append(path)

    return train_files, test_files, other_files


def detect_column_roles(df: pd.DataFrame) -> dict[str, Any]:
    """Heuristically detect timestamp, city, and pollutant columns."""
    normalised_columns = {_normalise_name(col): col for col in df.columns}

    timestamp_priority = (
        "timestamp",
        "datetime",
        "date_time",
        "measurement_time",
        "measured_at",
        "reading_time",
        "date",
        "time",
    )
    city_priority = (
        "city",
        "city_name",
        "station_city",
        "location",
        "site",
        "station",
    )

    timestamp_column = next(
        (normalised_columns[name] for name in timestamp_priority if name in normalised_columns),
        None,
    )

    if timestamp_column is None:
        datetime_like_cols = [
            col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])
        ]
        if datetime_like_cols:
            timestamp_column = datetime_like_cols[0]

    if timestamp_column is None:
        parseable_candidates: list[tuple[float, Any]] = []
        for col in df.columns:
            normalised = _normalise_name(col)
            if not any(token in normalised for token in ("date", "time")):
                continue
            parsed = pd.to_datetime(df[col], errors="coerce")
            success_rate = float(parsed.notna().mean())
            if success_rate >= 0.60:
                parseable_candidates.append((success_rate, col))
        if parseable_candidates:
            parseable_candidates.sort(reverse=True)
            timestamp_column = parseable_candidates[0][1]

    city_column = next(
        (normalised_columns[name] for name in city_priority if name in normalised_columns),
        None,
    )
    if city_column is None:
        city_like = [
            col
            for col in df.columns
            if any(token in _normalise_name(col) for token in ("city", "location", "station"))
        ]
        city_column = city_like[0] if city_like else None

    pollutant_patterns = {
        "pm2_5": re.compile(r"^(components_)?(pm_?2_?5|pm25|pm2p5)$"),
        "pm10": re.compile(r"^(components_)?pm_?10$"),
        "no": re.compile(r"^(components_)?no$"),
        "no2": re.compile(r"^(components_)?no_?2$"),
        "so2": re.compile(r"^(components_)?so_?2$"),
        "nh3": re.compile(r"^(components_)?nh_?3$"),
        "co": re.compile(r"^(components_)?co$"),
        "o3": re.compile(r"^(components_)?(o_?3|ozone)$"),
    }
    pollutant_columns: dict[str, Any] = {}
    for canonical, pattern in pollutant_patterns.items():
        for col in df.columns:
            if pattern.match(_normalise_name(col)):
                pollutant_columns[canonical] = col
                break

    return {
        "timestamp_column": timestamp_column,
        "city_column": city_column,
        "pollutant_columns": pollutant_columns,
    }


def inspect_data_files(raw_dir: str | Path = "data/raw") -> list[dict[str, Any]]:
    """Print and return a compact audit report for each discovered raw file."""
    files = list_data_files(raw_dir)
    if not files:
        print(
            f"[audit] No CSV/XLSX files found in {Path(raw_dir)}. "
            "Place the Pakistan air-quality raw files there before running the pipeline."
        )
        return []

    reports: list[dict[str, Any]] = []
    print(f"[audit] Found {len(files)} raw file(s) in {Path(raw_dir)}")
    for path in files:
        try:
            df = _read_with_inferred_city(path)
        except Exception as exc:  # pragma: no cover - defensive audit path
            print(f"[audit] {path.name}: failed to load ({exc})")
            reports.append({"file": path.name, "error": str(exc)})
            continue

        roles = detect_column_roles(df)
        timestamp_col = roles["timestamp_column"]
        city_col = roles["city_column"]

        parsed_ts = (
            parse_datetime_series(df[timestamp_col])
            if timestamp_col is not None
            else pd.Series(dtype="datetime64[ns]")
        )
        date_range = (
            (parsed_ts.min(), parsed_ts.max())
            if not parsed_ts.empty and parsed_ts.notna().any()
            else (None, None)
        )
        city_names = (
            sorted(df[city_col].dropna().astype(str).unique().tolist())[:20]
            if city_col is not None
            else []
        )
        missing_values = df.isna().sum()
        missing_summary = {
            str(col): int(count)
            for col, count in missing_values[missing_values > 0].sort_values(ascending=False).items()
        }

        report = {
            "file": str(path),
            "shape": df.shape,
            "columns": list(df.columns),
            "timestamp_column": timestamp_col,
            "date_range": date_range,
            "city_column": city_col,
            "city_names": city_names,
            "missing_values": missing_summary,
            "likely_pollutant_columns": roles["pollutant_columns"],
        }
        reports.append(report)

        print(f"\n[audit] File: {path}")
        print(f"        Shape: {df.shape}")
        print(f"        Columns: {list(df.columns)}")
        print(f"        Timestamp column: {timestamp_col}")
        print(f"        Date range: {date_range[0]} -> {date_range[1]}")
        print(f"        City column: {city_col}")
        print(f"        Cities (up to 20): {city_names}")
        print(f"        Missing values: {missing_summary or 'none'}")
        print(f"        Likely pollutant columns: {roles['pollutant_columns'] or 'none detected'}")

    return reports


def load_air_quality_data(
    raw_dir: str | Path = "data/raw",
    test_fraction: float = DEFAULT_TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load train/test data, or create a chronological split when needed."""
    files = list_data_files(raw_dir)
    if not files:
        raise FileNotFoundError(
            f"No CSV/XLSX files found in {Path(raw_dir)}. "
            "Place the dataset files in data/raw/ and rerun the pipeline."
        )

    train_files, test_files, other_files = identify_train_test_files(files)

    if train_files and test_files:
        city_specific_train_files = [path for path in train_files if _infer_city_from_path(path)]
        selected_train_files = city_specific_train_files or train_files
        ignored_generic_train_files = [
            path for path in train_files if path not in selected_train_files
        ]

        train_df = pd.concat(
            [_read_with_inferred_city(path) for path in selected_train_files],
            ignore_index=True,
        )
        test_df = pd.concat(
            [_read_with_inferred_city(path) for path in test_files],
            ignore_index=True,
        )

        train_roles = detect_column_roles(train_df)
        test_roles = detect_column_roles(test_df)
        if train_roles["timestamp_column"] and test_roles["timestamp_column"]:
            train_ts = parse_datetime_series(train_df[train_roles["timestamp_column"]])
            test_ts = parse_datetime_series(test_df[test_roles["timestamp_column"]])
            if test_ts.notna().any():
                test_start = test_ts.min()
                overlap_mask = train_ts >= test_start
                overlap_count = int(overlap_mask.sum())
                if overlap_count:
                    print(
                        f"[load] Warning: trimming {overlap_count} training row(s) at or after "
                        f"the test start {test_start} to prevent temporal leakage."
                    )
                    train_df = train_df.loc[~overlap_mask].copy()

        metadata = {
            "split_strategy": "filename",
            "train_files": [str(path) for path in selected_train_files],
            "test_files": [str(path) for path in test_files],
            "other_files_ignored": [str(path) for path in other_files],
            "generic_train_files_ignored": [str(path) for path in ignored_generic_train_files],
        }
        return train_df, test_df, metadata

    combined_df = pd.concat([_read_with_inferred_city(path) for path in files], ignore_index=True)
    roles = detect_column_roles(combined_df)
    timestamp_col = roles["timestamp_column"]
    if timestamp_col is None:
        raise ValueError(
            "Could not identify a timestamp column for chronological splitting. "
            "Use a column name such as timestamp/date_time or provide explicit train/test files."
        )

    timestamps = parse_datetime_series(combined_df[timestamp_col])
    invalid_count = int(timestamps.isna().sum())
    if timestamps.notna().sum() < 2:
        raise ValueError(
            f"Timestamp column {timestamp_col!r} does not contain enough valid dates "
            "to create a train/test split."
        )
    if invalid_count:
        print(
            f"[load] Warning: dropping {invalid_count} row(s) with invalid timestamps "
            f"before chronological splitting on {timestamp_col!r}."
        )

    valid_df = combined_df.loc[timestamps.notna()].copy()
    valid_df["_split_timestamp"] = timestamps[timestamps.notna()]
    valid_df = valid_df.sort_values("_split_timestamp").reset_index(drop=True)

    split_index = int(len(valid_df) * (1 - test_fraction))
    split_index = min(max(split_index, 1), len(valid_df) - 1)

    train_df = valid_df.iloc[:split_index].drop(columns="_split_timestamp").copy()
    test_df = valid_df.iloc[split_index:].drop(columns="_split_timestamp").copy()
    metadata = {
        "split_strategy": "chronological",
        "source_files": [path.name for path in files],
        "test_fraction": test_fraction,
        "split_timestamp": valid_df.iloc[split_index]["_split_timestamp"],
        "dropped_invalid_timestamp_rows": invalid_count,
    }
    return train_df, test_df, metadata
