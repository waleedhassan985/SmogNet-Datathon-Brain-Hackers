"""Shared dashboard utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go


def read_csv_if_exists(path: str | Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    """Load a CSV if present, otherwise return an empty dataframe."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=parse_dates)


def empty_figure(message: str) -> go.Figure:
    """Create a clean empty-state figure."""
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font={"size": 16},
    )
    figure.update_layout(
        template="plotly_white",
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
    )
    return figure


def filter_by_common_controls(
    df: pd.DataFrame,
    city: str | None = None,
    severity: str | None = None,
    source: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Apply dashboard filters that are shared across output tables."""
    if df.empty:
        return df.copy()
    filtered = df.copy()
    if city and city != "All":
        filtered = filtered.loc[filtered["city"] == city]
    if severity and severity != "All" and "severity" in filtered.columns:
        filtered = filtered.loc[filtered["severity"] == severity]
    if source and source != "All" and "probable_source" in filtered.columns:
        filtered = filtered.loc[filtered["probable_source"] == source]
    if "timestamp" in filtered.columns:
        timestamps = pd.to_datetime(filtered["timestamp"], errors="coerce")
        if start_date:
            filtered = filtered.loc[timestamps >= pd.Timestamp(start_date)]
            timestamps = pd.to_datetime(filtered["timestamp"], errors="coerce")
        if end_date:
            filtered = filtered.loc[timestamps <= pd.Timestamp(end_date) + pd.Timedelta(days=1)]
    return filtered.copy()

