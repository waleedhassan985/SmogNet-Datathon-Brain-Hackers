"""Deterministic public alert generation for classified pollution spikes."""

from __future__ import annotations

import re

import pandas as pd


def generate_alert(row: pd.Series) -> str:
    """Generate a calm, public-facing alert in four sentences."""
    severity = str(row["severity"]).lower()
    city = row["city"]
    probable_source = str(row["probable_source"]).lower()
    return (
        f"{city} is experiencing a {severity} air pollution spike, likely linked to {probable_source}. "
        "Children, elderly people, respiratory patients, and heart or lung patients may be more affected during this period. "
        "Residents should limit outdoor activity, keep windows closed where possible, and use a well-fitting mask if they need to travel. "
        "Continue monitoring local updates until pollution levels move closer to normal."
    )


def validate_alert(alert_text: str, row: pd.Series) -> str:
    """Validate the competition-required content of an alert."""
    sentences = [sentence for sentence in re.split(r"(?<=[.!?])\s+", alert_text.strip()) if sentence]
    checks = {
        "sentence_count": 3 <= len(sentences) <= 4,
        "city": str(row["city"]).lower() in alert_text.lower(),
        "source": str(row["probable_source"]).lower() in alert_text.lower(),
        "children": "children" in alert_text.lower(),
        "elderly": "elderly people" in alert_text.lower(),
        "respiratory": "respiratory patients" in alert_text.lower(),
        "heart_lung": "heart or lung patients" in alert_text.lower(),
        "protective_actions": all(
            phrase in alert_text.lower()
            for phrase in ("limit outdoor activity", "keep windows closed", "mask")
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return "valid" if not failed else f"invalid: {', '.join(failed)}"


def generate_alerts_dataframe(
    classified_spikes_df: pd.DataFrame,
    max_alerts: int | None = None,
) -> pd.DataFrame:
    """Generate alert rows for classified spikes."""
    columns = [
        "timestamp",
        "city",
        "severity",
        "probable_source",
        "alert_text",
        "validation_status",
    ]
    if classified_spikes_df.empty:
        return pd.DataFrame(columns=columns)

    alerts = classified_spikes_df.sort_values("timestamp").copy()
    if max_alerts is not None:
        alerts = alerts.head(max_alerts).copy()
    alerts["alert_text"] = alerts.apply(generate_alert, axis=1)
    alerts["validation_status"] = alerts.apply(
        lambda row: validate_alert(row["alert_text"], row), axis=1
    )
    return alerts[columns].reset_index(drop=True)

