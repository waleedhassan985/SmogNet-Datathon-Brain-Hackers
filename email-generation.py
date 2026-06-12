"""Email draft and SMTP sending helpers for SmogNet prediction alerts."""

from __future__ import annotations

from collections.abc import Mapping
from email.message import EmailMessage
from email.utils import parseaddr
import os
import smtplib
from urllib.parse import quote


DEFAULT_FROM_EMAIL = "info@dswaleed.live"
DEFAULT_SMTP_HOST = "smtp.office365.com"
DEFAULT_SMTP_PORT = 587
DEFAULT_SMTP_USERNAME = "info@dswaleed.live"


def _format_value(value: object, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _public_guidance(prediction: Mapping[str, object]) -> str:
    if not prediction.get("is_spike_detected", True):
        return (
            "No public alert is recommended for this reading because the model did not detect a spike. "
            "Continue routine air-quality monitoring and rerun prediction if pollutant readings change."
        )

    city = prediction.get("city", "the selected city")
    severity = str(prediction.get("severity", "detected")).lower()
    probable_source = str(prediction.get("probable_source", "an unknown source")).lower()
    return (
        f"{city} is experiencing a {severity} air pollution spike, likely linked to {probable_source}. "
        "Children, elderly people, respiratory patients, and heart or lung patients may be more affected during this period. "
        "Residents should limit outdoor activity, keep windows closed where possible, and use a well-fitting mask if they need to travel. "
        "Continue monitoring local updates until pollution levels move closer to normal."
    )


def generate_prediction_alert_email(prediction: Mapping[str, object]) -> dict[str, str]:
    """Create an email subject, body, and mailto link from a prediction payload."""
    is_spike_detected = bool(prediction.get("is_spike_detected", True))
    city = str(prediction.get("city", "Unknown city"))
    severity = str(prediction.get("severity", "spike")).title()
    timestamp = str(prediction.get("timestamp", "Unknown time"))
    dominant_pollutant = str(prediction.get("dominant_pollutant", "Unknown"))
    anomaly_score = _format_value(prediction.get("combined_anomaly_score"))
    probable_source = str(prediction.get("probable_source", "Unknown source"))
    source_confidence = str(prediction.get("source_confidence", "Unknown"))
    source_reason = str(prediction.get("source_reason", "No source reason available."))
    anomaly_explanation = str(
        prediction.get("anomaly_explanation", "No anomaly explanation available.")
    )

    if is_spike_detected:
        subject = f"SmogNet Alert: {severity} air pollution spike in {city}"
        heading = "SmogNet Prediction Alert"
        status = f"Spike detected: {severity}"
    else:
        subject = f"SmogNet Prediction: No spike detected in {city}"
        heading = "SmogNet Prediction Report"
        status = "No spike detected"

    lines = [
        heading,
        "",
        f"Prediction status: {status}",
        f"City: {city}",
        f"Timestamp: {timestamp}",
        f"Severity: {severity if is_spike_detected else 'Not applicable'}",
        f"Dominant pollutant: {dominant_pollutant}",
        f"Model score: {anomaly_score}",
        f"Probable source: {probable_source}",
        f"Source confidence: {source_confidence}",
        "",
        "Source reasoning:",
        source_reason,
        "",
        "Anomaly explanation:",
        anomaly_explanation,
        "",
        "Public alert text:",
        _public_guidance(prediction),
        "",
        "Pollutant details:",
    ]

    for reading in prediction.get("pollutant_readings", []):
        if not isinstance(reading, Mapping):
            continue
        label = reading.get("label", reading.get("pollutant", "Pollutant"))
        value = _format_value(reading.get("value"), digits=3)
        score = _format_value(reading.get("score"))
        threshold = _format_value(reading.get("moderate_threshold"))
        spike_text = "yes" if reading.get("is_spike") else "no"
        lines.append(
            f"- {label}: value {value}, score {score}, moderate threshold {threshold}, spike {spike_text}"
        )

    body = "\n".join(lines)
    mailto_href = f"mailto:?subject={quote(subject)}&body={quote(body)}"
    return {
        "subject": subject,
        "body": body,
        "mailto_href": mailto_href,
    }


def _valid_email_address(value: str, field_name: str) -> str:
    _, address = parseaddr(value or "")
    if "@" not in address or address.startswith("@") or address.endswith("@"):
        raise ValueError(f"Enter a valid {field_name} email address.")
    return address


def send_prediction_alert_email(
    email: Mapping[str, str],
    recipient: str,
) -> dict[str, str]:
    """Send a generated prediction email using Microsoft 365 SMTP."""
    smtp_host = os.getenv("SMOGNET_SMTP_HOST", DEFAULT_SMTP_HOST)
    smtp_port = int(os.getenv("SMOGNET_SMTP_PORT", str(DEFAULT_SMTP_PORT)))
    smtp_username = os.getenv("SMOGNET_SMTP_USERNAME", DEFAULT_SMTP_USERNAME)
    smtp_password = os.getenv("SMOGNET_SMTP_PASSWORD")
    from_email = os.getenv("SMOGNET_EMAIL_FROM", DEFAULT_FROM_EMAIL)

    if not smtp_password:
        raise ValueError(
            "Missing SMOGNET_SMTP_PASSWORD. Set it to the Microsoft 365 SMTP password "
            "or app password for info@dswaleed.live before sending email."
        )

    recipient_email = _valid_email_address(recipient, "recipient")
    sender_email = _valid_email_address(from_email, "sender")

    message = EmailMessage()
    message["From"] = f"SmogNet <{sender_email}>"
    message["To"] = recipient_email
    message["Subject"] = str(email["subject"])
    message.set_content(str(email["body"]))

    with smtplib.SMTP(smtp_host, smtp_port, timeout=30) as server:
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.send_message(message)

    return {
        "from": sender_email,
        "to": recipient_email,
        "subject": str(email["subject"]),
    }
