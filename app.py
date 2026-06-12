"""Plotly Dash dashboard for the SmogNet Datathon project."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, dcc, html

from src.anomaly_detection import (
    compute_rolling_baselines,
    detect_spikes,
    learn_anomaly_thresholds,
    score_test_data,
)
from src.config import (
    CANONICAL_POLLUTANTS,
    DEFAULT_ROLLING_WINDOW,
    OUTPUT_DATA_DIR,
    PROCESSED_DATA_DIR,
    SEVERITY_ORDER,
)
from src.preprocessing import assign_season
from src.source_classification import (
    attach_dust_ratio_thresholds,
    classify_all_spikes,
    fit_dust_ratio_reference,
)
from src.visualization_utils import empty_figure, filter_by_common_controls, read_csv_if_exists


def _load_email_generator():
    module_path = Path(__file__).with_name("email-generation.py")
    spec = importlib.util.spec_from_file_location("email_generation", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load email generator from {module_path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_prediction_alert_email, module.send_prediction_alert_email


generate_prediction_alert_email, send_prediction_alert_email = _load_email_generator()

detected_df = read_csv_if_exists(OUTPUT_DATA_DIR / "detected_spikes.csv", parse_dates=["timestamp"])
classified_df = read_csv_if_exists(OUTPUT_DATA_DIR / "classified_spikes.csv", parse_dates=["timestamp"])
alerts_df = read_csv_if_exists(OUTPUT_DATA_DIR / "public_alerts.csv", parse_dates=["timestamp"])
train_processed_df = read_csv_if_exists(PROCESSED_DATA_DIR / "train_processed.csv", parse_dates=["timestamp"])
trend_df = read_csv_if_exists(PROCESSED_DATA_DIR / "test_scored.csv", parse_dates=["timestamp"])
if trend_df.empty:
    trend_df = read_csv_if_exists(PROCESSED_DATA_DIR / "test_processed.csv", parse_dates=["timestamp"])


def _available_cities() -> list[str]:
    frames = [
        df
        for df in (trend_df, classified_df, alerts_df, train_processed_df)
        if not df.empty and "city" in df.columns
    ]
    if not frames:
        return []
    return sorted(pd.concat([df["city"] for df in frames], ignore_index=True).dropna().astype(str).unique())


def _available_pollutants() -> list[str]:
    columns = set(trend_df.columns) | set(detected_df.columns)
    return [pollutant for pollutant in CANONICAL_POLLUTANTS if pollutant in columns]


def _available_sources() -> list[str]:
    if classified_df.empty or "probable_source" not in classified_df.columns:
        return []
    return sorted(classified_df["probable_source"].dropna().astype(str).unique())


def _date_bounds() -> tuple[str | None, str | None]:
    frames = [df for df in (trend_df, classified_df) if not df.empty and "timestamp" in df.columns]
    if not frames:
        return None, None
    timestamps = pd.concat([pd.to_datetime(df["timestamp"], errors="coerce") for df in frames]).dropna()
    if timestamps.empty:
        return None, None
    return timestamps.min().date().isoformat(), timestamps.max().date().isoformat()


AVAILABLE_CITIES = _available_cities()
AVAILABLE_POLLUTANTS = _available_pollutants()
AVAILABLE_SOURCES = _available_sources()
MIN_DATE, MAX_DATE = _date_bounds()
DEFAULT_POLLUTANT = "pm2_5" if "pm2_5" in AVAILABLE_POLLUTANTS else (
    AVAILABLE_POLLUTANTS[0] if AVAILABLE_POLLUTANTS else None
)
PREDICTION_POLLUTANTS = [
    pollutant for pollutant in CANONICAL_POLLUTANTS if pollutant in train_processed_df.columns
]


def _pollutant_label(pollutant: str) -> str:
    labels = {
        "pm2_5": "PM2.5",
        "pm10": "PM10",
        "no": "NO",
        "no2": "NO2",
        "so2": "SO2",
        "nh3": "NH3",
        "co": "CO",
        "o3": "O3",
    }
    return labels.get(pollutant, pollutant.upper())


def _default_prediction_value(pollutant: str) -> float | None:
    for frame in (trend_df, train_processed_df):
        if not frame.empty and pollutant in frame.columns:
            value = pd.to_numeric(frame[pollutant], errors="coerce").median()
            if pd.notna(value):
                return round(float(value), 3)
    return None


@lru_cache(maxsize=1)
def _prediction_model() -> tuple[dict, dict, dict, tuple[str, ...]]:
    """Rebuild the training-derived references used by the pipeline."""
    if train_processed_df.empty:
        raise ValueError("Training data is missing. Run the pipeline before using live prediction.")
    if not PREDICTION_POLLUTANTS:
        raise ValueError("No pollutant columns are available for prediction.")

    train_scored, baseline_reference = compute_rolling_baselines(
        train_processed_df,
        PREDICTION_POLLUTANTS,
        window=DEFAULT_ROLLING_WINDOW,
    )
    thresholds = learn_anomaly_thresholds(train_scored, PREDICTION_POLLUTANTS)
    dust_reference = fit_dust_ratio_reference(train_processed_df)
    return baseline_reference, thresholds, dust_reference, tuple(PREDICTION_POLLUTANTS)


def _lookup_prediction_threshold(
    thresholds: dict,
    city: str,
    season: str,
    pollutant: str,
) -> dict | None:
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
    return None


def _score_table(row: pd.Series, thresholds: dict, pollutants: tuple[str, ...]) -> dbc.Table:
    records = []
    for pollutant in pollutants:
        threshold = _lookup_prediction_threshold(
            thresholds,
            str(row["city"]),
            str(row["season"]),
            pollutant,
        )
        score = float(row[f"{pollutant}_score"])
        moderate = float(threshold["moderate_threshold"]) if threshold else None
        records.append(
            html.Tr(
                [
                    html.Td(_pollutant_label(pollutant)),
                    html.Td(f"{float(row[pollutant]):,.3f}"),
                    html.Td(f"{score:,.2f}"),
                    html.Td(f"{moderate:,.2f}" if moderate is not None else "N/A"),
                    html.Td("Yes" if moderate is not None and score >= moderate else "No"),
                ]
            )
        )

    return dbc.Table(
        [
            html.Thead(
                html.Tr(
                    [
                        html.Th("Pollutant"),
                        html.Th("Input"),
                        html.Th("Score"),
                        html.Th("Moderate threshold"),
                        html.Th("Spike?"),
                    ]
                )
            ),
            html.Tbody(records),
        ],
        bordered=True,
        hover=True,
        responsive=True,
        size="sm",
        className="mb-0",
    )


def _prediction_email_payload(
    prediction: pd.Series,
    scored_row: pd.Series,
    thresholds: dict,
    pollutants: tuple[str, ...],
    is_spike_detected: bool,
) -> dict:
    readings = []
    for pollutant in pollutants:
        threshold = _lookup_prediction_threshold(
            thresholds,
            str(prediction["city"]),
            str(prediction["season"]),
            pollutant,
        )
        score = float(scored_row[f"{pollutant}_score"])
        moderate_threshold = (
            float(threshold["moderate_threshold"]) if threshold is not None else None
        )
        readings.append(
            {
                "pollutant": pollutant,
                "label": _pollutant_label(pollutant),
                "value": float(scored_row[pollutant]),
                "score": score,
                "moderate_threshold": moderate_threshold,
                "is_spike": moderate_threshold is not None and score >= moderate_threshold,
            }
        )

    return {
        "is_spike_detected": is_spike_detected,
        "timestamp": pd.Timestamp(prediction["timestamp"]).strftime("%Y-%m-%d %H:%M"),
        "city": str(prediction["city"]),
        "season": str(prediction["season"]),
        "severity": str(prediction["severity"]),
        "dominant_pollutant": _pollutant_label(str(prediction["dominant_pollutant"])),
        "combined_anomaly_score": float(prediction["combined_anomaly_score"]),
        "probable_source": str(prediction["probable_source"]),
        "source_confidence": str(prediction["source_confidence"]),
        "source_reason": str(prediction["source_reason"]),
        "anomaly_explanation": str(prediction["anomaly_explanation"]),
        "pollutant_readings": readings,
    }


def _prediction_input_card(pollutant: str) -> dbc.Col:
    return dbc.Col(
        [
            html.Label(_pollutant_label(pollutant), className="small text-muted"),
            dbc.Input(
                id=f"prediction-{pollutant}",
                type="number",
                min=0,
                step="any",
                value=_default_prediction_value(pollutant),
            ),
        ],
        md=3,
        sm=6,
        className="mb-3",
    )


def _prediction_layout() -> dbc.Container:
    default_date = MAX_DATE or pd.Timestamp.today().date().isoformat()
    default_city = AVAILABLE_CITIES[0] if AVAILABLE_CITIES else None

    return dbc.Container(
        [
            dcc.Store(id="prediction-email-payload"),
            dbc.Row(
                dbc.Col(
                    [
                        html.H1("Spike and Source Prediction", className="mt-4"),
                        html.P(
                            "Enter a single air-quality reading. The app scores it against the "
                            "training-derived spike detector, then runs source classification if a "
                            "spike is detected.",
                            className="text-muted",
                        ),
                    ]
                )
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Reading Details"),
                                    html.Label("City"),
                                    dcc.Dropdown(
                                        id="prediction-city",
                                        options=[
                                            {"label": city, "value": city}
                                            for city in AVAILABLE_CITIES
                                        ],
                                        value=default_city,
                                        clearable=False,
                                    ),
                                    html.Br(),
                                    html.Label("Date"),
                                    dcc.DatePickerSingle(
                                        id="prediction-date",
                                        date=default_date,
                                        display_format="YYYY-MM-DD",
                                    ),
                                    html.Br(),
                                    html.Br(),
                                    html.Label("Time"),
                                    dbc.Input(
                                        id="prediction-time",
                                        type="time",
                                        value="12:00",
                                    ),
                                ]
                            ),
                            className="shadow-sm h-100",
                        ),
                        md=3,
                    ),
                    dbc.Col(
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Pollutant Values"),
                                    dbc.Row(
                                        [
                                            _prediction_input_card(pollutant)
                                            for pollutant in PREDICTION_POLLUTANTS
                                        ]
                                    ),
                                    dbc.Button(
                                        "Run Prediction",
                                        id="prediction-button",
                                        color="primary",
                                    ),
                                ]
                            ),
                            className="shadow-sm h-100",
                        ),
                        md=9,
                    ),
                ],
                className="g-3",
            ),
            dbc.Row(
                dbc.Col(
                    html.Div(
                        id="prediction-output",
                        className="mt-4 mb-4",
                    )
                )
            ),
            dbc.Row(
                dbc.Col(
                    html.Div(
                        id="prediction-email-output",
                        className="mb-4",
                    )
                )
            ),
        ],
        fluid=True,
    )


def _navbar(active_path: str) -> dbc.NavbarSimple:
    return dbc.NavbarSimple(
        [
            dbc.NavItem(
                dbc.NavLink(
                    "Dashboard",
                    href="/",
                    active=active_path != "/prediction",
                )
            ),
            dbc.NavItem(
                dbc.NavLink(
                    "Prediction",
                    href="/prediction",
                    active=active_path == "/prediction",
                )
            ),
        ],
        brand="SmogNet",
        color="dark",
        dark=True,
    )


app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
app.title = "SmogNet"


def _kpi_card(title: str, value_id: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(title, className="text-muted small"),
                html.H4(id=value_id, className="mb-0"),
            ]
        ),
        className="shadow-sm h-100",
    )


app.layout = dbc.Container(
    [
        dcc.Store(id="simulation-index", data=-1),
        dbc.Row(
            dbc.Col(
                [
                    html.Div(
                        [
                            html.H1(
                                "SmogNet: Real-Time Air Quality Intelligence System",
                                className="mt-4 mb-0",
                            ),
                            dbc.Button(
                                "Open Prediction Page",
                                href="/prediction",
                                color="primary",
                                className="mt-4",
                            ),
                        ],
                        className="d-flex justify-content-between align-items-start gap-3 flex-wrap",
                    ),
                    html.P(
                        "Adaptive spike detection, probable-source classification, and public alerts "
                        "for Pakistan air-quality monitoring.",
                        className="text-muted",
                    ),
                ]
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            dbc.CardBody(
                                [
                                    html.H5("Filters"),
                                    html.Label("City"),
                                    dcc.Dropdown(
                                        id="city-filter",
                                        options=[{"label": "All", "value": "All"}]
                                        + [{"label": city, "value": city} for city in AVAILABLE_CITIES],
                                        value="All",
                                        clearable=False,
                                    ),
                                    html.Br(),
                                    html.Label("Pollutant"),
                                    dcc.Dropdown(
                                        id="pollutant-filter",
                                        options=[
                                            {"label": pollutant.upper(), "value": pollutant}
                                            for pollutant in AVAILABLE_POLLUTANTS
                                        ],
                                        value=DEFAULT_POLLUTANT,
                                        clearable=False,
                                    ),
                                    html.Br(),
                                    html.Label("Severity"),
                                    dcc.Dropdown(
                                        id="severity-filter",
                                        options=[{"label": "All", "value": "All"}]
                                        + [
                                            {"label": severity.title(), "value": severity}
                                            for severity in SEVERITY_ORDER
                                        ],
                                        value="All",
                                        clearable=False,
                                    ),
                                    html.Br(),
                                    html.Label("Probable source"),
                                    dcc.Dropdown(
                                        id="source-filter",
                                        options=[{"label": "All", "value": "All"}]
                                        + [
                                            {"label": source, "value": source}
                                            for source in AVAILABLE_SOURCES
                                        ],
                                        value="All",
                                        clearable=False,
                                    ),
                                    html.Br(),
                                    html.Label("Date range"),
                                    dcc.DatePickerRange(
                                        id="date-filter",
                                        min_date_allowed=MIN_DATE,
                                        max_date_allowed=MAX_DATE,
                                        start_date=MIN_DATE,
                                        end_date=MAX_DATE,
                                        display_format="YYYY-MM-DD",
                                    ),
                                ]
                            ),
                            className="shadow-sm",
                        )
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        dbc.Row(
                            [
                                dbc.Col(_kpi_card("Total spikes", "kpi-total-spikes"), md=2),
                                dbc.Col(_kpi_card("Cities affected", "kpi-cities"), md=2),
                                dbc.Col(_kpi_card("Most common source", "kpi-source"), md=3),
                                dbc.Col(_kpi_card("Most severe spike", "kpi-severe"), md=3),
                                dbc.Col(_kpi_card("Generated alerts", "kpi-alerts"), md=2),
                            ],
                            className="g-3",
                        ),
                        dbc.Row(
                            dbc.Col(dcc.Graph(id="pollution-trend"), className="mt-4"),
                        ),
                    ],
                    md=9,
                ),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="anomaly-timeline"), md=6),
                dbc.Col(dcc.Graph(id="source-pie"), md=6),
            ],
            className="mt-4",
        ),
        dbc.Row(
            [
                dbc.Col(dcc.Graph(id="city-source-summary"), md=6),
                dbc.Col(dcc.Graph(id="city-severity-summary"), md=6),
            ]
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Public Alerts"),
                                html.Div(id="public-alerts-container"),
                            ]
                        ),
                        className="shadow-sm",
                    ),
                    md=7,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H4("Simulated Near-Real-Time Panel"),
                                html.P(
                                    "Demonstrates the flow: incoming data → spike detection → "
                                    "probable-source classification → public alert.",
                                    className="text-muted",
                                ),
                                dbc.Button(
                                    "Simulate Next Detected Spike",
                                    id="simulate-button",
                                    color="primary",
                                    className="mb-3",
                                ),
                                html.Div(id="simulation-panel"),
                                html.Hr(),
                                html.Div(id="confidence-summary", className="small text-muted"),
                            ]
                        ),
                        className="shadow-sm h-100",
                    ),
                    md=5,
                ),
            ],
            className="mt-4 mb-4",
        ),
    ],
    fluid=True,
)

dashboard_page = app.layout
app.layout = html.Div(
    [
        dcc.Location(id="url"),
        html.Div(id="page-content"),
    ]
)


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
)
def render_page(pathname: str | None):
    active_path = pathname or "/"
    page = _prediction_layout() if active_path == "/prediction" else dashboard_page
    return [_navbar(active_path), page]


@app.callback(
    Output("kpi-total-spikes", "children"),
    Output("kpi-cities", "children"),
    Output("kpi-source", "children"),
    Output("kpi-severe", "children"),
    Output("kpi-alerts", "children"),
    Output("pollution-trend", "figure"),
    Output("anomaly-timeline", "figure"),
    Output("source-pie", "figure"),
    Output("city-source-summary", "figure"),
    Output("city-severity-summary", "figure"),
    Output("public-alerts-container", "children"),
    Output("confidence-summary", "children"),
    Input("city-filter", "value"),
    Input("pollutant-filter", "value"),
    Input("severity-filter", "value"),
    Input("source-filter", "value"),
    Input("date-filter", "start_date"),
    Input("date-filter", "end_date"),
)
def update_dashboard(
    city: str,
    pollutant: str | None,
    severity: str,
    source: str,
    start_date: str | None,
    end_date: str | None,
):
    filtered_spikes = filter_by_common_controls(
        classified_df, city=city, severity=severity, source=source, start_date=start_date, end_date=end_date
    )
    filtered_alerts = filter_by_common_controls(
        alerts_df, city=city, severity=severity, source=source, start_date=start_date, end_date=end_date
    )

    total_spikes = str(len(filtered_spikes))
    cities_affected = str(filtered_spikes["city"].nunique()) if not filtered_spikes.empty else "0"
    most_common_source = (
        filtered_spikes["probable_source"].mode().iat[0]
        if not filtered_spikes.empty and filtered_spikes["probable_source"].notna().any()
        else "None"
    )
    if not filtered_spikes.empty:
        most_severe_row = filtered_spikes.sort_values(
            ["combined_anomaly_score"], ascending=False
        ).iloc[0]
        most_severe = f"{most_severe_row['severity'].title()} · {most_severe_row['combined_anomaly_score']:.2f}"
    else:
        most_severe = "None"
    alert_count = str(len(filtered_alerts))

    trend_filtered = trend_df.copy()
    if not trend_filtered.empty:
        if city and city != "All":
            trend_filtered = trend_filtered.loc[trend_filtered["city"] == city]
        timestamps = pd.to_datetime(trend_filtered["timestamp"], errors="coerce")
        if start_date:
            trend_filtered = trend_filtered.loc[timestamps >= pd.Timestamp(start_date)]
            timestamps = pd.to_datetime(trend_filtered["timestamp"], errors="coerce")
        if end_date:
            trend_filtered = trend_filtered.loc[timestamps <= pd.Timestamp(end_date) + pd.Timedelta(days=1)]

    if pollutant and not trend_filtered.empty and pollutant in trend_filtered.columns:
        trend_fig = go.Figure()
        for city_name, group in trend_filtered.groupby("city", sort=True):
            trend_fig.add_trace(
                go.Scatter(
                    x=group["timestamp"],
                    y=group[pollutant],
                    mode="lines",
                    name=f"{city_name} {pollutant.upper()}",
                )
            )
            baseline_col = f"{pollutant}_baseline_median"
            if baseline_col in group.columns:
                trend_fig.add_trace(
                    go.Scatter(
                        x=group["timestamp"],
                        y=group[baseline_col],
                        mode="lines",
                        line={"dash": "dash"},
                        name=f"{city_name} rolling baseline",
                    )
                )

        if pollutant in filtered_spikes.columns:
            trend_fig.add_trace(
                go.Scatter(
                    x=filtered_spikes["timestamp"],
                    y=filtered_spikes[pollutant],
                    mode="markers",
                    marker={"size": 10, "color": "crimson", "symbol": "x"},
                    name="Detected spikes",
                )
            )
        trend_fig.update_layout(
            title=f"Pollution Trend and Rolling Baseline · {pollutant.upper()}",
            template="plotly_white",
            margin={"l": 40, "r": 20, "t": 50, "b": 40},
        )
    else:
        trend_fig = empty_figure("Run the pipeline with raw data to view pollution trends.")

    if filtered_spikes.empty:
        timeline_fig = empty_figure("No spikes match the current filters.")
        source_fig = empty_figure("No source classifications to display.")
        city_source_fig = empty_figure("No city/source summary available.")
        city_severity_fig = empty_figure("No city/severity summary available.")
        confidence_summary = "No classified spikes match the current filters."
    else:
        timeline_fig = px.scatter(
            filtered_spikes,
            x="timestamp",
            y="city",
            size="combined_anomaly_score",
            color="severity",
            category_orders={"severity": list(SEVERITY_ORDER)},
            title="Anomaly Timeline",
            hover_data=["dominant_pollutant", "probable_source", "combined_anomaly_score"],
        )
        timeline_fig.update_layout(template="plotly_white")

        source_fig = px.pie(
            filtered_spikes,
            names="probable_source",
            title="Probable Source Distribution",
        )
        source_fig.update_layout(template="plotly_white")

        source_counts = (
            filtered_spikes.groupby(["city", "probable_source"])
            .size()
            .reset_index(name="count")
        )
        city_source_fig = px.bar(
            source_counts,
            x="city",
            y="count",
            color="probable_source",
            title="City by Probable Source",
            barmode="stack",
        )
        city_source_fig.update_layout(template="plotly_white")

        severity_counts = (
            filtered_spikes.groupby(["city", "severity"])
            .size()
            .reset_index(name="count")
        )
        city_severity_fig = px.bar(
            severity_counts,
            x="city",
            y="count",
            color="severity",
            category_orders={"severity": list(SEVERITY_ORDER)},
            title="City by Severity",
            barmode="stack",
        )
        city_severity_fig.update_layout(template="plotly_white")

        confidence_counts = filtered_spikes["source_confidence"].value_counts().to_dict()
        confidence_summary = (
            "Source confidence summary: "
            + ", ".join(f"{label} {count}" for label, count in confidence_counts.items())
        )

    if filtered_alerts.empty:
        alert_cards = dbc.Alert(
            "No public alerts match the current filters.",
            color="light",
        )
    else:
        alert_cards = [
            dbc.Card(
                dbc.CardBody(
                    [
                        html.H6(
                            f"{row.city} · {pd.Timestamp(row.timestamp).strftime('%Y-%m-%d %H:%M')} · "
                            f"{str(row.severity).title()}",
                            className="mb-1",
                        ),
                        html.Div(row.probable_source, className="text-muted small mb-2"),
                        html.P(row.alert_text, className="mb-0"),
                    ]
                ),
                className="mb-2",
            )
            for row in filtered_alerts.sort_values("timestamp", ascending=False).head(10).itertuples()
        ]

    return (
        total_spikes,
        cities_affected,
        most_common_source,
        most_severe,
        alert_count,
        trend_fig,
        timeline_fig,
        source_fig,
        city_source_fig,
        city_severity_fig,
        alert_cards,
        confidence_summary,
    )


@app.callback(
    Output("simulation-index", "data"),
    Input("simulate-button", "n_clicks"),
    State("simulation-index", "data"),
    prevent_initial_call=True,
)
def advance_simulation(_: int, current_index: int) -> int:
    if classified_df.empty:
        return -1
    return (current_index + 1) % len(classified_df)


@app.callback(
    Output("simulation-panel", "children"),
    Input("simulation-index", "data"),
)
def update_simulation_panel(index: int):
    if classified_df.empty or index is None or index < 0:
        return dbc.Alert("No detected spikes are available yet. Run the pipeline after adding raw data.", color="light")

    row = classified_df.sort_values("timestamp").reset_index(drop=True).iloc[index]
    alert_match = alerts_df.loc[
        (alerts_df["timestamp"] == row["timestamp"])
        & (alerts_df["city"] == row["city"])
    ]
    alert_text = alert_match["alert_text"].iloc[0] if not alert_match.empty else "No alert generated."

    return dbc.Alert(
        [
            html.Strong(f"{row['city']} · {pd.Timestamp(row['timestamp']).strftime('%Y-%m-%d %H:%M')}"),
            html.Br(),
            f"Dominant pollutant: {str(row['dominant_pollutant']).upper()}",
            html.Br(),
            f"Anomaly score: {row['combined_anomaly_score']:.2f}",
            html.Br(),
            f"Probable source: {row['probable_source']} ({row['source_confidence']} confidence)",
            html.Hr(),
            html.Div(alert_text),
        ],
        color="warning",
    )


@app.callback(
    Output("prediction-output", "children"),
    Output("prediction-email-payload", "data"),
    Output("prediction-email-output", "children"),
    Input("prediction-button", "n_clicks"),
    State("prediction-city", "value"),
    State("prediction-date", "date"),
    State("prediction-time", "value"),
    *[State(f"prediction-{pollutant}", "value") for pollutant in PREDICTION_POLLUTANTS],
)
def run_prediction(
    n_clicks: int | None,
    city: str | None,
    date_value: str | None,
    time_value: str | None,
    *pollutant_values,
):
    if not n_clicks:
        return (
            dbc.Alert(
                "Enter pollutant readings and click Run Prediction.",
                color="light",
            ),
            None,
            None,
        )

    try:
        baseline_reference, thresholds, dust_reference, pollutants = _prediction_model()
        if not city:
            raise ValueError("Choose a city before running prediction.")
        if not date_value:
            raise ValueError("Choose a date before running prediction.")

        timestamp = pd.Timestamp(f"{date_value} {time_value or '00:00'}")
        if pd.isna(timestamp):
            raise ValueError("Enter a valid date and time.")

        row = {
            "timestamp": timestamp,
            "city": city,
            "hour": timestamp.hour,
            "day": timestamp.day,
            "month": timestamp.month,
            "year": timestamp.year,
            "date": timestamp.date(),
            "season": assign_season(timestamp.month),
        }

        for pollutant, value in zip(pollutants, pollutant_values, strict=False):
            if value is None:
                raise ValueError(f"Enter a value for {_pollutant_label(pollutant)}.")
            numeric_value = float(value)
            if numeric_value < 0:
                raise ValueError(f"{_pollutant_label(pollutant)} cannot be negative.")
            row[pollutant] = numeric_value

        scored = score_test_data(
            pd.DataFrame([row]),
            baseline_reference,
            list(pollutants),
        )
        spikes = detect_spikes(scored, thresholds, list(pollutants))
        scored_row = scored.iloc[0]
        top_pollutant = max(pollutants, key=lambda pollutant: scored_row[f"{pollutant}_score"])
        score_table = _score_table(scored_row, thresholds, pollutants)

        if spikes.empty:
            prediction_summary = scored_row.copy()
            prediction_summary["severity"] = "no spike"
            prediction_summary["dominant_pollutant"] = top_pollutant
            prediction_summary["combined_anomaly_score"] = float(
                scored_row[f"{top_pollutant}_score"]
            )
            prediction_summary["probable_source"] = "Not applicable"
            prediction_summary["source_confidence"] = "Not applicable"
            prediction_summary["source_reason"] = (
                "Source classification is only applied after a spike is detected."
            )
            prediction_summary["anomaly_explanation"] = (
                f"No spike was detected. Highest model score was "
                f"{_pollutant_label(top_pollutant)} at "
                f"{scored_row[f'{top_pollutant}_score']:.2f}."
            )
            email_payload = _prediction_email_payload(
                prediction_summary,
                scored_row,
                thresholds,
                pollutants,
                is_spike_detected=False,
            )

            return (
                dbc.Card(
                    dbc.CardBody(
                        [
                            dbc.Alert(
                                [
                                    html.Strong("No spike detected."),
                                    html.Br(),
                                    (
                                        f"Highest model score: {_pollutant_label(top_pollutant)} "
                                        f"at {scored_row[f'{top_pollutant}_score']:.2f}."
                                    ),
                                    html.Br(),
                                    "Source classification and alert email generation are only applied after a spike is detected.",
                                ],
                                color="success",
                            ),
                            dbc.Button(
                                "Generate Prediction Email",
                                id="generate-email-button",
                                color="secondary",
                                className="mb-3",
                            ),
                            html.H5("Model Scores"),
                            score_table,
                        ]
                    ),
                    className="shadow-sm",
                ),
                email_payload,
                None,
            )

        classified = classify_all_spikes(
            attach_dust_ratio_thresholds(spikes, dust_reference)
        )
        prediction = classified.iloc[0]
        severity = str(prediction["severity"])
        alert_color = {
            "moderate": "warning",
            "high": "danger",
            "severe": "dark",
        }.get(severity, "warning")

        email_payload = _prediction_email_payload(
            prediction,
            scored_row,
            thresholds,
            pollutants,
            is_spike_detected=True,
        )

        return (
            dbc.Card(
                dbc.CardBody(
                    [
                        dbc.Alert(
                            [
                                html.Strong(f"Spike detected: {severity.title()}"),
                                html.Br(),
                                (
                                    f"Dominant pollutant: "
                                    f"{_pollutant_label(str(prediction['dominant_pollutant']))}"
                                ),
                                html.Br(),
                                f"Anomaly score: {float(prediction['combined_anomaly_score']):.2f}",
                            ],
                            color=alert_color,
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [
                                        html.Div("Probable source", className="text-muted small"),
                                        html.H4(str(prediction["probable_source"])),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Div("Source confidence", className="text-muted small"),
                                        html.H4(str(prediction["source_confidence"])),
                                    ],
                                    md=4,
                                ),
                                dbc.Col(
                                    [
                                        html.Div("Season", className="text-muted small"),
                                        html.H4(str(prediction["season"])),
                                    ],
                                    md=4,
                                ),
                            ],
                            className="mb-3",
                        ),
                        html.P(str(prediction["source_reason"]), className="mb-1"),
                        html.P(str(prediction["anomaly_explanation"]), className="text-muted"),
                        dbc.Button(
                            "Generate Alert Email",
                            id="generate-email-button",
                            color="secondary",
                            className="mb-3",
                        ),
                        html.H5("Model Scores"),
                        score_table,
                    ]
                ),
                className="shadow-sm",
            ),
            email_payload,
            None,
        )
    except Exception as exc:
        return dbc.Alert(f"Prediction failed: {exc}", color="danger"), None, None


@app.callback(
    Output("prediction-email-output", "children", allow_duplicate=True),
    Input("generate-email-button", "n_clicks"),
    State("prediction-email-payload", "data"),
    prevent_initial_call=True,
)
def generate_prediction_email(n_clicks: int | None, payload: dict | None):
    if not n_clicks:
        return None
    if not payload:
        return dbc.Alert("Run a spike prediction before generating an alert email.", color="light")

    email = generate_prediction_alert_email(payload)
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div("Generated Alert Email", className="text-muted small"),
                html.H5(email["subject"]),
                dcc.Textarea(
                    value=email["body"],
                    readOnly=True,
                    style={
                        "width": "100%",
                        "height": "320px",
                        "fontFamily": "monospace",
                        "fontSize": "0.9rem",
                    },
                ),
                dbc.Button(
                    "Open Email Draft",
                    href=email["mailto_href"],
                    target="_blank",
                    color="success",
                    className="mt-3",
                ),
                html.Hr(),
                html.Label("Recipient email"),
                dbc.Input(
                    id="alert-email-recipient",
                    type="email",
                    placeholder="recipient@example.com",
                ),
                html.Div(
                    "Sender: info@dswaleed.live via Microsoft 365 Outlook SMTP",
                    className="small text-muted mt-2",
                ),
                dbc.Button(
                    "Send Email",
                    id="send-email-button",
                    color="primary",
                    className="mt-3",
                ),
                html.Div(id="email-send-status", className="mt-3"),
            ]
        ),
        className="shadow-sm",
    )


@app.callback(
    Output("email-send-status", "children"),
    Input("send-email-button", "n_clicks"),
    State("prediction-email-payload", "data"),
    State("alert-email-recipient", "value"),
    prevent_initial_call=True,
)
def send_prediction_email(
    n_clicks: int | None,
    payload: dict | None,
    recipient: str | None,
):
    if not n_clicks:
        return None
    if not payload:
        return dbc.Alert("Run prediction and generate the email first.", color="light")
    if not recipient:
        return dbc.Alert("Enter a recipient email address before sending.", color="warning")

    try:
        email = generate_prediction_alert_email(payload)
        result = send_prediction_alert_email(email, recipient)
        return dbc.Alert(
            f"Email sent from {result['from']} to {result['to']}.",
            color="success",
        )
    except Exception as exc:
        return dbc.Alert(f"Email was not sent: {exc}", color="danger")


if __name__ == "__main__":
    app.run(debug=False)
