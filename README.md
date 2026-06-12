# SmogNet: Real-Time Air Quality Intelligence System

SmogNet is an end-to-end Datathon project for the **Pakistan Air Quality & Pollutant Concentrations** dataset. It converts raw pollution readings into adaptive spike detections, interpretable probable-source labels, calm public alerts, and a demo-ready Plotly Dash dashboard.

## Objective

Build a transparent pipeline that answers four practical questions:

1. **When did pollution become abnormal for this city and season?**
2. **Which pollutant fingerprint most likely explains the spike?**
3. **What should the public be told right now?**
4. **How can judges explore the results interactively?**

## Problem Statement

Fixed pollution thresholds are weak for a country-wide monitoring task because the same pollutant value may be routine in one city-season context and unusual in another. SmogNet therefore learns historical behavior from training data only, then flags deviations in held-out test data without leaking future information into the baseline.

## End-to-End Pipeline

```text
raw CSV/XLSX files
        ↓
data audit + role detection
        ↓
preprocessing + safe imputation
        ↓
training-only rolling baselines
        ↓
adaptive anomaly scoring on test data
        ↓
probable-source classification
        ↓
public alert generation
        ↓
interactive Dash dashboard
```

## Methodology Summary

- **Preprocessing:** standardises heterogeneous column names, detects timestamp/city/pollutant columns, parses dates, adds temporal features, interpolates within each city, then fills remaining missing values with city and global medians.
- **Spike detection:** uses city-season rolling medians and rolling IQRs learned from training data only. The anomaly score is robust and thresholded by training quantiles rather than fixed AQI cutoffs.
- **Fallback logic:** if a city-season history is too small, the model falls back to city-level, then global-season, then global references.
- **Probable-source logic:** combines pollutant fingerprints:
  - crop burning → NH3 + CO
  - vehicular emissions → NO + NO2
  - industrial emissions → SO2
  - dust storms → PM10 much greater than PM2.5 using a training-derived ratio threshold
  - mixed sources → overlapping evidence or no dominant pattern
- **Alerts:** deterministic 4-sentence public messages that name the city, probable source, vulnerable groups, and protective actions.

## Installation

```bash
pip install -r requirements.txt
```

## Data Placement

Place the raw dataset files in:

```text
data/raw/
```

Supported formats:

- `.csv`
- `.xlsx`

If filenames contain `train` and `test`, SmogNet uses them directly. Otherwise it creates a chronological train/test split from the timestamp column.

## Run the Pipeline

```bash
python -m src.pipeline
```

The loader prints an audit report showing:

- file names
- shapes
- columns
- detected date ranges
- city names
- missing values
- likely pollutant columns

## Run the Dashboard

```bash
python app.py
```

Open the local Dash URL printed in the terminal.

## Output Files

Pipeline outputs are saved to:

```text
data/processed/train_processed.csv
data/processed/test_processed.csv
data/processed/train_scored.csv
data/processed/test_scored.csv
data/outputs/detected_spikes.csv
data/outputs/classified_spikes.csv
data/outputs/public_alerts.csv
```

## Dashboard Features

- KPI cards
- city, pollutant, severity, source, and date filters
- pollution trend chart with rolling baseline and anomaly markers
- anomaly timeline
- probable-source distribution
- city/source and city/severity summaries
- public alert cards
- simulated near-real-time panel with a **Simulate Next Detected Spike** button

## Assumptions

- Raw files contain at least one parseable timestamp field, one city/location field, and at least one supported pollutant field.
- Supported pollutant variants include common spellings such as `pm2.5`, `pm25`, `pm_2_5`, `pm10`, `no2`, `so2`, `nh3`, `co`, and `ozone`.
- For the supplied Datathon files, `components_*` pollutant fields are normalised automatically and city names are inferred from filenames such as `lahore_complete_data.xlsx` when no city column is present.
- When no explicit train/test files exist, the earlier chronological segment is treated as training data and the later segment as test data.
- Source classification is intended as an interpretable **probable source** estimate, not ground truth.

## Limitations

- Probable-source labels are inferential, not confirmed.
- No meteorological, traffic, industrial inventory, or satellite fire inputs are used.
- Sensor noise and missingness can still affect results.
- The project simulates near-real-time flow but does not ingest live APIs yet.

## Future Improvements

- Add weather covariates and wind direction
- Add satellite fire/hotspot feeds
- Deploy the dashboard publicly
- Add live API ingestion
- Improve source classification with labelled events
- Add richer validation once ground-truth event labels become available
