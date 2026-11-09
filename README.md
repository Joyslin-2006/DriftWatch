# DriftWatch — Fraud Model Monitoring & Observability

An experimental machine learning monitoring application that tracks data quality, feature drift, prediction drift, model performance, and alerts for a fraud detection model.

## Overview

Machine learning models can become less reliable when the data they receive changes over time. This is known as **data drift**. Even when a model continues generating predictions, changes in incoming data or prediction behaviour may indicate that its performance needs attention.

**DriftWatch** is a Streamlit-based monitoring dashboard built to demonstrate how a fraud detection model can be monitored after deployment.

Rather than focusing on training a model, this project focuses on the monitoring lifecycle.

## Key Features

- **Data Quality Monitoring:** Checks incoming transaction data for quality issues.
- **Feature Drift Detection:** Compares incoming feature distributions against reference data using Population Stability Index (PSI) and the Kolmogorov–Smirnov (KS) test.
- **Prediction Drift Monitoring:** Tracks changes in predicted fraud rates and prediction probability distributions.
- **Delayed-Label Performance Monitoring:** Evaluates model performance when actual transaction labels become available.
- **Alert Engine:** Generates monitoring alerts and assigns an overall health status.
- **Monitoring History:** Saves and displays reports from previously monitored batches.
- **Interactive Dashboard:** Provides separate views for overview, incoming data, data monitoring, prediction monitoring, model performance, history, and alerts.

## Tech Stack

- Python
- Pandas and NumPy
- Scikit-learn
- SciPy
- Streamlit
- Plotly
- Matplotlib and Seaborn
- Joblib

## Project Workflow

1. A trained fraud detection model and reference dataset are loaded.
2. A new batch of incoming transactions is uploaded.
3. The data quality checks are performed.
4. Incoming feature distributions are compared with reference distributions.
5. The model generates predictions for the incoming batch.
6. Prediction distributions are monitored for changes.
7. The monitoring report and predictions are saved.
8. When actual labels become available, model performance is evaluated.
9. The alert engine combines monitoring signals and generates an overall health status.

## Project Structure

```text
DriftWatch/
├── data/
│   ├── creditcard.csv
│   ├── train.csv
│   ├── reference.csv
│   ├── test.csv
│   ├── batches/
│   └── incoming_samples/
├── models/
│   └── fraud_model.pkl
├── src/
│   ├── inspect_data.py
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── save_baseline.py
│   ├── generate_batches.py
│   ├── data_quality.py
│   ├── drift_detection.py
│   ├── prediction_drift.py
│   ├── performance_monitor.py
│   ├── alert_engine.py
│   ├── monitoring_pipeline.py
│   └── run_monitor.py
├── reports/
├── app.py
├── requirements.txt
└── README.md
```

## Model Information

The project uses a Random Forest classifier trained on a credit card fraud dataset.

Model configuration:

- 100 estimators
- Random state: 42
- Class weighting: balanced

Baseline test performance:

| Metric | Baseline |
|---|---:|
| Precision | 91.53% |
| Recall | 76.06% |
| F1-score | 83.08% |
| PR-AUC | 79.67% |
| ROC-AUC | 95.43% |

The original dataset is highly imbalanced, with fraud representing approximately 0.17% of transactions.

## Drift Testing

To demonstrate monitoring behaviour, controlled transaction batches were generated.

- Normal batches were sampled from the reference dataset.
- Drifted batch 1 had its `Amount` feature multiplied by 2.5.
- Drifted batch 2 had its `Amount` feature multiplied by 4.

The generated batches contain 1,000 transactions each, including 995 legitimate and 5 fraudulent transactions. This controlled 0.5% fraud proportion is intended for demonstration and is not representative of the original dataset prevalence.

In one drifted-batch experiment, DriftWatch detected drift in the `Amount` feature while the prediction distribution test did not detect statistically significant prediction drift.

This demonstrates that **feature drift does not necessarily imply prediction drift**.

## Delayed-Label Evaluation

The application supports a separate labels file containing:

- `transaction_id`
- `actual_class`

DriftWatch matches actual outcomes with saved predictions and evaluates performance using precision, recall, F1-score, and PR-AUC.

This simulates a practical monitoring situation in which transaction outcomes become available after predictions have already been generated.

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd DriftWatch
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it in Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare the required project files

Ensure the required dataset, reference data, trained model, and supporting files are present in their expected folders.

If setting up from the original dataset, run the data preparation, model training, baseline generation, and batch-generation scripts in the appropriate order.

### 5. Launch the dashboard

```bash
streamlit run app.py
```

The Streamlit application will open in your browser.

## How to Use

1. Open **Incoming Data**.
2. Upload a CSV containing transaction IDs and model features.
3. Enter a batch name and run monitoring.
4. Inspect the results in Overview, Data Monitoring, Prediction Monitoring, and Alerts.
5. Open Model Performance when actual labels become available.
6. Upload the corresponding labels CSV and evaluate performance.
7. Review previous runs in Monitoring History.

## Important Notes and Limitations

- This is an experimental portfolio project, not a production deployment.
- The generated batches are controlled test data.
- Performance metrics from batches containing only five fraud examples can vary significantly and should be interpreted cautiously.
- A detected feature drift does not automatically mean model performance has degraded.
- Monitoring results should be interpreted alongside sample sizes, business context, and actual outcomes.
- The project demonstrates monitoring concepts; it does not implement a complete production model-serving or automated retraining system.

## Future Improvements

- Add configurable drift thresholds.
- Add more detailed monitoring and alert history.
- Support larger and more realistic evaluation batches.
- Add automated monitoring schedules.
- Add model version tracking.
- Add deployment using Streamlit Community Cloud or another hosting platform.

## Author

Developed as a hands-on machine learning portfolio project to explore model monitoring, data drift, prediction drift, delayed feedback, and ML observability.
