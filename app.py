
import json
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# --------------------------------------------------
# PATHS AND PIPELINE IMPORT
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from monitoring_pipeline import run_monitoring

REPORTS_DIR = PROJECT_ROOT / "reports"
INCOMING_DIR = PROJECT_ROOT / "data" / "incoming_samples"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
INCOMING_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DriftWatch | Model Observability",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #0b1120;
    color: #e5e7eb;
}
[data-testid="stSidebar"] {
    background-color: #111827;
    border-right: 1px solid #273449;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
}
h1, h2, h3 {
    color: #f8fafc !important;
}
p, label {
    color: #cbd5e1;
}
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #172338, #111c2e);
    border: 1px solid #293b54;
    padding: 18px;
    border-radius: 12px;
}
div[data-testid="stMetricLabel"] {
    color: #94a3b8;
}
div[data-testid="stMetricValue"] {
    color: #f8fafc;
}
hr {
    border-color: #273449;
}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def get_reports():
    files = sorted(
        REPORTS_DIR.glob("*_monitoring_report.json"),
        key=lambda file: file.stat().st_mtime,
        reverse=True
    )

    reports = {}

    for file in files:
        try:
            with open(file, "r") as f:
                reports[file.name.replace(
                    "_monitoring_report.json", ""
                )] = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

    return reports


def safe_batch_id(value):
    value = re.sub(r"[^a-zA-Z0-9_-]", "_", value.strip())
    return value[:60]


def style_chart(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1120",
        plot_bgcolor="#111827",
        font_color="#e5e7eb"
    )
    return fig


def show_status(status, alert_count):
    colors = {
        "HEALTHY": "#16a34a",
        "WARNING": "#d97706",
        "CRITICAL": "#dc2626"
    }

    color = colors.get(status, "#64748b")

    st.markdown(
        f"""
        <div style="
            background: linear-gradient(90deg, #172338, #111827);
            border: 1px solid {color};
            border-left: 6px solid {color};
            padding: 18px 22px;
            border-radius: 12px;
            margin-bottom: 22px;
        ">
            <span style="color:#94a3b8;font-size:13px;">
                OVERALL MODEL HEALTH
            </span>
            <h2 style="color:{color};margin:5px 0;">
                {status}
            </h2>
            <span style="color:#cbd5e1;">
                {alert_count} active alert(s)
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.markdown("# 🔍 DriftWatch")
st.sidebar.caption("ML MODEL OBSERVABILITY PLATFORM")
st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Overview",
        "Incoming Data",
        "Data Monitoring",
        "Prediction Monitoring",
        "Model Performance",
        "Monitoring History",
        "Alerts"
    ]
)

reports = get_reports()

st.sidebar.divider()
st.sidebar.markdown("### Monitoring Reports")

if reports:
    selected_batch = st.sidebar.selectbox(
        "Select monitoring batch",
        list(reports.keys())
    )
    report = reports[selected_batch]
else:
    selected_batch = None
    report = None
    st.sidebar.info("No monitoring reports yet.")


# --------------------------------------------------
# INCOMING DATA PAGE
# --------------------------------------------------

if page == "Incoming Data":

    st.title("Incoming Transaction Data")
    st.caption("Upload a batch of transactions and run DriftWatch monitoring.")

    st.info(
        "Upload feature data only. Do not include Class or actual_class. "
        "Keep transaction_id so predictions can be matched with delayed labels."
    )

    batch_name = st.text_input(
        "Batch name",
        value="uploaded_batch"
    )

    uploaded_file = st.file_uploader(
        "Upload incoming CSV",
        type=["csv"],
        key="incoming_csv"
    )

    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)

            st.markdown("### Data preview")
            st.write(
                f"Rows: {len(uploaded_df):,} | "
                f"Columns: {len(uploaded_df.columns)}"
            )
            st.dataframe(uploaded_df.head(10), use_container_width=True)

            if st.button("Run Monitoring", type="primary"):

                batch_id = safe_batch_id(batch_name)

                if not batch_id:
                    st.error("Please enter a valid batch name.")
                else:
                    try:
                        with st.spinner("Running DriftWatch monitoring..."):

                            # Save uploaded input for later label evaluation
                            input_path = (
                                INCOMING_DIR /
                                f"{batch_id}_incoming.csv"
                            )
                            uploaded_df.to_csv(input_path, index=False)

                            result = run_monitoring(
                                uploaded_df,
                                batch_id=batch_id
                            )

                        st.success("Monitoring completed successfully!")

                        st.subheader("Monitoring Results")

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Overall Status",
                            result["alerts"]["overall_status"]
                        )
                        col2.metric(
                            "Alerts",
                            result["alerts"]["alert_count"]
                        )
                        col3.metric(
                            "Label Status",
                            result["label_status"]
                        )

                        st.caption(
                            f"Report saved: reports/"
                            f"{batch_id}_monitoring_report.json"
                        )

                        st.info("Open Overview or Monitoring History to inspect the saved results.")

                    except Exception as error:
                        st.error(f"Monitoring failed: {error}")

        except Exception as error:
            st.error(f"Could not read CSV: {error}")


# --------------------------------------------------
# LOAD SELECTED REPORT FOR OTHER PAGES
# --------------------------------------------------

else:

    if not reports or report is None:
        st.title("DriftWatch")
        st.info(
            "No monitoring reports found. Go to Incoming Data "
            "to upload a batch and run monitoring."
        )
        st.stop()

    alerts = report.get("alerts", {})
    quality = report.get("data_quality", {})
    performance = report.get("performance")
    comparison = report.get("performance_comparison")
    feature_drift = pd.DataFrame(report.get("feature_drift", []))
    prediction_drift = report.get("prediction_drift", {})

    status = alerts.get("overall_status", "UNKNOWN")
    alert_count = alerts.get("alert_count", 0)
    label_status = report.get("label_status", "AVAILABLE" if performance else "PENDING")

    st.title("DriftWatch")
    st.caption("Fraud Model Monitoring and Observability")
    st.markdown(f"**Selected batch:** `{selected_batch}`")
    st.caption(f"Label status: {label_status}")

    show_status(status, alert_count)
    st.divider()


    # --------------------------------------------------
    # OVERVIEW
    # --------------------------------------------------

    if page == "Overview":

        st.subheader("Monitoring Overview")

        if performance:
            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Precision", f"{performance['precision']:.1%}")
            col2.metric("Recall", f"{performance['recall']:.1%}")
            col3.metric("F1 Score", f"{performance['f1_score']:.1%}")
            col4.metric("PR-AUC", f"{performance['pr_auc']:.1%}")
        else:
            st.info(
                "Performance metrics are pending. Upload delayed labels "
                "on the Model Performance page when they become available."
            )

        st.markdown("### System Health")

        col1, col2, col3 = st.columns(3)

        col1.metric("Data Quality", quality.get("status", "N/A"))
        col2.metric(
            "Drifted Features",
            int(feature_drift["drift_detected"].sum())
            if not feature_drift.empty else 0
        )
        col3.metric("Active Alerts", alert_count)

        if comparison:
            st.divider()
            st.markdown("### Model Performance vs Baseline")

            metrics = ["precision", "recall", "f1_score", "pr_auc"]

            chart_data = pd.DataFrame({
                "Metric": metrics,
                "Baseline": [comparison[m]["baseline"] for m in metrics],
                "Current": [comparison[m]["current"] for m in metrics]
            })

            chart_data = chart_data.melt(
                id_vars="Metric",
                var_name="Evaluation",
                value_name="Score"
            )

            fig = px.bar(
                chart_data,
                x="Metric",
                y="Score",
                color="Evaluation",
                barmode="group",
                title="Baseline vs Current Performance",
                color_discrete_sequence=["#64748b", "#38bdf8"]
            )
            fig.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(style_chart(fig), use_container_width=True)


    # --------------------------------------------------
    # DATA MONITORING
    # --------------------------------------------------

    elif page == "Data Monitoring":

        st.subheader("Data Quality Monitoring")

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", quality.get("total_rows", 0))
        col2.metric("Missing Values", quality.get("missing_values", 0))
        col3.metric("Duplicate Rows", quality.get("duplicate_rows", 0))

        if quality.get("issues"):
            for issue in quality["issues"]:
                st.warning(issue)
        else:
            st.success("No data quality issues detected.")

        st.divider()
        st.subheader("Feature Drift Analysis")

        if not feature_drift.empty:

            fig = px.bar(
                feature_drift.sort_values(
                    "psi", ascending=False
                ).head(15),
                x="feature",
                y="psi",
                color="drift_detected",
                title="Top 15 Features by PSI",
                color_discrete_map={
                    True: "#ef4444",
                    False: "#38bdf8"
                }
            )

            st.plotly_chart(style_chart(fig), use_container_width=True)

            st.markdown("### Feature Drift Details")
            st.dataframe(
                feature_drift.sort_values("psi", ascending=False),
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info("No feature drift results available.")


    # --------------------------------------------------
    # PREDICTION MONITORING
    # --------------------------------------------------

    elif page == "Prediction Monitoring":

        st.subheader("Prediction Distribution Monitoring")

        if prediction_drift:

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Reference Fraud Rate",
                f"{prediction_drift['reference_fraud_rate']:.3%}"
            )
            col2.metric(
                "Current Fraud Rate",
                f"{prediction_drift['current_fraud_rate']:.3%}"
            )
            col3.metric(
                "Fraud Rate Change",
                f"{prediction_drift['fraud_rate_change']:+.3%}"
            )

            st.divider()
            st.markdown("### Probability Distribution Test")

            col1, col2 = st.columns(2)

            col1.metric(
                "KS Statistic",
                f"{prediction_drift['probability_ks_statistic']:.4f}"
            )
            col2.metric(
                "P-value",
                f"{prediction_drift['probability_p_value']:.4g}"
            )

            if prediction_drift["prediction_drift_detected"]:
                st.warning("Prediction drift detected.")
            else:
                st.success("No prediction drift detected.")
        else:
            st.info("Prediction drift results are unavailable.")


    # --------------------------------------------------
    # MODEL PERFORMANCE AND DELAYED LABELS
    # --------------------------------------------------

    elif page == "Model Performance":

        st.subheader("Delayed-Label Performance Evaluation")

        st.info(
            "Upload a separate CSV containing transaction_id and "
            "actual_class. DriftWatch will match these labels with "
            "the saved incoming batch."
        )

        if performance and comparison:

            st.success("Labels have been evaluated for this batch.")

            metrics = ["precision", "recall", "f1_score", "pr_auc"]

            comparison_df = pd.DataFrame([
                {
                    "Metric": metric.upper(),
                    "Baseline": comparison[metric]["baseline"],
                    "Current": comparison[metric]["current"],
                    "Change": comparison[metric]["change"]
                }
                for metric in metrics
            ])

            st.dataframe(
                comparison_df.style.format({
                    "Baseline": "{:.3%}",
                    "Current": "{:.3%}",
                    "Change": "{:+.3%}"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.divider()
            st.markdown("### Confusion Matrix")

            cm = performance["confusion_matrix"]

            fig = go.Figure(
                data=go.Heatmap(
                    z=cm,
                    x=["Predicted Legitimate", "Predicted Fraud"],
                    y=["Actual Legitimate", "Actual Fraud"],
                    colorscale="Blues",
                    text=cm,
                    texttemplate="%{text}",
                    showscale=True
                )
            )

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1120",
                title="Actual vs Predicted Transactions"
            )

            st.plotly_chart(fig, use_container_width=True)

        else:

            st.warning("Performance evaluation is pending for this batch.")

            incoming_path = (
                INCOMING_DIR / f"{selected_batch}_incoming.csv"
            )

            if incoming_path.exists():

                label_file = st.file_uploader(
                    "Upload delayed labels CSV",
                    type=["csv"],
                    key=f"labels_{selected_batch}"
                )

                if label_file is not None:

                    labels_df = pd.read_csv(label_file)
                    st.write("Label data preview")
                    st.dataframe(labels_df.head(10), use_container_width=True)

                    if st.button("Evaluate Model Performance", type="primary"):

                        try:
                            with st.spinner("Evaluating delayed labels..."):

                                incoming_df = pd.read_csv(incoming_path)

                                result = run_monitoring(
                                    incoming_df,
                                    batch_id=selected_batch,
                                    labels_df=labels_df
                                )

                            st.success("Performance evaluation completed!")
                            st.metric(
                                "F1 Score",
                                f"{result['performance']['f1_score']:.3f}"
                            )
                            st.rerun()

                        except Exception as error:
                            st.error(f"Evaluation failed: {error}")

            else:
                st.warning(
                    "The original incoming CSV is not available for this "
                    "report. Upload the batch again through Incoming Data."
                )


    # --------------------------------------------------
    # MONITORING HISTORY
    # --------------------------------------------------

    elif page == "Monitoring History":

        st.subheader("Monitoring Run History")

        history = []

        for batch_id, item in reports.items():

            item_alerts = item.get("alerts", {})

            history.append({
                "Batch": batch_id,
                "Label Status": item.get("label_status", "AVAILABLE" if item.get("performance") else "PENDING"),
                "Overall Status": item_alerts.get("overall_status", "UNKNOWN"),
                "Alert Count": item_alerts.get("alert_count", 0),
                "Drifted Features": sum(
                    row.get("drift_detected", False)
                    for row in item.get("feature_drift", [])
                )
            })

        history_df = pd.DataFrame(history)

        st.dataframe(
            history_df,
            use_container_width=True,
            hide_index=True
        )


    # --------------------------------------------------
    # ALERTS
    # --------------------------------------------------

    elif page == "Alerts":

        st.subheader("Monitoring Alerts")

        alert_list = alerts.get("alerts", [])

        if not alert_list:
            st.success("No active alerts. All configured checks passed.")
        else:
            for alert in alert_list:

                if alert["severity"] == "CRITICAL":
                    st.error(
                        f"🚨 {alert['type']} — {alert['message']}"
                    )
                else:
                    st.warning(
                        f"⚠️ {alert['type']} — {alert['message']}"
                    )

        st.divider()
        st.markdown("### Alert Summary")

        col1, col2 = st.columns(2)
        col1.metric("Total Alerts", alert_count)
        col2.metric("Overall Status", status)

        if label_status == "PENDING":
            st.info(
                "Performance-related alerts are not evaluated until "
                "delayed labels become available."
            )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "DriftWatch | Fraud Model Monitoring | "
    "Experimental portfolio project"
)
