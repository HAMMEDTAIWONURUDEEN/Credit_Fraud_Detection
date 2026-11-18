
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from pathlib import Path


# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="FraudGuard AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "credit_fraud_detection.pkl"
SCALER_PATH = BASE_DIR / "credit_fraud_detection_scaler.pkl"
SAMPLE_PATH = BASE_DIR / "sample_transactions.csv"

THRESHOLD = 0.08

FEATURES = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


# =====================================================
# 2. APPLICATION DESIGN
# =====================================================

st.markdown("""
<style>

/* General application */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #55AA83;
}

/* Main workspace */

[data-testid="stMainBlockContainer"] {
    background: #FFFFFF;
    border-radius: 16px;
    padding: 30px 35px;
    max-width: 1450px;
    margin: 24px auto;
    box-shadow: 0 8px 30px rgba(20, 50, 40, 0.12);
}

/* Sidebar */

[data-testid="stSidebar"] {
    background: #303333;
}

[data-testid="stSidebar"] * {
    color: #F4F7F5;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #D3DBD6;
}

/* Headings */

.main-title {
    color: #293D34;
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -0.7px;
    margin-bottom: 5px;
}

.page-subtitle {
    color: #89958E;
    font-size: 13px;
    margin-bottom: 28px;
}

.section-title {
    color: #34483D;
    font-size: 19px;
    font-weight: 750;
    margin-top: 15px;
    margin-bottom: 15px;
}

/* Native Streamlit metric cards */

[data-testid="stMetric"] {
    background: #F7FAF8;
    border: 1px solid #E5ECE7;
    border-radius: 12px;
    padding: 19px;
    min-height: 110px;
}

[data-testid="stMetricLabel"] {
    color: #849188;
    font-size: 11px;
    font-weight: 700;
}

[data-testid="stMetricValue"] {
    color: #293E34;
    font-size: 27px;
    font-weight: 800;
}

/* Buttons */

div.stButton > button,
div.stDownloadButton > button {
    background: #32865F;
    color: white;
    border: none;
    border-radius: 8px;
    min-height: 43px;
    font-weight: 700;
}

div.stButton > button:hover,
div.stDownloadButton > button:hover {
    background: #246D4C;
    color: white;
    border: none;
}

/* Tables */

[data-testid="stDataFrame"] {
    border: 1px solid #E6ECE8;
    border-radius: 10px;
}

/* Select boxes */

div[data-baseweb="select"] > div {
    border-radius: 8px;
}

/* Dividers */

hr {
    border-color: #E8EEE9;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# 3. LOAD MODEL AND SCALER
# =====================================================

@st.cache_resource
def load_artifacts():

    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    if model.n_features_in_ != len(FEATURES):
        raise ValueError(
            "The model does not have the expected 30 features."
        )

    if scaler.n_features_in_ != len(FEATURES):
        raise ValueError(
            "The scaler does not have the expected 30 features."
        )

    return model, scaler


# =====================================================
# 4. LOAD SAMPLE TRANSACTIONS
# =====================================================

@st.cache_data
def load_sample_data():

    data = pd.read_csv(SAMPLE_PATH)

    required_columns = (
        ["Sample_ID"] + FEATURES + ["Actual_Class"]
    )

    missing_columns = [
        col for col in required_columns
        if col not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    data = data[required_columns].copy()

    data[FEATURES] = data[FEATURES].apply(
        pd.to_numeric,
        errors="coerce"
    )

    if data[FEATURES].isnull().any().any():
        raise ValueError(
            "The sample file contains invalid feature values."
        )

    if not data["Actual_Class"].isin([0, 1]).all():
        raise ValueError(
            "Actual_Class must contain only 0 and 1."
        )

    return data


try:
    model, scaler = load_artifacts()
    samples = load_sample_data()

except Exception as error:
    st.error(f"Unable to load application files: {error}")
    st.stop()


# =====================================================
# 5. PREDICTION FUNCTION
# =====================================================

def predict_transactions(data):

    X = data[FEATURES].copy()

    X_scaled = scaler.transform(X)

    fraud_index = list(model.classes_).index(1)

    probabilities = model.predict_proba(
        X_scaled
    )[:, fraud_index]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    return predictions, probabilities


# =====================================================
# 6. ANALYSE SAMPLE TRANSACTIONS
# =====================================================

predictions, probabilities = predict_transactions(samples)

results = samples[
    ["Sample_ID", "Amount", "Actual_Class"]
].copy()

results["Fraud Probability (%)"] = (
    probabilities * 100
).round(2)

results["Prediction"] = np.where(
    predictions == 1,
    "Suspicious",
    "Not Flagged"
)

results["Actual Label"] = np.where(
    results["Actual_Class"] == 1,
    "Fraud",
    "Legitimate"
)

results["Correct"] = (
    predictions == samples["Actual_Class"].to_numpy()
)


# =====================================================
# 7. SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.markdown("""
    <div style="padding: 20px 5px 25px 5px;">

        <div style="font-size:29px; font-weight:800;">
            <span style="color:#F4B942;">●</span>
            FraudGuard
        </div>

        <div style="font-size:10px; color:#AAB5AE;">
            TRANSACTION MONITORING
        </div>

    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "Dashboard",
            "Transaction Analysis",
            "Model Performance"
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        "<p style='font-size:10px; color:#AAB5AE;'>"
        "SYSTEM STATUS</p>",
        unsafe_allow_html=True
    )

    st.success("Prediction engine online")


# =====================================================
# 8. APPLICATION HEADER
# =====================================================

st.markdown(
    '<div class="main-title">FraudGuard AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="page-subtitle">'
    'Transaction monitoring and analysis'
    '</div>',
    unsafe_allow_html=True
)


# =====================================================
# 9. DASHBOARD
# =====================================================

if page == "Dashboard":

    st.markdown(
        '<div class="section-title">Dashboard Overview</div>',
        unsafe_allow_html=True
    )

    total_transactions = len(results)

    flagged = int((predictions == 1).sum())

    not_flagged = total_transactions - flagged

    flag_rate = (
        flagged / total_transactions * 100
        if total_transactions > 0 else 0
    )

    # Dashboard metric cards

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "TOTAL TRANSACTIONS",
        total_transactions
    )

    col2.metric(
        "FLAGGED TRANSACTIONS",
        flagged
    )

    col3.metric(
        "NOT FLAGGED",
        not_flagged
    )

    col4.metric(
        "FLAG RATE",
        f"{flag_rate:.1f}%"
    )

    st.markdown("---")

    left, right = st.columns([1.3, 1])

    with left:

        st.markdown(
            '<div class="section-title">Transaction Register</div>',
            unsafe_allow_html=True
        )

        dashboard_table = results[
            [
                "Sample_ID",
                "Amount",
                "Prediction",
                "Fraud Probability (%)"
            ]
        ].rename(
            columns={
                "Sample_ID": "Transaction ID"
            }
        )

        st.dataframe(
            dashboard_table,
            use_container_width=True,
            hide_index=True
        )

    with right:

        st.markdown(
            '<div class="section-title">Prediction Distribution</div>',
            unsafe_allow_html=True
        )

        chart_data = pd.DataFrame(
            {
                "Transactions": [
                    flagged,
                    not_flagged
                ]
            },
            index=[
                "Suspicious",
                "Not Flagged"
            ]
        )

        st.bar_chart(chart_data)

    st.markdown("---")

    st.markdown(
        '<div class="section-title">Export Report</div>',
        unsafe_allow_html=True
    )

    report = results.copy()

    report["Actual_Class"] = report["Actual_Class"].map({
        0: "Legitimate",
        1: "Fraud"
    })

    report["Correct"] = report["Correct"].map({
        True: "Yes",
        False: "No"
    })

    st.download_button(
        label="DOWNLOAD TRANSACTION REPORT",
        data=report.to_csv(index=False),
        file_name="fraudguard_analysis_report.csv",
        mime="text/csv"
    )

    st.caption(
        "The dashboard displays the supplied sample transactions, "
        "not live banking records."
    )


# =====================================================
# 10. TRANSACTION ANALYSIS
# =====================================================

elif page == "Transaction Analysis":

    st.markdown(
        '<div class="section-title">Transaction Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Select a transaction to view its analysis."
    )

    table = results[
        [
            "Sample_ID",
            "Amount",
            "Prediction",
            "Fraud Probability (%)"
        ]
    ].rename(
        columns={
            "Sample_ID": "Transaction ID",
            "Amount": "Transaction Amount"
        }
    )

    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.markdown("### Analyse Individual Transaction")

    selected_id = st.selectbox(
        "Select transaction ID",
        results["Sample_ID"].tolist()
    )

    if st.button(
        "ANALYSE TRANSACTION",
        use_container_width=True
    ):
        st.session_state["analysed_id"] = selected_id

    if "analysed_id" in st.session_state:

        selected_id = st.session_state["analysed_id"]

        position = samples.index[
            samples["Sample_ID"] == selected_id
        ][0]

        selected = results.loc[position]

        probability = float(
            selected["Fraud Probability (%)"]
        )

        prediction = int(predictions[position])

        actual = int(selected["Actual_Class"])

        st.markdown("---")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Transaction Amount",
            f"{selected['Amount']:,.2f}"
        )

        col2.metric(
            "Fraud Probability",
            f"{probability:.2f}%"
        )

        col3.metric(
            "Decision Threshold",
            "8.00%"
        )

        st.markdown("### Detection Result")

        if prediction == 1:

            st.error(
                "SUSPICIOUS TRANSACTION — The model flagged "
                "this transaction for further review."
            )

        else:

            st.success(
                "NOT FLAGGED — The model did not flag "
                "this transaction at the selected threshold."
            )

        st.markdown("### Dataset Reference")

        if actual == 1:
            st.write("Actual dataset label: **Fraud**")
        else:
            st.write("Actual dataset label: **Legitimate**")

        if prediction == actual:

            st.success(
                "The prediction matches the dataset label."
            )

        else:

            st.warning(
                "The prediction differs from the dataset label."
            )

        st.caption(
            "The actual dataset label is shown for comparison only. "
            "It is not used as a model input."
        )


# =====================================================
# 11. MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="section-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    col1.metric("ROC-AUC", "97.93%")
    col2.metric("PR-AUC", "78.28%")
    col3.metric("FRAUD RECALL", "83.67%")

    st.markdown("---")

    st.markdown("### Classification Metrics")

    performance = pd.DataFrame({
        "Evaluation Metric": [
            "Accuracy",
            "Fraud Precision",
            "Fraud Recall",
            "Fraud F1-score",
            "ROC-AUC",
            "PR-AUC"
        ],
        "Performance": [
            "99.91%",
            "70.69%",
            "83.67%",
            "76.64%",
            "97.93%",
            "78.28%"
        ]
    })

    st.dataframe(
        performance,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.markdown("### Confusion Matrix")

    confusion_matrix = pd.DataFrame(
        [
            [28415, 17],
            [8, 41]
        ],
        index=["Actual Legitimate", "Actual Fraud"],
        columns=["Predicted Legitimate", "Predicted Fraud"]
    )

    st.dataframe(
        confusion_matrix,
        use_container_width=True
    )

    st.caption(
        "Rows represent actual classes; columns represent model predictions."
    )

    st.markdown("---")

    st.markdown("### Model Configuration")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Algorithm:** Logistic Regression")
        st.write("**Regularisation parameter (C):** 0.01")
        st.write("**Solver:** lbfgs")

    with col2:
        st.write("**Class weight:** None")
        st.write("**Decision threshold:** 0.08")
        st.write("**Input features:** 30")

    st.caption(
        "ROC-AUC and PR-AUC are probability-based evaluation metrics "
        "and do not change when the classification threshold changes."
    )

