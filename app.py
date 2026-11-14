
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="BreastCare AI",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# 2. LOAD THE EXISTING MODEL AND SCALER
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "breast_cancer_prediction.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    return model, scaler

try:
    model, scaler = load_model()
    model_loaded = True
except Exception as error:
    model_loaded = False
    st.error(f"Unable to load the model or scaler: {error}")
    st.stop()

# =========================================================
# 3. FEATURE DEFINITIONS
# =========================================================

mean_features = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension"
]

error_features = [
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error"
]

worst_features = [
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension"
]

features = mean_features + error_features + worst_features

# =========================================================
# 4. CUSTOM NAVY AND TEAL DESIGN
# =========================================================

st.markdown("""
<style>

.stApp {
    background: #081321;
    color: #EAF2F8;
}

[data-testid="stHeader"] {
    background: rgba(8, 19, 33, 0.95);
}

[data-testid="stSidebar"] {
    background: #0D1B2A;
    border-right: 1px solid #20364D;
}

[data-testid="stSidebar"] * {
    color: #EAF2F8;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1450px;
}

h1, h2, h3, h4 {
    color: #F3F8FC !important;
}

p, label, span {
    color: #C5D3DF;
}

.brand-title {
    font-size: 29px;
    font-weight: 800;
    color: #42D6C5;
    letter-spacing: 0.5px;
}

.brand-subtitle {
    color: #91A9BC;
    font-size: 12px;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero {
    background: linear-gradient(120deg, #11283D, #123B4B);
    border: 1px solid #24566A;
    padding: 32px;
    border-radius: 20px;
    margin-bottom: 25px;
}

.hero-label {
    color: #42D6C5;
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
}

.hero h1 {
    font-size: 36px;
    margin-bottom: 10px;
}

.hero p {
    color: #C0D2DF;
    font-size: 15px;
    max-width: 750px;
}

.section-heading {
    color: #42D6C5;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.metric-card {
    background: #102438;
    border: 1px solid #29445B;
    border-radius: 15px;
    padding: 20px;
    min-height: 115px;
}

.metric-label {
    color: #91A9BC;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.metric-value {
    color: #42D6C5;
    font-size: 28px;
    font-weight: 800;
    margin-top: 8px;
}

.info-card {
    background: #102438;
    border: 1px solid #29445B;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 15px;
}

.result-card {
    background: linear-gradient(135deg, #102B40, #123C4A);
    border: 1px solid #328A89;
    border-radius: 18px;
    padding: 25px;
    margin-top: 15px;
}

.result-title {
    color: #42D6C5;
    font-size: 13px;
    font-weight: 800;
    letter-spacing: 1.5px;
    text-transform: uppercase;
}

.result-value {
    font-size: 30px;
    font-weight: 800;
    color: #FFFFFF;
    margin-top: 8px;
}

.small-note {
    color: #91A9BC;
    font-size: 12px;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #102438;
    border: 1px solid #29445B;
    border-radius: 15px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
    background: #0D1B2A;
    padding: 10px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    background: #142A3E;
    border-radius: 9px;
    color: #C5D3DF;
    padding: 12px 20px;
}

.stTabs [aria-selected="true"] {
    background: #126B70 !important;
    color: white !important;
}

.stButton > button {
    background: linear-gradient(90deg, #087E83, #18A99C);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 13px 25px;
    font-weight: 800;
    min-height: 48px;
    width: 100%;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #11999B, #37C9B7);
    color: white;
    border: none;
}

.stNumberInput input {
    background: #0B1B2B !important;
    color: white !important;
    border: 1px solid #34516A !important;
    border-radius: 8px !important;
}

.stProgress > div > div > div {
    background-color: #35C9B6;
}

hr {
    border-color: #29445B;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# 5. SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown('<div class="brand-title">🧬 BreastCare AI</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div class="brand-subtitle">Intelligent Research Platform</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Prediction Workspace",
            "Model Intelligence",
            "About System"
        ],
        label_visibility="visible"
    )

    st.markdown("---")

    st.markdown("### SYSTEM STATUS")

    st.success("Prediction model loaded")
    st.success("StandardScaler loaded")

    st.markdown(
        """
        <div class="small-note">
        Model: Logistic Regression<br>
        Input variables: 30<br>
        Preprocessing: StandardScaler
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown(
        '<div class="small-note">BREASTCARE AI · RESEARCH EDITION</div>',
        unsafe_allow_html=True
    )

# =========================================================
# 6. REUSABLE HERO SECTION
# =========================================================

def show_hero(label, heading, description):

    st.markdown(
        f"""
        <div class="hero">
            <div class="hero-label">{label}</div>
            <h1>{heading}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

# =========================================================
# 7. OVERVIEW PAGE
# =========================================================

if page == "Overview":

    show_hero(
        "INTELLIGENT ANALYSIS PLATFORM",
        "Breast Cancer Prediction System",
        "A machine-learning research interface for exploring "
        "breast tumour measurement patterns and model predictions."
    )

    st.markdown("### SYSTEM OVERVIEW")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Input variables</div>
            <div class="metric-value">30</div>
            <div class="small-note">Tumour measurements</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">ML algorithm</div>
            <div class="metric-value">LR</div>
            <div class="small-note">Logistic Regression</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Preprocessing</div>
            <div class="metric-value">Scaler</div>
            <div class="small-note">StandardScaler</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    st.markdown("### HOW THE SYSTEM WORKS")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class="info-card">
            <h4>01 · Measurement Input</h4>
            <p>Enter the 30 numerical features used by the trained model.</p>
        </div>
        <div class="info-card">
            <h4>02 · Data Preprocessing</h4>
            <p>The saved StandardScaler transforms the measurements
            before prediction.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="info-card">
            <h4>03 · Model Analysis</h4>
            <p>The Logistic Regression model processes the scaled
            measurements.</p>
        </div>
        <div class="info-card">
            <h4>04 · Prediction Output</h4>
            <p>The system displays its predicted class and estimated
            class probabilities.</p>
        </div>
        """, unsafe_allow_html=True)

    st.info(
        "Open Prediction Workspace from the sidebar to enter measurements."
    )

# =========================================================
# 8. INPUT FUNCTION
# =========================================================

def display_measurement_group(group_title, group_description, group_features, key_prefix):

    st.markdown(f"### {group_title}")
    st.caption(group_description)

    values = {}

    # Each measurement group is divided into two columns.
    left_column, right_column = st.columns(2)

    for index, feature in enumerate(group_features):

        target_column = left_column if index % 2 == 0 else right_column

        with target_column:

            with st.container(border=True):

                st.markdown(f"**{feature.replace('_', ' ').title()}**")

                values[feature] = st.number_input(
                    label=feature,
                    min_value=0.0,
                    value=0.0,
                    step=0.01,
                    format="%.4f",
                    key=f"{key_prefix}_{feature}",
                    label_visibility="collapsed"
                )

    return values

# =========================================================
# 9. PREDICTION WORKSPACE
# =========================================================

if page == "Prediction Workspace":

    show_hero(
        "CLINICAL MEASUREMENT INTERFACE",
        "Prediction Workspace",
        "Enter the tumour measurements below. The system will "
        "process the 30 features using your saved model and scaler."
    )

    st.markdown("### MEASUREMENT COMPLETION")

    # The three tabs are all executed by Streamlit.
    # This allows the app to collect values from every section.

    tab1, tab2, tab3 = st.tabs([
        "01 · Mean Measurements",
        "02 · Standard Error",
        "03 · Worst Measurements"
    ])

    input_values = {}

    with tab1:

        mean_values = display_measurement_group(
            "Mean Tumour Measurements",
            "Average measurements calculated from the tumour image.",
            mean_features,
            "mean"
        )

        input_values.update(mean_values)

    with tab2:

        error_values = display_measurement_group(
            "Standard Error Measurements",
            "Variation and standard error measurements.",
            error_features,
            "error"
        )

        input_values.update(error_values)

    with tab3:

        worst_values = display_measurement_group(
            "Worst Tumour Measurements",
            "The largest or most extreme measurements recorded.",
            worst_features,
            "worst"
        )

        input_values.update(worst_values)

    # =====================================================
    # 10. LIVE COMPLETION COUNTER
    # =====================================================

    completed_values = sum(
        1 for feature in features
        if input_values.get(feature, 0.0) > 0
    )

    completion_percentage = completed_values / len(features)

    st.markdown("---")

    counter_col, progress_col = st.columns([1, 3])

    with counter_col:

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">Measurements entered</div>
                <div class="metric-value">
                    {completed_values}/30
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with progress_col:

        st.markdown("**INPUT COMPLETION PROGRESS**")

        st.progress(completion_percentage)

        st.caption(
            f"{completion_percentage * 100:.1f}% of the measurements "
            "have been entered."
        )

    st.markdown("---")

    st.markdown("### RUN MODEL ANALYSIS")

    st.write(
        "Check that all 30 measurements have been entered before "
        "running the prediction."
    )

    analyze_button = st.button(
        "⚡ RUN BREAST CANCER ANALYSIS",
        use_container_width=True
    )

    # =====================================================
    # 11. MODEL PREDICTION
    # =====================================================

    if analyze_button:

        if completed_values != 30:

            st.warning(
                f"You have entered {completed_values} out of 30 "
                "measurements. Please complete all fields."
            )

        else:

            try:

                input_df = pd.DataFrame(
                    [[input_values[feature] for feature in features]],
                    columns=features
                )

                # Apply the same scaler used during model training.
                scaled_input = scaler.transform(input_df)

                # Generate prediction.
                prediction = model.predict(scaled_input)[0]

                # Generate class probabilities.
                probabilities = model.predict_proba(scaled_input)[0]

                # Map model classes to their probabilities.
                probability_map = {
                    int(class_value): float(probability)
                    for class_value, probability
                    in zip(model.classes_, probabilities)
                }

                # Confirmed output encoding:
                # 0 = Malignant
                # 1 = Benign

                prediction_name = {
                    0: "Malignant",
                    1: "Benign"
                }.get(int(prediction), "Unknown")

                malignant_probability = (
                    probability_map.get(0, 0.0) * 100
                )

                benign_probability = (
                    probability_map.get(1, 0.0) * 100
                )

                # Save the result so it remains visible on reruns.
                st.session_state["prediction_result"] = {
                    "prediction": prediction_name,
                    "malignant_probability": malignant_probability,
                    "benign_probability": benign_probability,
                    "input_values": input_values.copy()
                }

            except Exception as error:

                st.error(f"Prediction error: {error}")

    # =====================================================
    # 12. DISPLAY RESULTS AND GRAPH
    # =====================================================

    if "prediction_result" in st.session_state:

        result = st.session_state["prediction_result"]

        st.markdown("---")

        st.markdown("## ANALYSIS RESULTS")

        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-title">MODEL PREDICTION</div>
                <div class="result-value">{result["prediction"]}</div>
                <p>
                    This is the classification produced by the trained
                    machine-learning model.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("### CLASS PROBABILITY DISTRIBUTION")

        probability_col1, probability_col2 = st.columns(2)

        with probability_col1:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Malignant probability</div>
                    <div class="metric-value">
                        {result["malignant_probability"]:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with probability_col2:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">Benign probability</div>
                    <div class="metric-value">
                        {result["benign_probability"]:.2f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Build graph data from the actual model probabilities.
        graph_data = pd.DataFrame({
            "Class": ["Malignant", "Benign"],
            "Probability (%)": [
                result["malignant_probability"],
                result["benign_probability"]
            ]
        })

        st.markdown("### PROBABILITY COMPARISON GRAPH")

        st.bar_chart(
            graph_data,
            x="Class",
            y="Probability (%)",
            color="#35C9B6",
            height=350
        )

        st.caption(
            "The graph displays the two class probabilities returned "
            "by the trained model for this input."
        )

        with st.expander("VIEW ALL 30 SUBMITTED MEASUREMENTS"):

            submitted_df = pd.DataFrame({
                "Feature": features,
                "Entered Value": [
                    result["input_values"][feature]
                    for feature in features
                ]
            })

            st.dataframe(
                submitted_df,
                use_container_width=True,
                hide_index=True
            )

        st.warning(
            "Research and educational use only. This model is not a "
            "clinically validated diagnostic system. Its estimated "
            "probabilities must not be interpreted as a medical diagnosis."
        )

# =========================================================
# 13. MODEL INTELLIGENCE PAGE
# =========================================================

if page == "Model Intelligence":

    show_hero(
        "MODEL INFORMATION",
        "Model Intelligence",
        "Technical information about the machine-learning model "
        "and preprocessing components loaded by this application."
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="info-card">
            <h4>Machine Learning Algorithm</h4>
            <h2 style="color:#42D6C5;">Logistic Regression</h2>
            <p>A supervised classification algorithm.</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="info-card">
            <h4>Data Preprocessing</h4>
            <h2 style="color:#42D6C5;">StandardScaler</h2>
            <p>Transforms numerical features before prediction.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### MODEL CONFIGURATION")

    st.write("Number of input features:", len(features))
    st.write("Model classes:", model.classes_)
    st.write("Model file:", "breast_cancer_prediction.pkl")
    st.write("Scaler file:", "scaler.pkl")

    if hasattr(model, "n_features_in_"):
        st.write("Model expected features:", model.n_features_in_)

    if hasattr(scaler, "n_features_in_"):
        st.write("Scaler expected features:", scaler.n_features_in_)

    st.info(
        "The application loads the existing trained model. "
        "It does not retrain the model when a prediction is requested."
    )

# =========================================================
# 14. ABOUT PAGE
# =========================================================

if page == "About System":

    show_hero(
        "ABOUT THE PLATFORM",
        "BreastCare AI",
        "A machine-learning application built to demonstrate "
        "numerical feature processing and binary classification."
    )

    st.markdown("### SYSTEM COMPONENTS")

    st.markdown("""
    <div class="info-card">
        <h4>Python</h4>
        <p>Application programming language.</p>
    </div>

    <div class="info-card">
        <h4>Streamlit</h4>
        <p>Interactive application interface.</p>
    </div>

    <div class="info-card">
        <h4>Scikit-learn</h4>
        <p>Machine-learning model and preprocessing framework.</p>
    </div>

    <div class="info-card">
        <h4>Joblib</h4>
        <p>Loads the saved model and scaler files.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### IMPORTANT DISCLAIMER")

    st.warning(
        "BreastCare AI is an educational and research project. "
        "It is not a substitute for professional medical examination, "
        "diagnosis, or treatment. Its predictions should not be used "
        "to make medical decisions."
    )

# =========================================================
# 15. FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div style="text-align:center; padding:15px;">
        <div style="color:#42D6C5; font-weight:800;">
            BREASTCARE AI
        </div>
        <div style="color:#91A9BC; font-size:12px;">
            Machine Learning Research Platform
        </div>
    </div>
    """,
    unsafe_allow_html=True
)
