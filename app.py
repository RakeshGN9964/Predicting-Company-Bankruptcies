import traceback
import joblib
import streamlit as st

try:
    saved = joblib.load("bankruptcy_model.pkl")
    st.success("Model Loaded Successfully")

except Exception as e:
    st.error(str(e))
    st.code(traceback.format_exc())
# ==========================================================
# PAGE CONFIG
# ==========================================================
st.set_page_config(
    page_title="BankruptSense ",
    page_icon="📊",
    layout="wide"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================
st.markdown("""
<style>

.stApp{
    background-color:"#22c55e";
}

.title{
    text-align:center;
    font-size:42px;
    font-weight:bold;
    color:#38bdf8;
}

.subtitle{
    text-align:center;
    color:#cbd5e1;
    font-size:18px;
}

div[data-testid="metric-container"]{
    background-color:#1e293b;
    border:1px solid #334155;
    padding:15px;
    border-radius:15px;
    text-align:center;
}

.sidebar-title{
    font-size:22px;
    color:white;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL
# ==========================================================
@st.cache_resource
def load_artifacts():

    saved = joblib.load("bankruptcy_model.pkl")

    model = saved["model"]
    scaler = saved["scaler"]

    return model, scaler

model, scaler = load_artifacts()

# ==========================================================
# GET TRAINED FEATURES
# ==========================================================
feature_order = list(scaler.feature_names_in_)

# ==========================================================
# HEADER
# ==========================================================
st.markdown(
"""
<div class='title'>
📊 BankruptSense 
</div>

<div class='subtitle'>
Taiwan Company Bankruptcy Prediction Platform
</div>
<br>
""",
unsafe_allow_html=True
)

# ==========================================================
# SIDEBAR
# ==========================================================
st.sidebar.markdown(
"<div class='sidebar-title'>Financial Indicators</div>",
unsafe_allow_html=True
)

st.sidebar.markdown("---")

inputs = {}

for feat in feature_order:

    inputs[feat] = st.sidebar.number_input(
        label=feat,
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.001,
        format="%.4f"
    )

# ==========================================================
# PREDICTION BUTTON
# ==========================================================
if st.button(
    "🚀 Predict Bankruptcy Risk",
    use_container_width=True
):

    # ------------------------------------------------------
    # CREATE DATAFRAME
    # ------------------------------------------------------
    input_df = pd.DataFrame([inputs])

    # Exact order
    input_df = input_df.reindex(columns=feature_order)

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)[0]

    probability = model.predict_proba(input_scaled)[0][1]

    # ------------------------------------------------------
    # KPI CARDS
    # ------------------------------------------------------
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Bankruptcy Probability",
            f"{probability*100:.2f}%"
        )

    with col2:
        st.metric(
            "Prediction",
            "Bankrupt" if prediction == 1 else "Solvent"
        )

    with col3:
        st.metric(
            "Model",
            "Voting Classifier"
        )

    st.markdown("---")

    # ------------------------------------------------------
    # GAUGE COLOR
    # ------------------------------------------------------
    if probability < 0.30:
        gauge_color = "#22c55e"

    elif probability < 0.60:
        gauge_color = "#f59e0b"

    else:
        gauge_color = "#ef4444"

    # ------------------------------------------------------
    # GAUGE CHART
    # ------------------------------------------------------
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix":"%"},
            title={"text":"Bankruptcy Risk Score"},
            gauge={
                "axis":{"range":[0,100]},
                "bar":{"color":gauge_color},
                "steps":[
                    {"range":[0,30],"color":"#22c55e"},
                    {"range":[30,60],"color":"#facc15"},
                    {"range":[60,100],"color":"#ef4444"}
                ]
            }
        )
    )

    fig.update_layout(
        height=450,
        paper_bgcolor="#0f172a",
        font={"color":"white"}
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ------------------------------------------------------
    # RISK MESSAGE
    # ------------------------------------------------------
    if probability < 0.30:

        st.success(
            f"""
            ✅ LOW RISK COMPANY

            Probability of Bankruptcy:
            {probability*100:.2f}%
            """
        )

    elif probability < 0.60:

        st.warning(
            f"""
            ⚠️ MEDIUM RISK COMPANY

            Probability of Bankruptcy:
            {probability*100:.2f}%
            """
        )

    else:

        st.error(
            f"""
            🚨 HIGH RISK COMPANY

            Probability of Bankruptcy:
            {probability*100:.2f}%
            """
        )

    st.markdown("---")

    # ------------------------------------------------------
    # INPUT SUMMARY
    # ------------------------------------------------------
    st.subheader("📑 Financial Indicators Used")

    display_df = (
        input_df.T
        .rename(columns={0:"Value"})
        .reset_index()
    )

    display_df.columns = [
        "Financial Indicator",
        "Value"
    ]

    st.dataframe(
        display_df,
        use_container_width=True
    )

# ==========================================================
# FOOTER
# ==========================================================
st.markdown("---")

st.caption(
"""
BankruptSense  • Taiwan Company Bankruptcy Dataset

6819 Companies • 95 Financial Indicators

Voting Classifier • SMOTE • Bankruptcy Prediction
"""
)
