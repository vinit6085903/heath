import streamlit as st
import requests

# =========================
# CONFIG
# =========================
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(
    page_title="EquinoxSphere | AI Disease Prediction",
    page_icon="🧠",
    layout="centered"
)

# =========================
# BRAND HEADER
# =========================
st.markdown(
    """
    <div style="text-align:center;padding:10px 0">
        <h1 style="color:#1f4fd8;">🧠 EquinoxSphere</h1>
        <h4 style="color:#555;">AI-Powered Disease Prediction System</h4>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="background-color:#f6f8fc;padding:18px;border-radius:12px;
                border-left:6px solid #1f4fd8;">
    This professional AI system analyzes patient symptoms and predicts a possible
    disease with <b>confidence scoring</b> and <b>risk assessment</b>,
    designed for real-world healthcare AI demonstrations.
    </div>
    """,
    unsafe_allow_html=True
)

st.warning(
    "⚠️ **Medical Disclaimer:** This application is for educational and "
    "demonstration purposes only. It does NOT replace professional medical advice."
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🧪 Demo Test Cases")
st.sidebar.caption("Select a scenario for instant testing")

samples = {
    "🤒 Fever / Viral Infection": "fever, headache, body pain, fatigue",
    "🤢 Stomach Disorder": "vomiting, nausea, abdominal pain",
    "🧠 Neurological Risk": "joint pain, vomiting, fever, headache",
    "🦟 Malaria Symptoms": "high fever, chills, sweating, headache",
    "🚨 Emergency Warning": "headache, vomiting, weakness, confusion"
}

selected = st.sidebar.selectbox("Choose a case", samples.keys())

if st.sidebar.button("📥 Load Sample"):
    st.session_state["symptoms"] = samples[selected]

st.sidebar.markdown("---")
st.sidebar.caption("EquinoxSphere • Responsible AI")

# =========================
# INPUT SECTION
# =========================
st.markdown("### 🩺 Patient Symptom Input")

symptoms = st.text_area(
    label="Enter symptoms (comma separated)",
    value=st.session_state.get("symptoms", ""),
    placeholder="e.g. fever, headache, vomiting, joint pain",
    height=100
)

col_clear, col_predict = st.columns([1, 2])

with col_clear:
    if st.button("🧹 Clear Input"):
        st.session_state["symptoms"] = ""
        st.experimental_rerun()

with col_predict:
    predict_btn = st.button("🔍 Analyze Symptoms", use_container_width=True)

# =========================
# PREDICTION LOGIC
# =========================
if predict_btn:

    if symptoms.strip() == "":
        st.error("❌ Please enter patient symptoms before analysis.")

    else:
        with st.spinner("🧠 EquinoxSphere AI is analyzing data..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"symptoms": symptoms},
                    timeout=20
                )

                if response.status_code == 200:
                    result = response.json()

                    st.success("✅ Analysis Completed Successfully")

                    # =========================
                    # RESULTS PANEL
                    # =========================
                    st.markdown("## 📊 Clinical Prediction Summary")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("🦠 Predicted Disease", result["Predicted Disease"])
                    with col2:
                        conf = result["Confidence (%)"]
                        st.metric("📈 Confidence Level", f"{conf}%")

                    st.progress(int(conf))

                    # =========================
                    # RISK ASSESSMENT
                    # =========================
                    risk = result["Risk Level"]

                    st.markdown("### 🚦 Risk Assessment")

                    if risk == "HIGH":
                        st.error("🚨 HIGH RISK CONDITION")
                        st.info("👨‍⚕️ Immediate medical consultation is strongly advised.")
                    elif risk == "MEDIUM":
                        st.warning("⚠️ MODERATE RISK CONDITION")
                        st.info("👨‍⚕️ Monitor symptoms and consult a healthcare professional.")
                    else:
                        st.success("ℹ️ LOW RISK CONDITION")
                        st.info("👨‍⚕️ Rest, observe symptoms, and maintain a healthy routine.")

                    # =========================
                    # DESCRIPTION
                    # =========================
                    with st.expander("📖 Medical Description"):
                        st.write(result["Description"])

                    # =========================
                    # PRECAUTIONS
                    # =========================
                    st.markdown("### 🛡️ Recommended Preventive Measures")

                    precautions = result.get("Precautions", [])
                    if not precautions:
                        st.info("No specific precautions available.")
                    else:
                        for p in precautions:
                            st.markdown(f"- ✅ **{p.capitalize()}**")

                    # =========================
                    # NOTICE
                    # =========================
                    st.markdown("---")
                    st.caption(result["Medical_Notice"])

                else:
                    st.error("❌ Backend API error. Please ensure FastAPI is running.")

            except requests.exceptions.ConnectionError:
                st.error("❌ Unable to connect to backend service.")
            except Exception as e:
                st.error(f"❌ Unexpected error: {e}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<center>© 2026 <b>EquinoxSphere</b> | AI for Health & Trust</center>",
    unsafe_allow_html=True
)
