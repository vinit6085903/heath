import streamlit as st
import requests

# =========================
# CONFIG
# =========================
API_URL = "https://heath-2.onrender.com/predict"

st.set_page_config(
    page_title="EquinoxSphere | AI Disease Prediction",
    page_icon="🧠",
    layout="centered"
)

# =========================
# CUSTOM CSS (PRO UI)
# =========================
st.markdown("""
<style>
.main { background-color: #f9fafc; }
.card {
    background: white;
    padding: 20px;
    border-radius: 14px;
    box-shadow: 0px 6px 20px rgba(0,0,0,0.08);
    margin-bottom: 20px;
}
.title {
    text-align: center;
    color: #1f4fd8;
}
.subtitle {
    text-align: center;
    color: #555;
}
.badge {
    padding: 6px 12px;
    border-radius: 8px;
    font-weight: bold;
    color: white;
}
.high { background-color: #e74c3c; }
.medium { background-color: #f39c12; }
.low { background-color: #2ecc71; }
.footer {
    text-align:center;
    color:#777;
    font-size:13px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# HEADER
# =========================
st.markdown("""
<div class="card">
    <h1 class="title">🧠 EquinoxSphere</h1>
    <h4 class="subtitle">AI-Powered Disease Prediction System</h4>
    <p style="text-align:center;">
        Predict diseases from symptoms using deep learning with
        confidence & risk assessment.
    </p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "⚠️ **Medical Disclaimer:** This AI system is for educational and "
    "demonstration purposes only. Always consult a certified doctor."
)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("🧪 Quick Test Cases")

samples = {
    "🤒 Fever / Viral": "fever, headache, body pain",
    "🤢 Stomach Issue": "vomiting, nausea, abdominal pain",
    "🦟 Malaria": "high fever, chills, sweating, headache",
    "🧠 Neurological": "headache, dizziness, confusion",
    "🚨 Emergency": "chest pain, sweating, shortness of breath"
}

case = st.sidebar.selectbox("Select a demo case", list(samples.keys()))

if st.sidebar.button("📥 Load Sample"):
    st.session_state["symptoms"] = samples[case]

st.sidebar.markdown("---")
st.sidebar.caption("EquinoxSphere • Responsible AI")

# =========================
# INPUT CARD
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown("### 🩺 Patient Symptoms")

symptoms = st.text_area(
    "Enter symptoms (comma separated)",
    value=st.session_state.get("symptoms", ""),
    placeholder="e.g. fever, headache, vomiting",
    height=100
)

analyze = st.button("🔍 Analyze Symptoms", use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# PREDICTION
# =========================
if analyze:

    if symptoms.strip() == "":
        st.error("❌ Please enter symptoms before analysis.")

    else:
        with st.spinner("🧠 AI is analyzing symptoms..."):
            try:
                res = requests.post(
                    API_URL,
                    json={"symptoms": symptoms},
                    timeout=20
                )

                if res.status_code == 200:
                    result = res.json()

                    disease = result["Predicted Disease"]
                    confidence = result["Confidence (%)"]
                    risk = result["Risk Level"]

                    # =========================
                    # RESULT CARD
                    # =========================
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("## 📊 Prediction Result")

                    col1, col2 = st.columns(2)
                    col1.metric("🦠 Disease", disease)
                    col2.metric("📈 Confidence", f"{confidence}%")

                    st.progress(int(confidence))

                    # Risk Badge
                    if risk == "HIGH":
                        st.markdown('<span class="badge high">🚨 HIGH RISK</span>', unsafe_allow_html=True)
                    elif risk == "MEDIUM":
                        st.markdown('<span class="badge medium">⚠️ MEDIUM RISK</span>', unsafe_allow_html=True)
                    else:
                        st.markdown('<span class="badge low">✅ LOW RISK</span>', unsafe_allow_html=True)

                    st.markdown('</div>', unsafe_allow_html=True)

                    # =========================
                    # DESCRIPTION
                    # =========================
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### 📖 Disease Description")
                    st.write(result["Description"])
                    st.markdown('</div>', unsafe_allow_html=True)

                    # =========================
                    # PRECAUTIONS
                    # =========================
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("### 🛡️ Recommended Precautions")

                    precautions = result.get("Precautions", [])
                    if precautions:
                        for p in precautions:
                            st.markdown(f"- ✅ **{p.capitalize()}**")
                    else:
                        st.info("No specific precautions available.")

                    st.markdown('</div>', unsafe_allow_html=True)

                    st.caption(result["Medical_Notice"])

                else:
                    st.error("❌ Backend error. API not responding properly.")

            except requests.exceptions.ConnectionError:
                st.error("❌ Unable to connect to AI server.")
            except Exception as e:
                st.error(f"❌ Error: {e}")

# =========================
# FOOTER
# =========================
st.markdown("""
<div class="footer">
© 2026 <b>EquinoxSphere</b> • AI for Health, Trust & Innovation
</div>
""", unsafe_allow_html=True)
