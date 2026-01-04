# =========================
# 1️⃣ IMPORTS
# =========================
import os
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# =========================
# 2️⃣ MODEL PATH (CORRECT)
# =========================
MODEL_DIR = r"E:\clinq\data\2\saved_model"

MODEL_PATH = os.path.join(MODEL_DIR, "disease_lstm_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SUPPORT_PATH = os.path.join(MODEL_DIR, "support_data.pkl")

# Safety checks
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model not found at {MODEL_PATH}")


# =========================
# 3️⃣ LOAD MODEL & FILES
# =========================
model = load_model(MODEL_PATH)

with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

with open(ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

with open(SUPPORT_PATH, "rb") as f:
    support_data = pickle.load(f)

desc_map = support_data["desc_map"]
prec_map = support_data["prec_map"]
severity_map = support_data["severity_map"]


# =========================
# 4️⃣ HIGH-RISK DISEASE LIST
# =========================
HIGH_RISK_DISEASES = {
    "Paralysis (brain hemorrhage)",
    "Heart attack",
    "Brain tumor",
    "Stroke",
    "Tuberculosis",
    "Malaria"
}


# =========================
# 5️⃣ FASTAPI APP
# =========================
app = FastAPI(
    title="AI Disease Prediction API",
    description="AI-based disease prediction using Bi-LSTM (Educational purpose)",
    version="1.0"
)


# =========================
# 6️⃣ REQUEST MODEL
# =========================
class SymptomRequest(BaseModel):
    symptoms: str   # example: "fever, headache, vomiting"


# =========================
# 7️⃣ RISK CALCULATION (IMPROVED)
# =========================
def calculate_risk(symptom_text: str, disease: str, confidence: float):

    symptoms = [
        s.strip().lower().replace(" ", "_")
        for s in symptom_text.split(",")
    ]

    severity_score = sum(severity_map.get(s, 0) for s in symptoms)

    # Rule 1: Serious disease + high confidence
    if disease in HIGH_RISK_DISEASES and confidence >= 70:
        return "HIGH"

    # Rule 2: Symptom severity based
    if severity_score >= 30:
        return "HIGH"
    elif severity_score >= 15:
        return "MEDIUM"

    # Rule 3: High confidence but low severity
    if confidence >= 85:
        return "MEDIUM"

    return "LOW"


# =========================
# 8️⃣ ROOT ENDPOINT
# =========================
@app.get("/")
def home():
    return {"message": "AI Disease Prediction API is running 🚀"}


# =========================
# 9️⃣ PREDICTION ENDPOINT
# =========================
@app.post("/predict")
def predict_disease(data: SymptomRequest):

    # 🔹 Preprocess input (same as training)
    clean_text = " ".join(
        s.strip().lower().replace(" ", "_")
        for s in data.symptoms.split(",")
    )

    seq = tokenizer.texts_to_sequences([clean_text])
    pad = pad_sequences(seq, maxlen=30, padding="post")

    # 🔹 Predict
    probs = model.predict(pad, verbose=0)[0]
    idx = int(np.argmax(probs))

    disease = label_encoder.inverse_transform([idx])[0]
    confidence = round(float(probs[idx]) * 100, 2)

    risk = calculate_risk(data.symptoms, disease, confidence)

    precautions = [
        p for p in prec_map.get(disease, {}).values()
        if isinstance(p, str)
    ]

    return {
        "Predicted Disease": disease,
        "Confidence (%)": confidence,
        "Risk Level": risk,
        "Description": desc_map.get(disease, "N/A"),
        "Precautions": precautions,
        "Medical_Notice": (
            "This is an AI-based prediction for educational purposes only. "
            "It is not a medical diagnosis. Please consult a qualified doctor."
        )
    }
