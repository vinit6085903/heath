import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =========================
# BASE DIRECTORY
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "saved_model")

MODEL_PATH = os.path.join(MODEL_DIR, "disease_lstm_model.keras")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.pkl")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
SUPPORT_PATH = os.path.join(MODEL_DIR, "support_data.pkl")

# =========================
# SAFETY CHECK
# =========================
for file in [MODEL_PATH, TOKENIZER_PATH, ENCODER_PATH, SUPPORT_PATH]:
    if not os.path.exists(file):
        raise FileNotFoundError(f"Missing file: {file}")

# =========================
# LOAD MODEL & FILES
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
# HIGH RISK DISEASES
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
# FLASK APP
# =========================
app = Flask(__name__)

# =========================
# RISK CALCULATION
# =========================
def calculate_risk(symptom_text, disease, confidence):
    symptoms = [
        s.strip().lower().replace(" ", "_")
        for s in symptom_text.split(",")
    ]

    severity_score = sum(severity_map.get(s, 0) for s in symptoms)

    if disease in HIGH_RISK_DISEASES and confidence >= 70:
        return "HIGH"
    if severity_score >= 30:
        return "HIGH"
    elif severity_score >= 15:
        return "MEDIUM"
    if confidence >= 85:
        return "MEDIUM"
    return "LOW"

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return render_template("home.html")

@app.route("/favicon.ico")
def favicon():
    return "", 204

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    symptoms_text = data.get("symptoms", "")

    clean_text = " ".join(
        s.strip().lower().replace(" ", "_")
        for s in symptoms_text.split(",")
    )

    seq = tokenizer.texts_to_sequences([clean_text])
    pad = pad_sequences(seq, maxlen=30, padding="post")

    probs = model.predict(pad, verbose=0)[0]
    idx = int(np.argmax(probs))

    disease = label_encoder.inverse_transform([idx])[0]
    confidence = round(float(probs[idx]) * 100, 2)

    risk = calculate_risk(symptoms_text, disease, confidence)

    precautions = [
        p for p in prec_map.get(disease, {}).values()
        if isinstance(p, str)
    ]

    return jsonify({
        "disease": disease,
        "confidence": confidence,
        "risk": risk,
        "description": desc_map.get(disease, "N/A"),
        "precautions": precautions,
        "notice": "AI prediction only. Consult a doctor."
    })

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/how-it-works")
def how_it_works():
    return render_template("how_it_works.html")

@app.route("/research")
def research():
    return render_template("research.html")

@app.route("/contact", methods=["GET", "POST"])
def contact():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")
        message = request.form.get("message")

        # Abhi ke liye sirf console me print
        print("Contact Form Data:")
        print("Name:", name)
        print("Email:", email)
        print("Message:", message)

        return jsonify({
            "status": "success",
            "message": "Thank you for contacting us!"
        })

    return render_template("contact.html")


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    app.run(debug=True)
