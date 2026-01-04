# =========================
# 1️⃣ IMPORTS
# =========================
import os
import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Bidirectional, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =========================
# 2️⃣ DATASET PATH
# =========================
DATASET_PATH = r"E:\clinq\data\2"


# =========================
# 3️⃣ LOAD DATA
# =========================
df_main = pd.read_csv(os.path.join(DATASET_PATH, "dataset.csv"))
df_precaution = pd.read_csv(os.path.join(DATASET_PATH, "symptom_precaution.csv"))
df_severity = pd.read_csv(os.path.join(DATASET_PATH, "Symptom-severity.csv"))
df_description = pd.read_csv(os.path.join(DATASET_PATH, "symptom_Description.csv"))


# =========================
# 4️⃣ CLEAN & PREPARE DATA
# =========================
df_main.replace("NaN", "none", inplace=True)
df_main.fillna("none", inplace=True)

symptom_cols = [f"Symptom_{i}" for i in range(1, 18)]

df_main["all_symptoms"] = df_main[symptom_cols].apply(
    lambda row: " ".join(
        s.strip().lower().replace(" ", "_")
        for s in row if s != "none"
    ),
    axis=1
)

df_main = df_main[["Disease", "all_symptoms"]]

print("✅ Dataset Ready")
print(df_main.head())


# =========================
# 5️⃣ ENCODE LABELS
# =========================
le = LabelEncoder()
y = le.fit_transform(df_main["Disease"])
X = df_main["all_symptoms"]


# =========================
# 6️⃣ TOKENIZATION
# =========================
tokenizer = Tokenizer(num_words=7000, oov_token="<OOV>")
tokenizer.fit_on_texts(X)

X_seq = tokenizer.texts_to_sequences(X)
X_pad = pad_sequences(X_seq, maxlen=30, padding="post")


# =========================
# 7️⃣ TRAIN–TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X_pad, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# =========================
# 8️⃣ MODEL ARCHITECTURE
# =========================
model = Sequential([
    Embedding(7000, 128, input_length=30),
    Bidirectional(LSTM(64)),
    Dropout(0.4),
    Dense(64, activation="relu"),
    Dense(len(le.classes_), activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =========================
# 9️⃣ TRAIN MODEL
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32,
    callbacks=[early_stop]
)


# =========================
# 🔟 TRAIN & TEST ACCURACY
# =========================
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

print("\n📊 MODEL PERFORMANCE")
print("Train Accuracy:", round(train_acc * 100, 2), "%")
print("Test  Accuracy:", round(test_acc * 100, 2), "%")


# =========================
# 1️⃣1️⃣ SUPPORT MAPS
# =========================
desc_map = dict(zip(
    df_description["Disease"],
    df_description["Description"]
))

prec_map = df_precaution.set_index("Disease").to_dict("index")

severity_map = {
    k.strip().lower(): int(v)
    for k, v in zip(df_severity["Symptom"], df_severity["weight"])
}


# =========================
# 1️⃣2️⃣ RISK CALCULATION
# =========================
def calculate_risk(symptom_text):
    symptoms = [
        s.strip().lower().replace(" ", "_")
        for s in symptom_text.split(",")
    ]

    score = sum(severity_map.get(s, 0) for s in symptoms)

    if score >= 30:
        return "HIGH"
    elif score >= 15:
        return "MEDIUM"
    return "LOW"


# =========================
# 1️⃣3️⃣ AI SYMPTOM CHECKER
# =========================
def ai_symptom_checker(symptom_text):

    clean_text = " ".join(
        s.strip().lower().replace(" ", "_")
        for s in symptom_text.split(",")
    )

    seq = tokenizer.texts_to_sequences([clean_text])
    pad = pad_sequences(seq, maxlen=30, padding="post")

    probs = model.predict(pad, verbose=0)[0]
    idx = int(np.argmax(probs))

    disease = le.inverse_transform([idx])[0]
    confidence = round(float(probs[idx]) * 100, 2)

    precautions = [
        p for p in prec_map.get(disease, {}).values()
        if isinstance(p, str)
    ]

    return {
        "Predicted Disease": disease,
        "Confidence (%)": confidence,
        "Risk Level": calculate_risk(symptom_text),
        "Description": desc_map.get(disease, "N/A"),
        "Precautions": precautions
    }


# =========================
# 1️⃣4️⃣ TEST PREDICTION
# =========================
result = ai_symptom_checker(
    "joint pain, vomiting, fever, headache"
)

print("\n🧠 AI RESULT")
for k, v in result.items():
    print(f"{k}: {v}")


# =========================
# 1️⃣5️⃣ SAVE MODEL & FILES
# =========================
SAVE_DIR = "saved_model"
os.makedirs(SAVE_DIR, exist_ok=True)

# 🔹 Save Keras model
model.save(os.path.join(SAVE_DIR, "disease_lstm_model.keras"))

# 🔹 Save tokenizer & encoder
with open(os.path.join(SAVE_DIR, "tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

with open(os.path.join(SAVE_DIR, "label_encoder.pkl"), "wb") as f:
    pickle.dump(le, f)

# 🔹 Save support data
support_data = {
    "desc_map": desc_map,
    "prec_map": prec_map,
    "severity_map": severity_map
}

with open(os.path.join(SAVE_DIR, "support_data.pkl"), "wb") as f:
    pickle.dump(support_data, f)

print("\n✅ MODEL SAVED SUCCESSFULLY")
print("📁 saved_model/")
