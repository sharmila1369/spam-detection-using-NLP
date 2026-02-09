from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK data (only once)
nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

# ---------------- FILE PATHS ----------------
TOKENIZER_PATH = "new_tokenizer.pkl"   # use the one you just created
MODEL_PATH = "spam_lstm_model.h5"
MAX_LEN = 100
# --------------------------------------------

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# -------- SAFE CLEAN TEXT FUNCTION (NO ERRORS) --------
def clean_text(text):
    if pd.isna(text) or text is None:
        return ""

    text = text.lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()

    cleaned_words = []
    for w in words:
        try:
            cleaned_words.append(lemmatizer.lemmatize(w))
        except:
            cleaned_words.append(w)  # fallback if wordnet fails

    words = [w for w in cleaned_words if w not in stop_words]
    return " ".join(words)
# -----------------------------------------------------

# -------- LOAD TOKENIZER --------
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

# -------- LOAD MODEL --------
model = load_model(MODEL_PATH)
# -----------------------------------------------------

def predict_spam(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post")
    prob = model.predict(padded)[0][0]
    return "Spam 🚫" if prob > 0.5 else "Not Spam ✅"

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    message = ""

    if request.method == "POST":
        message = request.form["message"]
        prediction = predict_spam(message)

    return render_template("index.html", prediction=prediction, message=message)

if __name__ == "__main__":
    app.run(debug=True)
