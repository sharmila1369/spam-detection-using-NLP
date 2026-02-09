import pandas as pd
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer

# Load dataset (your downloaded spam.csv)
df = pd.read_csv("spam.csv", sep="\t", names=["label", "message"])

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

df["cleaned"] = df["message"].apply(clean_text)

# Create new tokenizer
MAX_WORDS = 2000
tokenizer = Tokenizer(num_words=MAX_WORDS)
tokenizer.fit_on_texts(df["cleaned"])

# Save new tokenizer
with open("new_tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("✅ New tokenizer created: new_tokenizer.pkl")
