import streamlit as st
import joblib
import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

def clean_text(txt):
    txt = txt.lower()
    txt = txt.translate(str.maketrans("", "", string.punctuation))
    txt = "".join(i for i in txt if not i.isdigit())
    words = [w for w in txt.split() if w not in stop_words]
    return " ".join(words)

model = joblib.load("lr_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

emotions = {
    0: "Sadness 😢",
    1: "Anger 😠",
    2: "Love ❤️",
    3: "Surprise 😲",
    4: "Fear 😨",
    5: "Joy 😄",
}

st.title("Emotion Detector")
text = st.text_area("Enter a sentence:")

if st.button("Analyze"):
    if text.strip():
        cleaned = clean_text(text)
        pred = model.predict(vectorizer.transform([cleaned]))[0]
        st.success(f"Emotion: {emotions[int(pred)]}")
    else:
        st.warning("Please enter some text.")