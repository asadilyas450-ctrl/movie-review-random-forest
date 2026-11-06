import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import zipfile
from urllib.request import urlopen
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.base import clone

st.set_page_config(page_title="Movie Review Sentiment Analyzer", page_icon="🎬", layout="wide")

DATA_URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"

@st.cache_data(show_spinner=False)
def load_dataset():
    raw = urlopen(DATA_URL, timeout=30).read()
    zf = zipfile.ZipFile(io.BytesIO(raw))
    frames = []
    for name in zf.namelist():
        if name.lower().endswith("imdb_labelled.txt"):
            df = pd.read_csv(zf.open(name), sep="\t", header=None, names=["review", "label"], encoding="utf-8")
            frames.append(df)
    if not frames:
        raise ValueError("The dataset files were not found in the downloaded archive.")
    data = pd.concat(frames, ignore_index=True)
    data["review"] = data["review"].astype(str).str.strip()
    data = data[data["review"].ne("")]
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data.dropna(subset=["label"])
    data["label"] = data["label"].astype(int).map({0: "Negative", 1: "Positive"})
    return data.dropna().reset_index(drop=True)

@st.cache_resource(show_spinner=False)
def train_model():
    data = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        data["review"], data["label"], test_size=0.20, random_state=42, stratify=data["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=25000, sublinear_tf=True)),
        ("rf", RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample", n_jobs=-1, max_features="sqrt"))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    matrix = confusion_matrix(y_test, predictions, labels=["Negative", "Positive"])
    return data, model, accuracy, report, matrix, len(X_train), len(X_test)

st.title("🎬 Movie Review Sentiment Analyzer")
st.write("A **Random Forest** text-classification project using **TF-IDF** features and labelled IMDb movie-review sentences from the UCI dataset.")
st.info("The model predicts whether a review expresses a positive or negative sentiment. The test set is held out from training.")

with st.spinner("Downloading the dataset and training the Random Forest model..."):
    try:
        data, model, accuracy, report, matrix, n_train, n_test = train_model()
    except Exception as e:
        st.error(f"Could not load or train the model: {e}")
        st.stop()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Test accuracy", f"{accuracy * 100:.2f}%")
m2.metric("Total reviews", f"{len(data):,}")
m3.metric("Training reviews", f"{n_train:,}")
m4.metric("Test reviews", f"{n_test:,}")

st.subheader("Model performance")
left, right = st.columns(2)
with left:
    st.write("**Classification report**")
    scores = pd.DataFrame(report).transpose().round(3)
    st.dataframe(scores, use_container_width=True)
with right:
    st.write("**Confusion matrix**")
    cm = pd.DataFrame(matrix, index=["Actual Negative", "Actual Positive"], columns=["Predicted Negative", "Predicted Positive"])
    st.dataframe(cm, use_container_width=True)

st.divider()
st.subheader("Try your own review")
review = st.text_area("Enter a movie review", placeholder="The story was engaging and the acting was excellent...", height=120)
if st.button("Analyze review", type="primary", disabled=not review.strip()):
    prediction = model.predict([review])[0]
    probabilities = model.predict_proba([review])[0]
    classes = model.named_steps["rf"].classes_
    confidence = float(probabilities[list(classes).index(prediction)])
    if prediction == "Positive":
        st.success(f"Prediction: {prediction}  |  Model confidence: {confidence:.1%}")
    else:
        st.error(f"Prediction: {prediction}  |  Model confidence: {confidence:.1%}")

st.caption("Dataset: UCI Machine Learning Repository — Sentiment Labelled Sentences. Split: 80% train / 20% test, stratified, random_state=42. Accuracy is computed live when the app trains.")
