from pathlib import Path

import joblib
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

try:
    import tensorflow as tf
except Exception:
    tf = None

MODEL_PATH = Path(__file__).with_name("fake_news_model.joblib")
LEGACY_MODEL_PATH = Path(__file__).with_name("fake_news_model.keras")

# Page configuration
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 AI Fake News Detector")
st.write("Enter a news headline or article below to check whether it is **Real** or **Fake**.")


def build_fallback_model():
    real_examples = [
        "Government officials confirmed a new policy after reviewing the latest data from the agency.",
        "The city council approved a public safety plan after a six-month review by local experts.",
        "Scientists published findings from a peer-reviewed study on climate and rainfall changes.",
        "The hospital reported that patient recovery improved after the treatment was updated by medical staff.",
        "Officials said the report was based on verified records and interviews from multiple sources.",
        "An international summit agreed on a new trade framework after months of negotiation.",
        "The police department released bodycam footage and a public statement after the incident.",
        "The university announced a research grant after review by its academic committee.",
        "The company confirmed the merger after filing documents with regulators and shareholders.",
        "The election commission published the final tally after verifying each ballot result.",
    ]
    fake_examples = [
        "Shocking secret revealed: government is hiding alien technology in a hidden bunker.",
        "Doctors say this miracle cure can stop cancer instantly without any tests.",
        "Breaking: celebrity exposed in bizarre scandal that authorities are trying to silence.",
        "A viral video proves the moon landing was filmed in a secret studio.",
        "Local residents claim the mayor is controlling the weather with a secret machine.",
        "Scientists discovered a magical ingredient that cures every disease overnight.",
        "The president announced a secret plan to erase memory using hidden satellites.",
        "An anonymous insider says the whole city is run by extraterrestrial operatives.",
        "A leaked memo reveals that the stock market was manipulated by invisible robots.",
        "Experts warn that a giant reptile is stalking the suburbs and officials are hiding it.",
    ]

    texts = real_examples + fake_examples
    labels = [1] * len(real_examples) + [0] * len(fake_examples)

    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    pipeline.fit(texts, labels)
    joblib.dump(pipeline, MODEL_PATH)
    return pipeline


@st.cache_resource
def load_trained_model():
    if LEGACY_MODEL_PATH.exists() and tf is not None:
        try:
            return tf.keras.models.load_model(str(LEGACY_MODEL_PATH))
        except Exception:
            pass

    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass

    return build_fallback_model()


model = load_trained_model()


def predict_probability(model_obj, text):
    if hasattr(model_obj, "predict_proba"):
        return float(model_obj.predict_proba([text])[0][1])

    if tf is not None and hasattr(model_obj, "predict"):
        try:
            prediction = model_obj.predict(tf.constant([text]), verbose=0)
            return float(prediction[0][0])
        except Exception:
            prediction = model_obj.predict([text])
            return float(prediction[0][0])

    raise ValueError("Unsupported model type. The model must provide predict_proba() or predict().")


# User input fields
headline = st.text_input("News Headline", placeholder="e.g., NASA confirms water on the Moon...")
article = st.text_area("Article Body (Optional)", height=200, placeholder="Paste the full news text here...")

# Predict button
if st.button("Check News", type="primary"):
    full_text = f"{headline} {article}".strip()

    if not full_text:
        st.warning("Please enter a headline or article text first.")
    else:
        prob = predict_probability(model, full_text)

        st.divider()
        if prob > 0.5:
            confidence = prob * 100
            st.success(f"### ✅ Result: REAL NEWS ({confidence:.1f}% confidence)")
        else:
            confidence = (1 - prob) * 100
            st.error(f"### 🚨 Result: FAKE NEWS ({confidence:.1f}% confidence)")

        st.caption(f"Real Score: {prob * 100:.1f}%  |  Fake Score: {(1 - prob) * 100:.1f}%")
        st.progress(prob)