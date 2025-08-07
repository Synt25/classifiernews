import streamlit as st
import joblib

model = joblib.load('pac_model.joblib')
vectorizer = joblib.load('tfidf_vectorizer.joblib')

st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.markdown("<h1 style='text-align:center;'>📰 Fake News Detection</h1>", unsafe_allow_html=True)

# Fungsi deteksi metadata otomatis
def detect_country(text):
    text = text.lower()
    if any(loc in text for loc in ["jakarta", "indonesia"]):
        return "Indonesia"
    elif any(loc in text for loc in ["white house", "biden", "trump", "usa"]):
        return "USA"
    elif any(loc in text for loc in ["london", "britain", "uk"]):
        return "UK"
    elif any(loc in text for loc in ["delhi", "india", "modi"]):
        return "India"
    else:
        return "Other"

def detect_category(text):
    text = text.lower()
    if any(kw in text for kw in ["president", "government", "election", "policy"]):
        return "Politics"
    elif any(kw in text for kw in ["vaccine", "covid", "doctor", "hospital", "virus"]):
        return "Health"
    elif any(kw in text for kw in ["startup", "ai", "software", "technology", "robot"]):
        return "Technology"
    elif any(kw in text for kw in ["movie",]()

