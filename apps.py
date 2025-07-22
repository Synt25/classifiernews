<import streamlit as st
import pickle

# Load model dan vectorizer
with open("best_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# UI
st.set_page_config(page_title="Fake News Classifier")
st.title("📰 Fake News Detection App")
st.write("Masukkan teks berita di bawah ini untuk diprediksi apakah FAKE atau REAL.")

text = st.text_area("Masukkan teks berita")

if st.button("Prediksi"):
    if not text.strip():
        st.warning("Teks tidak boleh kosong.")
    else:
        vectorized = vectorizer.transform([text])
        prediction = model.predict(vectorized)[0]
        if prediction.lower() == "real":
            st.success("✅ Prediksi: REAL (asli)")
        else:
            st.error("⚠️ Prediksi: FAKE (palsu)")
>