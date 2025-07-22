import streamlit as st
import pandas as pd
import pickle

# Load model dan vectorizer
with open("svm_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

st.title("📰 Fake News Detection App")

st.subheader("Pilih metode input:")
mode = st.radio("", ["Input Teks Manual", "Pilih Berita dari Daftar", "Upload File CSV"])

# Metadata input
st.markdown("### Informasi Tambahan (metadata)")
country = st.selectbox("Asal negara berita", ["USA", "UK", "Indonesia", "India", "Other"])
category = st.selectbox("Kategori berita", ["Politics", "Health", "Technology", "Entertainment", "Other"])
st.markdown(f"🗺️ Negara: `{country}` | 📚 Kategori: `{category}`")

# Contoh berita
sample_news = {
    "Biden signs new climate bill": "President Biden signed a new climate bill into law today...",
    "Aliens discovered in Indonesia": "Aliens were found walking around in Jakarta last night...",
    "NASA announces Mars mission": "NASA has confirmed the next manned mission to Mars will launch in 2026..."
}

text = ""

if mode == "Input Teks Manual":
    text = st.text_area("Masukkan teks berita:")

elif mode == "Pilih Berita dari Daftar":
    title = st.selectbox("Pilih judul berita:", list(sample_news.keys()))
    text = sample_news[title]
    st.write("Isi berita:")
    st.info(text)

elif mode == "Upload File CSV":
    uploaded_file = st.file_uploader("Upload file CSV (dengan kolom 'text')", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "text" in df.columns:
            X_batch = vectorizer.transform(df["text"])
            df["prediction"] = model.predict(X_batch)
            st.write("Hasil prediksi:")
            st.dataframe(df[["text", "prediction"]])
        else:
            st.error("Kolom 'text' tidak ditemukan di file.")

if text and mode != "Upload File CSV":
    vec = vectorizer.transform([text])
    prediction = model.predict(vec)[0]
    st.markdown("### Hasil Prediksi:")
    st.success(f"Berita ini terdeteksi sebagai **{prediction.upper()}**")

    # Tambahan info metadata
    st.markdown("### Rincian Input")
    st.write(f"🗺️ Negara: **{country}**")
    st.write(f"📚 Kategori: **{category}**")
