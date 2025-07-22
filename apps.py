import streamlit as st
import pickle

# Load model dan TF-IDF vectorizer
with open("svm_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("tfidf_vectorizer.pkl", "rb") as vec_file:
    vectorizer = pickle.load(vec_file)

# === Tampilan Utama ===
st.set_page_config(page_title="Fake News Detector", page_icon="🧠", layout="centered")
st.markdown(
    "<h1 style='text-align: center; color: #6c5ce7;'>📰 Fake News Detection</h1>",
    unsafe_allow_html=True
)
st.markdown("<p style='text-align: center; color: #a29bfe;'>Deteksi apakah sebuah berita palsu atau valid dengan cepat.</p>", unsafe_allow_html=True)

# === Pilihan Input ===
mode = st.radio("Pilih metode input:", ["📝 Input Teks Manual", "📚 Pilih dari Contoh Berita"])

# === Metadata sederhana ===
st.markdown("#### Informasi Tambahan")
col1, col2 = st.columns(2)
with col1:
    country = st.selectbox("🌍 Negara:", ["USA", "UK", "Indonesia", "India", "Other"])
with col2:
    category = st.selectbox("📖 Kategori:", ["Politics", "Health", "Tech", "Entertainment", "Other"])

# === Inputan Berita ===
sample_news = {
    "Biden signs new climate bill": "President Biden signed a new climate bill into law today...",
    "Aliens discovered in Indonesia": "Aliens were found walking around in Jakarta last night...",
    "NASA announces Mars mission": "NASA confirmed next manned Mars mission in 2026..."
}

text_input = ""

if mode == "📝 Input Teks Manual":
    text_input = st.text_area("Masukkan teks berita:", height=180)
    if st.button("🔍 Prediksi"):
        if text_input.strip():
            vectorized = vectorizer.transform([text_input])
            pred = model.predict(vectorized)[0]
            st.markdown(
                f"<div style='background-color: #ffeaa7; padding: 15px; border-radius: 10px;'>"
                f"<h4>Hasil Prediksi: <span style='color: #d63031'>{pred.upper()}</span></h4></div>",
                unsafe_allow_html=True
            )
        else:
            st.warning("Teks tidak boleh kosong.")

elif mode == "📚 Pilih dari Contoh Berita":
    title = st.selectbox("Pilih judul berita:", list(sample_news.keys()))
    text_input = sample_news[title]
    st.info(text_input)

    if st.button("🔍 Prediksi"):
        vectorized = vectorizer.transform([text_input])
        pred = model.predict(vectorized)[0]
        st.markdown(
            f"<div style='background-color: #a29bfe; padding: 15px; border-radius: 10px;'>"
            f"<h4>Hasil Prediksi: <span style='color: white'>{pred.upper()}</span></h4></div>",
            unsafe_allow_html=True
        )

# === Footer ===
st.markdown("<hr><small style='color:gray;'>Built with ❤️ using Streamlit</small>", unsafe_allow_html=True)
